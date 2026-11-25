"""Send checkpointed transfer metrics to W&B while the Colab train command runs."""

import argparse
from bisect import bisect_right
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from tempfile import TemporaryDirectory, gettempdir
from threading import Event, Thread
from traceback import format_exception
from urllib.parse import quote, quote_plus
from uuid import uuid4
from zipfile import ZipFile

# Keep the W&B runtime independent of the scientific Python environment.
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
from alignment_transfer_s2ut.process_logging import graceful_sigterm, log_event, run_process
from transfer_bleu import CHECKPOINT, PLAN, SHA256, baseline_dev_hash, evaluation_scope, publishable, result_hash
import transfer_metrics_v3 as v3
from wandb_history import DEV_UPDATE, DEV_WINDOW, LEGACY_TRAIN, TRAIN_UPDATE, TRAIN_WINDOW, number, reconcile_history
from wandb_writer import WriterBusy, shared_writer


REPORTER_SESSION = "wandb_reporter_session"
WRITER_BUCKET = "wandb_writer_lock_bucket"
WRITER_TOKEN = "wandb_writer_lock_token"
WRITER_CONTRACT = "wandb_writer_contract_hash"
IDLE_STATES = {"finished", "failed", "crashed", "killed"}
LEGACY_LOGICAL_TRAIN = {"train/learning_rate", "train/seconds"}
TRAIN_JSONL = "alignment-train.jsonl"
DEV_JSONL = "native-dev.jsonl"
EVAL_PROGRESS = "wandb_eval_update_v3"
GAP_OBSERVATION_SECONDS = 600


class WriterConfigurationError(RuntimeError):
    """A writer requires an operator to correct its configuration or ownership."""


class MetricReader:
    """Read complete JSONL rows once, including when dev logs are replaced atomically."""

    def __init__(self, path):
        self.path = path
        self.identity = None
        self.offset = 0

    def rows(self, maximum_update):
        try:
            source = self.path.open("rb")
        except FileNotFoundError:
            return
        with source:
            stat = os.fstat(source.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or self.offset > stat.st_size:
                self.identity, self.offset = identity, 0
            source.seek(self.offset)
            for line in source:
                if not line.endswith(b"\n"):
                    break
                if not line.strip():
                    self.offset += len(line)
                    continue
                try:
                    row = json.loads(line)
                    step = row["step"]
                except (ValueError, KeyError, TypeError):
                    self.offset += len(line)
                    continue
                if type(step) is not int or step < 1:
                    self.offset += len(line)
                    continue
                if step > maximum_update:
                    break
                self.offset += len(line)
                yield row


def training_command(command, action="train"):
    """Find and normalize the native trainer argv inside the notebook wrappers."""
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        return None
    for index in range(1, len(command) - 2):
        if command[index:index + 3] != ["-m", "alignment_transfer_s2ut.benchmark", action]:
            continue
        args = command[index - 1:]
        args[3] = "train"
        for position, arg in enumerate(args):
            if arg.partition("=")[0] == "--stop-after-update":
                del args[position:position + (1 if "=" in arg else 2)]
                break
        return args
    return None


PROGRESS = re.compile(
    rb"(?:^INFO:(?P<colab>train_inner|dev):\s*|(?:^| \| )(?P<pipe>train_inner|dev) \| )"
    rb"(?P<record>\{.*\})\s*$"
)
DURABLE = re.compile(rb"^\[recovery\] learned_at_least=\d+ durable_updates=(\d+)\b")
TRAIN_FIELDS = {
    "loss": "train/loss_fairseq", "nll_loss": "train/nll_loss", "ppl": "train/perplexity",
    "loss_scale": "train/loss_scale", "clip": "train/clip_fraction", "oom": "train/oom",
    "gnorm": "train/gradient_norm",
    "train/main_loss": "train/main_loss_fairseq",
    "train/unit_nll_nats": "train/unit_nll_nats",
    "train/optimization_loss_nats": "train/optimization_loss_nats",
}


def fairseq_row(line):
    match = PROGRESS.search(line)
    if not match:
        return None
    try:
        record = json.loads(match["record"])
    except (ValueError, UnicodeError):
        return None
    is_dev = (match["colab"] or match["pipe"]) == b"dev"
    # Fairseq's "update" is a fractional epoch. Its optimizer counter is num_updates.
    step = number(record.get("dev_num_updates" if is_dev else "num_updates"))
    if step is None or step <= 0 or step != int(step):
        return None
    if not is_dev and step != 1 and step % 100:
        return None
    values = {}
    for key, raw in record.items():
        if is_dev:
            name = key.removeprefix("dev_")
            if name == "train/main_loss":
                target = "dev/main_loss_fairseq"
            elif name in {"loss", "nll_loss", "ppl"} or name.startswith("multitask_"):
                target = "dev/" + name
            else:
                continue
        else:
            target = key if key.startswith("alignment/") else TRAIN_FIELDS.get(key)
            if key.startswith("multitask_"):
                target = "train/" + key
        value = number(raw)
        if target and value is not None:
            values[target] = value
    return (int(step), values) if values else None


class FairseqProgressReader:
    """Recover diagnostics only from checkpointed portions of matching process logs."""

    def __init__(self, logs, *, compare_attempts=False):
        self.logs = Path(logs).resolve()
        self.offsets = {}
        self.compare_attempts = compare_attempts

    def sources(self, expected, contract):
        plans = {}
        sources = {}
        for receipt in self.logs.glob("*.json"):
            try:
                record = json.loads(receipt.read_text(encoding="utf-8"))
                started = record["started_utc"]
                if type(started) is not str:
                    continue
                command = record.get("command")
                if training_command(command, "plan") == expected:
                    plans[started] = None
                    if record.get("status") == "succeeded":
                        output = Path(record["stdout_log"]).resolve()
                        if output.is_relative_to(self.logs):
                            plans[started] = json.loads(output.read_text(encoding="utf-8")).get("contract_hash")
                if training_command(command) != expected:
                    continue
                path = Path(record["log"]).resolve()
                if path.suffix == ".log" and path.is_relative_to(self.logs) and path.is_file():
                    sources[path] = started
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        timeline = sorted(plans)
        for path in sorted(sources, key=lambda path: (sources[path], path.name)):
            position = bisect_right(timeline, sources[path]) - 1
            if position >= 0 and plans[timeline[position]] == contract:
                yield path

    def rows(self, expected, contract, maximum_update):
        points = {}
        for path in self.sources(expected, contract):
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                old_identity, offset, pending, durable = self.offsets.get(path, (identity, 0, {}, 0))
                if old_identity != identity or offset > stat.st_size:
                    offset, pending, durable = 0, {}, 0
                stream.seek(offset)
                for line in stream:
                    if not line.endswith(b"\n"):
                        break
                    if match := DURABLE.match(line):
                        durable = max(durable, int(match[1]))
                    elif row := fairseq_row(line):
                        step, values = row
                        pending.setdefault(step, {}).update(values)
                    offset += len(line)
                limit = min(durable, maximum_update)
                for step in sorted(pending):
                    if step > limit:
                        break
                    values = pending.pop(step)
                    previous = points.setdefault(step, {})
                    if self.compare_attempts:
                        for key in previous.keys() & values.keys():
                            if previous[key] != values[key]:
                                raise ValueError(f"Conflicting local Fairseq metric {key} at update {step}")
                    previous.update(values)
                self.offsets[path] = (identity, offset, pending, durable)
        yield from sorted(points.items())


def published_snapshot(work, backup, plan_sha, minimum, method, *, exact=None, include_archive=False):
    """Read publication receipts without opening checkpoint archives."""
    folders = backup.glob(f"{exact:012d}-*") if exact is not None else backup.glob("*")
    for folder in sorted(folders, reverse=True):
        prefix = folder.name.split("-", 1)[0]
        if not prefix.isascii() or not prefix.isdecimal():
            continue
        update = int(prefix)
        if update <= minimum:
            break
        if exact is not None and update != exact:
            continue
        marker = folder / "snapshot.json"
        if not marker.is_file() or not (folder / "snapshot.zip").is_file():
            continue
        try:
            state = json.loads(marker.read_bytes())
            identity, files = state["identity"], state["files"]
            command = training_command(identity["command"])
            if (state["updates"] == update and state.get("storage") == "zip-v1"
                    and identity["work"] == str(work) and command is not None
                    and command[command.index("--output") + 1] == str(work)
                    and command[command.index("--method") + 1] == method
                    and CHECKPOINT in files and PLAN in files
                    and (SHA256.fullmatch(files[PLAN]) if plan_sha is None else files[PLAN] == plan_sha)):
                match = update, command, files
                return (*match, marker.parent / "snapshot.zip") if include_archive else match
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            continue
    return None


def saved_bleu_points(args, plan_sha, maximum, *, contract=None):
    points = {}
    for path in args.bleu_results.glob(f"{args.method}/*/*/result.json"):
        row = json.loads(path.read_bytes())
        if row["method"] != args.method or (contract and row["contract_hash"] != contract) or not publishable(row):
            continue
        update = row["update"]
        if type(update) is not int or not 0 < update <= maximum:
            continue
        scope = evaluation_scope(plan_sha, row["policy"], row["dev_sha256"], row.get("failure_policy"))
        if path.parent.name != f"{update:09d}" or path.parent.parent.name != scope:
            continue
        if row["result_hash"] != result_hash(row) or not SHA256.fullmatch(row["checkpoint_sha256"]):
            raise ValueError(f"Corrupt saved BLEU result at update {update}")
        exact = published_snapshot(args.work.resolve(), args.backup, plan_sha, update - 1, args.method, exact=update)
        if exact and exact[2][CHECKPOINT] != row["checkpoint_sha256"]:
            raise ValueError(f"Saved BLEU checkpoint differs from the receipt at update {update}")
        if v3.enabled(args):
            if not exact or update not in v3.MILESTONES:
                raise ValueError(f"Schema v3 BLEU requires its exact milestone receipt at {update}")
            v3.check_bleu(row)
        if update in points and points[update] != row:
            raise ValueError(f"Conflicting BLEU results at update {update}")
        points[update] = row
    return sorted(points.items())


def sync_bleu(run, points, sent, *, schema_v3=False):
    written = 0
    for update, row in points:
        marker = f"wandb_bleu_checkpoint_{update}"
        checkpoint = row["checkpoint_sha256"]
        previous = sent.get(update) or run.summary.get(marker)
        if previous == checkpoint:
            continue
        if previous is not None:
            raise ValueError(f"W&B already contains another BLEU checkpoint at update {update}")
        names = ("aux_bleu", "asr_bleu", "aux_failures", "asr_failures")
        if schema_v3:
            names += ("aux_samples", "asr_samples")
        values = {f"eval/{name}": row[name] for name in names}
        # W&B's internal step must increase. Only the chart coordinate may go backward.
        run.log({**values, "update": update, marker: checkpoint})
        sent[update] = checkpoint
        written += 1
    return written


def run_id(contract):
    return hashlib.sha256(contract.encode()).hexdigest()[:24]


def writer_path(contract, suffix):
    return Path(gettempdir()) / "alignment-wandb-writers" / (run_id(contract) + suffix)


def writer_lock(contract):
    from filelock import FileLock

    path = writer_path(contract, ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path), timeout=0)


def writer_identity(args, contract, entity):
    return dict(entity=entity, project=args.project, run_id=run_id(v3.namespace(contract, args)),
                contract_hash=contract, method=args.method)


def check_target(target, args, identity, *, require_group=False):
    if (target.id != identity["run_id"] or target.entity != identity["entity"] or target.project != identity["project"]
            or target.config.get("condition") != identity["method"]
            or target.config.get("alignment_contract_hash", identity["contract_hash"]) != identity["contract_hash"]
            or target.summary.get(WRITER_CONTRACT, identity["contract_hash"]) != identity["contract_hash"]
            or require_group and getattr(target, "group", None) != args.group):
        raise WriterConfigurationError("Existing W&B run identity does not match the verified training experiment")
    if v3.enabled(args):
        binding = v3.binding(identity["contract_hash"], args)
        if any(target.config.get(key) != value for key, value in binding.items()):
            raise WriterConfigurationError("Existing W&B run has a different Schema v3 binding")
    elif target.config.get("metric_schema") == 3:
        raise WriterConfigurationError("A Schema v3 W&B run cannot accept legacy metrics")
    if target.summary.get(WRITER_BUCKET) not in (None, getattr(args, "writer_bucket", None)):
        raise WriterConfigurationError("Set the shared writer bucket already bound to this W&B run")
    if target.state not in IDLE_STATES:
        raise WriterBusy("W&B run is active; wait for the previous writer to finish")


def writer_state(target):
    if target is not None:
        return tuple(target.summary.get(key) for key in (WRITER_BUCKET, WRITER_TOKEN, REPORTER_SESSION))


def verified_target(wandb, args, identity, *, require_group):
    target = wandb.Api(timeout=20).run(f"{identity['entity']}/{identity['project']}/{identity['run_id']}")
    check_target(target, args, identity, require_group=require_group)
    return target


def run_not_found(wandb, error):
    missing = wandb.apis.public.RunNotFoundError
    return isinstance(error, missing) or isinstance(getattr(error, "exc", None), missing)


def live_target(wandb, args, contract):
    api = wandb.Api(timeout=20)
    entity = args.entity or api.default_entity
    if not entity:
        raise WriterConfigurationError("Set the W&B entity for this reporter")
    identity = writer_identity(args, contract, entity)
    try:
        target = api.run(f"{identity['entity']}/{args.project}/{identity['run_id']}")
    except (wandb.apis.public.RunNotFoundError, wandb.errors.CommError) as error:
        if not run_not_found(wandb, error):
            raise
        return identity, None
    check_target(target, args, identity, require_group=bool(args.group))
    return identity, target


def store_writer_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def store_owner(contract, owner):
    previous = read_owner(contract) or {}
    if "history_step" in previous:
        owner = {**owner, "history_step": max(previous["history_step"], owner.get("history_step", -1))}
    store_writer_record(writer_path(contract, ".owner.json"), owner)


def read_owner(contract):
    try:
        return json.loads(writer_path(contract, ".owner.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def verify_owner_destination(owner, entity, args):
    if entity != owner["entity"] or args.project != owner["project"] or args.method != owner["method"]:
        raise WriterConfigurationError("W&B destination differs from the last training reporter in this runtime")


def check_pending_writer(contract):
    if writer_path(contract, ".pending.json").exists():
        raise WriterConfigurationError("A previous W&B init or finish was unconfirmed; stop that writer and use --recover-writer")


def start_writer(wandb, args, identity, lease, resume):
    contract = identity["contract_hash"]
    store_writer_record(writer_path(v3.namespace(contract, args), ".pending.json"), {**identity, "session": lease.token})
    lease.opening()
    return open_run(wandb, args, contract, entity=identity["entity"], resume=resume)


def finish_writer(contract, lease, run, *, track_delivery=False):
    path = writer_path(contract, ".pending.json")
    cursor = getattr(run, "step", None) if track_delivery else None
    if type(cursor) is int:
        receipt = {**json.loads(path.read_bytes()), "history_step": cursor - 1}
        store_writer_record(path, receipt)
    lease.finish(run)
    if type(cursor) is int:
        store_owner(contract, receipt)
    path.unlink(missing_ok=True)


def v3_history_floor(args, contract):
    key = v3.namespace(contract, args)
    owner = read_owner(key) or {}
    try:
        pending = json.loads(writer_path(key, ".pending.json").read_bytes())
    except FileNotFoundError:
        pending = {}
    return max(owner.get("history_step", -1), pending.get("history_step", -1))


def check_v3_resume_cursor(run, audit):
    # The SDK's resume cursor is independent of the Public API history scan.
    last = audit["history_step"]
    if last is not None and run.step != last + 1:
        raise WriterBusy("ACK_PENDING: the SDK resume cursor differs from the W&B history scan; run a read-only audit")


def bind_writer(run, bucket, lease, contract):
    if bucket:
        run.summary[WRITER_BUCKET] = bucket
        run.summary[WRITER_TOKEN] = lease.token
        run.summary[WRITER_CONTRACT] = contract


@contextmanager
def sync_writer(wandb, args, contract):
    bucket = getattr(args, "writer_bucket", None)
    recover = getattr(args, "recover_writer", False)
    generation = getattr(args, "recover_lock_generation", None)
    key = v3.namespace(contract, args)
    with writer_lock(key):
        owner = None
        if recover:
            if not args.entity or not args.group:
                raise RuntimeError("Writer recovery requires the existing --entity and --group")
            if not bucket and not getattr(args, "confirm_writers_stopped", False):
                raise RuntimeError("Without GCS, stop all writers and pass --confirm-writers-stopped")
            if generation is not None and not bucket:
                raise RuntimeError("Recovering a lock generation requires --writer-bucket")
            entity = args.entity
        else:
            check_pending_writer(key)
            owner = read_owner(key)
            if owner is None:
                raise RuntimeError("No local W&B ownership receipt; see the --recover-writer instructions")
            entity = owner["entity"]
            if not entity:
                raise WriterConfigurationError("The local W&B ownership receipt has no entity")
            verify_owner_destination(owner, args.entity or entity, args)
        identity = writer_identity(args, contract, entity)

        def get_target():
            try:
                current = verified_target(wandb, args, identity, require_group=recover or bool(getattr(args, "group", None)))
            except (wandb.apis.public.RunNotFoundError, wandb.errors.CommError) as error:
                if (v3.enabled(args) and recover and getattr(args, "confirm_writers_stopped", False)
                        and read_owner(key) is None and run_not_found(wandb, error)):
                    return None
                raise
            if owner is not None and current.summary.get(REPORTER_SESSION) != owner["session"]:
                raise RuntimeError("Another training reporter owns this W&B run. Leave the scores on Drive for that reporter.")
            return current

        target = get_target()
        state = writer_state(target)
        bound_bucket, token, session = state or (None, None, None)
        with shared_writer(bucket, identity, previous_token=token, recover_generation=generation) as lease:
            def recheck():
                if writer_state(get_target()) != state:
                    raise RuntimeError("Another W&B writer connected during recovery")

            def start_run(check=None):
                current = get_target()
                if writer_state(current) != state:
                    raise RuntimeError("Another W&B writer connected during recovery")
                if check is not None:
                    check(current)
                    recheck()  # A scan can take long enough for another VM to reconnect.
                run = start_writer(wandb, args, identity, lease, "must" if target is not None else "never")
                bind_writer(run, bucket, lease, contract)
                if recover or v3.enabled(args):
                    run.summary[REPORTER_SESSION] = lease.token
                return run

            yield target, start_run, lease
            if recover and not lease.unfinished:
                if not lease.opened:
                    if not isinstance(session, str) or not session or bucket != bound_bucket:
                        # Establish a remote session or first-time bucket binding without
                        # writing a history row.
                        finish_writer(key, lease, start_run())
                    else:
                        recheck()
                store_owner(key, dict(entity=entity, project=args.project, method=args.method,
                                      session=lease.token if lease.opened else session))
                writer_path(key, ".pending.json").unlink(missing_ok=True)
                log_event("wandb-writer", "recovered", args.logs, identity["run_id"])


def open_run(wandb, args, contract, *, entity=None, resume="allow"):
    identifier = run_id(v3.namespace(contract, args))
    options = {}
    if resume in ("allow", "never"):
        entity = entity or args.entity or None
        options = {"group": args.group, "name": args.method, "config": {"condition": args.method}}
        if v3.enabled(args):
            options["config"] = v3.binding(contract, args)
    if resume == "must" or v3.enabled(args) or getattr(args, "writer_bucket", None):
        options["mode"] = "online"
    run = wandb.init(project=args.project, entity=entity, id=identifier, resume=resume,
                     dir=str(args.logs.resolve()), save_code=False, **options,
                     settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                             x_disable_meta=True, x_disable_stats=True,
                                             x_service_wait=10, init_timeout=20, finish_timeout=20,
                                             finish_timeout_raises=True))
    # Historical rows still carry "update", but a late BLEU row must not
    # replace the latest training update in the Run Summary.
    run.define_metric("update", summary="none", overwrite=True)
    for prefix in ("train", "alignment", "dev", "eval"):
        run.define_metric(prefix + "/*", step_metric="update")
    log_event("wandb", "connected", args.logs, run.url or "Offline run")
    return run


def set_training_progress(run, published, *, schema_v3=False):
    name = "train/max_update"
    previous = number(run.summary.get(name)) or 0
    training = number(run.summary.get(LEGACY_TRAIN)) or 0
    maximum = int(max(published, previous, training, number(run.summary.get(TRAIN_UPDATE)) or 0))
    if maximum != previous:
        run.summary[name] = maximum
    if schema_v3:
        run.summary["train/target_update"] = v3.MAX_UPDATE
        run.summary["train/durable_update"] = max(number(run.summary.get("train/durable_update")) or 0, published)


def sync_saved_bleu(args):
    if v3.enabled(args):
        return sync_saved_metrics(args)
    import wandb

    snapshot = published_snapshot(args.work.resolve(), args.backup, None, 0, args.method)
    if snapshot is None:
        log_event("wandb-bleu", "skipped", args.logs, "No published training snapshot")
        return
    points = saved_bleu_points(args, snapshot[2][PLAN], snapshot[0], contract=args.contract)
    if not points:
        log_event("wandb-bleu", "skipped", args.logs, "No eligible saved BLEU results")
        return
    contracts = {row["contract_hash"] for _, row in points}
    if len(contracts) != 1:
        raise ValueError("Saved BLEU results must belong to a single training contract")
    contract = contracts.pop()
    with sync_writer(wandb, args, contract) as (_, start_run, lease):
        run = start_run()
        try:
            set_training_progress(run, snapshot[0])
            written = sync_bleu(run, points, {})
            log_event("wandb-bleu", "synced", args.logs, f"{written} new checkpoint results")
        finally:
            finish_writer(contract, lease, run)


@contextmanager
def metric_snapshot(args, *, plan_only=False):
    """Copy and verify only the archived plan and declared metric files."""
    snapshot = published_snapshot(args.work.resolve(), args.backup, None, 0, args.method, include_archive=True)
    if snapshot is None:
        raise ValueError("No matching published checkpoint for metric backfill")
    published, command, files, archive = snapshot
    if not isinstance(files[CHECKPOINT], str) or SHA256.fullmatch(files[CHECKPOINT]) is None:
        raise ValueError("Invalid published SHA-256 for the checkpoint")
    members = (PLAN,) if plan_only else (PLAN, *(name for name in (TRAIN_JSONL, DEV_JSONL) if name in files))
    with TemporaryDirectory(prefix="wandb-metrics-") as directory:
        local = Path(directory)
        with ZipFile(archive) as source:
            for name in members:
                if not isinstance(files[name], str) or SHA256.fullmatch(files[name]) is None:
                    raise ValueError(f"Invalid published SHA-256 for {name}")
                entries = [entry for entry in source.infolist() if entry.filename == name]
                if len(entries) != 1 or entries[0].is_dir():
                    raise ValueError(f"Expected one archived member for {name}")
                digest = hashlib.sha256()
                with source.open(entries[0]) as stream, (local / name).open("xb") as target:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        target.write(block)
                if digest.hexdigest() != files[name]:
                    raise ValueError(f"Archived {name} SHA-256 differs from the published receipt")
        plan = json.loads((local / PLAN).read_bytes())
        contract = plan.get("contract_hash")
        if (not isinstance(contract, str) or SHA256.fullmatch(contract) is None
                or getattr(args, "contract", None) not in (None, contract)
                or plan.get("transfer", {}).get("method") != args.method):
            raise ValueError("Archived metric plan does not match the expected training contract")
        yield contract, published, command, local, members


def audit_saved_metrics(args):
    legacy, unit_ce_updates = {}, set()
    with metric_snapshot(args) as (contract, published, command, local, members):
        for row in MetricReader(local / TRAIN_JSONL).rows(published):
            if row.get("alignment_contract_hash", contract) != contract:
                continue
            step = row["step"]
            if row.get("metric_schema") == 2 and number(row.get("train/unit_ce_nats")) is not None:
                unit_ce_updates.add(step)
            value = number(row.get("train/main_loss"))
            if row.get("metric_schema") is None and value is not None:
                legacy[step] = value
        dev_rows = sum(1 for _ in MetricReader(local / DEV_JSONL).rows(published))
    window = dict(FairseqProgressReader(args.logs).rows(command, contract, published))
    paired = [{"update": step, "legacy_update_main_loss_nats": legacy[step],
               "window_unit_ce_nats": row.get("train/main_loss_fairseq"),
               "window_unit_nll_nats": row["train/nll_loss"] * math.log(2)}
              for step, row in window.items() if step in legacy and "train/nll_loss" in row]
    print(json.dumps({"run_id": run_id(contract), "published_update": published,
                      "verified_snapshot_members": members,
                      "verified_dev_rows": dev_rows,
                      "legacy_main_loss_rows": len(legacy),
                      "legacy_main_loss_min": min(legacy.values(), default=None),
                      "verified_update_unit_ce_rows": len(unit_ce_updates),
                      "window_unit_ce_rows": sum("train/main_loss_fairseq" in row for row in window.values()),
                      "paired_update_and_window_rows": len(paired),
                      "examples": paired[-10:]}, allow_nan=False))


def v3_bleu_rows(args, plan_sha, maximum, contract):
    if not args.bleu_results:
        return []
    points = saved_bleu_points(args, plan_sha, maximum, contract=contract)
    if points:
        dev_sha = baseline_dev_hash(args.backup.parent.parent / "baseline", args.work.resolve(), v3.MAX_UPDATE)
        if any(row["dev_sha256"] != dev_sha for _, row in points):
            raise ValueError("Schema v3 BLEU Dev references do not match the registered Baseline")
    return [(EVAL_PROGRESS, update, {key: row[key.removeprefix("eval/")] for key in v3.BLEU})
            for update, row in points]


def v3_source_rows(args, plan_sha, maximum, contract, local, *, tau, dev_reference, allow_unpublished=False,
                   originals_only=False):
    train, train_status = v3.coverage(local / TRAIN_JSONL, maximum, args.method, contract, "train",
                                      tau=tau, allow_unpublished=allow_unpublished)
    dev, dev_status = v3.coverage(local / DEV_JSONL, maximum, args.method, contract, "dev",
                                  tau=tau, dev_reference=dev_reference, allow_unpublished=allow_unpublished)
    rows = [(TRAIN_UPDATE, update, values) for update, values in train.items()]
    rows.extend((DEV_UPDATE, update, values) for update, values in dev.items())
    if originals_only:
        return rows, [], {"train": train_status, "dev": dev_status}
    bleu = v3_bleu_rows(args, plan_sha, maximum, contract)
    scored = {update for _, update, _ in bleu}
    failed = set()
    if args.bleu_results:
        for path in args.bleu_results.glob(f"{args.method}/*/*/failed-*.json"):
            row = json.loads(path.read_bytes())
            update = row.get("update")
            if update in v3.MILESTONES and update <= maximum and row.get("status") == "failed":
                exact = published_snapshot(args.work.resolve(), args.backup, plan_sha, update - 1, args.method, exact=update)
                if exact and exact[2][CHECKPOINT] == row.get("checkpoint_sha256"):
                    failed.add(update)
    scheduled = {update for update in v3.MILESTONES if update <= maximum}
    reports = {"train": train_status, "dev": dev_status,
               "bleu": {"completed": sorted(scored), "failed": sorted((scheduled - scored) & failed),
                        "waiting": sorted(scheduled - scored - failed), "complete": scored == scheduled}}
    return rows, bleu, reports


def saved_metric_sources(args, *, compare_attempts=False):
    with metric_snapshot(args) as (contract, published, command, local, members):
        snapshot = list(logical_metric_rows(MetricReader(local / TRAIN_JSONL), MetricReader(local / DEV_JSONL),
                                           published, contract=contract))
    window = fairseq_metric_rows(FairseqProgressReader(args.logs, compare_attempts=compare_attempts),
                                 command, contract, published)
    return contract, published, members, {"snapshot_jsonl": snapshot, "local_fairseq": window}


def verified_v3_sources(args, audit_scope="snapshot", *, originals_only=False):
    with metric_snapshot(args) as (contract, published, command, local, members):
        if published > v3.MAX_UPDATE:
            raise ValueError("Schema v3 snapshot exceeds the registered budget")
        plan_bytes = (local / PLAN).read_bytes()
        plan_sha = hashlib.sha256(plan_bytes).hexdigest()
        plan = json.loads(plan_bytes)
        tau = v3.plan_tau(plan)
        source = args.work if audit_scope == "local" else local
        if audit_scope == "local" and (not (source / PLAN).is_file()
                or hashlib.sha256((source / PLAN).read_bytes()).hexdigest() != plan_sha):
            raise ValueError("Local metric plan differs from the published snapshot")
        metrics, bleu, reports = v3_source_rows(args, plan_sha, published, contract, source, tau=tau,
            dev_reference=v3.plan_dev_reference(plan), allow_unpublished=audit_scope == "local", originals_only=originals_only)
    windows = () if originals_only else fairseq_metric_rows(
        FairseqProgressReader(args.logs, compare_attempts=True), command, contract, published)
    return contract, published, members, {"original_jsonl": metrics, "local_fairseq": windows,
                                         "qualified_bleu": bleu}, reports


def verify_v3_plan(args):
    old = argparse.Namespace(work=args.previous_work, backup=args.previous_backup, method=args.method, contract=None)
    if not v3.scope(args.method, old.work, old.backup, v3.OLD_SCOPE):
        raise ValueError("Schema v3 preflight requires the original transfer-fixed-v1 stage")
    with metric_snapshot(old, plan_only=True) as (_, _, _, local, _):
        evidence = v3.scientific_plan(json.loads((local / PLAN).read_bytes()),
                                      json.loads(args.verify_v3_plan.read_bytes()))
    print(json.dumps(evidence))


def gap_sources(args, contract, update, sources):
    snapshot = published_snapshot(args.work.resolve(), args.backup, None, update - 1, args.method, exact=update)
    if snapshot is None:
        raise ValueError("Gap evidence requires the original published Snapshot")
    scope = dict(writer_identity(args, contract, args.entity), group=args.group, metric_schema=3,
                 update=update, members={name: snapshot[2].get(name)
                     for name in (PLAN, CHECKPOINT, TRAIN_JSONL, DEV_JSONL)})
    return scope, {key: list(sources[key]) for key in ("original_jsonl", "qualified_bleu")}


def gap_observation(scope, target, report):
    return {"scope": scope, "observed_at": time.time(), "writer_state": list(writer_state(target)),
            "unconfirmed": report["ack_pending"]}


def gap_points(observation):
    return {(update, key) for key, updates in observation.items() for update in updates}


def repeated_gaps(first, second, scope):
    if (first.get("scope") != scope or second.get("scope") != scope
            or first.get("writer_state") != second.get("writer_state")):
        raise ValueError("Gap observations must bind the same Snapshot, Run, Schema and stopped writer")
    times = [item.get("observed_at") for item in (first, second)]
    if (not all(v3.valid_number(value) for value in times) or times[1] > time.time()
            or times[1] - times[0] < GAP_OBSERVATION_SECONDS):
        raise ValueError(f"Gap observations must be at least {GAP_OBSERVATION_SECONDS} seconds apart")
    return gap_points(first["unconfirmed"]) & gap_points(second["unconfirmed"])


def repair_v3_gaps(args, wandb, contract, published, sources):
    if not args.recover_writer or not args.confirm_writers_stopped:
        raise ValueError("Gap repair requires explicit confirmation that all W&B writers are stopped")
    paths = args.repair_v3_gaps
    first, second = (json.loads(path.read_bytes()) for path in paths)
    scope, durable = gap_sources(args, contract, published, sources)
    confirmed = repeated_gaps(first, second, scope)
    if not confirmed:
        raise ValueError("No repeatedly observed gaps; W&B delivery is uncertain, so nothing was written")

    def repair_rows(target):
        if target is None:
            raise ValueError("Persistent gap recovery requires the existing verified W&B Run")
        audit = v3.remote_audit(target, durable, published, minimum_history_step=v3_history_floor(args, contract))
        if audit["status"] == "CONFLICT":
            raise ValueError(f"W&B value conflict or duplicate; nothing was written: {audit['errors']}")
        if not audit["history_complete"]:
            raise WriterBusy("ACK_PENDING: a partial W&B history scan cannot confirm a persistent gap")
        gaps = gap_points(audit["ack_pending"]) & confirmed
        if confirmed & gap_points(audit["missing"]):
            raise ValueError("A confirmed gap is no longer below its recorded watermark; collect fresh observations")
        if gaps and list(writer_state(target)) != second["writer_state"]:
            raise ValueError("Writer changed since the gap observations; collect fresh evidence")
        return ({(progress, update): subset for points in durable.values() for progress, update, values in points
                 if (subset := {key: value for key, value in values.items() if (update, key) in gaps})}, audit)

    _, target = live_target(wandb, args, contract)
    written = {}
    if repair_rows(target)[0]:
        with sync_writer(wandb, args, contract) as (_, start_run, lease):
            # Use a fresh uncached scan under the existing writer guards,
            # immediately before wandb.init. This also recomputes the missing cells.
            def check(current):
                rows, audit = repair_rows(current)
                written.update(rows)
                checked.update(audit)
                if not written:
                    raise WriterBusy("The confirmed gaps became visible; run the read-only audit again")

            checked = {}
            run = start_run(check)
            try:
                check_v3_resume_cursor(run, checked)
                watermarks = {}
                for (progress, update), values in sorted(written.items(), key=lambda item: (item[0][1], item[0][0])):
                    publish_metrics(run, progress, update, values, watermarks, backfill=True)
            finally:
                finish_writer(v3.namespace(contract, args), lease, run, track_delivery=True)
    _, target = live_target(wandb, args, contract)
    audit = v3.remote_audit(target, durable, published, minimum_history_step=v3_history_floor(args, contract))
    remaining = confirmed & (gap_points(audit["missing"]) | gap_points(audit["ack_pending"]))
    status = ("CONFLICT" if audit["status"] == "CONFLICT" else
              "ACK_PENDING" if remaining or not audit["history_complete"] else "RECOVERED")
    print(json.dumps({"status": status,
                      "repaired_rows": len(written), "remaining_confirmed_gaps": sorted(remaining),
                      "delivery": audit}, allow_nan=False, sort_keys=True), flush=True)
    if status != "RECOVERED":
        raise RuntimeError("W&B repair is not verified; wait and rerun the read-only audit before doing anything else")


def audit_v3(args):
    originals_only = getattr(args, "originals_only", False)
    record = getattr(args, "record_v3_gaps", None)
    contract, published, members, sources, reports = verified_v3_sources(
        args, args.audit_v3, originals_only=originals_only)
    original_ok = reports["train"]["complete"] and reports["dev"]["complete"]
    if getattr(args, "expected_update", None) not in (None, published):
        raise ValueError(f"Expected the milestone snapshot at {args.expected_update}, found {published}")
    report = {"scope": args.audit_v3, "published_update": published, "originals": reports,
              "verified_snapshot_members": members, "run_id": run_id(v3.namespace(contract, args))}
    remote_ok = True
    if args.audit_v3 == "wandb":
        import wandb
        _, target = live_target(wandb, args, contract)
        # Materialize once because exact history and transport comparison share the same scan.
        if target is None:
            remote_ok = False
            report["delivery"] = {"status": "NOT_PUBLISHED"}
        else:
            floor = v3_history_floor(args, contract)
            report["delivery"] = v3.remote_audit(target, sources, published, minimum_history_step=floor)
            progress = {key: number(target.summary.get(key)) for key in v3.SUMMARY}
            summary_ok = (progress["train/target_update"] == v3.MAX_UPDATE
                          and progress["train/durable_update"] == published
                          and published <= (progress["train/max_update"] or 0) <= v3.MAX_UPDATE)
            report["delivery"].update(summary=progress, summary_matches=summary_ok)
            remote_ok = report["delivery"]["status"] == "VERIFIED" and summary_ok
            if record and original_ok:
                # The evidence covers only durable sources available on a cold VM.
                scope, durable = gap_sources(args, contract, published, sources)
                delivery = v3.remote_audit(target, durable, published, minimum_history_step=floor)
                if delivery["status"] == "CONFLICT" or not delivery["history_complete"]:
                    raise ValueError("Conflicting, duplicate or partial history cannot produce gap recovery evidence")
                with record.open("x", encoding="utf-8") as output:
                    observation = gap_observation(scope, target, delivery)
                    json.dump(observation, output, allow_nan=False, sort_keys=True)
                report["gap_observation"] = str(record)
                if earlier := getattr(args, "compare_v3_gaps", None):
                    report["repeatedly_observed_gaps"] = sorted(repeated_gaps(
                        json.loads(earlier.read_bytes()), observation, scope))
    bleu_complete = originals_only or reports["bleu"]["complete"]
    report["status"] = ("FAILED" if not original_ok or args.final and not bleu_complete
                        else "INCOMPLETE_DELIVERY" if not remote_ok
                        else "VERIFIED" if bleu_complete else "EVALUATION_PENDING")
    print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
    return report["status"] == "VERIFIED" or not args.final and report["status"] == "EVALUATION_PENDING"


def require_history_destination(args):
    if not getattr(args, "entity", None) or not getattr(args, "group", None):
        raise ValueError("Legacy W&B history requires the existing --entity and --group")


def audit_metric_history(args):
    require_history_destination(args)
    contract, published, members, sources = saved_metric_sources(args, compare_attempts=True)
    import wandb
    identity = writer_identity(args, contract, args.entity)
    target = verified_target(wandb, args, identity, require_group=True)
    _, report = reconcile_history(target, sources, published=published, include_legacy=True)
    print(json.dumps({**identity, "group": args.group, "published_update": published,
                      "verified_snapshot_members": members, **report}, allow_nan=False, sort_keys=True))


def sync_saved_metrics(args):
    include_legacy = getattr(args, "include_legacy_history", False)
    if include_legacy:
        require_history_destination(args)
    schema_v3 = v3.enabled(args)
    if schema_v3:
        contract, published, _, sources, reports = verified_v3_sources(args)
        if not (reports["train"]["complete"] and reports["dev"]["complete"]):
            raise ValueError("Incomplete Schema v3 originals; run the read-only audit")
    else:
        contract, published, _, sources = saved_metric_sources(args, compare_attempts=include_legacy)
    import wandb
    if getattr(args, "repair_v3_gaps", None):
        return repair_v3_gaps(args, wandb, contract, published, sources)
    wrote = False
    with sync_writer(wandb, args, contract) as (target, start_run, lease):
        if schema_v3:
            missing, report = v3.reconcile_delivery(target, sources, published,
                minimum_history_step=v3_history_floor(args, contract))
        else:
            missing, report = reconcile_history(target, sources, published=published, include_legacy=include_legacy)
        progress_due = schema_v3 and target is not None and (
            target.summary.get("train/target_update") != v3.MAX_UPDATE
            or (number(target.summary.get("train/durable_update")) or 0) < published
            or (number(target.summary.get("train/max_update")) or 0) < published)
        if missing or progress_due:
            # Recheck ownership and idle state after the read-only history scan.
            run = start_run()
            try:
                if schema_v3:
                    check_v3_resume_cursor(run, report)
                watermarks = {}
                for (progress, update), values in missing.items():
                    publish_metrics(run, progress, update, values, watermarks, backfill=True)
                set_training_progress(run, published, schema_v3=schema_v3)
                wrote = True
            finally:
                finish_writer(v3.namespace(contract, args), lease, run, track_delivery=schema_v3)
    if schema_v3 and wrote:
        detail = f"SDK finished through update {published}; delivery unconfirmed. Wait and run a read-only W&B audit."
        try:
            _, target = live_target(wandb, args, contract)
        except WriterBusy as error:
            log_event("wandb-metrics", "ack-pending", args.logs, detail)
            raise WriterBusy("ACK_PENDING: " + detail) from error
        delivery = v3.remote_audit(target, sources, published, minimum_history_step=v3_history_floor(args, contract))
        if delivery["status"] == "CONFLICT":
            raise ValueError(f"W&B history conflict after SDK finish: {delivery['errors']}; run a read-only audit")
        if (delivery["status"] != "VERIFIED" or target is None
                or number(target.summary.get("train/target_update")) != v3.MAX_UPDATE
                or any((number(target.summary.get(key)) or 0) < published
                       for key in ("train/durable_update", "train/max_update"))):
            log_event("wandb-metrics", "ack-pending", args.logs, detail)
            raise WriterBusy("ACK_PENDING: " + detail)
    if include_legacy:
        print(json.dumps({"run_id": run_id(contract), "published_update": published, **report},
                         allow_nan=False, sort_keys=True), flush=True)
    log_event("wandb-metrics", "synced", args.logs, f"{len(missing)} historical metric rows through update {published}")


def retryable_preflight(error, wandb):
    from requests.exceptions import ConnectionError as RequestConnectionError, SSLError, Timeout

    if isinstance(error, WriterBusy):
        return True
    if isinstance(error, wandb.errors.CommError):
        error = getattr(error, "exc", None) or error
    if isinstance(error, SSLError):
        return False
    if isinstance(error, (RequestConnectionError, ConnectionError, Timeout, TimeoutError)):
        return True
    response = getattr(error, "response", None)
    # W&B 0.31 wraps the core's ApiErrorResponse. Google API errors expose .code.
    status = getattr(response, "http_status", None) or getattr(response, "status_code", None) or getattr(error, "code", None)
    return status in (408, 429, 500, 502, 503, 504) or (
        not status and type(error).__name__ == "WandbApiFailedError")


def retry_writer_preflight(stop, logs, delay, error):
    reason = str(error) if isinstance(error, WriterBusy) else type(error).__name__
    if stop.is_set():
        log_event("wandb", "skipped", logs, f"Writer unavailable at shutdown ({reason})")
        return True
    log_event("wandb", "retrying", logs, f"Writer preflight: {reason}; retry in {delay}s")
    return stop.wait(delay)


def reporter_preflight(wandb, args, writer_scope):
    key = v3.namespace(args.contract, args)
    check_pending_writer(key)
    owner = read_owner(key)
    if owner is not None:
        verify_owner_destination(owner, args.entity or owner["entity"], args)
    with ExitStack() as attempt:
        identity, target = live_target(wandb, args, args.contract)
        if owner is not None:
            verify_owner_destination(owner, identity["entity"], args)
            if target is None:
                raise WriterBusy("The previous W&B run is not visible yet")
        state = writer_state(target)
        token = state[1] if state is not None else None
        lease = attempt.enter_context(shared_writer(getattr(args, "writer_bucket", None), identity, previous_token=token))
        current_identity, current = live_target(wandb, args, args.contract)
        if current_identity != identity or writer_state(current) != state:
            raise WriterBusy("Another W&B writer connected while the reporter was opening")
        writer_scope.callback(attempt.pop_all().close)
        return identity, current, lease


def report(args, stop):
    import wandb

    work = args.work.resolve()
    train = MetricReader(work / TRAIN_JSONL)
    dev = MetricReader(work / DEV_JSONL)
    layers = FairseqProgressReader(args.logs)
    published = 0
    watermarks = {}
    sent_bleu = {}
    published_command = None
    run = None
    writer_scope = ExitStack()
    bucket = getattr(args, "writer_bucket", None)
    schema_v3 = v3.enabled(args)
    retry_delay = 2
    try:
        while True:
            finishing = stop.is_set()
            plan_path = work / "native-run.json"
            if plan_path.is_file():
                plan_bytes = plan_path.read_bytes()
                plan = json.loads(plan_bytes)
                if plan.get("contract_hash") == args.contract:
                    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
                    snapshot = published_snapshot(work, args.backup, plan_sha, published if run is not None else 0,
                                                  args.method)
                    if snapshot is not None:
                        published, published_command, _ = snapshot
                        if run is None:
                            try:
                                with ExitStack() as attempt:
                                    identity, current, lease = reporter_preflight(wandb, args, attempt)
                                    missing = {}
                                    audit = None
                                    if schema_v3 and current is not None:
                                        state = writer_state(current)
                                        history = list(logical_metric_rows(MetricReader(work / TRAIN_JSONL),
                                            MetricReader(work / DEV_JSONL), published, contract=args.contract,
                                            schema_v3=True, method=args.method, tau=v3.plan_tau(plan),
                                            dev_reference=v3.plan_dev_reference(plan)))
                                        # The trainer can replace the local logs while restoring this snapshot.
                                        # Missing originals would make already-delivered history look foreign.
                                        observed = {(progress, update) for progress, update, _ in history}
                                        required = {(TRAIN_UPDATE, 1)}
                                        required.update((TRAIN_UPDATE, update) for update in range(100, published + 1, 100))
                                        required.update((DEV_UPDATE, update) for update in range(1000, published + 1, 1000))
                                        if not required <= observed:
                                            raise WriterBusy(f"Waiting for local Schema v3 originals through update {published}")
                                        bleu_rows = v3_bleu_rows(args, plan_sha, published, args.contract)
                                        missing, audit = v3.reconcile_delivery(current, {"original_jsonl": history,
                                            "qualified_bleu": bleu_rows}, published,
                                            minimum_history_step=v3_history_floor(args, args.contract))
                                        watermarks.update(audit["history_watermarks"])
                                        _, checked = live_target(wandb, args, args.contract)
                                        if writer_state(checked) != state:
                                            raise WriterBusy("Another writer connected during history reconciliation")
                                    writer_scope.callback(attempt.pop_all().close)
                            except Exception as error:
                                if not retryable_preflight(error, wandb) or schema_v3 and retry_delay == 60:
                                    raise
                                if retry_writer_preflight(stop, args.logs, retry_delay, error):
                                    return
                                retry_delay = min(2 * retry_delay, 60)
                                continue
                            # After attempting init only confirmed finish can release this writer.
                            resume = "must" if current is not None else "never" if schema_v3 else "allow"
                            run = start_writer(wandb, args, identity, lease, resume)
                            session = lease.token
                            run.summary[REPORTER_SESSION] = session
                            bind_writer(run, bucket, lease, args.contract)
                            if audit is not None:
                                check_v3_resume_cursor(run, audit)
                            for (progress, update), values in missing.items():
                                publish_metrics(run, progress, update, values, watermarks, backfill=True)
                            if audit is not None:
                                for progress, update, _ in (*history, *bleu_rows):
                                    watermarks[progress] = max(watermarks.get(progress, 0), update)
                        set_training_progress(run, published, schema_v3=schema_v3)
                    if run is not None:
                        sync_rows(run, train, dev, published, watermarks, contract=args.contract,
                                  schema_v3=schema_v3, method=args.method, tau=v3.plan_tau(plan) if schema_v3 else None,
                                  dev_reference=v3.plan_dev_reference(plan) if schema_v3 else None)
                        sync_logged_layers(run, layers, published_command, args.contract, published,
                                           watermarks)
                        if getattr(args, "bleu_results", None) and (snapshot is not None or finishing):
                            try:
                                if schema_v3:
                                    for progress, update, values in v3_bleu_rows(args, plan_sha, published, args.contract):
                                        publish_metrics(run, progress, update, values, watermarks)
                                    written = 0
                                else:
                                    points = saved_bleu_points(args, plan_sha, published, contract=args.contract)
                                    written = sync_bleu(run, points, sent_bleu)
                                if written:
                                    log_event("wandb-bleu", "synced", args.logs, f"{written} new checkpoint results")
                            except (OSError, ValueError, TypeError, KeyError) as error:
                                log_event("wandb-bleu", "failed", args.logs, str(error))
            if finishing:
                break
            stop.wait(15)
    finally:
        try:
            if run is not None:
                finish_writer(v3.namespace(args.contract, args), lease, run, track_delivery=schema_v3)
            elif published == 0:
                log_event("wandb", "skipped", args.logs, "No published updates for this contract")
        finally:
            writer_scope.close()
    if run is not None:
        return {"entity": run.entity, "project": args.project, "method": args.method, "session": session}


def publish_metrics(run, progress, update, values, watermarks, *, backfill=False):
    previous = watermarks.setdefault(progress, number(run.summary.get(progress)) or 0)
    if not values or (not backfill and update <= previous):
        return
    run.log({**values, "update": update, progress: update})
    watermarks[progress] = max(previous, update)
    # Preserve the high-water mark when explicitly filling an earlier gap.
    if update < previous:
        run.summary[progress] = previous


def logical_train_metrics(row):
    modern = row.get("metric_schema") == 2
    output = {}
    for key, value in row.items():
        if type(value) not in (int, float) or number(value) is None:
            continue
        if key.startswith("train/") and (modern or key in LEGACY_LOGICAL_TRAIN):
            if key not in {"train/main_loss", "train/total_loss"}:
                destination = key if key == "train/unit_ce_nats" else "train/update/" + key.removeprefix("train/")
                output[destination] = value
        elif key.startswith("alignment/") and (modern or key != "alignment/worst_layer"):
            if key != "alignment/task_only_start" or value >= 0:
                output["alignment/update/" + key.removeprefix("alignment/")] = value
    return output


def logical_metric_rows(train, dev, published, *, contract=None, schema_v3=False, method=None, tau=None,
                        dev_reference=None):
    for reader, progress in ((train, TRAIN_UPDATE), (dev, DEV_UPDATE)):
        for row in reader.rows(published):
            update = row["step"]
            if schema_v3:
                split = "train" if reader is train else "dev"
                if row.get("alignment_contract_hash") != contract:
                    raise ValueError(f"Foreign Schema v3 measurement at update {update}")
                if (split == "train" and update != 1 and update % 100) or (split == "dev" and update % 1000):
                    continue
                values = v3.validate_row(row, method, contract, split, tau=tau, dev_reference=dev_reference)
                yield progress, update, values
                continue
            if row.get("metric_schema") == 3:
                raise ValueError("Legacy telemetry cannot publish Schema v3 measurements")
            if reader is train and contract is not None and row.get("alignment_contract_hash", contract) != contract:
                continue
            if reader is train and update != 1 and update % 100 and update != published:
                continue
            values = logical_train_metrics(row) if reader is train else {
                key: value for key, value in row.items()
                if key.startswith("dev/") and type(value) in (int, float) and number(value) is not None}
            if values:
                yield progress, update, values


def sync_rows(run, train, dev, published, watermarks, *, contract=None, schema_v3=False, method=None, tau=None,
              dev_reference=None):
    for progress, update, values in logical_metric_rows(train, dev, published, contract=contract,
                                                        schema_v3=schema_v3, method=method, tau=tau,
                                                        dev_reference=dev_reference):
        publish_metrics(run, progress, update, values, watermarks)


def window_alignment(values):
    output = {"alignment/window/" + key.removeprefix("alignment/"): value
              for key, value in values.items() if key.startswith("alignment/")
              and key not in {"alignment/worst_layer", "alignment/task_only_start"}}
    losses = {int(key.rsplit("_", 1)[1]): value for key, value in values.items()
              if re.fullmatch(r"alignment/D_\d+", key)}
    present = {layer: values.get(f"alignment/valid_rows_{layer}") for layer in losses}
    valid = sorted(layer for layer, count in present.items() if count is not None and count > 0)
    if valid and all(count is not None for count in present.values()):
        output["alignment/window/worst_layer"] = max(valid, key=losses.__getitem__)
    return output


def fairseq_metric_rows(reader, command, contract, published):
    """Keep train log-window metrics and full-Dev metrics separate from update rows."""
    for update, values in reader.rows(command, contract, published):
        window = {"train/window/" + key.removeprefix("train/"): value for key, value in values.items()
                  if key.startswith("train/")}
        if "train/window/clip_fraction" in window:
            window["train/window/clip_fraction"] /= 100
        window.update(window_alignment(values))
        full_dev = {"dev/full/" + key.removeprefix("dev/"): value for key, value in values.items()
                    if key.startswith("dev/")}
        if window:
            yield TRAIN_WINDOW, update, window
        if full_dev:
            yield DEV_WINDOW, update, full_dev


def sync_logged_layers(run, reader, command, contract, published, watermarks):
    for progress, update, values in fairseq_metric_rows(reader, command, contract, published):
        publish_metrics(run, progress, update, values, watermarks)


@graceful_sigterm
def supervise(args):
    stop = Event()

    def publish():
        from filelock import Timeout as LockTimeout

        try:
            delay = 2
            key = v3.namespace(args.contract, args)
            lock = writer_lock(key)
            while True:
                try:
                    lock.acquire()
                    break
                except LockTimeout as error:
                    if retry_writer_preflight(stop, args.logs, delay, error):
                        return
                    delay = min(2 * delay, 60)
            try:
                if owner := report(args, stop):
                    store_owner(key, owner)
            finally:
                lock.release()
        except Exception as error:
            detail = f"{type(error).__name__}: {error}\n" + "".join(format_exception(error)).rstrip()
            for name in ("WANDB_API_KEY", "GITHUB_TOKEN", "HF_TOKEN", "GOOGLE_OAUTH_ACCESS_TOKEN"):
                if secret := os.environ.get(name):
                    for value in {secret, quote(secret, safe=""), quote_plus(secret)}:
                        detail = detail.replace(value, "[REDACTED]")
            detail = re.sub(r"(?i)(\bauthorization[\s'\"]*:[\s'\"]*(?:basic|bearer)\s+)[^\s'\";,]+",
                            r"\1[REDACTED]", detail)
            detail = re.sub(r"(?i)(https?://)[^/\s@]+@",
                            r"\1[REDACTED]@", detail)
            detail = re.sub(r"(?i)([?&](?:api[_-]?key|access_token|token|x-goog-signature)=)[^&\s'\"]+",
                            r"\1[REDACTED]", detail)
            log_event("wandb", "failed", args.logs, detail + "\nW&B delivery stopped for this segment. Training and Snapshot publication continue.")

    worker = Thread(target=publish, name="wandb-metrics", daemon=True)
    worker.start()
    try:
        env = {key: value for key, value in os.environ.items() if key not in {"WANDB_API_KEY", "AT_LOG_WANDB_DRAIN"}}
        run_process(args.command, args.logs, name="wandb-training-child", env=env)
        if audit := getattr(args, "after_training_audit", None):
            run_process(audit, args.logs, name="schema-v3-milestone-originals", env=env)
        if args.after_training:
            run_process(args.after_training, args.logs, name="online-bleu-after-checkpoint", env=env)
    finally:
        stop.set()
        worker.join(timeout=55)
        if worker.is_alive():
            log_event("wandb", "failed", args.logs, "Timed out waiting for metric upload at shutdown")


@graceful_sigterm
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("work", "backup", "logs"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("method", "project"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--contract")
    parser.add_argument("--metric-schema", type=int, choices=(2, 3), default=2)
    parser.add_argument("--group")
    parser.add_argument("--entity")
    parser.add_argument("--writer-bucket", default=os.environ.get("WANDB_WRITER_BUCKET") or None)
    parser.add_argument("--recover-writer", action="store_true",
                        help="Recover ownership of an existing stopped run")
    parser.add_argument("--confirm-writers-stopped", action="store_true",
                        help="Confirm all previous writers are stopped for manual recovery without a shared bucket")
    parser.add_argument("--recover-lock-generation", type=int,
                        help="Replace this exact orphaned shared writer generation after confirming its VM is gone")
    parser.add_argument("--bleu-results", type=Path)
    parser.add_argument("--after-training", type=json.loads)
    parser.add_argument("--after-training-audit", type=json.loads)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sync-bleu-only", action="store_true")
    modes.add_argument("--sync-metrics-only", action="store_true")
    modes.add_argument("--audit-metrics-only", action="store_true")
    modes.add_argument("--audit-history-only", action="store_true",
                       help="Preview candidate and missing v2 metrics using read-only W&B history")
    modes.add_argument("--audit-v3", choices=("local", "snapshot", "wandb"))
    modes.add_argument("--verify-v3-plan", type=Path)
    parser.add_argument("--previous-work", type=Path)
    parser.add_argument("--previous-backup", type=Path)
    parser.add_argument("--final", action="store_true", help="Require all scheduled BLEU evaluations in a v3 audit")
    parser.add_argument("--originals-only", action="store_true", help="Audit the checkpoint's Train/Dev originals without BLEU or W&B")
    parser.add_argument("--expected-update", type=int, help="Require a publication at this exact milestone")
    parser.add_argument("--record-v3-gaps", type=Path, metavar="OUTPUT", help="Save a stopped Run's read-only gap observation locally")
    parser.add_argument("--compare-v3-gaps", type=Path, metavar="EARLIER", help="Report gaps absent in both qualified observations")
    parser.add_argument("--repair-v3-gaps", type=Path, nargs=2, metavar=("FIRST", "SECOND"),
                        help="Append proven gaps after explicit stopped-writer recovery")
    parser.add_argument("--include-legacy-history", action="store_true",
                        help="Use marked legacy W&B history as another source for --sync-metrics-only")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    v3.verify_destination(args)
    if (args.originals_only or args.expected_update is not None) and not (v3.enabled(args) and args.audit_v3 == "snapshot"):
        parser.error("--originals-only and --expected-update require Schema v3 --audit-v3 snapshot")
    if (args.record_v3_gaps or args.compare_v3_gaps) and not (
            v3.enabled(args) and args.audit_v3 == "wandb" and args.record_v3_gaps):
        parser.error("Gap observations require Schema v3 --audit-v3 wandb --record-v3-gaps OUTPUT")
    if args.repair_v3_gaps and not (v3.enabled(args) and args.sync_metrics_only
                                   and args.recover_writer and args.confirm_writers_stopped and not args.command):
        parser.error("Gap repair requires Schema v3 --sync-metrics-only --recover-writer --confirm-writers-stopped")
    if args.after_training_audit is not None and not (
            v3.enabled(args) and args.command and isinstance(args.after_training_audit, list)
            and all(isinstance(arg, str) for arg in args.after_training_audit)):
        parser.error("--after-training-audit requires Schema v3 and a JSON array of command arguments")
    if args.verify_v3_plan is not None:
        if not v3.enabled(args) or not args.previous_work or not args.previous_backup or args.command:
            parser.error("--verify-v3-plan requires Schema v3 and the original work and backup paths")
        verify_v3_plan(args)
        return
    if args.audit_v3:
        if not v3.enabled(args) or args.command or args.after_training:
            parser.error("--audit-v3 requires Schema v3 and no training command")
        if not audit_v3(args):
            raise SystemExit(1)
        return
    if args.recover_writer and not args.sync_metrics_only:
        parser.error("--recover-writer requires --sync-metrics-only")
    if args.confirm_writers_stopped and not args.recover_writer:
        parser.error("--confirm-writers-stopped requires --recover-writer")
    if args.include_legacy_history and not args.sync_metrics_only:
        parser.error("--include-legacy-history requires --sync-metrics-only")
    if args.recover_lock_generation is not None and (
            args.recover_lock_generation < 1 or not args.recover_writer or not args.writer_bucket):
        parser.error("--recover-lock-generation requires --recover-writer, --writer-bucket and a positive generation")
    if args.sync_metrics_only:
        if args.command or args.after_training:
            parser.error("--sync-metrics-only requires no training command")
        sync_saved_metrics(args)
        return
    if args.audit_metrics_only:
        if args.command or args.after_training:
            parser.error("--audit-metrics-only requires no training command")
        audit_saved_metrics(args)
        return
    if args.audit_history_only:
        if args.command or args.after_training:
            parser.error("--audit-history-only requires no training command")
        audit_metric_history(args)
        return
    if args.sync_bleu_only:
        if not args.bleu_results or args.command or args.after_training:
            parser.error("--sync-bleu-only requires --bleu-results and no training command")
        sync_saved_bleu(args)
        return
    if not all((args.command, args.contract, args.group)):
        parser.error("a training command, contract and group are required")
    if args.after_training is not None and (not isinstance(args.after_training, list)
                                            or not all(isinstance(arg, str) for arg in args.after_training)):
        parser.error("--after-training requires a JSON array of arguments")
    supervise(args)


if __name__ == "__main__":
    main()
