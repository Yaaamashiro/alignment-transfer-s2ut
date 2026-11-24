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
from tempfile import TemporaryDirectory, gettempdir
from threading import Event, Thread
from uuid import uuid4
from zipfile import ZipFile

# Keep the W&B runtime independent of the scientific Python environment.
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
from alignment_transfer_s2ut.process_logging import graceful_sigterm, log_event, run_process
from transfer_bleu import CHECKPOINT, PLAN, SHA256, evaluation_scope, publishable, result_hash
from wandb_writer import WriterBusy, shared_writer


REPORTER_SESSION = "wandb_reporter_session"
WRITER_BUCKET = "wandb_writer_lock_bucket"
WRITER_TOKEN = "wandb_writer_lock_token"
WRITER_CONTRACT = "wandb_writer_contract_hash"
IDLE_STATES = {"finished", "failed", "crashed", "killed"}
TRAIN_UPDATE = "wandb_train_update_v2"
TRAIN_WINDOW = "wandb_train_window_v2"
DEV_WINDOW = "wandb_dev_full_v2"
DEV_UPDATE = "wandb_dev_update"
LEGACY_LOGICAL_TRAIN = {"train/learning_rate", "train/seconds"}
TRAIN_JSONL = "alignment-train.jsonl"
DEV_JSONL = "native-dev.jsonl"


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


def number(value):
    if type(value) not in (int, float, str):
        return None
    try:
        value = float(value) if isinstance(value, str) else value
        return value if math.isfinite(value) else None
    except (ValueError, OverflowError):
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

    def __init__(self, logs):
        self.logs = Path(logs).resolve()
        self.offsets = {}

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
                    points.setdefault(step, {}).update(pending.pop(step))
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
        if update in points and points[update] != row:
            raise ValueError(f"Conflicting BLEU results at update {update}")
        points[update] = row
    return sorted(points.items())


def sync_bleu(run, points, sent):
    written = 0
    for update, row in points:
        marker = f"wandb_bleu_checkpoint_{update}"
        checkpoint = row["checkpoint_sha256"]
        previous = sent.get(update) or run.summary.get(marker)
        if previous == checkpoint:
            continue
        if previous is not None:
            raise ValueError(f"W&B already contains another BLEU checkpoint at update {update}")
        values = {f"eval/{name}": row[name] for name in ("aux_bleu", "asr_bleu", "aux_failures", "asr_failures")}
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
    return dict(entity=entity, project=args.project, run_id=run_id(contract), contract_hash=contract, method=args.method)


def check_target(target, args, identity, *, require_group=False):
    if (target.id != identity["run_id"] or target.entity != identity["entity"] or target.project != identity["project"]
            or target.config.get("condition") != identity["method"]
            or target.config.get("alignment_contract_hash", identity["contract_hash"]) != identity["contract_hash"]
            or target.summary.get(WRITER_CONTRACT, identity["contract_hash"]) != identity["contract_hash"]
            or require_group and getattr(target, "group", None) != args.group):
        raise WriterConfigurationError("Existing W&B run identity does not match the verified training experiment")
    if target.summary.get(WRITER_BUCKET) not in (None, getattr(args, "writer_bucket", None)):
        raise WriterConfigurationError("Set the shared writer bucket already bound to this W&B run")
    if target.state not in IDLE_STATES:
        raise WriterBusy("W&B run is active; wait for the previous writer to finish")


def writer_state(target):
    if target is not None:
        return tuple(target.summary.get(key) for key in (WRITER_BUCKET, WRITER_TOKEN, REPORTER_SESSION))


def live_target(wandb, args, contract):
    api = wandb.Api(timeout=20)
    entity = args.entity or api.default_entity
    if not entity:
        raise WriterConfigurationError("Set the W&B entity for this reporter")
    identity = writer_identity(args, contract, entity)
    try:
        target = api.run(f"{identity['entity']}/{args.project}/{identity['run_id']}")
    except (wandb.apis.public.RunNotFoundError, wandb.errors.CommError) as error:
        if not isinstance(error, wandb.apis.public.RunNotFoundError) and not isinstance(
                getattr(error, "exc", None), wandb.apis.public.RunNotFoundError):
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
    store_writer_record(writer_path(contract, ".pending.json"), {**identity, "session": lease.token})
    lease.opening()
    return open_run(wandb, args, contract, entity=identity["entity"], resume=resume)


def finish_writer(contract, lease, run):
    lease.finish(run)
    writer_path(contract, ".pending.json").unlink(missing_ok=True)


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
    with writer_lock(contract):
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
            check_pending_writer(contract)
            owner = read_owner(contract)
            if owner is None:
                raise RuntimeError("No local W&B ownership receipt; see the --recover-writer instructions")
            entity = owner["entity"]
            if not entity:
                raise WriterConfigurationError("The local W&B ownership receipt has no entity")
            verify_owner_destination(owner, args.entity or entity, args)
        identity = writer_identity(args, contract, entity)

        def get_target():
            current = wandb.Api(timeout=20).run(f"{entity}/{args.project}/{identity['run_id']}")
            check_target(current, args, identity, require_group=recover or bool(getattr(args, "group", None)))
            if owner is not None and current.summary.get(REPORTER_SESSION) != owner["session"]:
                raise RuntimeError("Another training reporter owns this W&B run. Leave the scores on Drive for that reporter.")
            return current

        target = get_target()
        state = writer_state(target)
        bound_bucket, token, session = state
        with shared_writer(bucket, identity, previous_token=token, recover_generation=generation) as lease:
            def recheck():
                if writer_state(get_target()) != state:
                    raise RuntimeError("Another W&B writer connected during recovery")

            def start_run():
                recheck()
                run = start_writer(wandb, args, identity, lease, "must")
                bind_writer(run, bucket, lease, contract)
                if recover:
                    run.summary[REPORTER_SESSION] = lease.token
                return run

            yield target, start_run, lease
            if recover and not lease.unfinished:
                if not lease.opened:
                    if not isinstance(session, str) or not session or bucket != bound_bucket:
                        # Establish a remote session or first-time bucket binding without
                        # writing a history row.
                        finish_writer(contract, lease, start_run())
                    else:
                        recheck()
                store_owner(contract, dict(entity=entity, project=args.project, method=args.method,
                                           session=lease.token if lease.opened else session))
                writer_path(contract, ".pending.json").unlink(missing_ok=True)
                log_event("wandb-writer", "recovered", args.logs, identity["run_id"])


def open_run(wandb, args, contract, *, entity=None, resume="allow"):
    identifier = run_id(contract)
    options = {}
    if resume == "allow":
        entity = entity or args.entity or None
        options = {"group": args.group, "name": args.method, "config": {"condition": args.method}}
    if resume == "must" or getattr(args, "writer_bucket", None):
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


def set_training_progress(run, published):
    name = "train/max_update"
    previous = number(run.summary.get(name)) or 0
    training = number(run.summary.get("wandb_train_update")) or 0
    maximum = int(max(published, previous, training, number(run.summary.get(TRAIN_UPDATE)) or 0))
    if maximum != previous:
        run.summary[name] = maximum


def sync_saved_bleu(args):
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
def metric_snapshot(args):
    """Copy and verify only the archived plan and declared metric files."""
    snapshot = published_snapshot(args.work.resolve(), args.backup, None, 0, args.method, include_archive=True)
    if snapshot is None:
        raise ValueError("No matching published checkpoint for metric backfill")
    published, command, files, archive = snapshot
    if not isinstance(files[CHECKPOINT], str) or SHA256.fullmatch(files[CHECKPOINT]) is None:
        raise ValueError("Invalid published SHA-256 for the checkpoint")
    members = (PLAN, *(name for name in (TRAIN_JSONL, DEV_JSONL) if name in files))
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


def missing_metrics(remote, candidates):
    """Compare sparse history by metric and update before opening a writer."""
    expected, coordinates = {}, {}
    for progress, update, values in candidates:
        for name, value in values.items():
            point = update, name
            if point in expected and expected[point] != value:
                raise ValueError(f"Conflicting source metric {name} at update {update}")
            expected[point] = value
            coordinates[point] = progress
    if not expected:
        return {}
    found = set()
    for row in remote.scan_history(page_size=1000, use_cache=False):
        step = number(row.get("update"))
        if step is None or step < 1 or step != int(step):
            continue
        step = int(step)
        for name, value in row.items():
            point = step, name
            if point in expected and value is not None:
                if number(value) != expected[point]:
                    raise ValueError(f"W&B metric conflict for {name} at update {step}")
                found.add(point)
    missing = {}
    for point, value in expected.items():
        if point not in found:
            update, name = point
            missing.setdefault((coordinates[point], update), {})[name] = value
    return missing


def sync_saved_metrics(args):
    with metric_snapshot(args) as (contract, published, command, local, _):
        candidates = list(logical_metric_rows(MetricReader(local / TRAIN_JSONL), MetricReader(local / DEV_JSONL),
                                              published, contract=contract))
    candidates.extend(fairseq_metric_rows(FairseqProgressReader(args.logs), command, contract, published))
    import wandb
    with sync_writer(wandb, args, contract) as (target, start_run, lease):
        missing = missing_metrics(target, candidates)
        if missing:
            # Recheck ownership and idle state after the read-only history scan.
            run = start_run()
            try:
                watermarks = {}
                for (progress, update), values in missing.items():
                    publish_metrics(run, progress, update, values, watermarks, backfill=True)
                set_training_progress(run, published)
            finally:
                finish_writer(contract, lease, run)
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
    check_pending_writer(args.contract)
    owner = read_owner(args.contract)
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
                                identity, current, lease = reporter_preflight(wandb, args, writer_scope)
                            except Exception as error:
                                if not retryable_preflight(error, wandb):
                                    raise
                                if retry_writer_preflight(stop, args.logs, retry_delay, error):
                                    return
                                retry_delay = min(2 * retry_delay, 60)
                                continue
                            # After attempting init only confirmed finish can release this writer.
                            run = start_writer(wandb, args, identity, lease, "must" if current is not None else "allow")
                            session = lease.token
                            run.summary[REPORTER_SESSION] = session
                            bind_writer(run, bucket, lease, args.contract)
                        set_training_progress(run, published)
                    if run is not None:
                        sync_rows(run, train, dev, published, watermarks, contract=args.contract)
                        sync_logged_layers(run, layers, published_command, args.contract, published,
                                           watermarks)
                        if getattr(args, "bleu_results", None) and (snapshot is not None or finishing):
                            try:
                                points = saved_bleu_points(args, plan_sha, published, contract=args.contract)
                                if written := sync_bleu(run, points, sent_bleu):
                                    log_event("wandb-bleu", "synced", args.logs, f"{written} new checkpoint results")
                            except (OSError, ValueError, TypeError, KeyError) as error:
                                log_event("wandb-bleu", "failed", args.logs, str(error))
            if finishing:
                break
            stop.wait(15)
    finally:
        try:
            if run is not None:
                finish_writer(args.contract, lease, run)
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


def logical_metric_rows(train, dev, published, *, contract=None):
    for reader, progress in ((train, TRAIN_UPDATE), (dev, DEV_UPDATE)):
        for row in reader.rows(published):
            update = row["step"]
            if reader is train and contract is not None and row.get("alignment_contract_hash", contract) != contract:
                continue
            if reader is train and update != 1 and update % 100 and update != published:
                continue
            values = logical_train_metrics(row) if reader is train else {
                key: value for key, value in row.items()
                if key.startswith("dev/") and type(value) in (int, float) and number(value) is not None}
            if values:
                yield progress, update, values


def sync_rows(run, train, dev, published, watermarks, *, contract=None):
    for progress, update, values in logical_metric_rows(train, dev, published, contract=contract):
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
            lock = writer_lock(args.contract)
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
                    store_owner(args.contract, owner)
            finally:
                lock.release()
        except Exception as error:
            detail = str(error) if isinstance(error, WriterConfigurationError) else type(error).__name__
            log_event("wandb", "failed", args.logs, detail)

    worker = Thread(target=publish, name="wandb-metrics", daemon=True)
    worker.start()
    try:
        env = {key: value for key, value in os.environ.items() if key not in {"WANDB_API_KEY", "AT_LOG_WANDB_DRAIN"}}
        run_process(args.command, args.logs, name="wandb-training-child", env=env)
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sync-bleu-only", action="store_true")
    modes.add_argument("--sync-metrics-only", action="store_true")
    modes.add_argument("--audit-metrics-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.recover_writer and not args.sync_metrics_only:
        parser.error("--recover-writer requires --sync-metrics-only")
    if args.confirm_writers_stopped and not args.recover_writer:
        parser.error("--confirm-writers-stopped requires --recover-writer")
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
