"""Send checkpointed transfer metrics to W&B while the Colab train command runs."""

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from tempfile import gettempdir
from threading import Event, Thread

# Keep the W&B runtime independent of the scientific Python environment.
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
from alignment_transfer_s2ut.process_logging import log_event, run_process
from transfer_bleu import CHECKPOINT, PLAN, SHA256, evaluation_scope, publishable, result_hash


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
    "train/main_loss": "train/main_loss_fairseq",
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


def published_snapshot(work, backup, plan_sha, minimum, method, *, exact=None):
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
                return update, command, files
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


def writer_lock(contract):
    from filelock import FileLock

    path = Path(gettempdir()) / "alignment-wandb-writers"
    path.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path / (run_id(contract) + ".lock")), timeout=0)


def open_run(wandb, args, contract, *, existing=False):
    identifier = run_id(contract)
    options = {}
    if existing:
        api = wandb.Api(timeout=20)
        entity = args.entity or api.default_entity
        target = api.run(f"{entity}/{args.project}/{identifier}")
        if target.state not in {"finished", "failed", "crashed", "killed"} or target.config.get("condition") != args.method:
            raise RuntimeError("W&B run is active or does not match the selected training condition")
    else:
        entity = args.entity or None
        options = {"group": args.group, "name": args.method, "config": {"condition": args.method}}
    run = wandb.init(project=args.project, entity=entity, id=identifier, resume="must" if existing else "allow",
                     dir=str(args.logs.resolve()), save_code=False, **options,
                     settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                             x_disable_meta=True, x_disable_stats=True,
                                             x_service_wait=10, init_timeout=20, finish_timeout=20,
                                             finish_timeout_raises=True))
    run.define_metric("update")
    for prefix in ("train", "alignment", "dev", "eval"):
        run.define_metric(prefix + "/*", step_metric="update")
    log_event("wandb", "connected", args.logs, run.url or "Offline run")
    return run


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
    with writer_lock(contract):
        run = open_run(wandb, args, contract, existing=True)
        try:
            written = sync_bleu(run, points, {})
            log_event("wandb-bleu", "synced", args.logs, f"{written} new checkpoint results")
        finally:
            run.finish()


def report(args, stop):
    import wandb

    work = args.work.resolve()
    train = MetricReader(work / "alignment-train.jsonl")
    dev = MetricReader(work / "native-dev.jsonl")
    layers = FairseqProgressReader(args.logs)
    published = 0
    watermarks, native_alignment = {}, {}
    sent_bleu = {}
    published_command = None
    run = None
    try:
        while True:
            finishing = stop.is_set()
            plan_path = work / "native-run.json"
            if plan_path.is_file():
                plan_bytes = plan_path.read_bytes()
                plan = json.loads(plan_bytes)
                if plan.get("contract_hash") == args.contract:
                    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
                    snapshot = published_snapshot(work, args.backup, plan_sha, published, args.method)
                    if snapshot is not None:
                        published, published_command, _ = snapshot
                        if run is None:
                            run = open_run(wandb, args, args.contract)
                    if run is not None:
                        sync_rows(run, train, dev, published, watermarks, native_alignment)
                        sync_logged_layers(run, layers, published_command, args.contract, published,
                                           watermarks, native_alignment)
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
        if run is not None:
            run.finish()
        else:
            log_event("wandb", "skipped", args.logs, "No published updates for this contract")


def sync_rows(run, train, dev, published, watermarks, native_alignment):
    for reader, prefixes, progress in ((train, ("train/", "alignment/"), "wandb_train_update"),
                                       (dev, ("dev/",), "wandb_dev_update")):
        sent = watermarks.get(progress)
        if sent is None:
            sent = run.summary.get(progress) or 0
            watermarks[progress] = sent
        for row in reader.rows(published):
            update = row["step"]
            if reader is train and update != 1 and update % 100 and update != published:
                continue
            values = {key: value for key, value in row.items()
                      if key.startswith(prefixes) and type(value) in (int, float) and math.isfinite(value)}
            if reader is train:
                native_alignment[update] = {key for key in values if key.startswith("alignment/")}
            if values and update > sent:
                run.log({**values, "update": update, progress: update})
                sent = watermarks[progress] = update


def sync_logged_layers(run, reader, command, contract, published, watermarks, native_alignment):
    """Backfill Fairseq diagnostics at their original optimizer updates."""
    progress = "wandb_layer_update"
    sent = watermarks.get(progress)
    if sent is None:
        sent = run.summary.get(progress) or 0
        watermarks[progress] = sent
    for update, values in reader.rows(command, contract, published):
        if update <= sent:
            continue
        native = native_alignment.get(update, ())
        values = {key: value for key, value in values.items() if key not in native}
        if values:
            run.log({**values, "update": update, progress: update})
            sent = watermarks[progress] = update


def supervise(args):
    stop = Event()

    def publish():
        try:
            with writer_lock(args.contract):
                report(args, stop)
        except Exception as error:
            log_event("wandb", "failed", args.logs, type(error).__name__)

    worker = Thread(target=publish, name="wandb-metrics", daemon=True)
    worker.start()
    try:
        env = {key: value for key, value in os.environ.items() if key != "WANDB_API_KEY"}
        run_process(args.command, args.logs, name="wandb-training-child", env=env)
        if args.after_training:
            run_process(args.after_training, args.logs, name="online-bleu-after-checkpoint", env=env)
    finally:
        stop.set()
        worker.join(timeout=55)
        if worker.is_alive():
            log_event("wandb", "failed", args.logs, "Timed out waiting for metric upload at shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("work", "backup", "logs"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("method", "project"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--contract")
    parser.add_argument("--group")
    parser.add_argument("--entity")
    parser.add_argument("--bleu-results", type=Path)
    parser.add_argument("--after-training", type=json.loads)
    parser.add_argument("--sync-bleu-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
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
