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
from threading import Event, Thread

# The tracking environment only needs the repository's standard-library logger.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alignment_transfer_s2ut.process_logging import log_event, run_process


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


def published_snapshot(work, backup, plan_sha, minimum, method):
    """Read publication receipts without opening checkpoint archives."""
    for folder in sorted(backup.iterdir(), reverse=True) if backup.is_dir() else ():
        prefix = folder.name.split("-", 1)[0]
        if not prefix.isascii() or not prefix.isdecimal():
            continue
        update = int(prefix)
        if update <= minimum:
            break
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
                    and "checkpoints/checkpoint_last.pt" in files
                    and files.get("native-run.json") == plan_sha):
                return update, command
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            continue
    return None


def report(args, stop):
    import wandb

    work = args.work.resolve()
    train = MetricReader(work / "alignment-train.jsonl")
    dev = MetricReader(work / "native-dev.jsonl")
    layers = FairseqProgressReader(args.logs)
    published = 0
    watermarks, native_alignment = {}, {}
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
                    snapshot = published_snapshot(work, args.backup, hashlib.sha256(plan_bytes).hexdigest(),
                                                  published, args.method)
                    if snapshot is not None:
                        published, published_command = snapshot
                        if run is None:
                            run_id = hashlib.sha256(args.contract.encode()).hexdigest()[:24]
                            run = wandb.init(project=args.project, entity=args.entity or None,
                                             group=args.group, name=args.method, id=run_id,
                                             resume="allow", dir=str(args.logs.resolve()), save_code=False,
                                             config={"condition": args.method},
                                             settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                                                     x_disable_meta=True, x_disable_stats=True,
                                                                     x_service_wait=10, init_timeout=20, finish_timeout=20,
                                                                     finish_timeout_raises=True))
                            run.define_metric("update")
                            for prefix in ("train", "alignment", "dev"):
                                run.define_metric(prefix + "/*", step_metric="update")
                            log_event("wandb", "connected", args.logs, run.url or "Offline run")
                    if run is not None:
                        sync_rows(run, train, dev, published, watermarks, native_alignment)
                        sync_logged_layers(run, layers, published_command, args.contract, published,
                                           watermarks, native_alignment)
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
            report(args, stop)
        except Exception as error:
            log_event("wandb", "failed", args.logs, type(error).__name__)

    worker = Thread(target=publish, name="wandb-metrics", daemon=True)
    worker.start()
    try:
        env = {key: value for key, value in os.environ.items() if key != "WANDB_API_KEY"}
        run_process(args.command, args.logs, name="wandb-training-child", env=env)
    finally:
        stop.set()
        worker.join(timeout=55)
        if worker.is_alive():
            log_event("wandb", "failed", args.logs, "Timed out waiting for metric upload at shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("work", "backup", "logs"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("contract", "group", "method", "project"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--entity")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a training command is required")
    supervise(args)


if __name__ == "__main__":
    main()
