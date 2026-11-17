"""Send checkpointed transfer metrics to W&B while the Colab train command runs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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
                row = json.loads(line)
                if row["step"] > maximum_update:
                    break
                self.offset += len(line)
                yield row


def published_update(work, backup, plan_sha, minimum):
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
        state = json.loads(marker.read_bytes())
        if (state["updates"] == update and state.get("storage") == "zip-v1"
                and state["identity"]["work"] == str(work)
                and state["files"].get("native-run.json") == plan_sha):
            return update
    return minimum


def report(args, stop):
    import wandb

    work = args.work.resolve()
    train = MetricReader(work / "alignment-train.jsonl")
    dev = MetricReader(work / "native-dev.jsonl")
    published = 0
    watermarks = {}
    run = None
    try:
        while True:
            finishing = stop.is_set()
            plan_path = work / "native-run.json"
            if plan_path.is_file():
                plan_bytes = plan_path.read_bytes()
                plan = json.loads(plan_bytes)
                if plan.get("contract_hash") == args.contract:
                    newest = published_update(work, args.backup, hashlib.sha256(plan_bytes).hexdigest(), published)
                    if newest > published:
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
                        published = newest
                    if run is not None:
                        sync_rows(run, train, dev, published, watermarks)
            if finishing:
                break
            stop.wait(15)
    finally:
        if run is not None:
            run.finish()
        else:
            log_event("wandb", "skipped", args.logs, "No published updates for this contract")


def sync_rows(run, train, dev, published, watermarks):
    for reader, prefixes, progress in ((train, ("train/", "alignment/"), "wandb_train_update"),
                                       (dev, ("dev/",), "wandb_dev_update")):
        sent = watermarks.get(progress)
        if sent is None:
            sent = run.summary.get(progress) or 0
            watermarks[progress] = sent
        for row in reader.rows(published):
            update = row["step"]
            if update <= sent or (reader is train and update != 1 and update % 100 and update != published):
                continue
            values = {key: value for key, value in row.items()
                      if key.startswith(prefixes) and type(value) in (int, float) and math.isfinite(value)}
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
