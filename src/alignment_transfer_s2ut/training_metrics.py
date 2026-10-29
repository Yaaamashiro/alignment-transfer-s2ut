"""Read-only local loss monitoring and small summaries of verified snapshots.

This module runs in the parent process, outside the recovered training source.
No checkpoint, metric row, training command or scientific contract is changed.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

from .io import file_hash, read_json, write_json
from .progress import progress

KIND = "t2tt_metrics_summary"
MAX_HISTORY = 2000


class MetricTail:
    """Follow complete JSONL rows with bounded state; never read Drive payloads."""
    def __init__(self, path, every=100):
        self.path, self.every = Path(path), every
        self.offset = self.reported = self.dev_reported = 0
        self.latest = self.latest_dev = self.best_dev = None

    def poll(self, force=False, stop=None):
        if not self.path.is_file():
            return
        if self.path.stat().st_size < self.offset:
            raise ValueError("local metrics file was truncated; refusing to mix histories")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while stop is None or not stop.is_set():
                start = stream.tell()
                line = stream.readline(1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 1024 * 1024:
                    raise ValueError("oversized metrics row")
                if not line.endswith(b"\n"):
                    break  # The writer may still be appending this row.
                if not line.strip():
                    self.offset = stream.tell()
                    continue
                row = json.loads(line)
                if type(row.get("step")) is not int or row["step"] <= 0:
                    raise ValueError("invalid metric update")
                if self.latest is not None and row["step"] != self.latest["step"] + 1:
                    raise ValueError("non-contiguous live metrics at byte " + str(start))
                for key in ("train/main_loss", "dev/main_loss", "train/learning_rate"):
                    if key in row and (type(row[key]) not in (float, int) or not math.isfinite(row[key])):
                        raise ValueError("invalid metric value: " + key)
                self.latest = {k: row[k] for k in ("step", "train/main_loss", "dev/main_loss", "train/learning_rate") if k in row}
                if "dev/main_loss" in row:
                    self.latest_dev = self.latest
                    if self.best_dev is None or row["dev/main_loss"] < self.best_dev["dev/main_loss"]:
                        self.best_dev = self.latest
                self.offset = stream.tell()
        if self.latest is None:
            return
        step = self.latest["step"]
        dev_step = self.latest_dev["step"] if self.latest_dev else 0
        if step > self.reported and (force or not self.reported or step - self.reported >= self.every or dev_step > self.dev_reported):
            message = dict(update=step, train_loss=self.latest.get("train/main_loss"),
                           learning_rate=self.latest.get("train/learning_rate"), source="local_metrics_not_durable")
            if self.latest_dev:
                message.update(dev_update=dev_step, dev_loss=self.latest_dev["dev/main_loss"],
                               best_observed_dev_loss=self.best_dev["dev/main_loss"], best_observed_dev_update=self.best_dev["step"])
            progress("T2TT loss", detail=json.dumps(message, allow_nan=False))
            self.reported, self.dev_reported = step, dev_step


@contextmanager
def live_metrics(work, every=100, interval=1.):
    """Logging failures are visible, but cannot stop or alter training."""
    reader = MetricTail(Path(work) / "metrics.jsonl", every)
    stop = threading.Event()
    failed = False
    def poll(force=False):
        nonlocal failed
        if failed:
            return
        try:
            reader.poll(force=force, stop=None if force else stop)
        except Exception as error:
            failed = True
            progress("T2TT loss: WARNING local monitoring disabled", detail=str(error))
    poll(force=True)
    def follow():
        while not stop.wait(interval):
            poll()
    thread = threading.Thread(target=follow, name="t2tt-local-loss", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        poll(force=True)


def owned_summary(path, receipt):
    summary = read_json(path)
    return (summary.get("kind") == KIND and summary.get("schema_version") == 1 and
            summary.get("current_update") == receipt["updates"] and
            summary.get("snapshot_zip_sha256") == receipt.get("zip_sha256") and
            summary.get("metrics_sha256") == receipt["files"]["metrics.jsonl"])


def publish_summary(files, snapshot, identity, updates):
    """Write once after verified ZIP publication, using its frozen local prefix.

    The JSON is diagnostic only. snapshot.json/ZIP remain the resume authority.
    Failures preserve the verified checkpoint and are reported without aborting it.
    """
    try:
        files, snapshot = Path(files), Path(snapshot)
        receipt = read_json(snapshot / "snapshot.json")
        if receipt["identity"] != identity or receipt["updates"] != updates or receipt["storage"] != "zip-v1":
            raise ValueError("metrics summary snapshot identity/update differs")
        output = snapshot / "metrics-summary.json"
        if output.exists():
            if not owned_summary(output, receipt):
                raise ValueError("existing metrics summary differs; retained without overwrite")
            return output
        path = files / "metrics.jsonl"
        if file_hash(path) != receipt["files"]["metrics.jsonl"]:
            raise ValueError("metrics summary source differs from verified snapshot")
        from .t2tt_extension import metric_rows
        from .training import load_checkpoint
        rows, gaps = metric_rows(path, updates)
        if gaps or not rows or rows[-1]["step"] != updates:
            raise ValueError("metrics summary requires a continuous committed prefix")
        best = load_checkpoint(files / "checkpoint_best.pt")
        if (not 0 < best["step"] <= updates or not math.isfinite(best["best_main_loss"]) or
                file_hash(files / "checkpoint_best.pt") != receipt["files"]["checkpoint_best.pt"]):
            raise ValueError("metrics summary best checkpoint differs")
        dev = [r for r in rows if "dev/main_loss" in r]
        # Preserve all validation points when they fit, then sample training rows.
        # For unusually frequent validation, evenly sample that history too.
        def sample(values, count):
            if count >= len(values):
                return values
            if count == 1:
                return values[-1:]
            return [values[i * (len(values) - 1) // (count - 1)] for i in range(count)]
        selected = {r["step"]: r for r in sample(dev, MAX_HISTORY // 2)}
        for row in sample(rows, MAX_HISTORY - len(selected)):
            selected[row["step"]] = row
        history = []
        for row in sorted(selected.values(), key=lambda r: r["step"]):
            point = dict(update=row["step"])
            for source, target in (("train/main_loss", "train_loss"), ("dev/main_loss", "dev_loss"), ("train/learning_rate", "learning_rate")):
                if source in row:
                    point[target] = row[source]
            history.append(point)
        lineage = read_json(files / "extension-lineage.json") if (files / "extension-lineage.json").is_file() else {"source_updates":0}
        plan = read_json(files / "extension-plan.json") if (files / "extension-plan.json").is_file() else {"target_updates":read_json(files / "experiment.json")["planned_max_updates"]}
        result = dict(kind=KIND, schema_version=1, current_update=updates, durable_update=updates,
            source_updates=lineage["source_updates"], target_updates=plan["target_updates"],
            latest_train_update=rows[-1]["step"], latest_train_loss=rows[-1].get("train/main_loss"),
            latest_dev_update=dev[-1]["step"] if dev else None, latest_dev_loss=dev[-1]["dev/main_loss"] if dev else None,
            best_dev_loss=best["best_main_loss"], best_checkpoint_step=best["step"],
            snapshot_directory=snapshot.name, snapshot_zip_sha256=receipt["zip_sha256"],
            metrics_sha256=receipt["files"]["metrics.jsonl"], best_checkpoint_sha256=receipt["files"]["checkpoint_best.pt"],
            contract_hash=best["contract_hash"], metric_rows=len(rows), history_points=len(history),
            history_sampling="sampled observations; all dev points if <=1000; no averaging or interpolation",
            history=history)
        write_json(output, result)
        progress("T2TT loss: published durable summary", detail=str(output))
        return output
    except Exception as error:
        progress("T2TT loss: WARNING summary unavailable; verified checkpoint retained", detail=str(error))
        return None
