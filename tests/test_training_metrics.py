"""Loss visibility without changing the recovered trainer or durable state."""
import json
from pathlib import Path
import subprocess

import pytest

from alignment_transfer_s2ut import training_metrics as m
from alignment_transfer_s2ut.io import file_hash, read_json, write_jsonl
from test_t2tt_extension import source_run, call
from test_benchmark_colab import benchmark_runtime


def test_live_loss_includes_validation_and_does_not_consume_partial_rows(tmp_path, capsys):
    path=tmp_path/"metrics.jsonl"
    rows=[dict(step=i,**{"train/main_loss":1/i,"train/learning_rate":.001}) for i in (1,2,3)]
    write_jsonl(path,rows[:1]);tail=m.MetricTail(path,every=100)
    before=file_hash(path);tail.poll();assert file_hash(path)==before
    assert '"update": 1' in capsys.readouterr().err
    line=json.dumps(rows[1])+"\n"
    with path.open("ab") as stream:stream.write(line[:-1].encode())
    offset=tail.offset;tail.poll();assert tail.offset==offset and not capsys.readouterr().err
    with path.open("ab") as stream:stream.write(b"\n")
    tail.poll();assert not capsys.readouterr().err  # No new dev point or 100-step interval.
    rows[2]["dev/main_loss"]=.7
    with path.open("a") as stream:stream.write(json.dumps(rows[2])+"\n")
    tail.poll();output=capsys.readouterr().err
    assert '"update": 3' in output and '"dev_loss": 0.7' in output
    assert 'local_metrics_not_durable' in output
    tail.poll(force=True);assert not capsys.readouterr().err


def test_monitor_thread_reports_final_metrics_and_cleans_up(tmp_path,capsys):
    import threading
    write_jsonl(tmp_path/"metrics.jsonl",[dict(step=1,**{"train/main_loss":2.})])
    with m.live_metrics(tmp_path,interval=.01):
        with (tmp_path/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=2,**{"train/main_loss":1.}))+"\n")
    assert '"update": 2' in capsys.readouterr().err
    assert not any(t.name=="t2tt-local-loss" for t in threading.enumerate())


def test_bad_local_metric_is_reported_without_stopping_training(tmp_path,capsys):
    (tmp_path/"metrics.jsonl").write_text('{bad json}\n',encoding="utf-8")
    with m.live_metrics(tmp_path,interval=.01):
        completed=True
    assert completed and 'WARNING local monitoring disabled' in capsys.readouterr().err


def test_snapshot_summary_is_exact_and_never_writes_original_source(source_run):
    from alignment_transfer_s2ut.training import load_checkpoint
    s=source_run
    original={p:file_hash(p) for root in (s["source_work"],s["source_backup"]) for p in root.rglob("*") if p.is_file()}
    assert call(s)["durable_updates"]==4
    for marker in s["backup"].glob("*/snapshot.json"):
        receipt=read_json(marker);summary=read_json(marker.parent/"metrics-summary.json")
        assert m.owned_summary(marker.parent/"metrics-summary.json",receipt)
        assert summary["current_update"]==summary["durable_update"]==receipt["updates"]
        assert summary["history"][-1]["update"]==receipt["updates"]
        assert summary["metrics_sha256"]==receipt["files"]["metrics.jsonl"]
        if receipt["updates"]==4:
            best=load_checkpoint(s["work"]/"checkpoint_best.pt")
            assert summary["best_checkpoint_step"]==best["step"]
            assert summary["best_dev_loss"]==best["best_main_loss"]
            rows=[json.loads(line) for line in (s["work"]/"metrics.jsonl").read_text().splitlines()]
            assert summary["latest_train_loss"]==rows[-1]["train/main_loss"]
            assert summary["latest_dev_loss"]==rows[-1]["dev/main_loss"]
    assert {p:file_hash(p) for root in (s["source_work"],s["source_backup"]) for p in root.rglob("*") if p.is_file()}==original


def test_summary_failure_does_not_lose_verified_checkpoint(source_run,monkeypatch,capsys):
    s=source_run
    original=m.write_json
    def fail(path,*args,**kwargs):
        if Path(path).name=="metrics-summary.json":raise OSError("summary destination unavailable")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(m,"write_json",fail)
    assert call(s)["durable_updates"]==4
    from alignment_transfer_s2ut.t2tt_extension import latest_snapshot
    assert latest_snapshot(s["backup"])[1]["updates"]==4
    assert 'verified checkpoint retained' in capsys.readouterr().err


def test_old_control_is_preserved_when_resuming_with_new_parent_monitor(source_run,monkeypatch):
    from alignment_transfer_s2ut import t2tt_extension as e
    s=source_run
    repo=Path(__file__).resolve().parents[1]
    old=subprocess.check_output(["git","-C",str(repo),"show","7d685bc:src/alignment_transfer_s2ut/t2tt_extension.py"])
    copy=e.shutil.copyfile
    def old_control(source,destination,*a,**kw):
        if Path(destination).name==".extension-control.py":
            Path(destination).write_bytes(old)
            return destination
        return copy(source,destination,*a,**kw)
    monkeypatch.setattr(e.shutil,"copyfile",old_control)
    assert call(s)["durable_updates"]==4
    assert (s["work"]/".extension-control.py").read_bytes()==old
    state=file_hash(s["work"]/"checkpoint_last.pt")
    control=file_hash(s["work"]/".extension-control.py")
    # Backfill a missing summary using the restored local files, with no updates.
    latest=e.latest_snapshot(s["backup"])[0]
    (latest/"metrics-summary.json").unlink()
    assert call(s)["durable_updates"]==4
    assert (latest/"metrics-summary.json").is_file()
    assert file_hash(s["work"]/"checkpoint_last.pt")==state
    assert file_hash(s["work"]/".extension-control.py")==control


def test_retention_understands_only_bound_owned_summaries(tmp_path):
    from alignment_transfer_s2ut.t2tt_extension import prune_target
    from alignment_transfer_s2ut.io import write_json
    identity={"owned":"extension"}
    for step in (1,2,3):
        folder=tmp_path/str(step)
        receipt=dict(identity=identity,updates=step,zip_sha256="zip",files={"metrics.jsonl":"metrics"})
        write_json(folder/"snapshot.json",receipt);(folder/"snapshot.zip").write_bytes(b"zip")
        write_json(folder/"metrics-summary.json",dict(kind=m.KIND,schema_version=1,current_update=step,
            snapshot_zip_sha256="zip" if step!=2 else "unrecognized",metrics_sha256="metrics"))
    prune_target(tmp_path,identity,1,[])
    assert not (tmp_path/"1").exists()
    assert (tmp_path/"2/snapshot.zip").is_file() and (tmp_path/"3/snapshot.zip").is_file()


def test_summary_history_is_bounded_and_keeps_real_validation_values(tmp_path,monkeypatch):
    import torch
    from alignment_transfer_s2ut.io import write_json
    files,snapshot=tmp_path/"files",tmp_path/"snapshot"
    identity={"owned":"test"}
    rows=[dict(step=i,**{"train/main_loss":1/i,"train/learning_rate":.001},
        **({"dev/main_loss":1+i/1000} if i%1000==0 else {})) for i in range(1,10001)]
    write_jsonl(files/"metrics.jsonl",rows)
    torch.save(dict(step=1000,best_main_loss=2.,contract_hash="contract"),files/"checkpoint_best.pt")
    write_json(files/"extension-lineage.json",dict(source_updates=2))
    write_json(files/"extension-plan.json",dict(target_updates=10000))
    receipt=dict(identity=identity,updates=10000,storage="zip-v1",zip_sha256="verified-by-caller",files={
        name:file_hash(files/name) for name in ("metrics.jsonl","checkpoint_best.pt")})
    write_json(snapshot/"snapshot.json",receipt)
    monkeypatch.setattr(m,"MAX_HISTORY",20)
    output=m.publish_summary(files,snapshot,identity,10000)
    result=read_json(output)
    assert len(result["history"])<=20 and result["metric_rows"]==10000
    assert [(p["update"],p["dev_loss"]) for p in result["history"] if "dev_loss" in p]==[(i,1+i/1000) for i in range(1000,10001,1000)]
    assert result["latest_train_loss"]==rows[-1]["train/main_loss"]
    assert result["best_checkpoint_step"]==1000
    assert output.stat().st_size<16384
    output.unlink()
    with (files/"metrics.jsonl").open("a") as stream:stream.write('{"step":10001}\n')
    assert m.publish_summary(files,snapshot,identity,10000) is None
    assert not output.exists()
