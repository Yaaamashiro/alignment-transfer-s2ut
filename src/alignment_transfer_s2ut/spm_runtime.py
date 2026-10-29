"""Fresh SentencePiece T2TT using the pinned benchmark session/ZIP runtime."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
from . import colab_runtime as c
from .cli import parser_for,run
from .io import read_json,write_json,file_hash
from .progress import progress
from .t2tt_extension import progress_update,prune_target,verify_state,report
from .training_metrics import live_metrics,publish_summary


def worker(plan):
    plan=read_json(plan)
    args=parser_for("train_t2tt").parse_args(plan["command"][3:])
    original=c.saved_checkpoint
    def scheduled(work,checkpoint,updates,force=False):
        path=Path(checkpoint);stat=path.stat();local=Path(work)/"training-local-progress.json"
        if updates%plan["snapshot_updates"]==0 or updates==args.max_updates or updates%50000==0:
            progress_update(lambda:original(work,checkpoint,updates,force=True),local)
        else:
            progress_update(lambda:write_json(local,dict(version=1,updates=updates,checkpoint=path.relative_to(work).as_posix(),
                bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),overwrite=True),local)
    c.saved_checkpoint=scheduled
    try:
        return run("train_t2tt",args)
    finally:
        c.saved_checkpoint=original


def start(command,work,backup,snapshot_updates=10000,keep_last=2):
    command=list(map(str,command));work,backup=c.session_paths(work,backup)
    if command[1:3] != ["-m","alignment_transfer_s2ut.train_t2tt"]:
        raise ValueError("SentencePiece runtime requires the standard T2TT CLI")
    args=parser_for("train_t2tt").parse_args(command[3:])
    if (args.tokenizer!="sentencepiece" or Path(args.output).resolve()!=work or not args.resume_if_exists or
            snapshot_updates<1 or snapshot_updates%args.save_interval_updates or keep_last<1):
        raise ValueError("SentencePiece runtime requires independent output, resume-if-exists and aligned snapshot interval")
    checkpoint=work/"checkpoint_last.pt"
    if not checkpoint.is_file():
        checkpoint=work/"checkpoint_best.pt"
    if checkpoint.is_file():
        from .training import load_checkpoint
        if "tokenizers" not in load_checkpoint(checkpoint):
            raise ValueError("existing character checkpoint is read-only; choose a new SentencePiece stage")
    # The operational plan stays outside the trainer output, which must be empty
    # for a fresh step-0 run. It is recreated from the bound command on each VM.
    plan=work.parent/(work.name+".spm-session.json")
    write_json(plan,dict(command=command,snapshot_updates=snapshot_updates),overwrite=True)
    milestones=[x for x in (50000,100000,150000,200000,args.max_updates) if x<=args.max_updates]
    def publish(files,destination,identity,updates):
        from .t2tt_extension import trim_frozen_metrics
        trim_frozen_metrics(Path(files)/"metrics.jsonl",updates)
        saved=c.runtime().publish_packed(files,destination,identity,updates)
        publish_summary(files,saved,identity,updates)
        prune_target(destination,identity,keep_last,milestones)
        return saved
    progress("SentencePiece T2TT: fresh stage or its own verified resume; character checkpoints are never seeded")
    with ExitStack() as stack:
        def launch(arguments,env):
            if list(arguments)!=command:
                raise ValueError("benchmark preparation changed the bound T2TT command")
            stack.enter_context(live_metrics(work))  # Restore must finish before following the new local file.
            return subprocess.Popen([command[0],"-m","alignment_transfer_s2ut.spm_runtime","_worker","--plan",str(plan)],env=env)
        return c.train_session(command,work,backup,popen=launch,publish=publish)


def status(work,target_updates):
    last,best=verify_state(Path(work),target_updates)
    if "tokenizers" not in last:
        raise ValueError("SentencePiece stage cannot use a character checkpoint")
    from .tokenization.factory import tokenizers_from_checkpoint
    tokenizers_from_checkpoint(last)
    return dict(completed_updates=last["step"],best_checkpoint_step=best["step"],best_main_loss=last["best_main_loss"],
        checkpoint_hash=file_hash(Path(work)/"checkpoint_best.pt"))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    train=sub.add_parser("train")
    for key in ("work","backup"):train.add_argument("--"+key,required=True)
    train.add_argument("--snapshot-updates",type=int,default=10000);train.add_argument("--keep-last",type=int,default=2)
    train.add_argument("command",nargs=argparse.REMAINDER)
    child=sub.add_parser("_worker");child.add_argument("--plan",required=True)
    child=sub.add_parser("status");child.add_argument("--work",required=True);child.add_argument("--target-updates",type=int,required=True)
    child=sub.add_parser("report");child.add_argument("--work",required=True);child.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.action=="train":
        result=start(args.command[1:] if args.command[:1]==["--"] else args.command,args.work,args.backup,args.snapshot_updates,args.keep_last)
    elif args.action=="_worker":result=worker(args.plan)
    elif args.action=="status":result=status(args.work,args.target_updates)
    else:result=report(args.work,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
