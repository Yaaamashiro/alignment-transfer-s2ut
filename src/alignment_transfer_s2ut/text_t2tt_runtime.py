"""Fresh text-only T2TT with the existing verified ZIP/session and loss monitor."""
import argparse
import json
from pathlib import Path
from .cli import parser_for
from .data.manifest import TEXT_IDENTITY
from .io import file_hash
from . import spm_runtime
from .t2tt_extension import verify_state,report
from .tokenization.factory import tokenizers_from_checkpoint


def start(command,work,backup,snapshot_updates=10000,keep_last=2):
    command=list(map(str,command))
    if command[1:3] != ["-m","alignment_transfer_s2ut.train_t2tt"]:
        raise ValueError("text runtime requires the standard T2TT CLI")
    args=parser_for("train_t2tt").parse_args(command[3:])
    checkpoint=Path(work)/"checkpoint_last.pt"
    if not checkpoint.is_file():checkpoint=Path(work)/"checkpoint_best.pt"
    if checkpoint.is_file():
        from .training import load_checkpoint
        if load_checkpoint(checkpoint).get("manifest_identity")!=TEXT_IDENTITY:
            raise ValueError("legacy audio-bound T2TT stage is read-only; use its original revision or a new text-only stage")
    return spm_runtime.start(command,work,backup,snapshot_updates,keep_last,expected_tokenizer=args.tokenizer)


def status(work,target_updates):
    last,best=verify_state(Path(work),target_updates)
    if last.get("manifest_identity")!=TEXT_IDENTITY:
        raise ValueError("text-only run must use the explicit text-v1 identity")
    tokenizers_from_checkpoint(last)
    return dict(completed_updates=last["step"],best_checkpoint_step=best["step"],best_main_loss=last["best_main_loss"],
        checkpoint_hash=file_hash(Path(work)/"checkpoint_best.pt"),manifest_identity=TEXT_IDENTITY,
        train_text_hash=last["train_hash"],dev_text_hash=last["dev_hash"])


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    train=sub.add_parser("train")
    for key in ("work","backup"):train.add_argument("--"+key,required=True)
    train.add_argument("--snapshot-updates",type=int,default=10000);train.add_argument("--keep-last",type=int,default=2)
    train.add_argument("command",nargs=argparse.REMAINDER)
    child=sub.add_parser("status");child.add_argument("--work",required=True);child.add_argument("--target-updates",type=int,required=True)
    child=sub.add_parser("report");child.add_argument("--work",required=True);child.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.action=="train":result=start(args.command[1:] if args.command[:1]==["--"] else args.command,args.work,args.backup,args.snapshot_updates,args.keep_last)
    elif args.action=="status":result=status(args.work,args.target_updates)
    else:result=report(args.work,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
