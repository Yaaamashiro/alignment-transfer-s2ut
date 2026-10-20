import sys


def progress(stage, completed=None, total=None):
    suffix = "" if completed is None else f" {completed}/{total}"
    print(f"[{stage}]{suffix}", file=sys.stderr, flush=True)
