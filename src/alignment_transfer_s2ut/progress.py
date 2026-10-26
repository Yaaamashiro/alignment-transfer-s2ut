import sys


def progress(stage, completed=None, total=None, *, detail=None):
    suffix = "" if completed is None else f" {completed}/{total}"
    if detail:
        suffix += " " + detail
    print(f"[{stage}]{suffix}", file=sys.stderr, flush=True)
