"""Stage an external text manifest once on the VM, without inspecting audio."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .data.manifest import load_manifest, TEXT_REQUIRED, text_manifest_hash, TEXT_IDENTITY
from .io import write_jsonl,file_hash
from .progress import progress


def prepare(source, output, resume=False):
    source, output = Path(source).resolve(),Path(output).resolve()
    if source == output:
        raise ValueError("text staging output must be separate from the read-only input")
    progress("text manifest: read and validate",detail=str(source))
    rows=load_manifest(source,require_audio=False)
    canonical=[{key:row[key] for key in TEXT_REQUIRED} for row in rows]
    counts=Counter(row["split"] for row in canonical)
    if not counts["train"] or not counts["dev"]:
        raise ValueError("T2TT text manifest requires train and dev; test is optional")
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if load_manifest(output,require_audio=False)!=canonical:
            raise ValueError("text manifest changed; use a new run/output directory")
        progress("text manifest: reused unchanged canonical text")
    else:
        write_jsonl(output,canonical)
    return dict(manifest=str(output),manifest_identity=TEXT_IDENTITY,text_manifest_hash=text_manifest_hash(canonical),
        rows=len(canonical),splits=dict(counts),canonical_sha256=file_hash(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True);parser.add_argument("--output",required=True);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();print(json.dumps(prepare(args.input,args.output,args.resume),ensure_ascii=False))


if __name__=="__main__":main()
