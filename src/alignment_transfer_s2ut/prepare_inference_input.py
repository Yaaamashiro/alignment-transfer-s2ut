"""Export source speech IDs/paths, without forwarding references to inference."""
import argparse
from pathlib import Path
from .data.manifest import load_manifest
from .io import read_jsonl, write_jsonl


def prepare_inference_input(manifest, split, output, resume=False, overwrite=False):
    rows = [{"pair_id": row["pair_id"], "ja_audio": row["ja_audio"]}
            for row in load_manifest(manifest, split, check_audio=False)]
    if not rows:
        raise ValueError("selected split is empty")
    if resume and Path(output).is_file():
        if read_jsonl(output) != rows:
            raise ValueError("source-only inference input changed on resume")
    else:
        write_jsonl(output, rows, overwrite)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    rows = prepare_inference_input(args.manifest, args.split, args.output, args.resume, args.overwrite)
    print(f"exported {len(rows)} source-only samples")


if __name__ == "__main__":
    main()
