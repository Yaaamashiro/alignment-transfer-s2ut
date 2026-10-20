"""Fail if a comparison changes a controlled variable beyond alignment transfer."""
import argparse
import json
from pathlib import Path
from .io import read_json, write_json

CONTROLLED = ("backbone", "backend", "modality", "source_vocabulary", "target_vocabulary", "recipe",
              "runtime", "implementation_hash", "train_hash", "dev_hash", "unit_config_hash",
              "initialization_hash", "checkpoint_selection", "planned_max_updates")


def compare_experiments(roots):
    experiments = [read_json(Path(root) / "experiment.json") for root in roots]
    if len(experiments) != 4 or {e["method"] for e in experiments} != {
            "baseline", "distillation", "guidance", "annealed_guidance"}:
        raise ValueError("supply one run for each of the four methods")
    differences = {key: [e.get(key) for e in experiments] for key in CONTROLLED
                   if any(e.get(key) != experiments[0].get(key) for e in experiments[1:])}
    teacher_signatures = [e["teacher_signature"] for e in experiments if e["method"] != "baseline"]
    if any(value != teacher_signatures[0] for value in teacher_signatures[1:]):
        differences["teacher_signature"] = teacher_signatures
    return {"fair_comparison": not differences, "methods": [e["method"] for e in experiments],
            "controlled_fields": list(CONTROLLED), "differences": differences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare_experiments(args.runs)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if not result["fair_comparison"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
