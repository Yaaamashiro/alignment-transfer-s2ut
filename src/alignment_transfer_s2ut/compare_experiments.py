"""Fail if a comparison changes a controlled variable beyond alignment transfer."""
import argparse
import json
from pathlib import Path
from .io import read_json, write_json

CONTROLLED = ("backbone", "backend", "modality", "source_vocabulary", "target_vocabulary", "recipe",
              "runtime", "implementation_hash", "train_hash", "dev_hash", "unit_config_hash",
              "initialization_hash", "checkpoint_selection", "planned_max_updates", "auxiliary")


def compare_experiments(roots, allow_scope_ablation=False):
    experiments = [read_json(Path(root) / "experiment.json") for root in roots]
    def selected_layer(e):
        transfer=e.get("transfer",{})
        if transfer.get("all_layers",False):
            return None
        layer=transfer.get("layer",-1)
        count=e.get("backbone",{}).get("decoder_layers")
        return layer % count if count else layer
    identities = [(e["method"], e.get("transfer", {}).get("prior_type", "t2tt")
                   if e["method"] != "baseline" else None, e.get("transfer",{}).get("all_layers",False),
                   selected_layer(e) if e["method"] != "baseline" else None) for e in experiments]
    if len(experiments) < 2 or len(set(identities)) != len(identities) or not any(
            e["method"] == "baseline" for e in experiments):
        raise ValueError("supply a baseline and distinct method/prior runs")
    differences = {key: [e.get(key) for e in experiments] for key in CONTROLLED
                   if any(e.get(key) != experiments[0].get(key) for e in experiments[1:])}
    transfers = [e.get("transfer", {}) for e in experiments if e["method"] != "baseline"]
    scopes=[dict(all_layers=scope,layer=layer) for method,prior,scope,layer in identities if method != "baseline"]
    scope_varies=any(s != scopes[0] for s in scopes[1:])
    ablation=allow_scope_ablation and scope_varies
    for key in ("source_mapping", "prior_smoothing", "layer", "heads"):
        if key=="layer" and ablation:
            continue
        if any(t.get(key) != transfers[0].get(key) for t in transfers[1:]):
            differences["transfer/" + key] = [t.get(key) for t in transfers]
    if not ablation and transfers and any(t.get("all_layers",False)!=transfers[0].get("all_layers",False) for t in transfers[1:]):
        differences["transfer/all_layers"]=[t.get("all_layers",False) for t in transfers]
    if ablation:
        conditions={(method,prior) for method,prior,scope,layer in identities if method != "baseline"}
        if len(conditions)!=1:
            differences["ablation/method_or_prior"] = sorted(conditions)
    # Runs using the same method must also use the same schedule/loss settings.
    for method in {e["method"] for e in experiments} - {"baseline"}:
        selected = [e["transfer"] for e in experiments if e["method"] == method]
        for key in ("lambda_align", "beta", "beta_start", "beta_end", "anneal_steps", "monotonic_sigma"):
            if any(t.get(key) != selected[0].get(key) for t in selected[1:]):
                differences[method + "/" + key] = [t.get(key) for t in selected]
    teacher_signatures = [e["teacher_signature"] for e in experiments if e["method"] != "baseline"
                          and e.get("transfer", {}).get("prior_type", "t2tt") == "t2tt"]
    if teacher_signatures and any(value != teacher_signatures[0] for value in teacher_signatures[1:]):
        differences["teacher_signature"] = teacher_signatures
    result = {"fair_comparison": not differences, "methods": [e["method"] for e in experiments],
            "conditions": [{"method": method, "prior_type": prior,"all_layers":scope,
                            **({"layer":layer} if allow_scope_ablation else {})} for method, prior,scope,layer in identities],
            "controlled_fields": list(CONTROLLED), "differences": differences}
    if allow_scope_ablation:
        result["independent_variables"]={"transfer_scope":scopes} if ablation else {}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--allow-scope-ablation",action="store_true")
    args = parser.parse_args()
    result = compare_experiments(args.runs,args.allow_scope_ablation)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if not result["fair_comparison"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
