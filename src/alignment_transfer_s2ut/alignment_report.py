"""Separate teacher-forced diagnostics and native loss/layer curves; never generation."""
import argparse
from pathlib import Path
import json
import torch
from . import benchmark as b
from .alignment.losses import layer_statistics, worst_layer, METHOD_LABELS
from .alignment_calibration import native_session, write_result
from .io import read_json, read_jsonl, file_hash
from .progress import progress


def _diagnose(plan,record,calibration,output,samples):
    from .s2ut.benchmark_extension import AlignmentDataset
    calibration_value=read_json(calibration)
    totals={key:[] for key in ("kl","mask","mass","weights","mask_weights","counts")}
    pairs=[]
    with native_session(plan,Path(output).parent/"diagnostic-model") as (trainer,model,criterion,extension):
        saved=b.load_native(record["checkpoint"])
        model.load_state_dict(saved["model"],strict=True);model.eval()
        trainer.task.load_dataset("dev")
        dataset=AlignmentDataset(trainer.task.dataset("dev"),plan["transfer"],diagnostic=True)
        with torch.no_grad():
            for i in range(min(samples,len(dataset))):
                sample=dataset.collater([dataset[i]])
                prior=sample["alignment_prior"]
                sample,_=trainer._prepare_sample(sample)
                sample["alignment_prior"]=prior.to(sample["target"].device).float()
                _,_,_,raw=criterion.components(model,sample)
                kl=layer_statistics(sample["alignment_prior"],raw)
                mask=layer_statistics(sample["alignment_prior"],raw,"mask",calibration_value["selected_top_p"])
                _,weights=worst_layer(kl["losses"],calibration_value["tau_kl"],kl["counts"])
                _,mask_weights=worst_layer(mask["losses"],calibration_value["tau_mask"],mask["counts"])
                totals["kl"].append(kl["losses"].cpu());totals["mask"].append(mask["losses"].cpu())
                totals["mass"].append(mask["region_mass"].cpu());totals["weights"].append(weights.cpu())
                totals["counts"].append(kl["counts"].cpu());totals["mask_weights"].append(mask_weights.cpu())
                pairs.append(dataset.dataset.ids[i]);progress("separate dev alignment diagnostics",i+1,min(samples,len(dataset)))
    if not pairs:raise ValueError("empty dev alignment diagnostic subset")
    counts=torch.stack(totals["counts"]).float()
    means={k:((torch.stack(v)*counts).sum(0)/counts.sum(0).clamp_min(1)).tolist() if k in {"kl","mask","mass"}
           else torch.stack(v).float().mean(0).tolist() for k,v in totals.items()}
    valid_layers = counts.sum(0) > 0
    valid_kl = torch.tensor(means["kl"])[valid_layers]
    return write_result(output,dict(format="alignment-diagnostics-v1",checkpoint_hash=record["checkpoint_hash"],
        calibration_hash=file_hash(calibration),split="dev",pair_ids=pairs,teacher_forced_diagnostic_only=True,
        per_layer_kl=means["kl"],kl_mean=float(valid_kl.mean()),kl_worst=float(valid_kl.max()),
        per_layer_mask_loss=means["mask"],teacher_region_mass=means["mass"],valid_rows=counts.sum(0).tolist(),
        kl_weight_distribution=means["weights"],mask_weight_distribution=means["mask_weights"],
        worst_layer_weight_distribution=means["mask_weights"] if record["method"]=="mask_distillation" else means["weights"],
        generation_uses_teacher=False))


def curves(runs,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for run in map(Path,runs):
        rows=read_jsonl(run/"alignment-train.jsonl")
        fig,axes=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
        for key in ("train/main_loss","train/total_loss","alignment/loss"):
            axes[0].plot([r["step"] for r in rows if key in r],[r[key] for r in rows if key in r],label=key)
        for key in sorted({k for r in rows for k in r if k.startswith("alignment/D_")}):
            axes[1].plot([r["step"] for r in rows if key in r],[r[key] for r in rows if key in r],label=key)
        for key in sorted({k for r in rows for k in r if k.startswith("alignment/w_")}):
            axes[2].plot([r["step"] for r in rows if key in r],[r[key] for r in rows if key in r],label=key)
        for axis in axes:
            axis.set_xlabel("Completed updates")
            if axis.lines:axis.legend()
        fig.suptitle(METHOD_LABELS.get(run.name,run.name));path=output/(run.name+".png");fig.savefig(path,dpi=140);plt.close(fig);paths.append(str(path))
    return dict(curves=paths)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    child=sub.add_parser("curves");child.add_argument("--runs",nargs="+",required=True);child.add_argument("--output",required=True)
    child=sub.add_parser("diagnose")
    for key in ("baseline","record","manifest","units","teacher","calibration","output"):
        child.add_argument("--"+key,required=True)
    child.add_argument("--samples",type=int,default=32)
    args=parser.parse_args(argv)
    if args.action=="curves":result=curves(args.runs,args.output)
    else:
        record=b.read_record(args.record)
        plan=b.make_plan(args.baseline,Path(args.output).parent/"diagnostic-model",manifest=args.manifest,
            units=args.units,teacher=args.teacher,calibration=args.calibration)
        if record["scientific_config"]!=plan["scientific_config"]:raise ValueError("diagnostic recipe differs")
        if record["method"]!="baseline" and (record["transfer"].get("calibration_hash")!=file_hash(args.calibration) or
                record["transfer"].get("artifacts")!=plan["transfer"]["artifacts"]):
            raise ValueError("diagnostic calibration/teacher differs from trained run")
        result=_diagnose(plan,record,args.calibration,args.output,args.samples)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":main()
