"""Teacher top-p analysis and no-update native fairseq gradient calibration."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import math
import torch
from . import benchmark as b
from .io import digest, read_json, write_json
from .progress import progress
from .alignment.losses import layer_statistics, worst_layer, top_p_mask, distributions
from .alignment.calibration import calibrate_tau, calibrate_lambda, gradient_norm, binding, runtime_identity


def write_result(output, result):
    result["artifact_hash"] = digest(result)
    if Path(output).is_file():
        if read_json(output) != result:
            raise ValueError("analysis/calibration changed; use a new output directory")
    else:
        write_json(output, result)
    return result


@contextmanager
def native_session(plan, directory):
    from copy import deepcopy
    from fairseq import tasks, utils
    from fairseq.trainer import Trainer
    from .s2ut import benchmark_extension as extension
    from .colab_runtime import native_inputs
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = deepcopy(plan)
    plan["config"]["task"]["alignment_config"] = str((directory / "native-run.json").resolve())
    plan["contract_hash"] = digest({k:v for k,v in plan.items() if k!="contract_hash"})
    write_json(directory / "native-run.json", plan, overwrite=True)
    cfg = b.native_config(plan["config"])
    if cfg.common.fp16 and not torch.cuda.is_available():
        raise RuntimeError("calibration requires the registered CUDA FP16 recipe; no precision fallback")
    with native_inputs():
        import random
        import numpy as np
        random.seed(cfg.common.seed)
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)
        task = tasks.setup_task(cfg.task)
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        trainer = Trainer(cfg, task, model, criterion)
        model = trainer.model
        model.train()
        for name in task.multitask_tasks:
            criterion.set_multitask_loss_weight(name, task.multitask_tasks[name].args.get_loss_weight(0))
        yield trainer, model, criterion, extension


def logical_batches(trainer, count):
    frequency = trainer.cfg.optimization.update_freq[0]
    epoch, iterator = 0, iter(())
    for _ in range(count):
        logical = []
        for _ in range(frequency):
            try:
                sample = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(trainer.get_train_iterator(epoch).next_epoch_itr(shuffle=True))
                try:
                    sample = next(iterator)
                except StopIteration as error:
                    raise ValueError("empty calibration training iterator") from error
            prior = sample["alignment_prior"]
            prepared, dummy = trainer._prepare_sample(sample)
            if dummy:
                raise ValueError("dummy sample is not valid calibration data")
            prepared["alignment_prior"] = prior.to(prepared["target"].device).float()
            logical.append(prepared)
        yield logical


def sample_ids(trainer, samples):
    dataset = trainer.task.dataset("train")
    return [[dataset.dataset.ids[int(i)] for i in sample["id"]] for sample in samples]


def coverage_summary(records):
    records = torch.cat(records)
    def statistics(selected):
        values, entropies = selected[:,0], selected[:,2]
        correlation = None
        if len(values)>1 and float(values.std())>0 and float(entropies.std())>0:
            correlation=float(torch.corrcoef(torch.stack([values,entropies]))[0,1])
        return dict(mean=float(values.mean()),median=float(values.median()),
            p25=float(values.quantile(.25)),p75=float(values.quantile(.75)),p90=float(values.quantile(.9)),
            valid_rows=len(selected),selected_frames=int(selected[:,1].sum()),
            mean_teacher_entropy=float(entropies.mean()),entropy_coverage_correlation=correlation,
            histogram=torch.histc(values.float(),bins=10,min=0,max=1).long().tolist())
    result=statistics(records)
    result["by_layer"]={str(int(layer)):statistics(records[records[:,3]==layer]) for layer in records[:,3].unique().tolist()}
    names={0:"<=100",1:"101-300",2:">300"}
    result["by_source_length"]={names[int(group)]:statistics(records[records[:,4]==group]) for group in records[:,4].unique().tolist()}
    return result


def diagnostics(plan, output, batches=16, candidates=(.8,.9,.95)):
    if batches < 1 or not {.8,.9,.95}.issubset(candidates) or any(not 0 < p <= 1 for p in candidates):
        raise ValueError("diagnostics need positive batches and the three primary top-p candidates")
    records = {str(p):[] for p in candidates}
    ids = []
    with native_session(plan, Path(output).parent / "diagnostic-session") as (trainer, model, criterion, extension):
        for index, samples in enumerate(logical_batches(trainer, batches)):
            ids.append(sample_ids(trainer, samples))
            for sample in samples:
                prior = sample["alignment_prior"].float()
                if prior.ndim == 3:
                    prior = prior[:,None]
                normalized, _, valid, _ = distributions(prior, prior)
                lengths = model.encoder.subsample.get_out_seq_lens_tensor(sample["net_input"]["src_lengths"])
                padding = torch.arange(prior.size(-1), device=prior.device)[None] >= lengths[:,None]
                entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(-1)
                for p in candidates:
                    mask = top_p_mask(prior, p, padding)
                    indexes=valid.nonzero()
                    row_lengths=lengths[indexes[:,0]]
                    selected=mask.sum(-1)[valid]
                    bins=(row_lengths>100).long()+(row_lengths>300).long()
                    records[str(p)].append(torch.stack([selected/row_lengths,selected,entropy[valid],indexes[:,1],bins],-1).detach().cpu().double())
            progress("top-p diagnostics", index+1, batches)
    if any(not values for values in records.values()):
        raise ValueError("top-p diagnostics found no valid teacher rows")
    return write_result(output, dict(format="alignment-top-p-v1", binding=binding(plan),
        implementation_hash=plan["implementation_hash"], seed=plan["config"]["common"]["seed"],
        calibration_pair_ids=ids, calibration_pair_hash=digest(ids), number_of_batches=batches,
        candidates={key:coverage_summary(values) for key,values in records.items()}, selected_top_p=None))


def calibrate(plan, output, analysis, top_p, batches=16, alpha=.5, ratio=3.):
    if batches < 1 or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("positive calibration batches/alpha and target ratio > 1 required")
    report = read_json(analysis)
    if (report.get("artifact_hash") != digest({k:v for k,v in report.items() if k != "artifact_hash"}) or
        report.get("binding") != binding(plan) or report.get("implementation_hash") != plan["implementation_hash"] or
        report.get("number_of_batches") != batches or str(top_p) not in report.get("candidates",{})):
        raise ValueError("matching top-p diagnostics and an explicit analyzed candidate are required")
    from fairseq import utils
    from .training import file_tensor_hash
    layer_batches = {"kl":[], "mask":[]}
    norms = {"kl":[], "mask":[]}
    observations, ids = [], []
    with native_session(plan, Path(output).parent / "calibration-session") as (trainer, model, criterion, extension):
        initial = digest({k:file_tensor_hash(v) for k,v in model.state_dict().items()})
        rng = extension.rng_state(model)
        parameters = [p for block in model.decoder.layers for p in block.encoder_attn.parameters() if p.requires_grad]
        # Pass 1 estimates tau. Same initialization/data/RNG are replayed in pass 2.
        with torch.no_grad():
            for index, samples in enumerate(logical_batches(trainer, batches)):
                utils.set_torch_seed(trainer.cfg.common.seed + index)
                ids.append(sample_ids(trainer, samples))
                total = sum(s["target"].size(0) if criterion.sentence_avg else s["ntokens"] for s in samples)
                values = {"kl":None, "mask":None}
                for sample in samples:
                    _, size, _, raw = criterion.components(model, sample)
                    for kind in values:
                        losses = layer_statistics(sample["alignment_prior"], raw, kind, top_p)["losses"].cpu().double()
                        values[kind] = losses * size/total if values[kind] is None else values[kind] + losses * size/total
                for kind in values:
                    layer_batches[kind].append(values[kind].tolist())
                progress("calibration: layer scale", index+1, batches)
        if ids != report["calibration_pair_ids"]:
            raise ValueError("diagnostic/calibration sample order differs")
        temperatures = {kind:calibrate_tau(values, ratio) for kind,values in layer_batches.items()}
        extension.restore_rng(model, rng)
        for index, samples in enumerate(logical_batches(trainer, batches)):
            utils.set_torch_seed(trainer.cfg.common.seed + index)
            if sample_ids(trainer,samples) != ids[index]:
                raise ValueError("calibration replay changed sample order")
            total = sum(s["target"].size(0) if criterion.sentence_avg else s["ntokens"] for s in samples)
            gradients = {key:[None]*len(parameters) for key in ("task","kl","mask")}
            weight_logs = {"kl":[],"mask":[]}
            for sample in samples:
                task_loss, size, _, raw = criterion.components(model, sample)
                objectives = {"task":task_loss / total}
                for kind in ("kl","mask"):
                    stats = layer_statistics(sample["alignment_prior"], raw, kind, top_p)
                    value, weights = worst_layer(stats["losses"], temperatures[kind]["tau"], stats["counts"])
                    objectives[kind] = value * size / total
                    weight_logs[kind].append(weights.detach().cpu().tolist())
                for key in ("task","kl","mask"):
                    grads = torch.autograd.grad(objectives[key], parameters, retain_graph=key!="mask", allow_unused=True)
                    for i,g in enumerate(grads):
                        if g is not None:
                            g = g.detach().float()
                            gradients[key][i] = g if gradients[key][i] is None else gradients[key][i] + g
            values = {key:gradient_norm(value) for key,value in gradients.items()}
            for kind in ("kl","mask"):
                norms[kind].append((values["task"],values[kind]))
            observations.append(dict(gradient_norms=values, worst_weights=weight_logs, virtual_seed=trainer.cfg.common.seed+index))
            progress("calibration: gradient norms (no optimizer step)", index+1,batches)
        after = digest({k:file_tensor_hash(v) for k,v in model.state_dict().items()})
        if after != initial or trainer.get_num_updates() != 0 or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("calibration changed model/update/gradient state")
        extension.restore_rng(model, rng)
    result = dict(format="alignment-calibration-v1", runtime=runtime_identity(), binding=binding(plan), implementation_hash=plan["implementation_hash"],
        initialization_identity=initial, baseline_checkpoint_identity=b.read_record(plan["baseline_record"])["checkpoint_hash"],
        seed=plan["config"]["common"]["seed"], calibration_pair_ids=ids, calibration_pair_hash=digest(ids),
        number_of_batches=batches, alpha=alpha,target_worst_median_ratio=ratio,
        lambda_kl=calibrate_lambda(norms["kl"],alpha),lambda_mask=calibrate_lambda(norms["mask"],alpha),
        tau_kl=temperatures["kl"]["tau"],tau_mask=temperatures["mask"]["tau"], tau_estimates=temperatures,
        per_layer_losses=layer_batches,per_batch=observations,top_p_diagnostics=report,
        selected_top_p=top_p, optimizer_updates=0, normalization="token-weighted microbatch LogSumExp; native logical accumulation")
    return write_result(output,result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["top-p","calibrate"])
    for key in ("baseline","manifest","units","teacher","output"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--batches",type=int,default=16)
    parser.add_argument("--diagnostics")
    parser.add_argument("--mask-top-p",type=float)
    parser.add_argument("--alpha",type=float,default=.5)
    parser.add_argument("--target-ratio",type=float,default=3.)
    args=parser.parse_args(argv)
    plan=b.make_plan(args.baseline,Path(args.output).parent / (args.action+"-session"),"distillation", "t2tt",
        args.manifest,args.units,args.teacher,all_layers=True,for_calibration=True)
    # Session path and config alignment_config must agree.
    directory=Path(args.output).parent / ("diagnostic-session" if args.action=="top-p" else "calibration-session")
    plan["config"]["task"]["alignment_config"]=str((directory/"native-run.json").resolve())
    if args.action=="top-p":result=diagnostics(plan,args.output,args.batches)
    else:
        if not args.diagnostics or args.mask_top_p is None:parser.error("calibrate requires --diagnostics and explicit --mask-top-p")
        result=calibrate(plan,args.output,args.diagnostics,args.mask_top_p,args.batches,args.alpha,args.target_ratio)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
