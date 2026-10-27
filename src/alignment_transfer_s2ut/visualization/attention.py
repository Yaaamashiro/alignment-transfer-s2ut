from pathlib import Path
import numpy as np
import torch
from ..training import load_checkpoint, checkpoint_model, collate, prior_builder
from ..data.manifest import load_manifest, manifest_hash
from ..config import BackboneConfig
from ..io import ArtifactStore, atomic_write, file_hash, read_json, digest


def native_inputs(args, saved, row):
    """Use the saved native task/frontend/dictionaries, then capture via the wrapper.

    Visualization requires the run plan and prepared data; teacher-free inference
    continues to use the independent stock generator path.
    """
    from fairseq import tasks, utils
    from .. import benchmark as b
    from ..s2ut import benchmark_extension  # Register local task/model.
    from ..s2ut.fairseq_extension import TransferAttention, verify_fairseq
    from ..s2ut.model import NativeModel
    from ..colab_runtime import native_inputs as physical_inputs
    verify_fairseq()
    plan_path = Path(args.checkpoint).resolve().parent.parent / "native-run.json"
    plan = read_json(plan_path)
    if (plan["contract_hash"] != digest({k:v for k,v in plan.items() if k != "contract_hash"}) or
            saved.get("extra_state",{}).get("alignment_contract_hash") != plan["contract_hash"]):
        raise ValueError("native visualization checkpoint/run contract mismatch")
    cfg = b.native_config(b.plain_config(saved))
    if b.scientific_config(b.plain_config(saved)) != plan["scientific_config"]:
        raise ValueError("native visualization scientific configuration differs")
    if cfg.task._name == "benchmark_alignment_speech":
        cfg.task.alignment_config = str(plan_path)
    utils.set_torch_seed(cfg.common.seed)
    with physical_inputs():
        task = tasks.setup_task(cfg.task)
        network = task.build_model(cfg.model)
        network.load_state_dict(saved["model"], strict=True)
        task.load_dataset(row["split"])
        dataset = task.dataset(row["split"])
        if isinstance(dataset, benchmark_extension.AlignmentDataset):
            dataset = dataset.dataset  # Construct the requested prior exactly once below.
        if not hasattr(dataset,"ids") or row["pair_id"] not in dataset.ids:
            raise ValueError("pair_id missing from the native prepared split")
        sample = dataset.collater([dataset[dataset.ids.index(row["pair_id"])]])
    config = b.backbone(b.plain_config(saved))
    for block in network.decoder.layers:
        block.encoder_attn.__class__ = TransferAttention
    model = NativeModel(network,config,"speech").to(args.device).eval()
    values = tuple(sample["net_input"][key].to(args.device) for key in ("src_tokens","src_lengths","prev_output_tokens"))
    transfer = {**plan["transfer"], "source_mapping":"frontend", "prior_smoothing":plan["transfer"].get("smoothing",.05)}
    args.teacher = args.teacher or transfer.get("teacher")
    normalized = {**saved,"modality":"speech","backend":"fairseq","backbone":config.asdict(),"recipe":{},
                  "transfer":transfer,"method":transfer["method"],"step":b.checkpoint_updates(saved)}
    return normalized,model,values


def visualize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load_manifest(args.manifest)
    selected = [r for r in rows if r["pair_id"] == args.pair_id]
    if len(selected) != 1:
        raise ValueError("pair_id must exist exactly once")
    row = selected[0]
    saved = load_checkpoint(args.checkpoint)
    values = None
    if "cfg" in saved and "modality" not in saved:
        saved, model, values = native_inputs(args,saved,row)
    else:
        model = checkpoint_model(saved, args.device).eval()
    if saved["modality"] != "speech":
        raise ValueError("S2UT checkpoint required")
    transfer = saved.get("transfer", {})
    all_layers = transfer.get("all_layers",False) or getattr(args,"all_layers",False)
    prior_type = transfer.get("prior_type", "t2tt")
    if prior_type == "t2tt" and not args.teacher:
        raise ValueError("T2TT visualization requires --teacher")
    units, teacher = ArtifactStore(args.units), ArtifactStore(args.teacher) if args.teacher else None
    cfg = BackboneConfig(**saved["backbone"])
    if values is None:
        values = collate(selected, "speech", None, None, saved["backend"], units, cfg.input_feat_per_channel,
                         args.device, saved["recipe"].get("speech_recipe"))
    layer = args.layer if args.layer is not None else transfer.get("layer", -1)
    if not -cfg.decoder_layers <= layer < cfg.decoder_layers:
        raise ValueError("visualization layer out of range")
    heads = args.heads if args.heads is not None else transfer.get("heads")
    builder = prior_builder(selected, teacher, units, cfg, manifest_hash(rows), saved["backend"], args.device,
                            values[1], transfer.get("source_mapping", "uniform"), prior_type,
                            transfer.get("monotonic_sigma", .1), all_layers=all_layers, layer=layer,
                            teacher_signature=transfer.get("artifacts",{}).get("teacher_signature",saved.get("teacher_signature")))
    with torch.no_grad():
        _, raw, guided, prior = model(*values[:3], builder, args.beta, layer, heads,
                                      transfer.get("prior_smoothing", 0),all_layers=all_layers)
    _, packet = (teacher if prior_type == "t2tt" else units).get(row["pair_id"])
    count = len(packet["reduced_units"])
    teacher_matrix = packet["mt_attention_layers"][layer] if prior_type=="t2tt" and "mt_attention_layers" in packet else packet.get("mt_attention")
    if all_layers:
        raw,guided,prior=[v[:,layer] for v in (raw,guided,prior)]
    matrices = {"t2tt" if prior_type == "t2tt" else "monotonic": teacher_matrix if prior_type == "t2tt" else
                    prior[0, :count].cpu().numpy(),
                "transferred": prior[0, :count].cpu().numpy(),
                "raw_s2ut": raw[0, :count].cpu().numpy(), "guided_s2ut": guided[0, :count].cpu().numpy()}
    metadata = {"split": row["split"], "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_hash": file_hash(args.checkpoint), "method": saved["method"], "step": saved["step"],
                "beta": args.beta, "lambda": saved["transfer"].get("lambda_align", 0),
                "source_text": row["ja_text"], "target_text": row["en_text"], "layer": layer, "heads": heads,"all_layers":all_layers,
                "prior_type": prior_type, "source_mapping": transfer.get("source_mapping", "uniform"),
                "prior_smoothing": transfer.get("prior_smoothing", 0)}
    store = ArtifactStore(args.output)
    store.put(row["pair_id"], metadata, matrices, args.overwrite)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (name, matrix) in zip(axes.flat, matrices.items()):
        heatmap = axis.imshow(matrix, aspect="auto", origin="upper", interpolation="nearest", vmin=0, vmax=1)
        axis.set(title=name, xlabel="Source text token" if name == "t2tt" else "Speech encoder position",
                 ylabel="Target text token" if name == "t2tt" else "Reduced target unit")
        if matrix.shape[1] <= 16:
            axis.set_xticks(np.arange(matrix.shape[1]))
        if matrix.shape[0] <= 16:
            axis.set_yticks(np.arange(matrix.shape[0]))
    fig.colorbar(heatmap, ax=axes.ravel().tolist(), label="Attention probability", shrink=0.85)
    fig.suptitle(f"{row['pair_id']} | decoder layer {layer % cfg.decoder_layers}")
    atomic_write(store.path(row["pair_id"]).with_suffix(".png"),
                 lambda stream: fig.savefig(stream, format="png", dpi=160), args.overwrite)
    plt.close(fig)
    return str(store.path(row["pair_id"]).with_suffix(".png").resolve())
