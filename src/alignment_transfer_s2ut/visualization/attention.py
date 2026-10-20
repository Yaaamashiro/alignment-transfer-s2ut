from pathlib import Path
import numpy as np
import torch
from ..training import load_checkpoint, checkpoint_model, collate, prior_builder
from ..data.manifest import load_manifest, manifest_hash
from ..config import BackboneConfig
from ..io import ArtifactStore, atomic_write, file_hash


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
    if saved["modality"] != "speech":
        raise ValueError("S2UT checkpoint required")
    model = checkpoint_model(saved, args.device).eval()
    transfer = saved.get("transfer", {})
    prior_type = transfer.get("prior_type", "t2tt")
    if prior_type == "t2tt" and not args.teacher:
        raise ValueError("T2TT visualization requires --teacher")
    units, teacher = ArtifactStore(args.units), ArtifactStore(args.teacher) if args.teacher else None
    cfg = BackboneConfig(**saved["backbone"])
    values = collate(selected, "speech", None, None, saved["backend"], units, cfg.input_feat_per_channel,
                     args.device, saved["recipe"].get("speech_recipe"))
    builder = prior_builder(selected, teacher, units, cfg, manifest_hash(rows), saved["backend"], args.device,
                            values[1], transfer.get("source_mapping", "uniform"), prior_type,
                            transfer.get("monotonic_sigma", .1))
    layer = args.layer if args.layer is not None else transfer.get("layer", -1)
    heads = args.heads if args.heads is not None else transfer.get("heads")
    with torch.no_grad():
        _, raw, guided, prior = model(*values[:3], builder, args.beta, layer, heads,
                                      transfer.get("prior_smoothing", 0))
    _, packet = (teacher if prior_type == "t2tt" else units).get(row["pair_id"])
    count = len(packet["reduced_units"])
    matrices = {"t2tt" if prior_type == "t2tt" else "monotonic": packet["mt_attention"] if prior_type == "t2tt" else
                    prior[0, :count].cpu().numpy(),
                "transferred": prior[0, :count].cpu().numpy(),
                "raw_s2ut": raw[0, :count].cpu().numpy(), "guided_s2ut": guided[0, :count].cpu().numpy()}
    metadata = {"split": row["split"], "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_hash": file_hash(args.checkpoint), "method": saved["method"], "step": saved["step"],
                "beta": args.beta, "lambda": saved["transfer"].get("lambda_align", 0),
                "source_text": row["ja_text"], "target_text": row["en_text"], "layer": layer, "heads": heads,
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
    fig.suptitle(row["pair_id"])
    atomic_write(store.path(row["pair_id"]).with_suffix(".png"),
                 lambda stream: fig.savefig(stream, format="png", dpi=160), args.overwrite)
    plt.close(fig)
    return str(store.path(row["pair_id"]).with_suffix(".png").resolve())
