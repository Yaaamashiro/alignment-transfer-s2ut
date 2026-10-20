"""Audit alignment coverage and uncertainty before model training (no GPU needed)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .alignment.timing import encoder_time_spans, output_lengths, monotonic_prior
from .alignment.projection import source_mapping, target_mapping
from .config import BackboneConfig
from .data.audio import features
from .data.manifest import load_manifest, manifest_hash
from .guidance.attention import smooth_prior, entropy
from .io import ArtifactStore, write_json, write_jsonl, file_hash, atomic_write
from .training import prior_builder
from .progress import progress


def diagnose(args):
    if args.max_samples < 1 or args.heatmaps < 0 or not 0 <= args.prior_smoothing <= 1:
        raise ValueError("invalid audit size, heatmap count or smoothing")
    rows = load_manifest(args.manifest)
    identity = manifest_hash(rows)
    population = [row for row in rows if row["split"] == args.split]
    if not population:
        raise ValueError("requested split is empty")
    indices = np.random.default_rng(args.seed).permutation(len(population))[:args.max_samples]
    selected = [population[int(i)] for i in indices]
    config = BackboneConfig.load(args.config)
    units, teacher = ArtifactStore(args.units), ArtifactStore(args.teacher)
    root = Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError("audit output exists; use a new directory or --overwrite")
    root.mkdir(parents=True, exist_ok=True)
    matrix_store = ArtifactStore(root / "heatmaps")
    metrics, signatures = [], []
    for index, row in enumerate(selected):
        meta, packet = teacher.get(row["pair_id"])
        signature = {key: meta[key] for key in ("mt_checkpoint_hash", "tokenizer_hash", "ctc_config_hash", "unit_config_hash")}
        if signatures and signature != signatures[0]:
            raise ValueError("mixed teacher artifacts in audit")
        signatures.append(signature)
        backend = meta["backend"]
        feature_length = len(features(row["ja_audio"], backend, config.input_feat_per_channel))
        length = int(output_lengths(feature_length, config, backend))
        builder = prior_builder([row], teacher, units, config, identity, backend, "cpu",
                                [feature_length], args.source_mapping)
        prior = builder(torch.tensor([length]), len(packet["reduced_units"]) + 1)[:, :-1]
        effective = smooth_prior(prior, alpha=args.prior_smoothing)
        duration = float(packet["source_duration"])
        if args.source_mapping == "frontend":
            frames = encoder_time_spans(feature_length, length, config, backend, duration)
        else:
            bounds = np.linspace(0, duration, length + 1)
            frames = np.stack([bounds[:-1], bounds[1:]], -1)
        src = source_mapping(packet["source_spans"], duration, length, frames)
        tgt = target_mapping(packet["target_spans"], packet["unit_spans"])
        mt = packet["mt_attention"]
        src_valid = (src.sum(-1) > 0) & (mt.sum(0) > 0)
        tgt_valid = (tgt.sum(0) > 0) & (mt.sum(-1) > 0)
        valid_rows = prior[0].sum(-1) > 0
        def span_valid(spans):
            return (spans[:, 0] >= 0) & (spans[:, 1] > spans[:, 0])
        values = {"pair_id": row["pair_id"], "split": row["split"], "backend": backend,
                  "source_tokens": len(src), "valid_source_tokens": int(src_valid.sum()),
                  "ctc_source_tokens": int(span_valid(packet["source_spans"]).sum()),
                  "target_tokens": mt.shape[0], "valid_target_tokens": int(tgt_valid.sum()),
                  "ctc_target_tokens": int(span_valid(packet["target_spans"]).sum()),
                  "unit_rows": prior.size(1), "valid_unit_rows": int(valid_rows.sum()),
                  "encoder_frames": length, "feature_frames": feature_length,
                  "prior_entropy_sum": float(entropy(prior[0])[valid_rows].sum()),
                  "smoothed_entropy_sum": float(entropy(effective[0])[valid_rows].sum()),
                  "zero_cells": int((prior[0][valid_rows] == 0).sum()),
                  "valid_cells": int(valid_rows.sum()) * length,
                  "teacher_packet_hash": file_hash(teacher.path(row["pair_id"]))}
        for side in ("source_tokens", "target_tokens", "unit_rows"):
            values["valid_" + side + "_ratio"] = values["valid_" + side] / max(1, values[side])
        metrics.append(values)
        if index < args.heatmaps:
            _, unit_packet = units.get(row["pair_id"])
            matrices = {"t2tt": mt, "projected": prior[0].numpy(), "smoothed": effective[0].numpy(),
                        "monotonic": monotonic_prior(packet["unit_spans"], float(unit_packet["duration"]),
                                                     frames, duration, .1)}
            matrix_store.put(row["pair_id"], {**values, "source_mapping": args.source_mapping,
                             "prior_smoothing": args.prior_smoothing}, matrices, args.overwrite)
            _plot(matrices, row["pair_id"], matrix_store.path(row["pair_id"]).with_suffix(".png"), args.overwrite)
        progress("alignment audit", index + 1, len(selected))
    count_keys = ("source_tokens", "valid_source_tokens", "ctc_source_tokens", "target_tokens",
                  "valid_target_tokens", "ctc_target_tokens", "unit_rows", "valid_unit_rows",
                  "prior_entropy_sum", "smoothed_entropy_sum", "zero_cells", "valid_cells")
    totals = {key: sum(row[key] for row in metrics) for key in count_keys}
    summary = {"split": args.split, "population_pairs": len(population), "audited_pairs": len(selected),
               "seed": args.seed, "manifest_hash": identity, "teacher_signature": signatures[0],
               "source_mapping": args.source_mapping, "prior_smoothing": args.prior_smoothing,
               "counts": totals, "heatmaps": min(args.heatmaps, len(selected)),
               "quality_scope": "Coverage and uncertainty diagnostics; not forced-alignment accuracy or ASR CER."}
    for side in ("source_tokens", "target_tokens", "unit_rows"):
        summary["valid_" + side + "_ratio"] = totals["valid_" + side] / max(1, totals[side])
    for key in ("prior", "smoothed"):
        summary[key + "_entropy"] = totals[key + "_entropy_sum"] / max(1, totals["valid_unit_rows"])
    summary["zero_cell_ratio"] = totals["zero_cells"] / max(1, totals["valid_cells"])
    write_jsonl(root / "pairs.jsonl", metrics, args.overwrite)
    write_json(root / "summary.json", summary, args.overwrite)
    return summary


def _plot(matrices, pair_id, path, overwrite):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (name, matrix) in zip(axes.flat, matrices.items()):
        heatmap = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        axis.set(title=name, xlabel="Source character" if name == "t2tt" else "Acoustic encoder position",
                 ylabel="Target character" if name == "t2tt" else "Reduced target unit")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(heatmap, ax=axes.ravel().tolist(), label="Probability")
    fig.suptitle(pair_id)
    atomic_write(path, lambda stream: fig.savefig(stream, format="png", dpi=140), overwrite)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "teacher", "units", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--heatmaps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-mapping", choices=["frontend", "uniform"], default="frontend")
    parser.add_argument("--prior-smoothing", type=float, default=.05)
    parser.add_argument("--overwrite", action="store_true")
    import json
    print(json.dumps(diagnose(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
