"""Audit alignment coverage and uncertainty before model training (no GPU needed)."""
import argparse
import csv
import io
import os
from pathlib import Path
import numpy as np
import torch
from .alignment.timing import encoder_time_spans, output_lengths, monotonic_prior
from .alignment.projection import normalize, source_mapping, target_mapping
from .alignment.layers import attention_spec, teacher_signature
from .config import BackboneConfig
from .data.audio import features
from .data.manifest import load_manifest, manifest_hash
from .guidance.attention import smooth_prior, entropy
from .io import ArtifactStore, write_json, write_jsonl, file_hash, atomic_write
from .training import prior_builder
from .progress import progress


def span_valid(spans):
    return (spans[:, 0] >= 0) & (spans[:, 1] > spans[:, 0])


def timing_metrics(packet, side, duration):
    tokens = packet[side + "_spans"]
    codepoints = packet.get(side + "_codepoint_spans", tokens)
    valid = codepoints[span_valid(codepoints)]
    gaps, end = [], 0.
    for start, stop in sorted(valid.tolist()):
        if start > end:
            gaps.append([end, start])
        end = max(end, stop)
    if end < duration:
        gaps.append([end, duration])
    codepoint_offsets = packet.get(side + "_token_spans")
    if codepoint_offsets is None:
        indices = np.arange(len(tokens))
        codepoint_offsets = np.stack([indices, indices + 1], axis=-1)
    widths = np.diff(codepoint_offsets, axis=1).ravel()
    result = {"ctc_duration_seconds": duration, "ctc_gap_intervals_seconds": gaps,
              "ctc_gap_seconds": sum(stop - start for start, stop in gaps),
              "masked_codepoints": len(codepoints) - len(valid),
              "token_codepoint_offsets": codepoint_offsets.tolist(), "token_time_spans_seconds": tokens.tolist()}
    for name, selected in (("single_codepoint", widths == 1), ("multi_codepoint", widths > 1)):
        spans = tokens[selected & span_valid(tokens)]
        result[name + "_tokens"] = int(selected.sum())
        result[name + "_valid_tokens"] = len(spans)
        result[name + "_span_seconds_sum"] = float((spans[:, 1] - spans[:, 0]).sum())
    return result


def projection_metrics(mt, src, tgt, projected):
    # Source rows with no time overlap are zero in the source mapping.
    teacher = normalize(mt)
    lost = teacher[:, src.sum(-1) == 0].sum(-1)
    text_rows = teacher.sum(-1) > 0
    unit_mass = tgt @ teacher.sum(-1)
    unit_lost = tgt @ lost
    rows = projected.sum(-1) > 0
    valid_cells = int(rows.sum()) * projected.shape[-1]
    return {"source_tokens": len(src), "valid_source_tokens": int(((src.sum(-1) > 0) & (teacher.sum(0) > 0)).sum()),
            "target_tokens": len(teacher), "valid_target_tokens": int(((tgt.sum(0) > 0) & text_rows).sum()),
            "unit_rows": len(projected), "valid_unit_rows": int(rows.sum()),
            "prior_entropy_sum": float(entropy(projected)[rows].sum()),
            "zero_cells": int((projected[rows] == 0).sum()), "valid_cells": valid_cells,
            "teacher_rows_with_mass": int(text_rows.sum()), "unmapped_source_mass_sum": float(lost.sum()),
            "unit_teacher_mass_sum": float(unit_mass.sum()),
            "unit_unmapped_source_mass_sum": float(unit_lost.sum()),
            "unit_rows_without_target_overlap": int((tgt.sum(-1) == 0).sum()),
            "unit_rows_with_teacher_mass": int((unit_mass > 0).sum()),
            "unit_rows_with_teacher_mass_but_zero_projection": int(((unit_mass > 0) & ~rows.numpy()).sum()),
            "projected_row_sum_max_error": float((projected[rows].sum(-1) - 1).abs().max()) if rows.any() else 0.}


def summarize_projection(records):
    counts = {key: sum(row[key] for row in records) for key in records[0]
              if key not in {"layer", "projected_row_sum_max_error"}}
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None
    return {**counts, "valid_source_tokens_ratio": ratio("valid_source_tokens", "source_tokens"),
            "valid_target_tokens_ratio": ratio("valid_target_tokens", "target_tokens"),
            "valid_unit_rows_ratio": ratio("valid_unit_rows", "unit_rows"),
            "prior_entropy": ratio("prior_entropy_sum", "valid_unit_rows"),
            "zero_cell_ratio": ratio("zero_cells", "valid_cells"),
            "unmapped_source_mass_ratio": ratio("unmapped_source_mass_sum", "teacher_rows_with_mass"),
            "unit_unmapped_source_mass_ratio": ratio("unit_unmapped_source_mass_sum", "unit_teacher_mass_sum"),
            "projected_row_sum_max_error": max(row["projected_row_sum_max_error"] for row in records)}


def diagnose(args):
    all_layers=getattr(args,"all_layers",False)
    layer=getattr(args,"layer",None)
    layer=-1 if layer is None else layer
    if all_layers and layer != -1:
        raise ValueError("--all-layers and --layer are mutually exclusive")
    if args.max_samples < 1 or args.heatmaps < 0 or not 0 <= args.prior_smoothing <= 1:
        raise ValueError("invalid audit size, heatmap count or smoothing")
    rows = load_manifest(args.manifest, check_audio=False)
    identity = manifest_hash(rows)
    population = [row for row in rows if row["split"] == args.split]
    if not population:
        raise ValueError("requested split is empty")
    indices = np.random.default_rng(args.seed).permutation(len(population))[:args.max_samples]
    selected = [population[int(i)] for i in indices]
    config = BackboneConfig.load(args.config)
    if not -config.decoder_layers <= layer < config.decoder_layers:
        raise ValueError("audit layer out of range")
    selected_layer = config.decoder_layers - 1 if all_layers else layer % config.decoder_layers
    units, teacher = ArtifactStore(args.units), ArtifactStore(args.teacher)
    root = Path(args.output).resolve()
    for path in (args.teacher, args.units, args.manifest, args.config):
        if Path(path).resolve().is_relative_to(root) or root.is_relative_to(Path(path).resolve()):
            raise ValueError("audit output must be separate from the inputs")
    corpus = os.environ.get("CORPUS_ROOT")
    if corpus and root.is_relative_to(Path(corpus).resolve()):
        raise ValueError("audit output must be outside CORPUS_ROOT")
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError("audit output exists; use a new directory or --overwrite")
    root.mkdir(parents=True, exist_ok=True)
    matrix_store = ArtifactStore(root / "heatmaps")
    metrics, signatures, selections = [], [], []
    for index, row in enumerate(selected):
        meta, packet = teacher.get(row["pair_id"])
        signature = teacher_signature(meta)
        if signatures and signature != signatures[0]:
            raise ValueError("mixed teacher artifacts in audit")
        signatures.append(signature)
        spec = attention_spec(meta)
        if selections and spec != selections[0]:
            raise ValueError("mixed teacher attention selections in audit")
        selections.append(spec)
        backend = meta["backend"]
        if row.get("ja_sha256") and file_hash(row["ja_audio"]) != row["ja_sha256"]:
            raise ValueError("source waveform does not match the audit manifest")
        feature_length = len(features(row["ja_audio"], backend, config.input_feat_per_channel))
        length = int(output_lengths(feature_length, config, backend))
        # This audit inspects the teacher, without a student head selection.
        heads = spec["heads"]
        builder = prior_builder([row], teacher, units, config, identity, backend, "cpu",
                                [feature_length], args.source_mapping, all_layers=all_layers, layer=layer,
                                heads=heads, manifest_hashes=True)
        projected = builder(torch.tensor([length]), len(packet["reduced_units"]) + 1)
        if projected[..., -1, :].count_nonzero():
            raise ValueError("unexpected nonzero teacher EOS row")
        prior = projected[:,-1,:-1] if all_layers else projected[:, :-1]
        effective = smooth_prior(prior, alpha=args.prior_smoothing)
        duration = float(packet["source_duration"])
        if args.source_mapping == "frontend":
            frames = encoder_time_spans(feature_length, length, config, backend, duration)
        else:
            bounds = np.linspace(0, duration, length + 1)
            frames = np.stack([bounds[:-1], bounds[1:]], -1)
        src = source_mapping(packet["source_spans"], duration, length, frames)
        tgt = target_mapping(packet["target_spans"], packet["unit_spans"])
        mt = packet["mt_attention_layers"][selected_layer] if spec["all_layers"] else packet["mt_attention"]
        src_valid = (src.sum(-1) > 0) & (mt.sum(0) > 0)
        tgt_valid = (tgt.sum(0) > 0) & (mt.sum(-1) > 0)
        valid_rows = prior[0].sum(-1) > 0
        _, unit_packet = units.get(row["pair_id"])
        values = {"pair_id": row["pair_id"], "split": row["split"], "backend": backend,
                  "layer": selected_layer, "heads": heads,
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
                  "eos_rows": projected.shape[1] if all_layers else 1,
                  "source_timing": timing_metrics(packet, "source", duration),
                  "target_timing": timing_metrics(packet, "target", float(unit_packet["duration"])),
                  "teacher_packet_hash": file_hash(teacher.path(row["pair_id"]))}
        values["layers"] = ([dict(layer=l, **projection_metrics(matrix, src, tgt, prior_layer))
                             for l, (matrix, prior_layer) in enumerate(zip(packet["mt_attention_layers"], projected[0, :, :-1], strict=True))]
                            if all_layers else [dict(layer=selected_layer, **projection_metrics(mt, src, tgt, prior[0]))])
        for side in ("source_tokens", "target_tokens", "unit_rows"):
            values["valid_" + side + "_ratio"] = values["valid_" + side] / max(1, values[side])
        metrics.append(values)
        if index < args.heatmaps:
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
               "layer": selected_layer, "heads": heads,
               "counts": totals, "heatmaps": min(args.heatmaps, len(selected)),
               "quality_scope": "Coverage and uncertainty diagnostics; not forced-alignment accuracy or ASR CER."}
    if all_layers:
        summary["all_layers"]=True
        summary["compatibility_heatmap_layer"]=config.decoder_layers-1
    summary["layers"] = [dict(layer=layer["layer"], **summarize_projection([row["layers"][i] for row in metrics]))
                         for i, layer in enumerate(metrics[0]["layers"])]
    summary["pair_ids"] = [row["pair_id"] for row in metrics]
    for side in ("source", "target"):
        timing = [row[side + "_timing"] for row in metrics]
        counts = {key: sum(row[key] for row in timing) for key, value in timing[0].items()
                  if isinstance(value, (float, int))}
        counts["ctc_gap_ratio"] = counts["ctc_gap_seconds"] / counts["ctc_duration_seconds"]
        summary[side + "_timing"] = counts
    for side in ("source_tokens", "target_tokens", "unit_rows"):
        summary["valid_" + side + "_ratio"] = totals["valid_" + side] / max(1, totals[side])
    for key in ("prior", "smoothed"):
        summary[key + "_entropy"] = totals[key + "_entropy_sum"] / max(1, totals["valid_unit_rows"])
    summary["zero_cell_ratio"] = totals["zero_cells"] / max(1, totals["valid_cells"])
    write_jsonl(root / "pairs.jsonl", metrics, args.overwrite)
    layer_rows = [{"pair_id": row["pair_id"], "layer": layer["layer"], **summarize_projection([layer])}
                  for row in metrics for layer in row["layers"]]
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=list(layer_rows[0]))
    writer.writeheader()
    writer.writerows(layer_rows)
    atomic_write(root / "layers.csv", lambda stream: stream.write(csv_text.getvalue().encode("utf-8")), args.overwrite)
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
        axis.set(title=name, xlabel="Source text token" if name == "t2tt" else "Acoustic encoder position",
                 ylabel="Target text token" if name == "t2tt" else "Reduced target unit")
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
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all-layers",action="store_true")
    scope.add_argument("--layer",type=int,help="Teacher decoder layer (zero-based; default -1 is final)")
    import json
    print(json.dumps(diagnose(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
