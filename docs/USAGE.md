# Usage

## Manifest and T2TT

External Japanese–English manifests without audio use the [text-only T2TT workflow](TEXT_ONLY_T2TT.md). It supports character and SentencePiece tokenization, training-only vocabulary, dev BLEU, and all-layer attention extraction. S2UT, CTC, unit extraction, and teacher construction require an audio manifest.

## Native research

Register a separately trained Baseline with `benchmark register`, then prepare the full train/dev manifest, units, and all-layer teacher. See the [Colab Baseline snapshot workflow](BENCHMARK_COMPARISON.md).

```bash
python -m alignment_transfer_s2ut.alignment_calibration top-p \
  --baseline baseline.json --manifest manifest.jsonl --units units --teacher teacher \
  --batches 16 --output analysis/top-p/top-p.json
```

Inspect coverage and choose p explicitly. The command below uses 0.9 as an example selection.

```bash
python -m alignment_transfer_s2ut.alignment_calibration calibrate \
  --baseline baseline.json --manifest manifest.jsonl --units units --teacher teacher \
  --batches 16 --diagnostics analysis/top-p/top-p.json --mask-top-p 0.9 \
  --output analysis/calibration/alignment-calibration.json

python -m alignment_transfer_s2ut.benchmark train \
  --baseline baseline.json --manifest manifest.jsonl --units units --teacher teacher \
  --method distillation --all-layers --calibration analysis/calibration/alignment-calibration.json \
  --output runs/distillation --resume
```

Use separate output directories for `annealed_distillation` and `mask_distillation` and share the same calibration JSON. The native path derives the training budget and precision from Baseline registration and loads the calibrated coefficients from JSON. `train_s2ut --trainer fairseq --baseline-record ... --calibration ...` uses the same native path.

## Evaluation and diagnostics

Register each best checkpoint with `benchmark register-transfer`, then run `benchmark infer`, `evaluate`, and `benchmark compare` in order. The generator takes source-only inputs.

```bash
python -m alignment_transfer_s2ut.alignment_report diagnose \
  --baseline baseline.json --record registered-distillation.json \
  --manifest manifest.jsonl --units units --teacher teacher \
  --calibration analysis/calibration/alignment-calibration.json --samples 32 \
  --output diagnostics/distillation.json

python -m alignment_transfer_s2ut.alignment_report curves \
  --runs runs/distillation runs/annealed_distillation runs/mask_distillation \
  --output diagnostics/curves
```

Per-layer diagnostics run on teacher-forced dev inputs independently from generation. Use `visualize_attention` for heatmaps. Native checkpoint generation runs teacher-free.

## Historical standalone tools

`--trainer legacy`, the smoke backend, and the old pilot and pipeline notebooks support historical controlled experiments and regression testing. Guidance names belong to these historical workflows. Resume a saved run with its original source revision. Use `training.ipynb` for training and `evaluate.ipynb` for inference and comparison.

FP16 transfer requires a precision-qualified v3 calibration JSON with microbatch τ estimates. See [FP16 recalibration](CALIBRATION_PRECISION.md) for verified Top-p reuse and comparison with v2. To change annealing, edit the fraction and schedule in `configs/experiments/conditions.yaml` and start a separate run.
