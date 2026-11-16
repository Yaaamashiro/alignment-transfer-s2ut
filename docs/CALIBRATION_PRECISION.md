# FP16 calibration precision

The saved `experiment-001` / `transfer-main-v1` calibration used FP16 backward at scale 1. On the seven L4 diagnostic batches, using scale 1024 with the same forward reduced the Mask gradient-ratio range from 0.0043–18.83 to 5.95–14.41 at the old λ. The extreme ratios therefore affected how the old λ was estimated.

## Measurement

The forward pass for τ and the fixed 16 logical batches are reused. For each microbatch, calibration differentiates the normalized task, KL, and Mask objectives on one FP16 forward at scales 128, 1024, and 2048. It converts the returned gradients to FP32, divides each by its scale, and then accumulates and measures the logical norm in FP32. Each λ uses the same chosen scale for both its numerator and denominator. Model state, update count, and parameter gradients are checked before publication.

The FP32 reference widens the same initialized values and inputs, starts at the same RNG state, and disables TF32. Dropout masks can still differ across dtypes. The tight comparison therefore uses the shared FP16 forward. To qualify, a scale needs agreement with another FP16 scale and with the FP32 reference on all batches and objectives. The shared-forward limits are cosine ≥ 0.995, relative vector L2 error ≤ 10%, and a zero-fraction difference ≤ 1 percentage point. The FP32 reference limits are cosine ≥ 0.95, relative norm error ≤ 5%, and a zero-fraction difference ≤ 2 percentage points. These are diagnostic criteria, not training hyperparameters. A qualifying 1024 is preferred. Missing FP32 reference measurements, nonfinite or zero-norm gradients in the candidate, and lack of agreement prevent qualification.

The native fairseq optimizer applies its dynamic scale to the combined training loss, accumulates native gradients, moves them to FP32, and accounts for scale and sample-size normalization on the optimizer path. The calibration measurement uses individually normalized objectives and unscales each microbatch in FP32. It leaves the model, training optimizer, and overflow retries untouched.

## Colab L4

Open the branch's pinned `notebooks/colab/training.ipynb` in an L4 runtime. For the saved run, set `EXPERIMENT_ID='experiment-001'`, `TRANSFER_EXPERIMENT_ID='transfer-main-v1'`, and your existing corpus and teacher settings. Keep `RUN_BASELINE=False` and all three transfer switches off. Set `RUN_ALIGNMENT_CALIBRATION=True`, `CALIBRATION_BATCHES=16`, and `MASK_TOP_P=0.8`. Keep `RUN_MASK_TOPP_DIAGNOSTICS=False` to restore the completed report. Run through **Top-p diagnostics and calibration**. This restores the matching v2 calibration read-only for comparison and publishes the new stage separately under `research-v6-calibration-fp16-*`.

Top-p reuse requires the same registration, manifest, teacher/units contract, sample order and report integrity. The migration checks the exact reviewed versions of the changed files, then reconstructs the prior implementation fingerprint with the two original calibration source hashes and without the new precision helper. The Top-p calculation itself is unchanged. A different code fingerprint or changed inputs require a new Top-p report in a distinct stage. Baseline, Teacher, and saved checkpoints retain their original paths and contracts.

If inputs are already restored on the VM, run the following through the notebook's logged `run` helper. The previous artifact must be in a separate directory.

```python
output = RESEARCH_ROOT / 'pilot-calibration-fp16' / 'alignment-calibration.json'
previous = RESEARCH_ROOT / 'pilot-calibration-masked' / 'alignment-calibration.json'
run('research-v6-calibration-fp16' + SCOPE_SUFFIX,
    'alignment_transfer_s2ut.alignment_calibration', 'calibrate',
    '--baseline', BASELINE, '--manifest', MANIFEST, '--units', Path(ROOT) / 'units',
    '--teacher', TEACHER_ROOT, '--diagnostics', TOPP_REPORT, '--mask-top-p', .8,
    '--batches', 16, '--previous-calibration', previous, '--output', output, gpu=True)
```

`alignment-calibration-precision.json` reports per-batch finite counts, zero fractions, L2 norms, and cosine and relative errors for every tested scale and the reference. It records τ and any qualification failures even when the calibration cannot be published. `alignment-calibration.json` is the qualified v3 artifact. It reports the old and new λ and τ, exact τ equality, and the mean, median, minimum, and maximum of each calibrated gradient ratio. Review both files before enabling any Pilot run. For a previously started transfer run, keep its pinned code/calibration to reproduce it, or use a new transfer output ID with v3.
