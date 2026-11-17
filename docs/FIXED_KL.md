# Explicit transfer coefficients

The fixed-coefficient condition uses λ = 1 for KL-based Attention Distillation and starts Annealed KL-based Attention Distillation at λ = 1. The 400,000-update budget and cosine schedule end annealing at update 320,000. The last 80,000 updates have λ = 0. This is a reproducible condition with reduced auxiliary weighting. Its optimality is unmeasured.

Mask Distillation uses an independent fixed λ = 10. The existing precision-qualified calibration supplies KL τ = 0.5538942450051301 and Mask τ = 0.04495423606858799. Its original KL λ = 12.781413823176955 and Mask λ = 145.51251933764968 remain recorded in the artifact. The loss definitions, layer aggregation, top-p 0.8, normalization, and native fairseq training recipe are retained.

## Colab

Open the main training notebook on L4. Use the saved corpus and T2TT settings. Set the following in **Experiment and training stages**.

```python
EXPERIMENT_ID = 'experiment-001'
TRANSFER_EXPERIMENT_ID = 'transfer-fixed-v1'
CALIBRATION_EXPERIMENT_ID = 'transfer-main-v1'
PREPARE_S2UT = True
PREPARE_AUDIO_PACKS = False
RUN_BASELINE = False
RUN_T2TT = False
RUN_TEACHER_ARTIFACTS = False
RUN_MASK_TOPP_DIAGNOSTICS = False
RUN_ALIGNMENT_CALIBRATION = False
MASK_TOP_P = 0.8
TRANSFER_KL_LAMBDA = 1.0
TRANSFER_MASK_LAMBDA = 10.0
RUN_DISTILLATION = True
RUN_ANNEALED_DISTILLATION = True
RUN_MASK_DISTILLATION = True
TRANSFER_RUN_MODE = 'full'
TRANSFER_RECOVERY_CHECK = 'off'
```

The Drive search on 2026-11-18 found no names containing `transfer-fixed-v1` or `transfer-lambda1-v1`. A saved-run search establishes published progress. Live Colab execution was not inspected. The old Mask stage under `transfer-main-v1` had a completed receipt at update 6,000 and an incomplete `snapshot.zip.tmp` under the 6,400-update folder. The new ID separates all three conditions from those files. Registration and runtime contract checks also run locally when a session starts.

Run setup and input restoration through **Top-p diagnostics and calibration**. The Baseline, corpus preparation, Teacher, Top-p, and calibration use their existing verified snapshots. Calibration stages are restored with `--restore-only` when the source ID differs from the training ID. The notebook checks calibration integrity, runtime, data/Teacher/initialization binding, and the exact reviewed source transition. The calibrated coefficients remain in the original JSON.

Run **Train the transfer conditions** to start the selected runs. Its plan log shows method, experiment IDs, initial λ, coefficient source, calibrated λ, τ, annealing end, full budget, fresh initialization, and checkpoint directory. With the settings above, each condition begins at update 0 in its own new directory. The notebook trains selected methods sequentially. A disconnect resumes the matching new run's own checkpoint.

The old 20k KL Pilot and calibrated Mask Pilot have different coefficients and contracts. Reproduce or continue existing runs with their original pinned source, IDs, and settings. The new conditions start from the Baseline's registered fresh initialization. A copied old checkpoint is rejected by the contract check.

## CLI and recorded policy

`benchmark plan`, `benchmark train`, and native `train_s2ut` accept `--kl-lambda 1.0` and `--mask-lambda 10.0`. KL methods read only the KL option, and Mask reads only the Mask option. Omitting a method's coefficient preserves calibrated behavior. The calibration remains mandatory and is validated before the explicit coefficient is applied.

```bash
python -m alignment_transfer_s2ut.benchmark plan \
  --baseline registrations/baseline.json --manifest manifest/manifest.jsonl \
  --units units --teacher teacher-all-layers --all-layers \
  --calibration research-v3-spm-all-layers-run-transfer-main-v1/pilot-calibration-fp16/alignment-calibration.json \
  --method annealed_distillation --kl-lambda 1.0 \
  --output research-v3-spm-all-layers-run-transfer-fixed-v1/annealed_distillation
```

The plan stores `transfer.kl_lambda` for the two KL methods or `transfer.mask_lambda` for Mask and a resolved `transfer.coefficient_policy` with `initial_lambda`, `lambda_source`, `calibration_lambda`, τ, and schedule. The plan hash binds those values. `alignment-train.jsonl` records the applied per-update λ and the policy with run identity. Checkpoint extra state records `alignment_coefficient_policy` and the contract hash. Pilot gradient diagnostics use the same resolved coefficient.

The existing v3 calibration is accepted only when every unchanged source fingerprint matches and all modified sources match `coefficient-compatibility.json`. Changed artifacts, inputs, runtime, and unreviewed source edits remain errors. The earlier Top-p migration checks are retained.
