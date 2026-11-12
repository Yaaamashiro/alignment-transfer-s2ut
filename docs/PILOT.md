# Native transfer pilot

Use the Pilot after the 400,000-update Baseline has published its final Drive snapshot. Use the original Baseline experiment ID, corpus, prepared data, and snapshot location. The training notebook verifies the completed Baseline and displays the recorded recipe, runtime revision, fairseq revision, data identity, seed, batch settings, optimizer, scheduler, and precision. Older Baseline metadata may omit the alignment repository revision. Use the original notebook or execution log to identify that revision.

## Configure the experiment

Open [training.ipynb](../notebooks/colab/training.ipynb) in a GPU runtime. Keep `RUN_BASELINE=False` and use the existing SentencePiece T2TT teacher. Use **Top-p diagnostics and calibration** before enabling a transfer condition.

1. Enable `RUN_MASK_TOPP_DIAGNOSTICS`. Compare the occupancy, retained probability mass, selected frames, entropy, and invalid rows for 0.8, 0.9, and 0.95. Start by reviewing 0.8. Choose `MASK_TOP_P` explicitly after review.
2. Enable `RUN_ALIGNMENT_CALIBRATION`. The calibration uses 16 logical train batches, fixed initial gradient ratio 0.5, and a worst-to-median layer weight target of 3. Review the calibrated λ and τ, batch gradient ratios and cosines, and layer weight statistics.
3. Enable `RUN_DISTILLATION`, `RUN_ANNEALED_DISTILLATION`, and/or `RUN_MASK_DISTILLATION`. Each condition starts from the registered fresh seed and uses a separate run directory.

## Pause and review

Set `TRANSFER_RUN_MODE='pilot'`, `TRANSFER_PILOT_UPDATES=10000`, and `TRANSFER_DIAGNOSTIC_STEPS=[1000, 5000, 10000]`. The notebook also probes the final Pilot step if it is absent from the list. The update limit for the scientific run remains 400,000. Annealed KL follows the full cosine schedule through 320,000 updates.

At each probe boundary, the Trainer completes the logical update and writes its native checkpoint. The Drive session verifies publication before reporting `PILOT_PAUSED`. The checkpoint retains the model, optimizer, scheduler, FP16 scaler counters, FP32 optimizer master weights, data iterator, RNG, and best-checkpoint state. The additional master weights increase FP16 checkpoint size. Pausing off the normal validation cadence writes a last checkpoint using the normal save path and preserves the selection schedule.

Each probe runs in a separate process. It reads the saved model and the first calibration logical batch, measures gradients on the cross-attention parameters, and uses the λ at the checkpoint's completed-update count. It records the weighted gradient norm ratio, the unweighted cosine, elapsed GPU computation time, peak allocated CUDA memory, and undefined-metric reasons. Negative cosine is a warning. Nonfinite values and checkpoint or contract mismatches are errors.

**Training diagnostics and evaluation** shows the curves and a fixed 32-example dev subset with task loss, teacher/student KL, teacher-region student mass, and valid rows by layer. Full dev unit loss appears when the original validation schedule covers the checkpoint. The completed Baseline uses the same fixed dev subset. Optionally set `EVALUATE_TRANSFER_PILOT_DEV=True` for full dev UER and `TRANSFER_PILOT_ASR_BLEU=True` for full dev ASR-BLEU. The optional scores use the common teacher-free generator. The Pilot uses train and dev, and its results describe the current checkpoints.

## Continue to 400,000 updates

Set `TRANSFER_RUN_MODE='full'`, keep the same selected conditions and paths, and rerun from setup on a fresh runtime or rerun the settings and transfer sections in the existing runtime. The verified snapshot restores the last durable update. The optimizer, scheduler, and sampler continue from that point. For a new calibration or top-p trial, set a new `TRANSFER_EXPERIMENT_ID` and keep `EXPERIMENT_ID` to reuse the original Baseline and corpus. Use the same `TRANSFER_EXPERIMENT_ID` in the evaluation notebook. A different corpus, seed, or recipe requires a matching Baseline registration.

The native CLI accepts `benchmark train --stop-after-update N` as an operational limit. The same command without that argument resumes to the registered training budget. After successful publication the session reports `COMPLETED` at the full budget. Errors and runtime interruptions retain verified snapshots for recovery. The final registration and [evaluate.ipynb](../notebooks/colab/evaluate.ipynb) require a completed transfer run.

The original standalone Pilot is documented at its [historical revision](https://github.com/Yaaamashiro/alignment-transfer-s2ut/blob/1c4ed71964aaabf32a2db38e0b3a4186122afc3a/docs/PILOT.md).
