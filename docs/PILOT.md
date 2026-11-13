# Native transfer pilot

Use the Pilot after the 400,000-update Baseline has published its final Drive snapshot. Use the original Baseline experiment ID, corpus, prepared data, and snapshot location. In **Baseline**, the notebook checks the last and best checkpoints against the snapshot receipt, recipe, runtime lock, and registered scientific configuration. It saves a JSON report under the transfer run's `baseline-provenance` directory.

Set `BASELINE_NOTEBOOK_REVISION` to the full alignment commit SHA from the original Baseline notebook when available. Set `BASELINE_EXECUTION_LOGS` to the exported `execution` directory from the original run to verify the checkout and matching Baseline launch. The report labels the source as `launch_log`, `metadata`, `notebook_pin`, or `unavailable`. A matching notebook pin establishes adapter compatibility for that candidate commit. The launch log or recorded metadata supplies evidence of the executed revision. The report stops on conflicting evidence or an incompatible lock.

## Configure the experiment

Open [training.ipynb](../notebooks/colab/training.ipynb) in a GPU runtime. Keep `RUN_BASELINE=False` and use the existing SentencePiece T2TT teacher. Use **Top-p diagnostics and calibration** before enabling a transfer condition.

1. Enable `RUN_MASK_TOPP_DIAGNOSTICS`. Compare the occupancy, retained probability mass, selected frames, entropy, and invalid rows for 0.8, 0.9, and 0.95. Start by reviewing 0.8. Choose `MASK_TOP_P` explicitly after review.
2. Enable `RUN_ALIGNMENT_CALIBRATION`. The calibration uses 16 logical train batches, fixed initial gradient ratio 0.5, and a worst-to-median layer weight target of 3. Review the calibrated λ and τ, batch gradient ratios and cosines, and layer weight statistics.
3. Enable `RUN_DISTILLATION`, `RUN_ANNEALED_DISTILLATION`, and/or `RUN_MASK_DISTILLATION`. Each condition starts from the registered fresh seed and uses a separate run directory.

## Pause and review

Set `TRANSFER_RUN_MODE='pilot'`. The defaults are `TRANSFER_PILOT_UPDATES=20000` and `TRANSFER_DIAGNOSTIC_STEPS=[1000, 5000, 10000, 20000]`. The 20k probe observes training after the registered 10k warmup. For a 10k smoke run, set both variables to end at 10k. The notebook also probes the final Pilot step if it is absent from the list and warns when every probe is within warmup. The scientific run remains 400,000 updates. Annealed KL follows the full cosine schedule through 320,000 updates.

At each probe boundary, the Trainer completes the logical update and writes its native checkpoint. The Drive session verifies publication before reporting `PILOT_PAUSED`. The checkpoint retains the model, optimizer, scheduler, FP16 scaler counters, FP32 optimizer master weights, data iterator, RNG, and best-checkpoint state. The additional master weights increase FP16 checkpoint size. Pausing off the normal validation cadence writes a last checkpoint using the normal save path and preserves the selection schedule.

Each probe runs in a separate process. It reads the saved model and the first calibration logical batch, measures gradients on the cross-attention parameters, and uses the λ at the checkpoint's completed-update count. It records the weighted gradient norm ratio, the unweighted cosine, elapsed GPU computation time, peak allocated CUDA memory, and undefined-metric reasons. Negative cosine is a warning. Nonfinite values and checkpoint or contract mismatches are errors.

**Training diagnostics and evaluation** shows the curves and a fixed 32-example dev subset with task loss, teacher/student KL, teacher-region student mass, and valid rows by layer. Full dev unit loss appears when the original validation schedule covers the checkpoint. The completed Baseline uses the same fixed dev subset as a reference at a different training stage. Use the gradient ratio and cosine alongside task loss and Attention KL. Use the completed checkpoints to compare translation performance. Optionally set `EVALUATE_TRANSFER_PILOT_DEV=True` for full dev UER and `TRANSFER_PILOT_ASR_BLEU=True` for full dev ASR-BLEU with the common teacher-free generator.

## Check recovery in Colab

1. Use the original `EXPERIMENT_ID`, the same `TRANSFER_EXPERIMENT_ID` planned for the Pilot, and the prepared real corpus and calibration. Enable one transfer condition, keep `RUN_BASELINE=False` and `TRANSFER_RUN_MODE='pilot'`, and set `TRANSFER_RECOVERY_CHECK='save'`. Run through **Train the transfer conditions**. Wait for `CHECKPOINT_SAVED` at update 2.
2. Open a fresh GPU runtime, keep the same settings, and set `TRANSFER_RECOVERY_CHECK='resume'`. Run through **Train the transfer conditions** again. `RECOVERY_CHECK_PASSED` reports the verified update-4 checkpoint, optimizer history showing resume at update 2, and exactly one log row for each update 1 through 4.
3. Set `TRANSFER_RECOVERY_CHECK='off'` and continue the same condition into the Pilot. Keep the original scientific settings and storage IDs. Use the execution log's `[training-loader]` messages to see the worker count the pinned runtime selected.

The short check exercises real inputs, CUDA FP16, and Drive recovery under the registered recipe. The configured worker count is a ceiling for the pinned runtime's adaptive worker controller. Multi-epoch recovery on the full corpus requires a longer run.

## Continue to 400,000 updates

Set `TRANSFER_RUN_MODE='full'`, keep the same selected conditions and paths, and rerun from setup on a fresh runtime or rerun the settings and transfer sections in the existing runtime. The verified snapshot restores the last durable update. The optimizer, scheduler, and sampler continue from that point. For a new calibration or top-p trial, set a new `TRANSFER_EXPERIMENT_ID` and keep `EXPERIMENT_ID` to reuse the original Baseline and corpus. Use the same `TRANSFER_EXPERIMENT_ID` in the evaluation notebook. A different corpus, seed, or recipe requires a matching Baseline registration.

The native CLI accepts `benchmark train --stop-after-update N` as an operational limit. The same command without that argument resumes to the registered training budget. After successful publication the session reports `COMPLETED` at the full budget. Errors and runtime interruptions retain verified snapshots for recovery. The final registration and [evaluate.ipynb](../notebooks/colab/evaluate.ipynb) require a completed transfer run.

The original standalone Pilot is documented at its [historical revision](https://github.com/Yaaamashiro/alignment-transfer-s2ut/blob/1c4ed71964aaabf32a2db38e0b3a4186122afc3a/docs/PILOT.md).
