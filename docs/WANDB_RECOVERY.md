# W&B recovery for pinned transfer training

This is a **logging-only** repair: the scientific `src/alignment_transfer_s2ut/` implementation, loss, optimizer, fairseq runtime and on-disk checkpoint contract are unchanged.

## Metrics audited

The native `alignment-train.jsonl`/W&B bridge already sends: `train/total_loss`, `train/learning_rate`, `train/gradient_norm`, `train/seconds`, the numeric `train/main_loss` value (which may use its existing NLL fallback), `alignment/lambda`, `alignment/tau`, `alignment/worst_layer`, `alignment/worst_layer_fraction_*`, and available numeric alignment fields. `native-dev.jsonl` is still sent as `dev/*`.

The recovered Fairseq stdout metrics include **all finite numeric `alignment/*` keys** not already supplied above: per-layer `D_*`, `w_*`, `kl_layer_*`, `valid_rows_*`; `kl_mean`, `kl_worst`, `loss`, `task_loss`, `teacher_entropy`, `student_entropy`; Mask `region_mass_*`, `coverage_*`, `selected_frames_*`; and Annealed KL `annealing_progress`, `task_only_start`, `task_only_phase`. It additionally sends `train/main_loss_fairseq` read from native stdout, **without silently overwriting** the historical `train/main_loss` series.

Only matching training-method and output-path process logs are used; only finite numeric values are sent. Older updates can be replayed after a restart in the *same runtime*, up to the latest verified Drive snapshot. W&B resumes the original deterministic contract-derived Run ID and stores `wandb_layer_update` as an independent upload watermark.

These recovered stdout metrics are sampled at Fairseq's progress reporting frequency, **not necessarily every optimizer update**. In particular, the stdout record's `alignment/w_*` should not be interpreted as a direct per-parameter gradient measurement.

## Recover without disconnecting the Colab runtime

1. Stop the **currently running training cell**, **not** the runtime; keep the Drive mount and `/content/alignment-transfer-work/<EXPERIMENT_ID>/execution/` intact. Check for a completed matching `snapshot.json` *and* `snapshot.zip` under `/content/drive/MyDrive/alignment-transfer-s2ut-data/<EXPERIMENT_ID>/stages/research-v3-<method>-spm-all-layers-run-<TRANSFER_EXPERIMENT_ID>/`.
2. In each Colab VM, fetch and check out the pinned logging-fix SHA shown in the updated `notebooks/colab/release.json`. **Do not reset or wipe** the current working data, execution log directory, or installed runtime. The `src/` sources are unchanged, so the already installed training wheel does not need to change.
3. In the existing Notebook's `Storage and pinned environment` cell, change `REPO_REVISION` to that same SHA. Rerun that cell and `Verify Git revisions and record the notebook update time`. Keep `EXPERIMENT_ID`, `TRANSFER_EXPERIMENT_ID`, `CALIBRATION_EXPERIMENT_ID`, `MASK_TOP_P`, coefficients and method switches unchanged.
4. Reexecute only `Train the selected transfer conditions` in this live runtime with **one transfer condition enabled per Colab** and the existing W&B Project/Entity/secret. The native checkpoint loader restores the newest verified matching snapshot, while the new `tools/wandb_transfer.py` reads the old stdout logs and restores missing charts after a snapshot has been published.
5. Confirm `[wandb] connected` under `execution/notebook-events.log` and verify the extra `alignment/D_*`, `alignment/w_*`, `alignment/kl_mean`, or Mask/Annealed fields under that existing W&B Run.

If training is interrupted between saved snapshots, **some optimizer updates may have to be replayed**. If Colab itself disconnects, `/content` stdout logs can be lost; the native Drive snapshot still allows checkpoint recovery, but unsaved per-layer history cannot be inferred from it.

Do not run the same method from multiple Colab instances simultaneously; all three distinct methods can run on separate VMs with one shared W&B Project/Entity.
