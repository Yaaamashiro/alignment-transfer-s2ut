# W&B legacy history migration

This migration appends Schema v2 points to the existing transfer runs in `direct-s2st`. The three sources are archived JSONL verified against the published Snapshot Receipt, local Fairseq logs matched to a plan and a durable-update record, and marked legacy history from the verified W&B Run. Baseline history is outside this migration.

## Identify the legacy record

| Row marker | Producer | Migration rule |
| --- | --- | --- |
| `wandb_train_update` | The legacy Reporter copied finite numbers from `alignment-train.jsonl`, using the JSONL `step`, the completed optimizer update. | Admit only the documented logical-update keys. |
| `wandb_dev_update` | The old Reporter copied `native-dev.jsonl`. The native `dev/main_loss` already has the same key and definition. | Keep it under its existing key. It is never a source for `dev/full/*`. |
| `wandb_layer_update` | The original Reporter at `5f91c5f` and `884ee13` read Fairseq's fractional epoch field, `update`, and reused this marker. Starting at `0f3d247`, the parser used `num_updates` for training and `dev_num_updates` for Dev, and added the loss, auxiliary and optimizer metrics below. | Require at least one finite field introduced by the optimizer-counter parser. The early Reporter could also emit `train/main_loss_fairseq`, so that key by itself does not qualify the row. A row may contain both a training window and a full-Dev record. |

The marker must be the only recognized training or Dev stream marker populated in the row and equal its positive integer `update`. Missing, mismatched, mixed, or early epoch-only markers do not qualify. Training windows are restricted to update 1 and multiples of 100. Every candidate is bounded by the verified published Snapshot. Older rows carry no per-row Reporter SHA. The field rules identify the compatible producer family. They establish neither an exact commit for each row nor a hash check on W&B's legacy payload. [Early Reporter](https://github.com/Yaaamashiro/alignment-transfer-s2ut/blob/884ee13191ca144dd75ad2d2444e1f8a4475751f/tools/wandb_transfer.py) · [Optimizer-counter Reporter](https://github.com/Yaaamashiro/alignment-transfer-s2ut/blob/0f3d2474c56fe92b2fd407ad4a2dcbe7aee761fe/tools/wandb_transfer.py) · [Pre-v2 Reporter](https://github.com/Yaaamashiro/alignment-transfer-s2ut/blob/c371938b89eab48f5c8690d32d7d0c53d01a963b/tools/wandb_transfer.py)

## Eligible metrics

Let `N` be the sum of main target-unit sample sizes for the recorded window, logical batch, or full Dev. Auxiliary Task Loss uses each task's own sample size. A Fairseq window aggregates the meter's observations since the previous reset. Full Dev aggregates the prepared Dev split. Fairseq values retain the printed precision. [Pinned Fairseq criterion](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/criterions/speech_to_speech_criterion.py) · [Pinned Fairseq trainer](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/trainer.py)

| Legacy metric | Schema v2 destination | Calculation and scope |
| --- | --- | --- |
| `train/main_loss_fairseq` | `train/window/main_loss_fairseq` | Label-smoothed Unit CE sum divided by `N`, in nats. Fairseq training window. |
| `train/loss_fairseq` | `train/window/loss_fairseq` | Fairseq logged optimization loss sum, including weighted auxiliary and alignment terms where applicable, divided by `N × ln(2)`. Training window, in bits. |
| `train/nll_loss`, `train/perplexity` | `train/window/nll_loss`, `train/window/perplexity` | Unit NLL sum divided by target-token count and `ln(2)`. Perplexity is `2^NLL(bits)`. Training window. |
| `train/multitask_*_loss`, `train/multitask_*_loss_weight` | Add `window/` after `train/`. | Source Letter, Target Letter and decoder-target CTC. Loss is the raw task-loss sum divided by its own sample size and `ln(2)`. Weight is the reported recipe coefficient. Training window. |
| `train/gradient_norm`, `train/loss_scale`, `train/oom` | Add `window/` after `train/`. | Fairseq meters. Gradient norm is the window's mean L2 gradient norm, loss scale is the reported FP16 meter, and OOM retains the original reported meter when present. |
| `train/clip_fraction` | `train/window/clip_fraction` | Fraction of measured gradient norms above the clip threshold. The old Fairseq meter used 0–100 percent, so divide by 100. |
| `dev/loss` | `dev/full/loss` | Fairseq weighted task-total sum divided by `N × ln(2)`, in bits over full Dev. |
| `dev/main_loss_fairseq`, `dev/nll_loss`, `dev/ppl` | Add `full/` after `dev/`. | Full-Dev label-smoothed Unit CE in nats, Unit NLL in bits per target token, and perplexity derived from the NLL. |
| `dev/multitask_*_loss`, `dev/multitask_*_loss_weight` | Add `full/` after `dev/`. | The three auxiliary losses in bits per task sample and reported recipe weights, over full Dev. |
| `train/learning_rate`, `train/seconds` | Add `update/` after `train/`. | Native JSONL post-update scheduler LR and cumulative training wall-clock seconds. |
| `alignment/task_loss` | `alignment/update/task_loss` for native JSONL or `alignment/window/task_loss` for Fairseq. | Weighted main-plus-auxiliary task loss per main unit in nats. At each recorded aggregate, the losses are weighted by the native sample size. |
| Other qualified `alignment/*` | Add `update/` or `window/` according to the marker. | `loss`, `D_*`, `w_*`, KL, entropies, valid-row counts, selected frames, Mask region mass and coverage, worst-layer fractions, λ, τ, Top-p, and annealing controls. Native JSONL records a logical update, Fairseq records its meter's window observations. The documented per-layer and sample-size weighting is unchanged for these exported fields. Only native `task_only_start` is usable, and the legacy negative sentinel is ignored. [Alignment definitions](WANDB_METRIC_AUDIT.md#metric-definitions) |

The old `train/main_loss` can contain Unit NLL or a fallback zero. It never produces `train/unit_ce_nats`. An actual Fairseq CE window also never produces a logical-update CE point. Other exclusions include legacy `train/total_loss` and native `train/gradient_norm` with fallback values, categorical `alignment/worst_layer`, window `alignment/task_only_start`, and metrics outside the allowlist. Existing scores and loss metrics are kept in place. This migration does not recompute BLEU.

## Audit and append in Colab

1. Keep the original experiment IDs and W&B destination. Stop all writers to the selected Run, including writers in other VMs. Use the main training notebook and run through **Verify Git revisions and record the notebook update time**. A CPU runtime is sufficient. Leave `WANDB_WRITER_BUCKET` empty for an unbound Run. A GCS-bound Run still requires its existing bucket.
2. In **Recover W&B writer and sync saved metrics**, enable `RECOVER_WANDB_METRICS` and choose the method. Keep `AUDIT_WANDB_METRICS_ONLY=True`. The cell verifies the archived plan and declared JSONL SHA-256 values, reads the existing Run with uncached `scan_history`, and prints source-specific counts and update ranges for each metric. `missing_metrics` lists the eligible points absent from W&B.
3. To apply the audited migration, turn `AUDIT_WANDB_METRICS_ONLY=False` and confirm the stopped-writer setting. Run the same cell. It recomputes the reconciliation under the PR #10 writer lock, verifies the Run again before opening it, and appends missing points at their original `update` with W&B's increasing internal step. Manual recovery without GCS requires one writer across all VMs. [Writer recovery](WANDB_RECOVERY.md#restore-writer-ownership-on-a-cpu-runtime)

For the CLI, audit uses `--audit-history-only`. Apply uses `--sync-metrics-only --include-legacy-history`, with the existing recovery flags as needed. Both require the original entity and group. The older `--audit-metrics-only` remains available as an offline archive audit.

The audit separates `snapshot_jsonl`, `local_fairseq`, and `wandb_legacy_history`. Counts in these sources can overlap. Fairseq's rounded values are compared as printed, even when an unrounded native metric exists in another stream. Matching numeric values compare exactly apart from one float64 ULP for converting the clip percentage to a fraction. A value conflict stops the operation before opening a writer. The existing v2 point is kept when it matches. Legacy stream watermarks stay intact, v2 watermarks and `train/max_update` never move backward, and a successful rerun adds zero duplicate points.
