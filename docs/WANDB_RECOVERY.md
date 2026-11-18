# W&B metric recovery

The telemetry reader uses native JSONL and Fairseq process logs. Each condition resumes its existing contract-derived W&B run.

| Source | Metrics |
| --- | --- |
| Training JSONL | Task or historical NLL fallback, total loss, learning rate, gradient norm, elapsed time, and available finite alignment scalars such as λ, τ, and worst-layer fractions |
| Dev JSONL | Dev unit loss at the existing validation interval |
| Fairseq training logs | Main loss, Fairseq total and NLL losses, perplexity, auxiliary-task losses, FP16 loss scale, clipping and OOM counters when present |
| Fairseq alignment logs | Per-layer loss, weights, KL, and valid rows. Aggregate KL, task and alignment loss, entropy, Mask region mass, coverage, selected frames, and Annealed KL progress and task-only flags when present |
| Fairseq dev logs | Fairseq validation loss, NLL, perplexity, and auxiliary-task losses when present |

The backfill reads Colab's `INFO:train_inner:` and `INFO:dev:` prefixes as well as the pipe-delimited Fairseq format. It uses `num_updates` for training and `dev_num_updates` for validation. Training is sampled at update 1 and every 100 updates where a Fairseq record exists. Dev contributes its recorded validation points. The canonical dev unit loss stays in `dev/main_loss`. The native Fairseq task-loss values have separate `train/main_loss_fairseq` and `dev/main_loss_fairseq` series. Evaluation metrics such as ASR-BLEU and UER are produced by the separate evaluator.

A Fairseq log can contribute when the recorded trainer command matches the published command and the preceding notebook plan records the same contract hash. A Pilot stop is execution control and is ignored for matching. An update must be covered by a completed matching Drive receipt and confirmed durable in that log's own training attempt. Matching copies are combined, native metric points take precedence, and W&B run summaries supply the progress markers on reconnect. Keep the process receipts, plan stdout files, and Fairseq logs together in `execution/` to backfill history.

## Resume in the current Colab runtime

1. In each Colab, wait for the desired `[recovery] ... durable_updates=...` line. Check that the corresponding folder under Drive's training stage has both `snapshot.json` and `snapshot.zip`. Interrupt the running cell and wait for it to exit. Keep the runtime connected and Drive mounted.
2. In the existing notebook, edit **Storage and pinned environment**. Copy `REPO_REVISION` and `NOTEBOOK_UPDATED_AT` from [release.json](../notebooks/colab/release.json). Keep the original experiment IDs, inputs, transfer conditions, coefficients, mode, W&B project/entity, and secret. Run that cell and **Verify Git revisions and record the notebook update time**. This fetches the fixed SHA and verifies both checkouts. The already installed training package comes from byte-identical sources.
3. Run **Train the selected transfer conditions** in each runtime with that runtime's original method selection. It reuses `/content/alignment-transfer-work/<EXPERIMENT_ID>`, including `execution/`, and the same Drive stage. The trainer restores the latest verified checkpoint and the W&B reader backfills the saved history.
4. Check `[wandb] connected` and the added metrics in the existing W&B run. The link also appears in `execution/notebook-events.log`. Optimizer updates after the last durable checkpoint may be replayed. A runtime reset can remove the local Fairseq history needed for backfill.

Run each method in one runtime so a single trainer owns its checkpoint and W&B run.
