# W&B metric recovery

Use [the metric audit and schema-v2 migration](WANDB_METRIC_AUDIT.md) for corrected series and stand-alone historical sync. The metric names below describe the legacy reporter. Its native `train/main_loss` can contain the Unit NLL fallback and should be excluded from matched Unit CE charts.

The telemetry reader uses native JSONL and Fairseq process logs. Each condition resumes its existing contract-derived W&B run.

For joint Baseline and transfer reporting, use the [matched S2UT metric definitions and importer conventions](WANDB_BASELINE_COMPARISON.md).

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
2. In the existing notebook, edit **Storage and pinned environment**. Copy `REPO_REVISION` and `NOTEBOOK_UPDATED_AT` from [release.json](../notebooks/colab/release.json). Keep the original experiment IDs, inputs, transfer conditions, coefficients, mode, W&B project/entity, and secret. Run that cell and **Verify Git revisions and record the notebook update time**. This fetches the fixed SHA and verifies both checkouts. The notebook's editable install reads the updated checkout in the next process. An environment using a non-editable build must reinstall from the fixed revision before training.
3. Run **Train the selected transfer conditions** in each runtime with that runtime's original method selection. It reuses `/content/alignment-transfer-work/<EXPERIMENT_ID>`, including `execution/`, and the same Drive stage. The trainer restores the latest verified checkpoint and the W&B reader backfills the saved history.
4. Check `[wandb] connected` and the added metrics in the existing W&B run. The link also appears in `execution/notebook-events.log`. Optimizer updates after the last durable checkpoint may be replayed. A runtime reset can remove the local Fairseq history needed for backfill.

Run each method in one runtime so a single trainer owns its checkpoint and W&B run.

When updating from a release older than writer recovery, load and run the updated **Process logging and progress** and **Run, save, and restore helpers** cells as well. Those cells are embedded in the notebook and supply the longer W&B shutdown grace. For runs using a shared bucket, authenticate with `google.colab.auth.authenticate_user()` in each runtime before starting a W&B writer.

## Restore writer ownership on a CPU runtime

`No local W&B ownership receipt` means the runtime lacks the record written after `wandb.finish()`. A SIGTERM, forced exit, or VM replacement can leave that record missing. The W&B-wrapped notebook command now allows up to 75 seconds for its reporter to drain. The trainer child retains its original five-second termination grace. The reporter writes its local receipt atomically after confirmed finalization. A forced kill can still skip this path.

1. Stop every W&B writer for the selected method, including older notebooks and manual scripts. Wait until the matching W&B Run is `finished`, `failed`, `crashed`, or `killed`. Leave all other writers for it stopped until recovery completes.
2. Open the updated `training.ipynb` in the existing runtime or a fresh CPU Colab. Run **Process logging and progress**, **Mount Google Drive**, **Experiment and training stages**, **Storage and pinned environment**, and **Verify Git revisions and record the notebook update time**. Keep the original experiment IDs and W&B destination. The matched comparison uses project `direct-s2st` and entity `yaaamashiro-national-institute-of-technology-okinawa-college`. Supply the existing `WANDB_API_KEY` Colab secret. Leave `WANDB_WRITER_BUCKET` blank for an unbound Run, or set it to the exact bucket already bound to the Run.
3. Open **Recover W&B writer and sync saved metrics**. Set `RECOVER_WANDB_METRICS=True`, select `RECOVERY_WANDB_METHOD`, and first keep `AUDIT_WANDB_METRICS_ONLY=True` for a read-only preview. After reviewing the counts, turn audit off and set `CONFIRM_WANDB_WRITERS_STOPPED=True` after confirming step 1. Run only this recovery cell. It verifies the Snapshot and sets up its own W&B environment. [Legacy history migration](WANDB_LEGACY_MIGRATION.md#audit-and-append-in-colab)
4. Look for `[wandb-metrics] synced` and `[wandb-writer] recovered`. The local receipt is restored even when the log says zero historical rows. Once the Run shows a stopped state again, use the [BLEU-only workflow](TRANSFER_BLEU.md#sync-scores-already-on-drive) on this same VM. Keep one writer for the Run until it finishes.

To keep the original runtime and its local Fairseq logs, follow the pinned revision update above and copy the recovery cell from the updated notebook into the existing one. Run it in the existing runtime after revision verification.

Manual recovery holds the local FileLock and checks the existing Run's contract-derived ID, entity, project, method, group, bucket and stopped state. It reads uncached history and checks the Run again immediately before opening the Writer or restoring the receipt. The local lock coordinates processes on this VM. A Writer on another VM can still race the W&B status checks. Use manual recovery only with one Writer across all VMs. A Run already bound to GCS requires its matching bucket.

The CPU command verifies the archived plan and declared metric JSONL against the SHA-256 in the published receipt. Local Fairseq windows come from retained execution logs and receipts. Qualified old W&B rows are a separate source. Matching history is kept, missing keys are appended by `(update, metric key)`, and a conflicting value aborts before writes. If the existing W&B reporter session is available and no history rows are needed, the operation writes only the local receipt. A legacy Run without a session opens and finishes a metadata-only W&B session. [Metric definitions and audit](WANDB_METRIC_AUDIT.md#colab)

### Optional shared lock

Use one GCS bucket for all participating writers to a Run, including training and saved BLEU sync. Stop older writers before first binding the Run. Credentials need object read, create, and delete access. Keep the lock prefix outside auto-deletion lifecycle rules. With a bucket set, the recovery cell prompts for Google authentication. If Application Default Credentials have no project, set `GOOGLE_CLOUD_PROJECT` to the bucket's project ID. The shared lock uses a conditional create and releases the exact generation only after confirmed finish or an unused claim. [Cloud Storage generation preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions)

An uncertain `wandb.init()` or `wandb.finish()` stops automatic connection attempts and keeps the pending local record and any GCS lock. A shared lock never expires automatically. If a writer is gone and an orphaned lock remains, confirm in W&B that the Run is stopped and read the object's live **Generation** at the `gs://` path printed in the error. Add `'--recover-lock-generation', '<generation>'` to the recovery cell's sync command for this one attempt. The command verifies the lock identity and conditionally replaces that exact generation. Direct CLI callers use `--sync-metrics-only --recover-writer` with the existing `--entity` and `--group`, and add `--confirm-writers-stopped` for manual recovery without a bucket.

During training the Reporter retries transient W&B lookup errors, delayed stopped states, and writer-lock contention before opening a Writer. The delay starts at two seconds and doubles up to 60 seconds. Training and its local metrics continue while the Reporter waits. A contract, destination, or bucket mismatch stops the Reporter and requires correction.
