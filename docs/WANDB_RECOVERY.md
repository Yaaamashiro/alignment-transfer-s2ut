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

Cold recovery uses a shared Google Cloud Storage bucket as a writer lock. W&B's idle state and a local file lock cannot reserve a writer across VMs. Use one bucket for all writers that target these runs, including future training and saved BLEU sync. Credentials need object read, create and delete access in that bucket. Keep the lock prefix out of auto-deletion lifecycle rules, and use a bucket where the lock generation can be deleted after release. Cloud Storage creates the lock only if no live generation exists and deletes only the acquired generation. An unconfirmed `wandb.init()` or `wandb.finish()` leaves the lock in place. [Cloud Storage generation preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions)

1. Stop every previous W&B writer for the selected method, including older notebook revisions or manual write scripts. Wait until the target Run is `finished`, `failed`, `crashed`, or `killed`. This is required before first adopting the shared bucket, because older clients do not participate in its lock.
2. Open the updated `training.ipynb` on a CPU Colab. Run **Process logging and progress**, **Mount Google Drive**, **Experiment and training stages**, **Storage and pinned environment**, and **Verify Git revisions and record the notebook update time**, in notebook order. Keep the original experiment IDs and use the existing project and entity. The comparison in this repository uses `direct-s2st` under `yaaamashiro-national-institute-of-technology-okinawa-college`. Fill `WANDB_WRITER_BUCKET` with the shared bucket's name. The Python/Fairseq setup and training cells can be skipped.
3. Add and run this cell. It checks the selected archive in read-only audit mode before opening the existing W&B Run. Keep `method`, the experiment IDs, and destination set to the original values. If ADC does not supply a project, set `os.environ['GOOGLE_CLOUD_PROJECT']` to the bucket's Google Cloud project ID before running the cell.

```python
import os
import sys
from pathlib import Path
from google.colab import auth, userdata

method = 'distillation'
project = WANDB_PROJECT.strip()
entity = WANDB_ENTITY.strip()
bucket = WANDB_WRITER_BUCKET.strip()
if not all((project, entity, bucket)):
    raise ValueError('Set the existing W&B project, entity and shared writer bucket.')

backup = Path(PERSISTENT) / 'stages' / ('research-v3-' + method + SCOPE_SUFFIX)
command = [REPO / 'tools/wandb_transfer.py', '--work', RESEARCH_ROOT / method,
           '--backup', backup, '--logs', LOGS, '--method', method, '--project', project,
           '--entity', entity, '--group', f'{EXPERIMENT_ID}/{TRANSFER_EXPERIMENT_ID or "default"}']
execute(sys.executable, *command, '--audit-metrics-only', name='wandb-audit-' + method)

auth.authenticate_user()
sdk = Path('/content/alignment-wandb-recovery-venv')
python = sdk / 'bin/python'
execute(sys.executable, '-m', 'pip', 'install', 'uv', name='install-sync-uv')
execute('uv', 'venv', '--python', sys.executable, '--allow-existing', sdk, name='sync-environment')
execute('uv', 'pip', 'install', '--python', python, 'wandb==0.31.0', 'filelock==4.1.0',
        'google-cloud-storage==3.4.1', name='sync-dependencies')
execute(python, *command, '--sync-metrics-only', '--recover-writer', '--writer-bucket', bucket,
        env=dict(os.environ, WANDB_API_KEY=userdata.get('WANDB_API_KEY'), AT_LOG_WANDB_DRAIN='1'),
        name='wandb-cold-sync-' + method)
```

The recovery command derives the same Run ID from the verified archived contract and checks W&B entity, project, Run ID, method, group, terminal state, and any previously recorded contract and lock bucket. It holds the lock throughout history reconciliation and, when rows are needed, until `wandb.finish()` confirms completion. On success it can recreate the local ownership record. A fully synchronized run is a no-op. Existing values are compared by `(update, metric key)` and a conflict aborts before writes. The console reports `[wandb-metrics] synced` and the number of appended rows.

If a previous writer died after acquiring the shared lock, the next sync reports its `gs://` object path. Confirm that the original writer process or VM is gone and that W&B shows a terminal state, then read the object's live **Generation** in the Cloud Storage console. Add `--recover-lock-generation` and that exact numeric value to the final command. Recovery validates the orphan's identity and replaces that generation conditionally. An occupied lock never expires automatically. Use the same bucket in future notebook runs. The stored bucket binding makes updated reporters reject a different or omitted bucket for that Run.

The CPU command verifies and reads the necessary JSONL members directly from the existing snapshot. Fairseq window metrics require the separately retained execution logs and receipts. The command leaves Drive source data and checkpoints intact. Saved BLEU still uses its dedicated sync mode or the training reporter. [Metric definitions and audit](WANDB_METRIC_AUDIT.md#colab)
