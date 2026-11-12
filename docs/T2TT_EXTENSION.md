# Verified T2TT 20k → 200k extension

Continue the saved 20k T2TT run to 200k using its existing training state. The 20k stage remains the read-only source for a separate continuation stage. Measure the 200k results and BLEU in Colab.

## Colab workflow

Open the latest `notebooks/legacy/benchmark_comparison.ipynb`.

1. Connect Drive and set `ROOT`, `PERSISTENT`, and `BENCHMARK` to their original values.
2. Run the settings cell and the T2TT storage-path cell immediately below it. Defaults are listed in the table.
3. In section 2 run the environment check. In sections 3–5 prepare the original inputs, Baseline metadata, manifest, and backbone.
4. Enable `RUN_T2TT_EXTENSION=True` when continuing training and run **6.1a**. The first run starts from a verified 20k snapshot. Later runs resume from the 200k stage.
5. In **6.1b**, verify completion at 200k and measure in-domain dev BLEU for the selected best checkpoint.
6. In **6.1c**, enable `RUN_EXTERNAL_TEXT=True` for external BLEU and translations on the same saved 8C 300 samples.
7. In **6.1d**, review the loss and LR curves and the saved 20k and 200k best comparison.
8. After reviewing evaluation, set `RUN_TEACHER_ARTIFACTS=True` and run **6.2**.
9. Extract attention, align CTC, construct and diagnose the teacher, then proceed to S2UT if needed.

Run all stops at the 6.2 teacher gate until 200k and matching dev and external results are complete for the selected best checkpoint. The Monotonic-only workflow skips teacher preparation.

| Setting | Default | Purpose |
|---|---:|---|
| `T2TT_BASE_UPDATES` | 20000 | Source budget. Also checked against the last checkpoint step |
| `T2TT_TARGET_UPDATES` | 200000 | End of the continuation stage |
| `RUN_T2TT_EXTENSION` | False | Enable to run extension in 6.1a |
| `USE_EXTENDED_T2TT` | True | Use the best checkpoint from the 200k stage for evaluation and downstream stages |
| `T2TT_SNAPSHOT_UPDATES` | 10000 | Update interval between full Drive ZIPs |
| `T2TT_KEEP_LAST` | 2 | Recent snapshots to retain in addition to milestones |
| `RUN_TEACHER_ARTIFACTS` | False | Explicit switch to prepare attention after evaluation |

`T2TT_UPDATES` is a legacy alias for the 20k budget. The 200k limit comes from `T2TT_TARGET_UPDATES`. Rerun the storage-path cell after changing `USE_EXTENDED_T2TT`. To evaluate a completed 200k run, set `RUN_T2TT_EXTENSION=False` and `USE_EXTENDED_T2TT=True`.

## Source verification

Use the latest verified ZIP in `PERSISTENT/stages/t2tt`. The pinned benchmark's `latest/restore` checks the ZIP SHA-256, member list, and individual file hashes.

- Verify the benchmark revision, training module, logical manifest and config paths and their hashes, output, planned update budget, and original input identity including the interpreter.
- Require `checkpoint_last.pt`, `checkpoint_best.pt`, `experiment.json`, and `metrics.jsonl`.
- Require 20,000 updates in the snapshot and last checkpoint. The best checkpoint must share the contract and have a step at or before the last checkpoint.
- Require model and Adam state, epoch, cursor, best loss, bad validation count, and Python, NumPy, Torch, and CUDA RNG state in the last checkpoint.
- Verify the scientific contract and hash across last, best, and experiment metadata. Training metrics must contain a contiguous prefix from step 1 to the last update.

Missing files, corruption, and configuration mismatches stop the extension. A verified last checkpoint and its original contract are required.

## Recovering the original training code

The trainer's scientific contract binds the `implementation_hash` of all Python sources. Evaluation and persistence changes can change this hash. Continuation requires the original implementation.

The extension looks for the complete source tree matching the checkpoint hash. It copies an exact current tree or exports the matching revision from local Git history to `.training-runtime/` inside the new stage. It stops if the tree cannot be verified.

Run the isolated CLI, parser, model, and trainer in a separate process. On first launch and resume, run a zero-update preflight with the limit set to the saved step. The original trainer verifies the contract and loads the optimizer and RNG. After checking source and target contract equality, reload the same last checkpoint and continue.

Only the ending update limit changes. The continuation uses the original manifest, architecture, tokenizer, seed, LR and warmup, batch and update frequency, max tokens, clipping, label smoothing, validation and save intervals, patience, backend, splits, and implementation. LR follows the global optimizer update. The new stage records `planned_max_updates=200000` in the experiment metadata, outside the scientific contract.

## Continuation and best-checkpoint selection

Resume from `checkpoint_last.pt` to restore the model, optimizer, step, epoch and cursor, best dev loss, bad validation count, and all four RNG states.

Seed `checkpoint_best.pt` from the source. The original trainer updates it only when `dev/main_loss` improves after 20k. The final teacher is the best dev-loss checkpoint observed through 200k and may have a step before 20k.

`extension-lineage.json` stores source stage and snapshot identity, ZIP and checkpoint hashes, source and target contract hashes, and budgets. `extension-plan.json` stores the original command, isolated implementation, and run and save policies. Recovery ZIPs include both files, milestones, and the controller.

## Snapshot frequency and retention

Validation runs every 1000 updates and local checkpoint saves every 100. The extension wraps the trainer's completed-save callback and publishes full Drive ZIPs every 10,000 updates by default, at milestones, and at the final update. Local progress uses the existing schema. The snapshot interval can use another multiple of the local save interval, such as 5000.

The pinned benchmark handles the queue, freeze, upload, checksum, and recovery for the extension stage. The continuation uses the original training code and save policy for the source 20k and S2UT stages.

After a new ZIP uploads and passes checksum verification, retain the latest two snapshots plus the 50k, 100k, 150k, and 200k milestones within the extension stage. Keep all 20k source snapshots and directories with unknown files. Log deletion failures and continue.

`extension-milestones.json` records the actual best loss and step at 20k and each milestone. Each milestone ZIP contains its then-current best checkpoint for reevaluation on the same fixed 300 samples.

On failure, the final backup trims metrics beyond the recovery checkpoint from the frozen backup copy. Keep the original VM log. The learning-curve report flags missing and duplicate steps.

## Separate storage paths

| Purpose | Local VM | Drive stage |
|---|---|---|
| Source 20k | `ROOT/t2tt` (read-only to the extension) | `PERSISTENT/stages/t2tt` (read-only) |
| extended training | `ROOT/t2tt-200k` | `PERSISTENT/stages/t2tt-200k` |
| extended dev | `ROOT/text-evaluation/t2tt-200k/dev-text.json` | `t2tt-dev-200k` |
| extended FLEURS | `ROOT/text-evaluation/t2tt-200k/flores-fleurs/flores-fleurs-text.json` | `t2tt-flores-fleurs-200k` |
| Attention | `ROOT/t2tt-attention-200k` | `t2tt-attention-200k` |
| CTC | `ROOT/speech-text-alignment-200k` | `speech-text-alignment-200k` |
| projected teacher | `ROOT/teacher-alignment-200k` | `teacher-alignment-200k` |
| alignment audit | `ROOT/audit-200k` | `diagnostics-200k` |

Abbreviated Drive stages in the table are under `PERSISTENT/stages/`. Both runs share `ROOT/external-text-input/fixed-8c.json` and the same saved 8C set of 300 examples.

`T2TT_CHECKPOINT = T2TT_ACTIVE_ROOT / 'checkpoint_best.pt'` supplies dev evaluation, external evaluation, and attention extraction. Teacher construction and S2UT use the selected attention and teacher directories. S2UT with the extended teacher writes to `annealed-t2tt-200k`.

## Live loss and Drive summaries

The parent process tails the VM-local `metrics.jsonl` once per second. Every 100 updates or on a new dev measurement, it writes `[T2TT loss]` with train loss, LR, latest dev loss and update, and best observed loss to stderr. The shared logger displays and saves it. The monitor reports at startup and exit and waits for incomplete JSONL lines. Live observations can be ahead of the latest durable checkpoint.

After publishing and verifying each full ZIP, write `metrics-summary.json` once in its snapshot directory. Use the frozen metrics copy trimmed to the checkpoint update. Bind the summary to the ZIP, metrics, and best-checkpoint hashes, contract, and durable update. Include latest train and dev values, dev measurement step, actual best step and loss, and history.

Keep up to 2000 measured points. Retain all dev measurements up to 1000 and fill remaining slots with evenly sampled training points. Sample dev points when there are more than 1000. The ZIP's `metrics.jsonl` contains the complete history. Summary failure emits a warning and the published checkpoint remains usable. Retention removes summaries with their corresponding snapshots.

```text
PERSISTENT/stages/t2tt-200k/<update>-<id>/
  snapshot.json
  snapshot.zip
  metrics-summary.json
```

The verified ZIP and receipt remain the recovery source. Monitoring lives in the parent process and uses the original trainer and worker contract. It can monitor a resumed run with an older saved controller. For completed stages missing summaries, regenerate them from restored local history.

An active process uses its loaded code. To view live history, download `ROOT/t2tt-200k/metrics.jsonl` from Colab's Files panel. Use the updated notebook at the next normal resume. A second code cell waits while the training cell is active.

## Resume

Rerun with the same revision, settings, and input paths. If `t2tt-200k` has a verified snapshot, restore its last checkpoint and original implementation directly from the saved revision and hash. Completed 200k stages perform zero optimizer updates.

Before the first verified continuation snapshot, seed from verified 20k and retain unpublished local work in `.interrupted-*`. If the 200k stage has ZIPs but none pass verification, stop for inspection. If the VM is lost, recompute updates after the last durable snapshot.

## CLI

```bash
python -m alignment_transfer_s2ut.t2tt_extension extend \
  --source-work /local/work/t2tt \
  --source-backup /drive/experiment/stages/t2tt \
  --work /local/work/t2tt-200k \
  --backup /drive/experiment/stages/t2tt-200k \
  --manifest /local/work/manifest/manifest.jsonl \
  --config /local/work/backbone.json \
  --source-updates 20000 --target-updates 200000 \
  --snapshot-updates 10000 --keep-last 2
```

In Colab, run 6.1a through the shared logger with GPU monitoring. The dedicated extension runner owns the budget change and uses a separate stage.

```bash
python -m alignment_transfer_s2ut.t2tt_extension status \
  --work /local/work/t2tt-200k --target-updates 200000
python -m alignment_transfer_s2ut.t2tt_extension report \
  --work /local/work/t2tt-200k --output /local/work/t2tt-200k/diagnostics
```

The report plots `train/main_loss`, `dev/main_loss`, and `train/learning_rate` against optimizer update. It reads 20k comparison values from saved local JSON and leaves missing entries blank.

## CPU continuation tests

Set one CPU thread in both the pytest baseline and the extension subprocess for exact equality. `torch.set_num_threads(1)` is process-local, so the test fixture also sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for child processes and restores them at exit. Compare tensors with `torch.equal`.

All-layer runs add `-all-layers` to the attention, teacher, audit, and transfer paths. They share the T2TT training, dev and external evaluation, and CTC paths. [All-layer settings](ALL_LAYER_TRANSFER.md)
