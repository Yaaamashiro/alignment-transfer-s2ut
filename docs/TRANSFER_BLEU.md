# Transfer BLEU during training

Full transfer training can pause at 1k, 5k, 10k, 25k, 50k, 100k, 200k, and 400k optimizer updates. A pause uses the existing Fairseq training stage, its complete 400k schedule, and its verified Drive snapshot protocol. The evaluator loads the exact published last checkpoint in a separate process. Successful evaluation is followed by the next training segment. BLEU is report-only. Checkpoint selection still uses the registered training criterion.

## Resume an existing transfer run

1. Open the updated `notebooks/colab/training.ipynb` on a GPU runtime. Keep the experiment, transfer and calibration IDs, corpus paths, Top-p, coefficients, and Teacher settings from the run you are resuming. For the current runs use `experiment-001` and `transfer-fixed-v1`.
2. Keep `RUN_BASELINE=False` and select the transfer conditions to resume. Set `TRANSFER_RUN_MODE='full'`, `TRANSFER_RECOVERY_CHECK='off'`, and `TRANSFER_ONLINE_BLEU=True`. Keep the previously configured settings for restoring inputs and training artifacts. Enable W&B and grant the notebook access to the `WANDB_API_KEY` Colab secret to publish metrics.
3. Run the setup and input restoration sections, then **Train the selected transfer conditions**. The notebook computes the existing scientific plan and checks it against the receipts before starting training. A run already between evaluation milestones starts with the next milestone. At an exact milestone the notebook reuses the already published checkpoint and any completed evaluation.

The first evaluation fetches or uses cached copies of the pinned reference vocoder and Whisper model. Recognized generation failures, such as a decoder missing EOS or producing invalid units, enter the fixed denominator as empty hypotheses. They are listed in `generation_failures` in the saved report. An ASR, vocoder, input, CUDA, timeout or unclassified error aborts the evaluation and leaves a `failed-*.json` receipt without publishing a score. Review the command log, then rerun the training or backfill cell. Successful predictions are reused from the local journal while the runtime survives. If CUDA remains unhealthy, restart the runtime and restore the same run.

For a new full run, keep `TRANSFER_ONLINE_BLEU=True`. Empty training and backup locations allow the existing trainer to initialize at update 0, with the first evaluation at 1k. If training files exist without a usable published snapshot, the notebook stops and preserves them for recovery.

## Backfill earlier checkpoints

In `notebooks/colab/evaluate.ipynb`, use the same experiment and corpus settings, enable `EVALUATE_S2UT` and `BACKFILL_TRANSFER_BLEU`, and select the transfer conditions. Leave `TRANSFER_BLEU_UPDATE=0` to score all available milestones or enter one exact milestone. Run through **Backfill Aux-BLEU and ASR-BLEU** on a GPU runtime. This mode uses the existing speech preparation and then restores historical ZIP payloads into isolated scratch space. Missing checkpoints are reported and skipped. The ordinary final-model comparison workflow is a separate mode.

### Colab L4 smoke test at 25k

1. Open the current PR revision of `evaluate.ipynb` on an L4 runtime. Use `EXPERIMENT_ID='experiment-001'`, `TRANSFER_EXPERIMENT_ID='transfer-fixed-v1'`, and the original corpus settings. Enable `EVALUATE_S2UT`, `BACKFILL_TRANSFER_BLEU`, and `COMPARE_DISTILLATION`. Leave W&B disabled during the smoke test.
2. Run the notebook up to and including **Restore speech data and WAV ZIPs**. Run the following in a new code cell. The separate output path makes the smoke test run actual scoring even if the normal backfill already contains a result.

   ```python
   method = 'distillation'
   stage = 'research-v3-' + method + SCOPE_SUFFIX
   execute(PYTHON, REPO / 'tools/transfer_bleu.py', 'backfill',
           '--method', method, '--update', 25000,
           '--work', RESEARCH_ROOT / method,
           '--backup', Path(PERSISTENT) / 'stages' / stage,
           '--baseline-backup', BACKUP,
           '--scratch', Path(ROOT) / 'transfer-bleu-smoke-work',
           '--output', Path(PERSISTENT) / 'transfer-bleu-smoke' / TRANSFER_EXPERIMENT_ID,
           '--logs', LOGS, name='bleu-smoke-kl-25000')
   ```

3. In the resulting JSON, verify `update=25000`, `aux_samples=500`, and `asr_samples=100`. The KL Snapshot receipt records `ab9012de0422927b952decbaedd36f1ebb3365ba8806bbc3d656dcf8f9e48f4c` for `checkpoints/checkpoint_last.pt`. Verify that `checkpoint_sha256` matches and that any reported failures appear in `generation_failures`. Run the same cell again and check for `Reused BLEU at update 25000` and an unchanged `result_hash`. If the first attempt fails in the evaluation environment, retain its log and rerun the cell after fixing the issue. A failed attempt should only add a failure receipt in the smoke-test directory.

## Metrics and records

- **Aux-BLEU** uses the encoder-attached target-letter decoder and a fixed seed-1 hash sample of 500 Dev pairs.
- **ASR-BLEU** uses beam-10 unit generation, the benchmark's official unit vocoder and pinned English Whisper ASR. It scores the first 100 pairs from the same fixed sample. Both metrics use the benchmark's English normalization and SacreBLEU 13a tokenizer.
- Drive stores an immutable `result.json` and `result.csv` for each completed evaluation under `transfer-bleu/<transfer-experiment>/<method>/<evaluation-scope>/<update>/`. Each result binds the exact checkpoint SHA256, scientific contract, Dev reference SHA256, selected pair IDs, evaluation policy and evaluator revision.
- W&B publishes eligible scores to the existing training run for each method. Plot `eval/aux_bleu`, `eval/asr_bleu`, `eval/aux_failures`, or `eval/asr_failures` with `update` as the x-axis. Earlier completed results with zero failures can be reused. Earlier results with unclassified failures remain on Drive and need evaluation under the current failure policy before sync.

## Sync scores already on Drive

During training, use `ENABLE_WANDB=True` in the updated training notebook. It picks up saved BLEU results for the same contract when a checkpoint is published and when the reporter finishes. The backfill evaluation notebook produces Drive results and leaves publishing to that reporter.

After the selected training cells and their W&B reporters have stopped, you can sync saved scores in the same Colab runtime where the latest reporter finished. In the updated `evaluate.ipynb`, use the original experiment IDs, method switches, W&B project and entity. Set `SYNC_SAVED_TRANSFER_BLEU=True`, `ENABLE_WANDB=True`, `EVALUATE_S2UT=False`, `EVALUATE_T2TT_EXTERNAL=False`, and `BACKFILL_TRANSFER_BLEU=False`. Run through **Backfill Aux-BLEU and ASR-BLEU or sync saved scores**. This mode resumes an existing idle run after checking the local ownership receipt against W&B.

To stay in the owning Colab runtime, you can also run this cell in the updated training notebook after the training cell has stopped:

```python
method = 'distillation'
tracking = prepare_wandb()
if tracking:
    python, key = tracking
    backup = Path(PERSISTENT) / 'stages' / ('research-v3-' + method + SCOPE_SUFFIX)
    results = Path(PERSISTENT) / 'transfer-bleu' / (TRANSFER_EXPERIMENT_ID or 'default')
    execute(python, REPO / 'tools/wandb_transfer.py', '--sync-bleu-only', '--method', method,
            '--work', RESEARCH_ROOT / method, '--backup', backup, '--bleu-results', results,
            '--logs', LOGS, '--project', WANDB_PROJECT.strip(), '--entity', WANDB_ENTITY.strip(),
            env=dict(os.environ, WANDB_API_KEY=key), name='wandb-sync-saved-bleu-' + method)
```

Across runtimes, leave saved scores on Drive for the training reporter to pick up. Wait for sync-only to finish before starting a trainer for the same method anywhere. W&B status checks and the runtime's local file lock do not provide a distributed lock across Colab VMs. If the original runtime is gone, the next training reporter picks up the existing scores.

Historical BLEU is appended to W&B history. W&B's internal step still advances, and each BLEU row uses its actual checkpoint update as the chart coordinate. The Run Summary uses `train/max_update` for the largest published training update the reporter has observed. A stored checkpoint hash prevents duplicate points on later syncs. Earlier standalone BLEU runs and their history remain available.
