# Transfer BLEU during training

Full transfer training can pause at 1k, 5k, 10k, 25k, 50k, 100k, 200k, and 400k optimizer updates. A pause uses the existing Fairseq training stage, its complete 400k schedule, and its verified Drive snapshot protocol. The evaluator loads the exact published last checkpoint in a separate process. Successful evaluation is followed by the next training segment. BLEU is report-only. Checkpoint selection still uses the registered training criterion.

## Resume an existing transfer run

1. Open the updated `notebooks/colab/training.ipynb` on a GPU runtime. Keep the experiment, transfer and calibration IDs, corpus paths, Top-p, coefficients, and Teacher settings from the run you are resuming. For the current runs use `experiment-001` and `transfer-fixed-v1`.
2. Keep `RUN_BASELINE=False` and select the transfer conditions to resume. Set `TRANSFER_RUN_MODE='full'`, `TRANSFER_RECOVERY_CHECK='off'`, and `TRANSFER_ONLINE_BLEU=True`. Keep the previously configured settings for restoring inputs and training artifacts. Enable W&B and grant the notebook access to the `WANDB_API_KEY` Colab secret to publish metrics.
3. Run the setup and input restoration sections, then **Train the selected transfer conditions**. The notebook computes the existing scientific plan and checks it against the receipts before starting training. A run already between evaluation milestones starts with the next milestone. At an exact milestone the notebook reuses the already published checkpoint and any completed evaluation.

The first evaluation fetches or uses cached copies of the pinned reference vocoder and Whisper model. GPU OOM and incomplete evaluations stop at the durable checkpoint. Review the command log, then rerun the training cell. The evaluator reuses successful predictions from its local journal while the runtime survives. If CUDA remains unhealthy, restart the runtime and restore the same run. An isolated decoding failure contributes an empty hypothesis and is counted in the report. Failed evaluation attempts have their own JSON records and are eligible for retry.

For a new full run, keep `TRANSFER_ONLINE_BLEU=True`. Empty training and backup locations allow the existing trainer to initialize at update 0, with the first evaluation at 1k. If training files exist without a usable published snapshot, the notebook stops and preserves them for recovery.

## Backfill earlier checkpoints

In `notebooks/colab/evaluate.ipynb`, use the same experiment and corpus settings, enable `EVALUATE_S2UT` and `BACKFILL_TRANSFER_BLEU`, and select the transfer conditions. Leave `TRANSFER_BLEU_UPDATE=0` to score all available milestones or enter one exact milestone. Run through **Backfill Aux-BLEU and ASR-BLEU** on a GPU runtime. This mode uses the existing speech preparation and then restores historical ZIP payloads into isolated scratch space. Missing checkpoints are reported and skipped. The ordinary final-model comparison workflow is a separate mode.

## Metrics and records

- **Aux-BLEU** uses the encoder-attached target-letter decoder and a fixed seed-1 hash sample of 500 Dev pairs.
- **ASR-BLEU** uses beam-10 unit generation, the benchmark's official unit vocoder and pinned English Whisper ASR. It scores the first 100 pairs from the same fixed sample. Both metrics use the benchmark's English normalization and SacreBLEU 13a tokenizer.
- Drive stores an immutable `result.json` and `result.csv` for each completed evaluation under `transfer-bleu/<transfer-experiment>/<method>/<evaluation-scope>/<update>/`. Each result binds the exact checkpoint SHA256, scientific contract, Dev reference SHA256, selected pair IDs, evaluation policy and evaluator revision.
- W&B uses one `<method>-bleu` run per training contract. Either notebook syncs all published scores for that contract, including immutable results written by earlier notebook versions. Plot `eval/aux_bleu` or `eval/asr_bleu` with `eval/checkpoint_update` as the x-axis. `eval/aux_failures` and `eval/asr_failures` report failed samples. Saved scores can be synced later using either notebook or `tools/transfer_bleu.py wandb` with the original output, method, group and project.
