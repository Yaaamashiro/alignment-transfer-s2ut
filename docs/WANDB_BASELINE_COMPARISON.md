# W&B metrics for the matched S2UT comparison

This specification is for the Baseline importer and reporter in `ja-en-direct-s2st-benchmark`. The comparison is with `alignment-transfer-s2ut` PR #7 and the Baseline trained under the registered `paper_fisher_v1` recipe. Use the registered corpus, Fairseq revision, and scientific config when checking another Baseline. The pinned Baseline runtime is `0c0024ea888206149a3119bf096b86ba66d63b66`, with Fairseq at `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`.

## Run and update conventions

| Setting | Value |
| --- | --- |
| W&B project | `direct-s2st` |
| W&B entity | `yaaamashiro-national-institute-of-technology-okinawa-college` |
| Group for this comparison | `experiment-001/transfer-fixed-v1` |
| Run names | `baseline`, `distillation`, `annealed_distillation`, `mask_distillation` |

Use `ENABLE_WANDB=True`, `WANDB_PROJECT`, `WANDB_ENTITY`, and the Colab secret `WANDB_API_KEY`. The transfer group is `<EXPERIMENT_ID>/<TRANSFER_EXPERIMENT_ID or 'default'>`. The Baseline reporter should accept that group explicitly so both sides can join the same comparison.

- `update` is the number of completed optimizer updates. Use the monitor's `update`, the verified checkpoint's optimizer history, or Fairseq's `num_updates` and `dev_num_updates`. One update can contain four microbatches in the registered recipe. Fairseq's printable fractional epoch field and W&B's internal `_step` do not represent optimizer updates.
- Define `update` with `summary="none"` and `overwrite=True`. Register `train/*`, `dev/*`, and `eval/*` with `step_metric="update"` and put the actual update in every history row. Let W&B append rows with its own internal step so old BLEU points can arrive late.
- Set the Summary key `train/max_update` to the maximum of its previous value, the verified training watermark, and the latest matching published checkpoint. A BLEU backfill leaves that progress unchanged.
- PR #7 derives each transfer Run ID as `sha256(contract_hash.encode()).hexdigest()[:24]`. Preserve those IDs. For a Baseline with an existing W&B Run, configure its exact ID and resume it with `resume="must"`. For the first authorized import of a Baseline without a Run, choose a stable namespaced ID from the original Baseline run key and training recipe fingerprint. For example, hash the UTF-8 JSON array `["baseline", baseline_run_key, training_fingerprint]` with fixed compact serialization and take the first 24 hex digits. Use that same ID for import and live reporting. The display name, group, current checkpoint hash, and reporter revision do not identify the run.

## Loss and learning-rate mapping

The registered recipe uses the same native S2UT Unit and three auxiliary losses in all four conditions. Unit loss is label-smoothed cross-entropy with smoothing 0.2. For each reported component, sum criterion numerators and sample sizes before dividing. Main Unit and total losses use Fairseq's main `ntokens` under this token-averaged recipe. Each auxiliary loss uses its own criterion `sample_size`. Baseline monitor losses are in bits. Convert bits to nats by multiplying by `ln(2)`.

The raw task numerator is `Unit + 8 × source_letter + 8 × target_letter + 1.6 × decoder_target_ctc`. Apply these weights to raw sums before dividing by the main sample size. The three separately normalized auxiliary losses are unweighted. The transfer optimizer adds the alignment term during training. Validation evaluates the full Dev set with teacher-forced target units and uses the Unit loss for checkpoint selection.

| Signal | Common metric or separate series | Baseline source and compatibility |
| --- | --- | --- |
| Train Unit loss | **`train/main_loss`** | `train.unit_loss × ln(2)` from `training-monitor.json`, or `train_unit_loss × ln(2)` from its CSV. One logical update, label-smoothed Unit loss in nats per main target token. Use only transfer rows known to contain the smoothed Unit term. |
| Dev Unit loss | **`dev/main_loss`** | `dev.unit_loss` / `dev_unit_loss`. Full-Dev label-smoothed Unit loss in bits per main target token. |
| Learning rate | **`train/learning_rate`** | `learning_rate` from the monitor. Both reporters read Fairseq's `get_lr()` after the completed update. The registered run uses inverse square root scheduling with 10,000 warmup updates, initial warmup rate `1e-7`, and configured rate `0.0005`. |
| Dev weighted task loss | **`dev/loss`** | `dev.loss` / `dev_loss`. Both include the weighted S2UT auxiliaries, in bits per main target token. The transfer reporter reads Fairseq's full-Dev log and its value may be rounded. |
| Dev auxiliary losses | **`dev/multitask_<task>_loss`** | `dev.<task>` / `dev_<task>` for the three tasks above. Both use full-Dev, unweighted bits per auxiliary sample unit. Transfer points depend on available Fairseq logs. |
| Train weighted task loss | Proposed **`train/task_loss`** | `train.loss × ln(2)` / `train_loss × ln(2)`. Its transfer counterpart is the existing `alignment/task_loss`. A consumer can compare these two series or expose a common alias. Keep the existing transfer history keys. |
| Train optimization loss | Keep **`train/total_loss`** and **`train/loss_fairseq`** on transfer | The transfer objective includes the weighted alignment loss. `train/total_loss` is the logical update in nats. `train/loss_fairseq` is the Fairseq logging window in bits. Import Baseline task loss under `train/task_loss`. |
| Train auxiliary losses | **`train/aux/<task>_loss_bits`** for Baseline monitor values | The sparse monitor captures one logical update. The existing transfer `train/multitask_<task>_loss` aggregates Fairseq's logging window. A future Baseline reporter may use the latter key when it uses a matching Fairseq window. |
| Fairseq Unit loss diagnostics and NLL | Keep existing Fairseq series distinct | `train/main_loss_fairseq` is the smoothed Unit loss in nats over the Fairseq logging window. `dev/main_loss_fairseq` is the corresponding full-Dev diagnostic in nats when recorded. Any legacy Unit NLL fallback belongs under a separate key such as `train/unit_nll_legacy_nats` in a derived export. Monitor `unit_loss` cannot reconstruct `train/nll_loss`, `dev/nll_loss`, or perplexity. |

The Baseline monitor samples training every 100 updates and Dev every 1,000. The transfer reporter can also log update 1 and the current published checkpoint. Blank Dev cells in the Baseline CSV mean validation was unscheduled at that update. Preserve gaps and compare like aggregation windows. Attention metrics under `alignment/*` are transfer diagnostics; the Baseline has no measurements for them.

## BLEU mapping

Publish **`eval/aux_bleu`**, **`eval/asr_bleu`**, **`eval/aux_failures`**, and **`eval/asr_failures`** when the corresponding metric record meets the following contract. Check each metric's own evaluation status, counts, and journal.

| Component | Matched protocol |
| --- | --- |
| Evaluation set | Same Dev manifest hash and references. Sort unique Dev pairs by SHA256 of UTF-8 `1:<pair_id>`. Aux scores the first 500. ASR scores the first 100 of those 500. Require matching Pair IDs and reference text. |
| Aux decoding | Source-only, greedy encoder-attached target-letter decoder, maximum 256 tokens, and EOS required. |
| ASR decoding | Source-only Unit generation with beam 10, `max_len_a=1`, `max_len_b=200`, and EOS. Use the pinned official LJSpeech Unit vocoder with duration prediction at 16 kHz and English `openai/whisper-large-v3-turbo` at revision `60be3615a4d667e1258e8ad29130467587c489aa`. |
| Scoring | Corpus SacreBLEU with one reference, tokenizer `13a`, exponential smoothing, and effective order off. Apply NFKC and lowercase, preserve Unicode word characters, straight apostrophes and whitespace, replace other characters with spaces, then collapse and trim whitespace. Require the recorded signature to match, including the SacreBLEU version. |
| Failures | Keep every requested Pair ID in the corpus. A verified generation failure contributes an empty hypothesis. Treat ASR, vocoder, input, CUDA, timeout, and other unclassified errors as failed evaluation attempts. Hold the affected metric out of the shared series until the journal verifies it or a validated retry exists. |

The transfer reporter uses milestones 1k, 5k, 10k, 25k, 50k, 100k, 200k, and 400k. Baseline evaluates Aux every 1k and ASR at those eight milestones. Compare the same milestone and bind the value to the model actually scored at that update. Use a distinct name such as `eval/asr_bleu_full_dev` for the 1,453-pair full-Dev score. External evaluation and any legacy score with unclassified failures stay separate, for example `eval/aux_bleu_legacy_unclassified`.

## Backfill and protection

1. Read published receipt metadata and verify the report against the matching archived file hash. Drive previews are convenient copies. Use the immutable snapshot as the source if the preview and receipt disagree. Keep metric definitions and the loss/Dev/evaluator identity with the import manifest.
2. Only append finite points at their actual optimizer updates, bounded by the matching durable snapshot. Carry the source checkpoint SHA256 when available. For earlier Baseline scores preserved inside a later snapshot, bind the point to its immutable report, model weight hash, evaluation protocol, and update. A later checkpoint SHA256 cannot identify the earlier model.
3. Deduplicate within the target Run using the metric family, update, and a stable source identity. For BLEU, use the checkpoint SHA256 or verified earlier evaluation identity. For loss and LR, use the lineage and the normalized record at that update. Repeating identical data is a no-op. A conflicting value or identity at the same metric/update should stop that import for review. Keep the per-family watermark or per-point marker in W&B across restarts.
4. Keep one W&B writer per training run. During training, queue saved data for that reporter. When importing to an idle Run, resolve its exact destination and verify the Run ID, condition, and lineage before using append-only `resume="must"`. Never rewind, replace, or delete existing history. A local lock and an API status read cannot serialize different Colab VMs.

## Evidence checked for this comparison

The published 400k Baseline receipt names all four loss/BLEU CSV and JSON reports. The Drive loss CSV contains 4,000 sampled training updates and 400 Dev records. The Baseline and the saved KL 50k result have the same Dev manifest hash and BLEU policy. The Baseline BLEU CSV matches the published receipt hash and records 500 Aux and 100 ASR examples at all eight matched milestones. ASR has zero recorded failures at each. Aux has recorded but unclassified failures at every matched milestone, including 500 failures at 1k. Audit the archived `aux.jsonl` for each point before importing that Aux score and its failure count under the common keys. The report CSV alone cannot establish the failure causes or the scoring signature.

An unequal-microbatch CPU check against the pinned Fairseq criterion verified the Unit bits-to-nats conversion, the distinction between smoothed Unit loss and NLL, the weighted objective, and the auxiliary denominators. This comparison did not execute a Baseline importer or write to live W&B.

Source references [PR #7](https://github.com/Yaaamashiro/alignment-transfer-s2ut/pull/7) · [Pinned Baseline loss and BLEU monitoring](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/tree/0c0024ea888206149a3119bf096b86ba66d63b66/src/direct_s2st/s2ut) · [Pinned Fairseq criterion](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/criterions/speech_to_speech_criterion.py) · [W&B custom axes](https://docs.wandb.ai/models/track/log/customize-logging-axes)
