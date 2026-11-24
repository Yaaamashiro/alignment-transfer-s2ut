# W&B metrics for the matched S2UT comparison

Use the schema-v2 mapping below for corrected transfer comparisons. Baseline PR #1 has verified Unit CE in `train/main_loss`. The legacy transfer key can contain NLL. Keep previous history and append the verified Baseline values to the matching schema-v2 series. [Metric audit](WANDB_METRIC_AUDIT.md)

This specification is for the Baseline importer and reporter in `ja-en-direct-s2st-benchmark`. The comparison is with `alignment-transfer-s2ut` PR #7 and the Baseline trained under the registered `paper_fisher_v1` recipe. Use the registered corpus, Fairseq revision, and scientific config when checking another Baseline. The pinned Baseline runtime is `0c0024ea888206149a3119bf096b86ba66d63b66`, with Fairseq at `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`.

## W&B destination

| Setting | Value |
| --- | --- |
| Operator-configured project for `transfer-fixed-v1` | `direct-s2st` |
| Operator-configured entity for `transfer-fixed-v1` | `yaaamashiro-national-institute-of-technology-okinawa-college` |
| Group for this comparison | `experiment-001/transfer-fixed-v1` |
| Run names | `baseline`, `distillation`, `annealed_distillation`, `mask_distillation` |

Use `ENABLE_WANDB=True`, set `WANDB_PROJECT` and `WANDB_ENTITY` explicitly, and use the Colab secret `WANDB_API_KEY`. The checked-in notebook defaults are `alignment-transfer-s2ut` and an empty entity. The training notebook passes the selected values to the reporter, which logs the W&B-resolved Run URL in the `[wandb] connected` event. The transfer group is `<EXPERIMENT_ID>/<TRANSFER_EXPERIMENT_ID or 'default'>`. The Baseline reporter should accept that group explicitly. [Notebook settings](../notebooks/colab/training.ipynb) · [Reporter](../tools/wandb_transfer.py)

The destination above is the operator's stated configuration. Direct W&B access was unavailable during this audit, and the saved Drive records do not establish which destination accepted the transfer Runs. The authority for each existing Run is its W&B entity, project and Run ID, or the matching `[wandb] connected` Run URL from `execution/notebook-events.log`. Check those records for the selected transfer conditions before sending the Baseline import. A project name in a Notebook or a Run display name alone cannot identify the destination.

## Run ID and Baseline identity

PR #7 derives each transfer Run ID as `sha256(contract_hash.encode()).hexdigest()[:24]`. Preserve those IDs. If the Baseline already has a W&B Run, use its verified W&B Run ID and `resume="must"`. The derivation below applies on first authorized publication of a Baseline without a W&B Run.

| Input | Authoritative source |
| --- | --- |
| `baseline_run_key` | Use the original per-run snapshot publication directory in the run descriptor or launch configuration (`BACKUP` in the alignment Notebook, `--backup-root` in the benchmark launcher). Strip the exact `/content/drive/MyDrive/` prefix once. Use forward slashes with nonempty path components, preserve case and Unicode code points, and reject `.`, `..`, and leading or trailing slashes. Freeze this key when first binding the Baseline to W&B. A move, ZIP restore, or new runtime keeps the original key. For the registered Baseline here the key is `alignment-transfer-s2ut-data/experiment-001/baseline`. |
| `training_fingerprint` | Use `research-metadata.json["training_fingerprint"]` from a verified published snapshot. Verify that it equals `s2ut-recipe-lock.json["recipe"]["training_fingerprint"]` and `checkpoint_last.pt["extra_state"]["s2st_recipe_fingerprint"]` when establishing the lineage. A verified existing registration may supply `baseline.json["baseline_fingerprint"]`, which the registration code already checked against the lock, metadata and registered checkpoint. Use the stored original lowercase 64-digit SHA256. A resumed import or reporter reads that same fingerprint even if the launcher path or its source revision changed. |

The Import Notebook and live reporter use the same serialized array and hash. Store the two inputs, the derived or adopted Run ID, and its exact entity and project in the reporter's persistent W&B binding alongside the run's metadata. Reuse that binding for later imports and reporting. A missing input or a conflicting binding requires verification of the original run before publishing.

```python
import hashlib
import json
import re


def baseline_wandb_run_id(baseline_run_key, training_fingerprint):
    if "\\" in baseline_run_key or any(part in ("", ".", "..") for part in baseline_run_key.split("/")):
        raise ValueError("expected the original nonempty MyDrive-relative publication key")
    if not re.fullmatch(r"[0-9a-f]{64}", training_fingerprint):
        raise ValueError("expected the saved lowercase SHA256 training fingerprint")
    payload = json.dumps(
        ["baseline", baseline_run_key, training_fingerprint],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]
```

Hash the UTF-8 bytes as emitted, with zero extra whitespace, no byte-order mark and no terminating newline. With a synthetic fingerprint of 64 zeroes, the example Run key above yields `047024bc03277652fa2aa9fe`. The key `benchmark-data/実験/baseline` yields `22eba1499911c50df601ae81` with the same synthetic fingerprint.

The original publication directory and the archived 400k Baseline registration were checked read-only. The registration archive and `baseline.json` matched the published SHA256 values, and `baseline_fingerprint` matched the embedded research metadata. [Baseline registration](../src/alignment_transfer_s2ut/benchmark.py) · [Pinned recipe fingerprint binding](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/0c0024ea888206149a3119bf096b86ba66d63b66/src/direct_s2st/s2ut/recipe_identity.py)

## Update and Summary conventions

- `update` is the number of completed optimizer updates. Use the monitor's `update`, the verified checkpoint's optimizer history, or Fairseq's `num_updates` and `dev_num_updates`. One update can contain four microbatches in the registered recipe. Fairseq's printable fractional epoch field and W&B's internal `_step` do not represent optimizer updates.
- Define `update` with `summary="none"` and `overwrite=True`. Register `train/*`, `dev/*`, and `eval/*` with `step_metric="update"` and put the actual update in every history row. Let W&B append rows with its own internal step so old BLEU points can arrive late.
- Set the Summary key `train/max_update` to the maximum of its previous value, the verified training watermark, and the latest matching published checkpoint. A BLEU backfill leaves that progress unchanged.

## Metric schema v2 for the Baseline importer and reporter

The registered recipe uses the same native S2UT Unit and three auxiliary losses in all four conditions. Unit loss is label-smoothed cross-entropy with smoothing 0.2. For each reported component, sum criterion numerators and sample sizes before dividing. Main Unit and total losses use Fairseq's main `ntokens` under this token-averaged recipe. Each auxiliary loss uses its own criterion `sample_size`. Baseline monitor losses are in bits. Convert bits to nats by multiplying by `ln(2)`.

The raw task numerator is `Unit + 8 × source_letter + 8 × target_letter + 1.6 × decoder_target_ctc`. Apply these weights to raw sums before dividing by the main sample size. The three separately normalized auxiliary losses are unweighted. The transfer optimizer adds the alignment term during training. Validation evaluates the full Dev set with teacher-forced target units and uses the Unit loss for checkpoint selection.

| Signal | Shared schema-v2 key | Baseline source, units, and scope |
| --- | --- | --- |
| Train Unit CE | `train/unit_ce_nats` | `train.unit_loss × ln(2)` from `training-monitor.json`, or `train_unit_loss × ln(2)` from its CSV. One logical update, label-smoothed Unit CE in nats per main target token. |
| Train weighted Task Loss | `alignment/update/task_loss` | `train.loss × ln(2)` / `train_loss × ln(2)`. One logical update in nats per main target token. The Transfer key records the shared weighted S2UT task term before the alignment loss is added. |
| Learning Rate | `train/update/learning_rate` | The monitor's `learning_rate`. Both use Fairseq's `get_lr()` after the completed optimizer update. No unit conversion. |
| Dev Unit CE | `dev/main_loss` | `dev.unit_loss` / `dev_unit_loss`. Already shared. Full-Dev label-smoothed Unit CE in bits per main target token. |
| Dev Total Loss | `dev/full/loss` | `dev.loss` / `dev_loss`. Full-Dev weighted S2UT task loss in bits per main target token. Preserve the available precision of each reporter. |
| Dev Source Letter Loss | `dev/full/multitask_source_letter_loss` | `dev.source_letter` / `dev_source_letter`. Full-Dev unweighted bits per Source Letter criterion sample unit. |
| Dev Target Letter Loss | `dev/full/multitask_target_letter_loss` | `dev.target_letter` / `dev_target_letter`. Full-Dev unweighted bits per Target Letter criterion sample unit. |
| Dev decoder-target CTC Loss | `dev/full/multitask_decoder_target_ctc_loss` | `dev.decoder_target_ctc` / `dev_decoder_target_ctc`. Full-Dev unweighted bits per decoder-target CTC criterion sample unit. |

The Baseline importer and reporter should use the same mapping and the existing durable outbox on the existing Baseline Run ID. Keep legacy Baseline keys `train/main_loss`, `train/task_loss`, `train/learning_rate`, `dev/loss`, and `dev/multitask_*` available for existing charts. `dev/main_loss` and the qualified BLEU keys already match. For schema-v2 append, compare existing values by `(update, metric key)` so a partial record does not suppress missing keys. On disagreement, stop before publishing. Keep `update` as the metric axis and preserve the `train/max_update` watermark.

- Transfer optimization loss includes the alignment term and remains in `train/update/optimization_loss_nats` and `train/window/loss_fairseq`. Baseline uses the shared Task Loss key above.
- Baseline's logical-update auxiliary keys `train/aux/<task>_loss_bits` and Transfer's Fairseq-window keys `train/window/multitask_<task>_loss` have different scopes. A matching Baseline window can use the latter when recorded with the same aggregation.
- Use `train/window/main_loss_fairseq` for a matching recorded CE window. NLL and perplexity require their own measurements. Omit them when the Baseline has only CE. Full-Dev NLL and perplexity require separately measured full-Dev data.

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
