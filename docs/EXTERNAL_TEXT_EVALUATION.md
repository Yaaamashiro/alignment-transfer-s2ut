# FLEURS/FLORES external text evaluation for T2TT

Evaluate a trained T2TT checkpoint on the same saved 300-sample 8C set used for prior Cascade and S2UT evaluation. This text-only workflow uses the existing checkpoint, vocabulary, and sample selection.

## Current notebook

Set `EVALUATE_T2TT_EXTERNAL=True` and the training run's `T2TT_RUN_ID`. Set `EVALUATE_S2UT=False` for T2TT-only evaluation. The notebook restores the fixed 300 samples, translates, and scores them. [evaluate.ipynb](../notebooks/colab/evaluate.ipynb)

## Legacy evaluation

Run **6.1c FLEURS/FLORES external text BLEU** in `notebooks/legacy/benchmark_comparison.ipynb`.

1. Prepare the environment, mount Drive, and load the `run(...)` helper. Use the saved checkpoint from 6.1a.
2. Set `ROOT`, `PERSISTENT`, and the T2TT selection. `USE_EXTENDED_T2TT=True` selects the completed 200k best checkpoint. `False` selects the existing 20k best. Restore it from the selected stage's completed ZIP if needed.
3. Set `RUN_EXTERNAL_TEXT=True` in 6.1c. If multiple saved 8C runs exist, set `SOURCE_8C` to the existing identity folder.
4. Run the cell to verify the fixed 300 samples, generate, score, publish a ZIP, and display scores and ten examples selected by a fixed seed.

`restore-inputs` first verifies the ZIP and member checksums using the pinned benchmark's `latest/restore`. It verifies paths, the T2TT stage, benchmark revision, and completed budget, and evaluates the saved `checkpoint_best.pt`. An existing local best remains in place. Incomplete local training directories are retained under `.interrupted-*` as needed. A missing model, corrupt ZIP, unfinished run, or missing best checkpoint stops evaluation. Restore only the model and evaluation inputs.

If `PERSISTENT/stages/t2tt-dev` contains a completed dev evaluation, verify its best checkpoint and restore the metrics and predictions. Only the two files needed from the final incremental ZIP are restored, preserving the parent directory and any external evaluation. A conflicting local dev result stops restoration.

If dev results are absent, evaluate the fixed external 300 with the existing `evaluate_t2tt` generation and BLEU settings and report that the in-domain comparison was skipped. When dev is present, reject corruption, missing predictions, and checkpoint or settings mismatches.

Before displaying results, run the read-only `verify` command even if a restored ZIP skipped generation. It checks checkpoint and fixed-set hashes, available dev evaluation and prediction hashes, external prediction IDs and hashes, and corpus BLEU. Any mismatch stops display.

Display the actual update step of the selected best checkpoint. Dev and external results must share its checkpoint SHA.

## Verify saved 8C inputs

Read the following files from `BENCHMARK/diagnostics/flores-fleurs-cascade/<identity>/` as evidence for the fixed set.

- `identity.json` records Japanese and English text, IDs, and text SHA-256 in `selected`.
- `dataset-audit.json` records `selected_samples` and `selected=300`.
- `metrics.json` contains the completed report, `identity_sha256`, `metrics.samples=300`, and per-sample results.
- `per_sample.jsonl` records the actual evaluated IDs and text.

Use the pinned `direct_s2st.journal.digest` to verify identity and folder name. The four ID sets must match exactly, with unique IDs, no omissions, matching text hashes, and matching historical BLEU implementation. Sort by `sample_id` to fix prediction order.

A missing completed 300-sample report raises `No completed saved 8C FLORES/FLEURS 300-sample identity found...`. Multiple reports require an explicit saved identity. The workflow requires the complete original set.

Write the verified evidence to `ROOT/external-text-input/fixed-8c.json` on the VM. The file becomes an evaluation argument and its hash enters the existing runtime identity. Changed 8C evidence requires a new output location.

## Inference and metrics

The legacy workflow uses `evaluation.text.generate_text` with the checkpoint tokenizer for source-only autoregressive generation. Dev defaults are beam 5, `max_chars=1000`, and length penalty 1.0. A saved dev contract must match these settings.

The generation loop receives only `sample_id` and `source_text`. Join English references for scoring after generating all 300 predictions. Intermediate caches contain source and prediction data, and scoring uses the generated text.

| Output | Meaning |
|---|---|
| `text_bleu` | Same SacreBLEU as dev, with 13a, case sensitivity, exp smoothing, and `effective_order=False` |
| `benchmark_normalized_bleu` | Apply pinned `normalize_english` to hypotheses and references, then score with `corpus_bleu`. Main metric for comparison with previous Cascade and S2UT external evaluations |
| `sentence_bleu` | Per-sample diagnostic with `effective_order=True` for short text. Report corpus BLEU for the experiment |

Benchmark normalization uses NFKC, lowercasing, its existing punctuation rules, and whitespace cleanup. Characters such as apostrophes follow the pinned implementation. Differences between T2TT text BLEU and Cascade or S2UT ASR-BLEU also include the effects of speech generation and ASR.

Unknown characters use the checkpoint tokenizer's UNK. Record Unicode codepoint counts including spaces, UNK counts and rates, sentence counts and rates with UNK, and common unknown characters.

## Storage and recovery

Dedicated VM directory

```text
ROOT/text-evaluation/flores-fleurs/
  flores-fleurs-text.json
  flores-fleurs-text-predictions.jsonl
  flores-fleurs-text-cache/*.json
```

Dedicated Drive stage

```text
PERSISTENT/stages/t2tt-flores-fleurs/<snapshot>/
  snapshot.json
  snapshot.zip
```

`run(...)` saves and restores cached predictions and results in verified incremental ZIPs. External text evaluation uses its own destination alongside the T2TT, dev, and Cascade and S2UT results.

If overriding output names in the CLI, the derived metrics, prediction, and cache paths must remain separate from dev results, dev predictions and caches, the checkpoint, and fixed 8C inputs. For example, `dev-text.jsonl` can collide with an existing dev prediction filename despite a different extension.

On the same VM, rerun 6.1c. On a fresh VM, prepare the environment and Drive connection in sections 1 and 2 using the same revision, paths, and settings, then run 6.1c. It restores the required best checkpoint and saved dev results. The resume contract binds dev presence and hashes. Reuse verified caches and completed results. Standalone CLI resume also checks the contract, prediction hashes, and recomputed corpus BLEU. Predictions lost since the last ZIP are recomputed.

Training resume binds all Python sources. Finish unfinished T2TT or S2UT training with its original revision. Use the evaluation revision to load completed checkpoints for external evaluation.

## CLI

With the pinned benchmark runtime installed, restore saved inputs, freeze the fixed set, and evaluate in order.

```bash
python -m alignment_transfer_s2ut.evaluate_t2tt_external restore-inputs \
  --work /local/work/t2tt --backup /drive/experiment/stages/t2tt \
  --expected-updates 20000 --dev-backup /drive/experiment/stages/t2tt-dev \
  --dev-output /local/work/text-evaluation/dev-text.json

python -m alignment_transfer_s2ut.evaluate_t2tt_external prepare \
  --benchmark /path/to/benchmark-experiment \
  --output /local/work/external-text-input/fixed-8c.json

python -m alignment_transfer_s2ut.evaluate_t2tt_external evaluate \
  --fixed-set /local/work/external-text-input/fixed-8c.json \
  --checkpoint /local/work/t2tt/checkpoint_best.pt \
  --dev-evaluation /local/work/text-evaluation/dev-text.json \
  --output /local/work/text-evaluation/flores-fleurs/flores-fleurs-text.json \
  --device cuda --resume
```

In Colab, run the CLI through the common logger and stage wrapper in 6.1c.

## Evaluate the 200k teacher

`USE_EXTENDED_T2TT=True` sets `T2TT_CHECKPOINT` to `ROOT/t2tt-200k/checkpoint_best.pt`. Dev uses `text-evaluation/t2tt-200k/dev-text.json`, external results use `text-evaluation/t2tt-200k/flores-fleurs/`, and Drive uses `t2tt-dev-200k` and `t2tt-flores-fleurs-200k`. Both runs share the fixed 8C packet. `False` selects the 20k paths. Rerun the path cell after changing the setting. [Extension, verification, and attention workflow](T2TT_EXTENSION.md)
