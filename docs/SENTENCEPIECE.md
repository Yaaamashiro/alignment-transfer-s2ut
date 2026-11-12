# SentencePiece T2TT

This T2TT experiment compares character and SentencePiece tokenization. Keep the existing character checkpoints, teacher artifacts, evaluation results, and stages. Start SentencePiece training from a fresh model.

Use `training.ipynb`, set `T2TT_TOKENIZER='sentencepiece'`, select a new `T2TT_RUN_ID`, and set `RUN_T2TT=True`. [Training notebook](../notebooks/colab/training.ipynb) and [text-only T2TT](TEXT_ONLY_T2TT.md)

## Legacy comparison workflow

1. Select `T2TT_TOKENIZER='sentencepiece'`. Use `RUN_T2TT_SENTENCEPIECE=True` to train or `False` to evaluate a completed model. [Legacy comparison notebook](../notebooks/legacy/benchmark_comparison.ipynb)
2. Rerun the storage-path cell, then run the environment check and prepare the same manifest and config.
3. In 6.1a-S, train `ROOT/t2tt-spm-200k` from 0 to 200k. The SentencePiece workflow skips character training and extension. It uses the pinned benchmark's resident session, local freeze, and verified Drive ZIPs.
4. Use 6.1b for dev, 6.1c for the saved fixed 300 samples, and 6.1d for curves and saved character and SentencePiece scores. Reuse the fixed 8C samples.
5. Review BLEU, translations, and repetition diagnostics, then set `RUN_TEACHER_ARTIFACTS=True` for 6.2 onward. Build SentencePiece attention and teacher artifacts for this run.

`RUN_T2TT_EXTENSION` and `USE_EXTENDED_T2TT` are character settings. SentencePiece uses its own new 200k stage and completion check. The CLI and legacy notebook default to character for compatibility.

| Purpose | Separate stage / local path |
|---|---|
| T2TT training | `t2tt-spm-200k` |
| dev | `t2tt-dev-spm-200k` / `text-evaluation/t2tt-spm-200k/dev-text.json` |
| Fixed 300-sample evaluation | `t2tt-flores-fleurs-spm-200k` / `text-evaluation/t2tt-spm-200k/flores-fleurs/` |
| Attention | `t2tt-attention-spm-200k` |
| teacher | `teacher-alignment-spm-200k` |
| audit | `diagnostics-spm-200k` / `audit-spm-200k` |
| KL-based Attention Distillation / Annealed KL-based Attention Distillation / Attention-level Knowledge Distillation | Separate stages under `research-v3-spm-200k-all-layers` |

All-layer attention, teacher, audit, and S2UT paths add `-all-layers`. CTC uses `speech-text-alignment-200k` and units use their existing path. Reuse requires matching input and configuration hashes. Drive stages are under `PERSISTENT/stages/` and local files under `ROOT/`. The workflow rejects character `t2tt` and `t2tt-200k` destinations for SentencePiece.

## Tokenizer specification

- Use SentencePiece **0.2.0**, Unigram, and separate source and target models, each targeting **8000** pieces. `hard_vocab_limit=False` allows smaller vocabularies. Coverage is 1.0. If required characters exceed the vocabulary limit, training stops.
- Fit on `train` alone in manifest order, with one thread and no shuffling or sampling. Dev, test, and fixed FLEURS/FLORES stay out of vocabulary fitting. The required alphabet contains all non-whitespace training characters, including those in literal strings resembling special tokens.
- Use `normalization_rule_name=identity`, `remove_extra_whitespaces=False`, and `add_dummy_prefix=False`. Preserve repeated, leading, and trailing spaces. Byte fallback is disabled.
- Reserve `<bos>=0`, `<pad>=1`, `<eos>=2`, and `<unk>=3`. SentencePiece uses its own whitespace handling. Fixed user-defined symbols preserve tab, LF, and CR in training text.
- Check Python immutable proto surfaces and offsets against the original text. Convert UTF-8 byte offsets to codepoint boundaries when needed. Reject zero-width dummy-prefix tokens and spans that fail to cover the text. Keep the owning proto alive while reading C++ piece views.

Check codepoint spans for all train and dev rows. Rows without UNK must satisfy `decode(encode(text)) == text`. With a train-only vocabulary and no byte fallback, unknown dev characters remain UNK spans at the original positions. Record unknown codepoint counts and exact round trips.

Literal U+2581 `▁` and NUL raise explicit errors because this representation cannot encode them losslessly. If either occurs in production input, choose a reversible representation before training.

The preprocess extra pins protobuf 4.25.3. Startup verifies Unigram, normalization, and special IDs in the model. [Python API](https://github.com/google/sentencepiece/blob/v0.2.0/python/README.md) and [model schema](https://github.com/google/sentencepiece/blob/v0.2.0/src/sentencepiece_model.proto)

## Training and storage

Use 12 encoder and 6 decoder layers, dimension 256, FFN 2048, 4/8 heads, ReLU, dropout 0.1, pre-norm, and sinusoidal positions. The recipe uses batch 8, update frequency 4, max tokens 20000, and 200k updates, with the same Adam, learning rate, warmup, clipping, label smoothing, validation, and best selection. Vocabulary-dependent embedding and projection rows and token sequence lengths differ.

Compare BLEU with shared sources, references, and generation settings because character and SentencePiece losses use different token units. `max_chars` remains the CLI name for the decoder token step limit, including EOS. The default is 1000 steps for both tokenizers.

Checkpoint and experiment `tokenizers.source/target` store the spec and serialized model proto in base64. The contract binds type, Unigram, identity, requested and actual vocabulary size, model SHA-256, SentencePiece version, and normalizer flags. A checkpoint contains enough data to restore its tokenizers.

Also save `tokenizers/ja.model`, `ja.vocab`, `en.model`, and `en.vocab` and check byte parity with the checkpoint on resume. Vocabulary TSVs store JSON-quoted pieces and scores. `tokenization.json` records mean codepoints, mean tokens, codepoints per token, UNK codepoints, and exact round trips by train/dev and source/target. Recovery ZIPs include these files.

New SentencePiece training starts at update 0. Resume uses that stage's last checkpoint, optimizer, RNG, and cursor and checks the tokenizer model, vocabulary request, spec, and contract. Use a new stage to switch tokenizer type. Drive snapshots default to every 10k updates, 50k milestones, and completion. Retain the latest two and milestones. Loss logs and `metrics-summary.json` use the shared implementation.

A checkpoint without tokenizer metadata loads as a legacy `CharacterTokenizer`. Dev and external evaluation and extraction remain available. Resume old training with its original revision. T2TT extension isolates sources that match the saved hash. Reuse old evaluation JSON for comparisons and use a separate evaluation destination with different implementation hashes.

## Attention and speech timing

Attention uses `[T_subword,S_subword]` or `[L,T_subword,S_subword]` for all layers. Save source and target `[begin,end)` codepoint spans, tokenizer type, and source, target, and combined hashes. Mask UNK rows and columns.

CTC aligns original characters. Each subword spans the earliest start and latest end of its valid characters. Ignore invalid characters such as spaces and punctuation, and use `[-1,-1]` if none are valid. The teacher preserves aggregated timing, original CTC timing, and token spans.

`M_tgt @ A_MT @ M_src` projects to HuBERT reduced units by speech encoder positions. The native S2UT recipe supplies the unit vocabulary, model, and task loss. The three transfer conditions use λ and τ from a separate calibration artifact. Projection matches corresponding layers and averages all heads. Tests cover the native fairseq criterion.

Accept legacy character artifacts as one codepoint per token only when attention dimensions match original text lengths. SentencePiece requires spans. The workflow rejects mismatched type, spec, hash, or shape, mixed tokenizer artifacts, and tokenizer changes on resume.

## Evaluation and validation

Dev uses SacreBLEU 13a, case-sensitive exp smoothing, beam 5, and length penalty 1. External evaluation reuses the fixed 300 samples and the same native and benchmark-normalized BLEU. JSON reports BLEU in source length buckets 0–19, 20–39, 40–79, and 80+, along with empty outputs, unique outputs, and repeated 4-gram rate. The tokenizers share metric definitions, source text, and references. Use repetition for diagnosis and dev loss for checkpoint selection.

Small CPU models validate implementation. Measure vocabulary construction with an 8000-piece target, mean sequence lengths, 200k training, GPU memory, and BLEU on the real corpus in Colab.

| File | Role |
|---|---|
| `tokenization/character.py`, `sentencepiece.py`, `factory.py` | Shared API, identity and offsets, checkpoint restore, and diagnostics |
| `tokenization/alignment.py`, `preprocessing.py` | Original character span checks, subword timing, and mismatch rejection |
| `training.py`, `cli.py` | Tokenizer selection, contract, and subword attention extraction |
| `evaluation/text.py`, `evaluate_t2tt_external.py` | Checkpoint factory, codepoint OOVs, shared BLEU, and diagnostics |
| `spm_runtime.py`, `colab_runtime.py`, `training_metrics.py`, `t2tt_extension.py` | Runtime adapter, model and diagnostic recovery, shared loss logs and curves |
| `pyproject.toml`, `environment.py` | Pinned dependencies and import checks |
| `notebooks/legacy/benchmark_comparison.ipynb` | Separate stage, fresh 200k training, matched evaluation, and teacher and S2UT workflow |
| `diagnose_alignment.py` | Text-token labels for character and subword plots |
| `tests/test_sentencepiece*.py`, `tests/test_notebook_commands.py` | Unit, training, external evaluation, native, recovery, and notebook tests |
| `README.md`, `docs/SENTENCEPIECE.md`, `docs/DEVIATIONS.md`, `docs/VALIDATION.md` | Design, workflow, limits, and validation |

Python paths are relative to `src/alignment_transfer_s2ut/`. [Validation records](VALIDATION.md)
