# External text-only T2TT

Use externally generated Japanese–English JSONL to train T2TT, compute dev BLEU, and extract attention before preparing audio. This workflow covers text model training and evaluation. Corpus filtering, TTS, Whisper QC, S2UT training, and xCOMET and COMETKiwi are separate workflows.

## Inputs

```json
{"pair_id":"p0001","split":"train","ja_text":"日本語文","en_text":"English sentence."}
{"pair_id":"p0002","split":"dev","ja_text":"別の文","en_text":"Another sentence."}
```

Each row needs nonempty strings for `pair_id`, `split`, `ja_text`, and `en_text`. Train and dev are required, and test is optional. The importer preserves split labels and text verbatim. It rejects duplicate IDs and translation pairs repeated across splits. Build the vocabulary on train and select the best checkpoint on dev.

## Manifest API

```python
from alignment_transfer_s2ut.data.manifest import (
    load_manifest, validate, text_manifest_hash, speech_manifest_hash,
)

text_rows = load_manifest("manifest.jsonl", require_audio=False)
speech_rows = load_manifest("manifest-with-audio.jsonl")
# Validate speech metadata without opening audio payloads.
speech_metadata = load_manifest("manifest-with-audio.jsonl", check_audio=False)
```

`TEXT_REQUIRED` lists the four text fields and `AUDIO_REQUIRED` lists `ja_audio` and `en_audio`. `REQUIRED` combines both. `require_audio=True` is the default, and `check_audio=False` disables the payload check while still requiring audio fields.

With `require_audio=False`, optional audio paths are left untouched and audio existence, hashes, and duplicate checks are skipped. T2TT, dev evaluation, and attention extraction use text mode. CTC, HuBERT and unit extraction, teacher construction, S2UT, speech evaluation, and audio visualization use speech mode and require audio fields. CTC and unit extraction validate those fields before downloading models.

## Identity and adding audio

`text_manifest_hash(rows)` hashes the following four fields in row order.

```text
pair_id, split, ja_text, en_text
```

The hash binds IDs, original text, splits, row order, and row membership. Audio paths, hashes, and other fields are outside the text identity.

New T2TT checkpoints and experiments bind `manifest_identity="text-v1"` and store text identities as `train_hash` and `dev_hash` for character and SentencePiece models. Dev evaluation and attention metadata also record the identity type. Attention metadata includes the full `text_manifest_hash`.

When adding audio to the same ordered IDs, splits, and texts, verify T2TT attention with text identity. CTC, units, and the resulting teacher use the existing identity that includes audio.

`speech_manifest_hash` aliases the original `manifest_hash` calculation, including audio paths and available SHA values. Existing S2UT checkpoints, CTC, units, and teachers retain speech identity. Pilot T2TT dev matching uses the added diagnostic `dev_text_hash` while S2UT retains its scientific contract and speech hash.

## Legacy checkpoints and artifacts

T2TT checkpoints without an identity type use the legacy identity that includes audio metadata. Evaluation and extraction verify the saved hash from the original metadata without reading audio payloads. Relative paths become lexical absolute paths. If the old experiment stored symlink-resolved paths, supply the same canonical metadata.

A four-field manifest fails the legacy audio-inclusive hash check. Keep the original manifest and revision to resume legacy training, or start a separate stage for external text. Old attention artifacts still use speech identity and can resume with their original inputs.

## CLI

Pass a JSONL manifest containing the text fields directly.

```bash
python -m alignment_transfer_s2ut.train_t2tt \
  --manifest /data/text.jsonl --config configs/t2tt/fisher.yaml \
  --output outputs/t2tt-external-text --device cuda --max-updates 200000

python -m alignment_transfer_s2ut.evaluate_t2tt \
  --manifest /data/text.jsonl --checkpoint outputs/t2tt-external-text/checkpoint_best.pt \
  --split dev --beam-size 5 --length-penalty 1 --device cuda \
  --output outputs/t2tt-external-dev/dev-text.json

python -m alignment_transfer_s2ut.extract_t2tt_attention \
  --manifest /data/text.jsonl --checkpoint outputs/t2tt-external-text/checkpoint_best.pt \
  --all-layers --device cuda --output outputs/t2tt-external-attention
```

The CLI defaults to character tokenization. For SentencePiece, add `--tokenizer sentencepiece --source-vocab-size 8000 --target-vocab-size 8000 --sentencepiece-model-type unigram`. The workflow supports SacreBLEU, prediction JSONL, EOS rate, generation metadata, and teacher-forced all-layer extraction. The fixed 300-item external evaluation uses the checkpoint-owned tokenizer and text inputs. [SentencePiece identity, UNK, and character spans](SENTENCEPIECE.md)

## Colab and recovery

Prepare, train, and restore T2TT in [training.ipynb](../notebooks/colab/training.ipynb).

1. Select a GPU runtime and set `PREPARE_AUDIO_PACKS=False` and `PREPARE_S2UT=False`.
2. Set `T2TT_SOURCE_MANIFEST_OVERRIDE` to the external text JSONL. Choose a new `T2TT_RUN_ID` and set `RUN_T2TT=True`.
3. In the T2TT section, create the canonical text manifest and start training. Set `EVALUATE_T2TT=True` to view dev BLEU, translation examples, and learning curves.
4. For speech experiments, continue through preprocessing, Baseline training, teacher construction, and transfer training.

T2TT can run independently. Local training uses `TEXT_ROOT/t2tt-<tokenizer>` and Drive uses `TEXT_PERSISTENT/stages/t2tt-<tokenizer>`. Both roots derive from `T2TT_RUN_ID`. After a disconnect, rerun with the original revision and settings to restore the stage's last checkpoint, optimizer, RNG, and cursor.

To use a trained T2TT model, set `RUN_T2TT=False` and enable `EVALUATE_T2TT` or the teacher switches. `T2TT_RUN_ID` defaults to `t2tt-001`. Specify the actual saved model ID and original text manifest. [Legacy text-only notebook](../notebooks/legacy/jparacrawl_t2tt.ipynb)

## Changed files

| File | Change |
|---|---|
| `data/manifest.py`, `data/__init__.py` | Validation by input mode, text and speech identity, and explicit legacy identity handling |
| `training.py`, `evaluation/text.py` | Audio-free input, the text-v1 contract, and compatibility paths for evaluation and extraction |
| `preprocessing.py` | Early speech validation and the identity boundary between text attention and speech teacher artifacts |
| `pilot.py`, `t2tt_extension.py` | Pilot dev text matching and old and new checkpoint checks |
| `text_manifest.py` | Read-only source staging for canonical text |
| `text_t2tt_runtime.py`, `spm_runtime.py` | Shared fresh start, session, and ZIP adapters for character and SentencePiece |
| `notebooks/legacy/jparacrawl_t2tt.ipynb` | Inputs, training, recovery, BLEU, examples, and optional attention extraction |
| `tests/test_text_manifest.py`, `test_text_t2tt.py`, `test_text_runtime.py`, `test_text_notebook.py` | API, audio-free inputs, adding audio, teacher connection, recovery, and notebook tests |
| `tests/test_t2tt_external.py` | BLEU comparison tables and examples checked independently of diagnostic table order |
| `README.md`, `docs/TEXT_ONLY_T2TT.md`, `docs/DEVIATIONS.md`, `docs/VALIDATION.md` | Workflow, identity, compatibility, and validation |

Python paths in the table are relative to `src/alignment_transfer_s2ut/`. CPU tiny tests check workflow connectivity. The external corpus, 200k GPU training, and BLEU require separate runs.
