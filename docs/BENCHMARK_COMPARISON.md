# Training and comparison

Use [training.ipynb](../notebooks/colab/training.ipynb) for preparation and training and [evaluate.ipynb](../notebooks/colab/evaluate.ipynb) for inference and comparison.

## Inputs and experiment IDs

The defaults are `EXPERIMENT_ID=experiment-001` and `T2TT_RUN_ID=t2tt-001`. Each ID names a storage location. Select the corpus and manifest separately.

Select a data YAML with `DATA_CONFIG`. Set `corpus_root` and `accepted_manifest` there, or set `CORPUS_ROOT_OVERRIDE` and `SOURCE_MANIFEST_OVERRIDE` in the notebook. Set the T2TT source text with `teacher_source_manifest` or `T2TT_SOURCE_MANIFEST_OVERRIDE`. [Default configuration](../configs/data/colab.yaml)

The speech manifest is JSONL with `pair_id`, `split`, `ja_text`, `en_text`, `ja_audio`, `en_audio`, `ja_sha256`, `en_sha256`, `ja_duration`, and `en_duration`. Durations are in seconds. Hashes are lowercase hexadecimal SHA-256 of the complete WAV. Use 16 kHz PCM16 WAVs and paths relative to the corpus root or absolute paths under it. The reader also accepts `ja_wav_16k` and `en_wav_16k`. Preserve the `train`, `dev`, and `test` labels. Preparation and training use `train` and `dev`. `corpus` and `source_id` are optional. If present, `qc_status` must be `accepted`.

`release_summary` is optional. For a QC release, the importer checks `complete` and accepted counts by split. Set `expected_pairs` to verify the combined train and dev count. Otherwise the importer uses the manifest count.

Resume saved runs with the original experiment ID and input and output settings. [Example corpus configuration](../configs/data/kokoro_jparacrawl.example.yaml)

## training.ipynb

1. Set `EXPERIMENT_ID` and run a CPU runtime with `PREPARE_AUDIO_PACKS=True` to prepare audio ZIPs. `AUDIO_PACK_WORKERS` accepts integers from 1 to 4 and defaults to 2 for CPU preparation. Completed ZIPs are reused.
2. Switch to GPU, set `PREPARE_AUDIO_PACKS=False`, and rerun from the top with the same settings. The preflight uses synthetic audio to check a small FP16 update, save, and resume sequence. Verify production execution on the real corpus separately.
3. Restore a trained SentencePiece T2TT model from its configured location. For new training, use a new `T2TT_RUN_ID` and set `RUN_T2TT=True`. Use `EVALUATE_T2TT=True` to view dev BLEU, translation examples, and learning curves.
4. Prepare units, dictionaries, and fairseq data, then check the registration preflight. Set `RUN_BASELINE=True` to train the Baseline for 400,000 updates.
5. Set `RUN_TEACHER_ARTIFACTS=True` to verify all IDs, splits, and texts against the original T2TT manifest and build all-layer attention, CTC alignments, and the projected teacher.
6. Inspect the top-p diagnostics, set `MASK_TOP_P` explicitly, and calibrate λ and τ.
7. Use `RUN_DISTILLATION`, `RUN_ANNEALED_DISTILLATION`, and `RUN_MASK_DISTILLATION` to select transfer conditions. Run the Pilot, review the diagnostics, then switch `TRANSFER_RUN_MODE` to `full` to continue. Set `RUN_ALIGNMENT_DIAGNOSTICS=True` for completed-model layer diagnostics and visualizations. [Native transfer pilot](PILOT.md)

To reuse trained T2TT, specify its `T2TT_RUN_ID` and original text manifest. The Baseline and transfer conditions share the 400,000-update budget, initialization policy, and native fairseq recipe. Each model uses its own verified snapshots.

## evaluate.ipynb

1. Use the same `EXPERIMENT_ID`, `TRANSFER_EXPERIMENT_ID`, and corpus settings as training.
2. Restore the preprocessing artifacts and Baseline, then restore selected transfer models from completed, verified snapshots.
3. Generate dev predictions with the common native fairseq generator and compare UER and optional ASR-BLEU.

Set `EVALUATE_S2UT=True` to evaluate S2UT. The three `COMPARE_*` transfer switches default to `False`, which evaluates the Baseline alone. Enable only completed transfer conditions. Validation and inference run teacher-free.

Set `EVALUATE_T2TT_EXTERNAL=True` to compute BLEU on the fixed 300-item T2TT external benchmark. Set `EVALUATE_S2UT=False` for a T2TT-only evaluation.

## Storage and recovery

On GPU startup, the workflow indexes the audio ZIPs and fetches and verifies each pack on demand. The fairseq loader prefetches training data. Extracted data accumulates on the VM, which needs space for accessed audio, checkpoints, and teacher artifacts. The four conditions use a total budget of 1,600,000 updates. [Colab I/O and recovery](COLAB_RUNTIME.md)

Work stays on the VM and verified ZIPs from the pinned benchmark runtime persist in Drive. Resume S2UT with the original experiment name and settings. Builds verified as compatible with pre-streaming revision `74c5efa` check saved S2UT contracts against their existing identity. Resume T2TT and other legacy runs with their original revision. [Historical notebooks](../notebooks/legacy)

[Method](METHOD.md) and [validation records](VALIDATION.md)
