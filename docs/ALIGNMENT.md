# Alignment and Artifacts

## CTC

`align_speech_text` computes a Viterbi path from the log posteriors of a pinned `AutoModelForCTC`, accounting for blanks and repeated labels. The dynamic program advances through emission frames. It maps spans back to the original codepoint positions and stores them as `[start_sec,end_sec]`.

Set `text_transform: upper` for the English CTC vocabulary. This transform applies only to CTC vocabulary lookup. A character that expands to multiple codepoints is ineligible for a single label. With `unknown_policy: mask`, out-of-vocabulary characters and punctuation receive `[-1,-1]` spans, and the metadata records their original indices. The `error` policy stops the stage. The stage reports failure for samples with no valid labels or too few frames to align all labels, including repeats.

The default checkpoints are [Japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese/tree/cf031e020336460d15a417eba710bbc5bb43be9a) and [English](https://huggingface.co/facebook/wav2vec2-base-960h/tree/22aad52d435eb6dbaf354bdad9b0da84ce7d6156). The official model API confirmed these revisions on 2026-10-20. Alignment quality requires evaluation with real speech.

## HuBERT units

Apply the checksum-pinned set of 100 official centroids to `hidden_states[6]` from the pinned official HuBERT Base checkpoint. Convert audio to 16 kHz mono and apply the processor's normalization. Unit IDs range from 0 to 99.

Each sample stores original and reduced units, half-open run intervals `[start,end)`, original frame spans, and reduced run spans. The HuBERT convolution kernels and strides determine the receptive field and hop size. Frame timing uses a one-hop interval around each frame center.

## Projection

Row-normalize the overlap between source token spans and `F` audio time bins to obtain `M_src[S,F]`. CPU prefetch derives `F` from the actual feature length and frontend geometry, and each forward pass checks it against the measured encoder length. The source artifact stores times, and the loader derives encoder indices at runtime. Row-normalize the overlap between reduced run spans and English token spans to obtain `M_tgt[U,T]`.

The default `--source-mapping frontend` computes the acoustic support of each encoder position from the FBANK frame count and the convolution kernels, strides, and padding. With a 25 ms FBANK window, 10 ms shift, kernels 5/5, and strides 2/2, position `j` is centered on frame `4j` and covers `4j-6` through `4j+6`. Clip the indices to valid frames and use each FBANK window's start and end times. The implementation excludes convolution padding from observed speech and stops if the predicted and measured lengths differ. These intervals describe the frontend's local receptive field. Transformer self-attention can subsequently mix information across positions.

`--source-mapping uniform` divides the audio duration into `F` equal intervals. The smoke backend uses a separate timing formula for centered STFT and the stride-4 test frontend. The selected mapping is part of the checkpoint resume contract and experiment comparison.

The projected teacher is

\[
\tilde A_l^{\mathrm{T2TT}}=M_{\mathrm{tgt}}A_l^{\mathrm{T2TT}}M_{\mathrm{src}}
\]

Renormalize valid rows. Source or target tokens with no valid time overlap contribute zero. A subword with some valid characters uses the interval from its first to its last aligned character. Padded columns and target padding and EOS rows carry zero mass. The native loss and both calibration passes use the same source-padding mask.

The primary comparison uses the raw projected prior in the loss. Historical smoothed views remain available for diagnostics. Individual CTC character spans exclude blank frames. A multi-character token can span intervening blanks, and an acoustic frontend window can overlap both a character and a blank interval.

The teacher CLI stores a projection packet with time spans and MT attention. CPU prefetch and visualization build the dense U×F matrix after the encoder length is known.

## Integrity

Artifact filenames use the SHA-256 of `pair_id` to prevent path traversal. NPZ loads disable pickle and check internal hashes for metadata and array shape, dtype, and contents. The loader also verifies the external `.sha256.json` when present. The internal hash lets a completed NPZ be reused after an interruption before the external receipt was written.

Teacher metadata binds the pair ID, split, manifest, MT checkpoint, MT and S2UT architectures, tokenizer, CTC artifacts, and unit configuration and contents. Mismatches are rejected. CTC and unit artifacts also record input audio hashes.

## Alignment quality diagnostics

`diagnose_alignment` defaults to at most 500 dev pairs with seed 1. It writes `summary.json`, `pairs.jsonl`, `layers.csv`, and PNG/NPZ heatmaps for the first 20 selected pairs by default. For all-layer audits the JSON and CSV cover every decoder layer and the heatmaps show the final layer.

| Metric | Numerator / denominator |
|---|---|
| Valid source token rate | Tokens with frontend overlap and positive MT attention mass / Japanese tokens |
| Valid target token rate | Tokens with unit overlap and positive MT row mass / English tokens |
| Valid unit row rate | Rows with positive mass after projection / reduced units, excluding EOS |
| Prior entropy | Sum of entropy over valid unit rows / valid unit rows |
| Zero-cell rate | Zero cells in valid rows / all cells in valid rows |
| Unmapped source mass | MT attention on source tokens with no frontend overlap / total normalized MT mass |
| Unit-weighted unmapped source mass | Lost source mass weighted by the target unit mapping / teacher mass available to mapped units before the source mapping |

`pairs.jsonl` records CTC character-span gaps, masked codepoint counts, token offsets and time spans, and separate statistics for one-codepoint and multi-codepoint tokens. Gaps are the complement of saved character intervals. The packets do not store the original per-frame CTC path or distinguish masked unknowns from ignored characters. `layers.csv` distinguishes units with no target overlap from units that receive teacher mass and still project to zero. Corpus rates sum numerators and denominators across samples. Entropy uses natural logarithms. Valid rows should sum to one within FP32 rounding.

### Run on saved production artifacts in Colab

Use a separate CPU runtime with the branch's installed alignment package, PyTorch, Torchaudio, and Matplotlib. Use the benchmark runtime at its existing pin if restoring ZIPs or using the on-demand WAV mapping. The diagnostic itself uses CPU FBANK and projection. CUDA is required for the separate native FP16 calibration and Pilot.

1. Prepare VM-local inputs from the original experiment. Use the **full, ordered speech manifest** used to build the teacher, the matching Teacher and Units artifact directories with their checksum files, and the matching S2UT backbone configuration. The diagnostic also reads the sampled Japanese WAVs for actual feature lengths. Mount Drive and use the existing verified WAV ZIP index and local cache when those WAVs are archived. The original English WAVs, CTC model, HuBERT model, and T2TT checkpoint are unnecessary for this projection audit.
2. Reuse existing local artifacts when available. On a fresh CPU runtime, run the setup and path cells and the helper definitions. For archived WAVs, call `corpus_common(restore_only=True)` and `corpus_stage_audio(create=False)`. For the manifest, units, and teacher, use the original stage commands with `restore_only=True` as shown below. Keep the original logical paths and use the original `BACKBONE_CONFIG`. The restore-only option requires completed stage snapshots.
3. Run the diagnostic directly from a Colab code cell with the notebook's logging helper. Set the variables to the restored input paths and use a new local output directory. The command samples dev internally and preserves the manifest identity.

```python
# Restore completed stages only. Skip these calls if the corresponding local inputs exist.
run("full-manifest", "alignment_transfer_s2ut.benchmark_data", "manifest",
    "--common", COMMON, "--prepared", PREPARED, "--output", MANIFEST, "--resume",
    restore_only=True)
run("import-units", "alignment_transfer_s2ut.benchmark_data", "units",
    "--manifest", MANIFEST, "--benchmark-units", BENCHMARK_UNITS, "--prepared", PREPARED,
    "--output", Path(ROOT) / "units", "--resume", restore_only=True)
run(TEACHER_STAGE, "alignment_transfer_s2ut.build_teacher_alignment",
    "--manifest", MANIFEST, "--mt-attention", T2TT_ATTENTION_ROOT,
    "--alignment", CTC_ROOT, "--units", Path(ROOT) / "units", "--config", BACKBONE_CONFIG,
    "--output", TEACHER_ROOT, "--resume", restore_only=True)
```

If the teacher stage's identity check requires the original backbone file, use the original copy or recreate the same JSON from the registered Baseline backbone. If a stage receipt is missing, supply the existing verified local artifacts directly. A projection audit only requires the completed packets.

```python
AUDIT_OUTPUT = Path("/content/attention-audit/dev-seed1")
execute(PYTHON, "-m", "alignment_transfer_s2ut.diagnose_alignment",
        "--manifest", str(MANIFEST), "--teacher", str(TEACHER_ROOT),
        "--units", str(Path(ROOT) / "units"), "--config", str(BACKBONE_CONFIG),
        "--split", "dev", "--seed", "1", "--max-samples", "500",
        "--all-layers", "--heatmaps", "20", "--source-mapping", "frontend",
        "--prior-smoothing", "0", "--output", str(AUDIT_OUTPUT),
        name="teacher-projection-audit")
```

Open `summary.json` for the sampled IDs and corpus and layer totals. Sort `layers.csv` by lost mass and valid row rate to find outliers. High lost mass can be hidden by the final renormalization. Sparse rows and low entropy can arise from confident alignments or from truncated support. Review outliers together with ordinary pairs and compare their token spans and raw/projected heatmaps against the audio. The PNGs show an additional monotonic reference. Copy the resulting report directory to a separate Drive location if needed.

Coverage and entropy describe the projection, while text BLEU measures translation quality. To assess whether T2TT attention corresponds to translation alignment, check the matching checkpoint's saved dev BLEU and translations, then inspect bilingual source/target links on the same fixed dev sample. Review ordinary examples and mass-loss outliers across layers, including reordering and ignored or unknown characters. With annotated links, report alignment precision and recall on those examples and compare against a monotonic reference. [Saved T2TT text evaluation](EXTERNAL_TEXT_EVALUATION.md)

### Regenerate calibration for the masked implementation

The correction changes the implementation hash used by top-p reports, calibration artifacts, and native transfer checkpoints. Generate the top-p report and calibration again on this branch. The training notebook writes them under the new masked calibration paths. Keep earlier artifacts and checkpoints at their original revision. For an experiment with existing transfer checkpoints, use a new `TRANSFER_EXPERIMENT_ID` to train with the new calibration. Teacher and Units artifacts and the registered Baseline retain their existing identities.
