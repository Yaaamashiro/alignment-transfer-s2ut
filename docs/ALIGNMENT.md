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

Renormalize valid rows, excluding invalid source and target spans and unknown characters. Padding carries zero mass, and epsilon stabilizes only the arguments to logarithms in the loss. Exclude invalid unit rows and EOS from the alignment loss.

The primary comparison uses the raw projected prior in the loss. Historical smoothed views remain available for diagnostics. Projection preserves the CTC blanks and original token spans.

The teacher CLI stores a projection packet with time spans and MT attention. CPU prefetch and visualization build the dense U×F matrix after the encoder length is known.

## Integrity

Artifact filenames use the SHA-256 of `pair_id` to prevent path traversal. NPZ loads disable pickle and check internal hashes for metadata and array shape, dtype, and contents. The loader also verifies the external `.sha256.json` when present. The internal hash lets a completed NPZ be reused after an interruption before the external receipt was written.

Teacher metadata binds the pair ID, split, manifest, MT checkpoint, MT and S2UT architectures, tokenizer, CTC artifacts, and unit configuration and contents. Mismatches are rejected. CTC and unit artifacts also record input audio hashes.

## Alignment quality diagnostics

`diagnose_alignment` draws up to 500 samples from the selected split using a fixed seed. It writes `pairs.jsonl`, `summary.json`, and heatmaps and NPZ files for up to the first 20 samples, covering raw MT attention, projected attention, smoothed attention, and the monotonic prior.

| Metric | Numerator / denominator |
|---|---|
| Valid source token rate | Characters with frontend overlap and positive MT attention mass / original Japanese characters |
| Valid target token rate | Characters with unit overlap and valid MT rows / original English characters |
| Valid unit row rate | Rows with positive mass after projection / reduced units, excluding EOS |
| Prior entropy | Sum of entropy over valid unit rows / valid unit rows |
| Zero-cell rate | Zero cells in valid rows / all cells in valid rows |

The output also records source and target character counts with valid CTC spans. Corpus rates sum the numerators and denominators across samples. Entropy uses natural logarithms. These metrics characterize coverage and uncertainty. Check boundary timing against the audio and heatmaps to evaluate forced alignment, since ASR CER measures recognition errors.
