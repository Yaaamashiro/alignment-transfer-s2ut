# alignment-transfer-s2ut

A research implementation of layer-wise cross-attention transfer from text-to-text translation (T2TT) to speech-to-unit translation (S2UT) for Japanese-to-English direct speech-to-speech translation. Teacher attention provides a learned routing signal. Experiments use matched S2UT training settings from pinned fairseq and benchmark implementations.

$$
\boxed{\mathcal L=\mathcal L_{\mathrm{S2UT}}+\lambda\mathcal L_{\mathrm{align}}}
$$

## Experimental conditions

| Condition | Supervision | Forward pass |
|---|---|---|
| Baseline | S2UT task loss | Raw student attention |
| KL-based Attention Distillation | Per-layer KL divergence | Raw student attention |
| Annealed KL-based Attention Distillation | Per-layer KL divergence with calibrated λ decaying to zero over 80% of training using a cosine schedule | Raw student attention |
| Attention-level Knowledge Distillation | Student attention mass within explicitly selected teacher top-p regions | Raw student attention |

All three transfer conditions aggregate per-layer losses using calibrated worst-layer LogSumExp. Each T2TT decoder layer supervises the corresponding S2UT decoder layer. Attention is averaged across heads and projected into S2UT coordinates through source and target mappings. Validation and inference are teacher-free.

## Getting started with Colab

| Notebook | Purpose |
|---|---|
| [training.ipynb](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/training.ipynb) | T2TT training, speech preprocessing, S2UT Baseline training, teacher construction, calibration, and training the three transfer conditions |
| [evaluate.ipynb](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/evaluate.ipynb) | Checkpoint restoration, shared inference, dev evaluation, and comparison |

1. Open `training.ipynb`, configure the data sources and experiment IDs, and enable the stages to run. The default IDs are `experiment-001` and `t2tt-001`. To reuse a trained T2TT model, specify its saved run ID and update count.
2. Prepare audio ZIPs on a CPU runtime, then use a GPU runtime for training. `AUDIO_PACK_WORKERS` defaults to 2 and accepts values from 1 to 4. The Baseline trains for 400,000 updates. Review the top-p diagnostics, run calibration, and train the selected transfer conditions.
3. Open `evaluate.ipynb`, use the same `EXPERIMENT_ID`, and select the conditions to compare.

Set the corpus root and input manifest in the data configuration or notebook. See the [data configuration](configs/data/colab.yaml) and [input formats and examples](docs/BENCHMARK_COMPARISON.md).

Audio ZIP preparation supports resumption and bounded parallel workers. See [audio ZIP preparation](docs/AUDIO_PACKS.md) for details and measurements.

Rerun the notebook with the same revision, settings, and storage paths to resume from verified Google Drive snapshots. Training switches default to `False`. Earlier experimental notebooks are available in [legacy](notebooks/legacy).

For private GitHub access, configure the `GITHUB_TOKEN` secret in Colab before running setup. See the [GitHub access guide](docs/GITHUB_ACCESS.md).

External text-only inputs use four fields, `pair_id`, `split`, `ja_text`, and `en_text`. See the [T2TT workflow](docs/TEXT_ONLY_T2TT.md).

## Environment

The Colab environment uses Python 3.10.22, PyTorch 2.7.1, and pinned fairseq dependencies. The notebooks install the dependencies and run environment checks. Local environments must also verify the pinned revisions and compiled batch sampler. See the [fairseq requirements](requirements-fairseq.txt) and [Python requirements](requirements.txt).

## Documentation

- [Method, calibration, and microbatch definitions](docs/METHOD.md)
- [Experimental conditions](docs/EXPERIMENTS.md) and [all-layer transfer](docs/ALL_LAYER_TRANSFER.md)
- [Baseline comparison and recovery](docs/BENCHMARK_COMPARISON.md)
- [CLI reference](docs/USAGE.md) and [Colab runtime and storage](docs/COLAB_RUNTIME.md)
- [SentencePiece](docs/SENTENCEPIECE.md) and [validation records](docs/VALIDATION.md)
- [Implementation definitions and limitations](docs/DEVIATIONS.md)

## License

[MIT](LICENSE)
