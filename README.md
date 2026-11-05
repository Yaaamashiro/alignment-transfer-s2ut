# alignment-transfer-s2ut

T2TTのlearned cross-attention routing/alignment patternを、対応するS2UT Decoder層へ転移する研究実装。日本語→英語Direct S2STを対象に、固定fairseq/benchmarkのS2UT学習条件を揃えて比較する。Attentionをground-truth alignmentとは扱わない。

\[
\boxed{\mathcal L=\mathcal L_{\mathrm{S2UT}}+\lambda\mathcal L_{\mathrm{align}}}
\]

## 主要4条件

| Condition | 教師信号 | Forward |
|---|---|---|
| Baseline | なし | raw Attention |
| KL-based Attention Distillation | 層別KL | raw Attention |
| Annealed KL-based Attention Distillation | 同じKL、校正λを0.8Tまでcosine減衰 | raw Attention |
| Attention-level Knowledge Distillation | 明示top-p領域内のstudent Attention mass | raw Attention |

3条件とも校正したworst-layer LogSumExpで層lossを集約する。T2TT層l→S2UT層l、全head平均、既存の両側modality projectionを維持する。validation/inferenceはteacher-free。

## Colabで始める

| Notebook | 用途 |
|---|---|
| [training.ipynb](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/training.ipynb) | T2TT・音声前処理・Baseline・Teacher・校正・転移3条件の学習 |
| [evaluate.ipynb](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/evaluate.ipynb) | 学習済みモデルの復元・共通推論・dev評価・比較 |

1. trainingを開き、実験名と実行する学習スイッチを設定する。学習済みT2TTは保存先から復元できる。
2. 音声ZIPはCPU、学習はGPUで実行する。Baselineは400k updates。top-p診断を確認して校正し、必要な転移条件を学習する。
3. evaluateを開き、同じEXPERIMENT_IDと比較する条件を選ぶ。

同じrevision・設定・保存先で再実行すると、検証済みDrive ZIPから再開する。学習スイッチは既定False。元の実験用Notebookは [legacy](notebooks/legacy) に保存している。

外部text-only入力は pair_id/split/ja_text/en_text の4列。[T2TT手順](docs/TEXT_ONLY_T2TT.md)。

## Environment

Colabの固定環境はPython 3.10.22・Torch 2.7.1と[固定fairseq依存](requirements-fairseq.txt)を使う。[Python依存](requirements.txt)とpreflightを含むNotebookで構築する。ローカル環境も同じrevisionとcompiled batch samplerを検証する。

## Documentation

- [研究の数式・校正・microbatch定義](docs/METHOD.md)
- [実験条件](docs/EXPERIMENTS.md)・[全層転移](docs/ALL_LAYER_TRANSFER.md)
- [Baselineとの比較・保存と再開](docs/BENCHMARK_COMPARISON.md)
- [CLI](docs/USAGE.md)・[環境と保存](docs/COLAB_RUNTIME.md)
- [SentencePiece](docs/SENTENCEPIECE.md)・[検証記録](docs/VALIDATION.md)
- [実装上の定義と制約](docs/DEVIATIONS.md)

## License

[MIT](LICENSE)
