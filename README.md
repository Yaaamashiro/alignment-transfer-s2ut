# alignment-transfer-s2ut

T2TTのlearned cross-attention routing/alignment patternを、対応するS2UT Decoder層へ転移する研究実装。日本語→英語Direct S2STを対象に、固定fairseq/benchmarkのS2UT学習条件を揃えて比較する。Attentionをground-truth alignmentとは扱わない。

## 主要4条件

| Condition | 教師信号 | Forward |
|---|---|---|
| Baseline | なし | raw Attention |
| KL-based Attention Distillation | 層別KL | raw Attention |
| Annealed KL-based Attention Distillation | 同じKL、校正λを0.8Tまでcosine減衰 | raw Attention |
| Attention-level Knowledge Distillation | 明示top-p領域内のstudent Attention mass | raw Attention |

3条件とも校正したworst-layer LogSumExpで層lossを集約する。T2TT層l→S2UT層l、全head平均、既存の両側modality projectionを維持する。validation/inferenceはteacher-free。

## Colabで始める

[主研究Notebook](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/benchmark_comparison.ipynb)をGPUランタイムで開く。

1. 環境を構築し、別学習済みBaselineを復元・登録する。
2. T2TT checkpoint・dev翻訳を確認し、全層Attentionとprojected teacherを準備/再利用する。
3. top-p診断を見て`MASK_TOP_P`を明示選択する。
4. λ/τ calibrationを実行する。
5. 必要な学習スイッチを有効にし、KL-based Attention Distillation → Annealed KL-based Attention Distillation → Attention-level Knowledge Distillationの順に実行する。
6. 共通評価・比較表・loss/layer diagnosticsを確認する。

本学習スイッチは既定False。λ/τを手動設定せず、3条件で同じcalibration artifactを共有する。作業はVM、永続化は既存benchmark方式の検証付きDrive ZIP。同じrevision・設定・保存先で再実行すると保存済み更新から再開する。

## その他のNotebook

| Notebook | 用途 |
|---|---|
| [jparacrawl_t2tt](notebooks/colab/jparacrawl_t2tt.ipynb) | 音声なし外部日英manifestからT2TT学習・dev BLEU・Attention抽出 |
| [pilot](notebooks/colab/pilot.ipynb) | historical standaloneの少量接続試験 |
| [pipeline](notebooks/colab/pipeline.ipynb) | historical standaloneの前処理/評価例 |

外部text-only入力は`pair_id/split/ja_text/en_text`の4列。後から音声を追加しても新T2TTのtext identityは変わらない。[手順](docs/TEXT_ONLY_T2TT.md)。

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
