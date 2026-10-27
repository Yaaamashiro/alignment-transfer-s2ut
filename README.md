# Alignment Transfer for S2UT

Text-to-Text Translation（T2TT）のCross-Attentionを、日本語音声から英語音声unitを生成するSpeech-to-Unit Translation（S2UT）の学習へ利用する研究実装。

[Colabで実行](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/benchmark_comparison.ipynb) · [比較手順](docs/BENCHMARK_COMPARISON.md) · [MIT License](LICENSE)

## 概要

T2TT Attentionを、日英のCTC forced alignmentとHuBERT unitの時刻情報でS2UTの時間座標へ写像する。得られた分布を追加損失またはCross-Attention priorとして使う。推論は全条件でteacher-free。

```text
日本語text → T2TT → 英語text
               │ Attention + CTC alignment
               ▼
日本語音声 → S2UT → 英語unit → unit vocoder → 英語音声
```

本比較では、[ja-en-direct-s2st-benchmark](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark)で別途学習したモデルをBaselineとして登録する。転移モデルは同じ保存設定・preparedデータ・辞書を使い、同じseedから新規学習する。公式fairseqのモデル・Trainer・FP16・補助loss・optimizer・scheduler・batch・更新数・checkpoint選択を引き継ぐ。

| 条件 | 学習時のalignment |
|---|---|
| S2UT Baseline | benchmarkの学習済みモデル |
| Annealed T2TT Guidance | T2TT由来priorを徐々に減衰 |
| Annealed Monotonic Guidance（任意のcontrol） | 音声時間から作る単調priorを徐々に減衰 |

Alignment Distillation・固定係数のGuidanceにも対応する。モデル構成・数式・実装範囲は[ARCHITECTURE](docs/ARCHITECTURE.md)、[METHOD](docs/METHOD.md)、[DEVIATIONS](docs/DEVIATIONS.md)を参照。

## Colabで始める

1. [benchmark比較ノートブック](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/benchmark_comparison.ipynb)をGPUランタイムで開く。
2. 既存実験の`BENCHMARK`・`RUN_NAME`と、今回の`ROOT`（VMローカル）・`PERSISTENT`（Drive）を設定する。
3. `RUN_TRANSFER=True`で主比較を実行する。3条件を比較する場合は`RUN_MONOTONIC=True`も指定する。
4. 上から順に環境検査・データ準備・学習・dev評価を実行する。

両学習スイッチがFalseの既定では、**T2TT学習・teacher準備とplan確認まで**を行う。S2UT本学習は開始しない。ASR-BLEUを測る場合は、最初から`GENERATE_WAVEFORMS=True`と`ASR_BLEU=True`を指定する。

音声ZIPが未準備なら、CPUで`PREPARE_AUDIO_PACKS=True`にして入力準備まで実行し、GPUへ切り替えてFalseへ戻す。作業・checkpoint・pair artifactはVMへ保存し、固定revisionのbenchmark runtimeで検証付きZIPをDriveへ保存する。切断後は同じコードrevision・設定・保存先で上から再実行する。[保存・再開の詳細](docs/COLAB_RUNTIME.md)

### ノートブック一覧

| ノートブック | 用途 | 学習・保存 |
|---|---|---|
| [benchmark_comparison](notebooks/colab/benchmark_comparison.ipynb) | 学習済みBaselineとの本比較 | native fairseq・ローカル作業＋Drive ZIP |
| [pilot](notebooks/colab/pilot.ipynb) | 少量実データの診断と3条件の機能検証 | 旧standalone trainer・Drive直接保存 |
| [pipeline](notebooks/colab/pipeline.ipynb) | 前処理から5条件の評価までの実行例 | 旧standalone trainer・Drive直接保存 |

## ローカル環境

本実験は**Linux / Python 3.10 / CUDA**を対象とする。PyTorchとtorchaudioは2.7.1、fairseqは[固定commit](https://github.com/facebookresearch/fairseq/tree/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99)を使用する。古いOmegaConfの依存metadataに対応するためpip 23.3.2で導入する。

```bash
git clone https://github.com/Yaaamashiro/alignment-transfer-s2ut.git
cd alignment-transfer-s2ut
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==23.3.2' 'setuptools<81' wheel
python -m pip install torch==2.7.1 torchaudio==2.7.1 numpy==1.26.4 Cython==3.2.9
python -m pip install --no-build-isolation -r requirements-fairseq.txt
python -m pip install -r requirements.txt
python -m alignment_transfer_s2ut.environment --native --cuda
```

Colabの保存・復元機能をローカルから使用する場合は、[固定benchmark runtimeの導入手順](docs/COLAB_RUNTIME.md)も必要。

### 動作確認

```bash
python -m pip check
python -m pytest -q
python -m alignment_transfer_s2ut.smoke --output outputs/smoke
```

smokeは人工データで4条件の学習からteacher削除後の推論まで検査する。外部モデルを取得しない。CUDAとモデルassetが必要なテスト、検証済み範囲は[VALIDATION](docs/VALIDATION.md)に記載する。実コーパスの本学習・翻訳品質の比較は未実施。

### T2TTの外部評価

比較ノートブックの **6.1.1 FLEURS/FLORES external text BLEU** で、学習済み `checkpoint_best.pt` を以前のCascade/S2UTと同じ保存済み8C 300件で評価できる。VMにbestがなければ完了済みT2TT ZIPから復元し、保存済みdev結果もあれば復元する。`RUN_EXTERNAL_TEXT=True` にすると、devと同じnative BLEU、benchmark正規化BLEU、未知文字率、生成例を表示する。再学習・新しい300件の選択・音声取得は行わない。[固定セットの確認・保存・再開](docs/EXTERNAL_TEXT_EVALUATION.md)

## CLIとデータ

`python -m alignment_transfer_s2ut.benchmark --help`からBaseline登録・学習・共通推論・比較を実行できる。[実行例](docs/BENCHMARK_COMPARISON.md)

standalone前処理の入力は1行1pairのJSONL。`pair_id`を全stageで保持し、T2TT語彙はtrainのみから構築する。

```json
{"pair_id":"sample-0001","split":"train","ja_text":"こんにちは。","en_text":"Hello.","ja_audio":"audio/ja/0001.wav","en_audio":"audio/en/0001.wav"}
```

## ドキュメント

| 文書 | 内容 |
|---|---|
| [BENCHMARK_COMPARISON](docs/BENCHMARK_COMPARISON.md) | Baseline登録・native学習・共通評価 |
| [COLAB_RUNTIME](docs/COLAB_RUNTIME.md) | 入出力・ZIP保存・切断後の再開 |
| [EXTERNAL_TEXT_EVALUATION](docs/EXTERNAL_TEXT_EVALUATION.md) | 保存済み8C 300件によるT2TT外部BLEU・未知文字診断 |
| [LOGGING](docs/LOGGING.md) | 全段階の進捗・生存通知・実行ログ |
| [USAGE](docs/USAGE.md) | standalone前処理・学習・推論のCLI |
| [METHOD](docs/METHOD.md) / [ARCHITECTURE](docs/ARCHITECTURE.md) | 手法とモデル構成 |
| [ALIGNMENT](docs/ALIGNMENT.md) | CTC・unit timing・投影・品質診断 |
| [EXPERIMENTS](docs/EXPERIMENTS.md) / [STREAMING](docs/STREAMING.md) | 独立trainerのレシピ・実験管理・入力先読み |
| [PILOT](docs/PILOT.md) | 少量実データの診断手順 |
| [DEVIATIONS](docs/DEVIATIONS.md) / [VALIDATION](docs/VALIDATION.md) | 実装範囲・差異・検証結果 |

## ライセンス

[MIT License](LICENSE)。外部モデル・データは各配布元のライセンスに従う。
