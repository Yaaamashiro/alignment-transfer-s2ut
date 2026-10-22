# Alignment Transfer for S2UT

[MIT License](LICENSE) · [Google Colab](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/benchmark_comparison.ipynb) · [Baseline比較ガイド](docs/BENCHMARK_COMPARISON.md)

学習済みbenchmark S2UTをBaselineとして使う場合は[専用Colab](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/benchmark_comparison.ipynb)を使用する。既存checkpointを登録し、保存された学習設定・preparedデータ・辞書を使って転移モデルを新規学習する。両モデルの推論・評価も共通化する。[比較手順](docs/BENCHMARK_COMPARISON.md)を参照。

少量データでの機能診断には[Strong S2UT Pilot](docs/PILOT.md)を使用できる。独立trainerを使う旧経路で、[順序付きCPU先読み・キャッシュ](docs/STREAMING.md)に対応する。benchmark比較の入力並列読み込みはnative fairseq DataLoaderを使用する。

比較用Colabは、benchmarkの固定revisionから入力ZIP staging・cache/worker制御・定期ZIP backup・復元処理を再利用する。作業はColabローカル、永続保存はDriveへ分離する。[通常運用と再開手順](docs/COLAB_RUNTIME.md)を参照。

Text-to-Text Translation（T2TT）のCross-Attentionを、Speech-to-Unit Translation（S2UT）の学習に利用するアラインメント転移学習の研究実装。

T2TTとS2UTのTransformer構成を統一し、入力・出力モダリティの違いとアラインメント転移の効果を比較する。日本語音声から英語の離散音声unitを生成し、unit vocoderで英語音声へ変換する。

## 概要

```text
T2TT: Japanese text   → Text embedding  → Transformer → English text
S2UT: Japanese speech → Speech frontend → Transformer → English units → Vocoder
                                         共通構成
```

T2TTのAttentionを、日英それぞれのCTC forced alignmentとHuBERT unitの時刻情報を使ってS2UTのAttention座標へ写像する。

```text
A_T = M_tgt @ A_MT @ M_src
```

モデルの重み転移は行わず、得られた分布を学習時の教師信号またはAttention priorとして用いる。Cross-Attentionはlearned soft alignment / routing patternとして扱う。

| 条件 | 学習方法 |
|---|---|
| Baseline | 通常のS2UT学習 |
| Alignment Distillation | 転移分布とS2UT Attention間のKL損失を追加 |
| Alignment Guidance | 転移分布をCross-Attentionのpriorに使用 |
| Annealed Alignment Guidance | priorの係数を学習とともに減衰 |

推論は全条件でteacher-free。日本語音声、S2UT checkpoint、unit vocoderのみを使用する。

benchmark比較は公式fairseq Trainer・FP16・optimizer・scheduler・token-budget batch・補助lossを使用し、保存Baselineの設定を引き継ぐ。T2TT由来priorと時間単調prior、prior平滑化、frontendの受容野に基づく時刻投影、alignment品質診断に対応する。旧standalone経路には`controlled`と`strong`のレシピを残す。

## モデル構成

fairseqの`s2ut_transformer_fisher`を基準に、T2TTとS2UTで同一のEncoder/Decoder層とCross-Attention実装を使用する。

| 項目 | T2TT / S2UT共通 |
|---|---|
| Encoder / Decoder layers | 12 / 6 |
| Model dimension | 256 |
| FFN dimension | 2048 |
| Encoder / Decoder attention heads | 4 / 8 |
| Activation / Layer normalization | ReLU / pre-norm |
| Positional encoding | Sinusoidal |

T2TTはUnicode codepoint単位の文字tokenを使用する。S2UTの出力はHuBERT Baseの第6層から抽出したKM100 unitで、連続する重複unitを縮約する。構成の詳細は[ARCHITECTURE](docs/ARCHITECTURE.md)を参照。

## インストール

実験環境はPython 3.10、PyTorch / torchaudio 2.7.1、固定commitのfairseq。Linux / CUDA環境を想定する。

```bash
git clone https://github.com/Yaaamashiro/alignment-transfer-s2ut.git
cd alignment-transfer-s2ut

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==23.3.2' 'setuptools<81' wheel
python -m pip install torch==2.7.1 torchaudio==2.7.1 numpy==1.26.4 Cython==3.2.9
python -m pip install --no-build-isolation -r requirements-fairseq.txt
python -m pip install -r requirements.txt
```

Google Colabでは[比較ノートブック](notebooks/colab/benchmark_comparison.ipynb)がPython 3.10環境を構築し、各CLIを実行する。

## クイックスタート

```bash
python -m pytest -q
python -m alignment_transfer_s2ut.smoke --output outputs/smoke
```

smokeは小規模モデルと人工データによるpipeline検証。外部の学習済みモデルを取得せず、4条件の学習からteacher削除後の推論まで実行する。実音声による翻訳品質評価は含まない。

既存Baselineとの実データ比較は[比較ガイド](docs/BENCHMARK_COMPARISON.md)、独立trainerの前処理・診断は[利用ガイド](docs/USAGE.md)に記載する。

```text
Manifest → T2TT学習 → Attention抽出 → CTC alignment / HuBERT unit抽出
         → Alignment投影 → S2UT学習 → 音声生成 → 評価
```

`train_s2ut`は既定でnative fairseq経路を使用し、`--baseline-record`を必須とする。更新数・batch等は登録Baselineに固定する。旧CLIは`--trainer legacy`を明示し、既定2更新から`--max-updates`で変更する。

## データ形式

入力は1行1pairのJSONL。`split`は`train`、`dev`、`test`を使用する。

```json
{"pair_id":"sample-0001","split":"train","ja_text":"こんにちは。","en_text":"Hello.","ja_audio":"audio/ja/0001.wav","en_audio":"audio/en/0001.wav"}
```

T2TTとS2UTは同じtrain pairを使用し、全stageで`pair_id`を保持する。T2TT語彙はtrainのみから構築する。benchmark比較のS2UTは既存の辞書・文字ラベル・SentencePieceをそのまま使い、辞書構築範囲もdata lockに記録する。

## 推論

```bash
python -m alignment_transfer_s2ut.benchmark infer \
  --record outputs/comparison/baseline.json --input outputs/comparison/dev-source.jsonl \
  --output outputs/comparison/baseline-predictions.jsonl --units-only --device cuda
```

転移モデルの登録JSONにも同じコマンドを使用する。音声生成には`--units-only`を`--vocoder-config configs/s2ut/vocoder.yaml`へ変更する。共通ASR-BLEU評価は[比較ガイド](docs/BENCHMARK_COMPARISON.md)、旧checkpointの推論・Attention可視化は[利用ガイド](docs/USAGE.md#推論と評価)を参照。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [BENCHMARK_COMPARISON](docs/BENCHMARK_COMPARISON.md) | 既存Baselineの登録、native fairseq学習、共通推論・評価 |
| [COLAB_RUNTIME](docs/COLAB_RUNTIME.md) | benchmarkの入出力・保存・再開処理の再利用 |
| [USAGE](docs/USAGE.md) | 前処理・学習・推論・評価の実行手順 |
| [METHOD](docs/METHOD.md) | アラインメント転移と4つの比較条件 |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | T2TT / S2UTの共有構成とモダリティ固有部分 |
| [ALIGNMENT](docs/ALIGNMENT.md) | CTC、unit timing、行列投影、artifact管理 |
| [EXPERIMENTS](docs/EXPERIMENTS.md) | 共通レシピ、checkpoint選択、再開、ログ |
| [PILOT](docs/PILOT.md) | 実データ診断、strongの3条件pilot、GPU/時間の記録 |
| [STREAMING](docs/STREAMING.md) | 学習と入力取得の重ね合わせ、cache、順序・再開 |
| [DEVIATIONS](docs/DEVIATIONS.md) | 設計上の差異と実装範囲 |
| [VALIDATION](docs/VALIDATION.md) | 検証環境とテスト結果 |

本学習・実コーパスでの性能比較は未実施。native比較経路と旧独立trainerの範囲、残る検証上の限界は[DEVIATIONS](docs/DEVIATIONS.md)に記載する。

## ライセンス

[MIT License](LICENSE)
