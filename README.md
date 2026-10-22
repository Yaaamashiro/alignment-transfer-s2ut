# Alignment Transfer for S2UT

[MIT License](LICENSE) · [Google Colab](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/pipeline.ipynb) · [利用ガイド](docs/USAGE.md)

実データでの短期比較は[Strong S2UT Pilot（Colab）](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/pilot.ipynb)を使用する。[実行手順](docs/PILOT.md)に、CUDA接続試験・alignment診断・3条件pilot・dev評価をまとめる。

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

学習レシピは、転移の効果を比較する`controlled`と、文字補助decoder・decoder CTC・CMVN・SpecAugmentを加えた`strong`を選択できる。T2TT由来priorと時間単調priorの対照実験、prior平滑化、frontendの受容野に基づく時刻投影、alignment品質診断に対応する。[実験手順](docs/USAGE.md#強いbaselineと単調priorの対照実験)を参照。

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

Google Colabでは[ノートブック](notebooks/colab/pipeline.ipynb)がPython 3.10環境を構築し、各CLIを実行する。

## クイックスタート

```bash
python -m pytest -q
python -m alignment_transfer_s2ut.smoke --output outputs/smoke
```

smokeは小規模モデルと人工データによるpipeline検証。外部の学習済みモデルを取得せず、4条件の学習からteacher削除後の推論まで実行する。実音声による翻訳品質評価は含まない。

実データの実行手順は[利用ガイド](docs/USAGE.md)に記載する。

```text
Manifest → T2TT学習 → Attention抽出 → CTC alignment / HuBERT unit抽出
         → Alignment投影 → S2UT学習 → 音声生成 → 評価
```

学習CLIの既定上限は2更新。本学習では`--max-updates`を指定する。`--resume`または`--resume-if-exists`でcheckpointから再開できる。

## データ形式

入力は1行1pairのJSONL。`split`は`train`、`dev`、`test`を使用する。

```json
{"pair_id":"sample-0001","split":"train","ja_text":"こんにちは。","en_text":"Hello.","ja_audio":"audio/ja/0001.wav","en_audio":"audio/en/0001.wav"}
```

T2TTとS2UTは同じtrain pairを使用し、語彙もtrain splitのみから構築する。文字列に暗黙の正規化を行わず、全stageで`pair_id`を保持する。

## 推論

```bash
python -m alignment_transfer_s2ut.fetch_assets --config configs/s2ut/vocoder.yaml
python -m alignment_transfer_s2ut.infer \
  --audio /path/to/ja.wav \
  --checkpoint outputs/annealed_guidance/checkpoint_best.pt \
  --vocoder-config configs/s2ut/vocoder.yaml \
  --output outputs/predictions.jsonl --device cuda
```

複数音声の推論、ASR-BLEU評価、Attention可視化は[利用ガイド](docs/USAGE.md#推論と評価)を参照。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [USAGE](docs/USAGE.md) | 前処理・学習・推論・評価の実行手順 |
| [METHOD](docs/METHOD.md) | アラインメント転移と4つの比較条件 |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | T2TT / S2UTの共有構成とモダリティ固有部分 |
| [ALIGNMENT](docs/ALIGNMENT.md) | CTC、unit timing、行列投影、artifact管理 |
| [EXPERIMENTS](docs/EXPERIMENTS.md) | 共通レシピ、checkpoint選択、再開、ログ |
| [PILOT](docs/PILOT.md) | 実データ診断、strongの3条件pilot、GPU/時間の記録 |
| [DEVIATIONS](docs/DEVIATIONS.md) | 設計上の差異と実装範囲 |
| [VALIDATION](docs/VALIDATION.md) | 検証環境とテスト結果 |

S2UTには独立trainerを使用する。本学習・実コーパスでの性能評価は未実施。benchmarkとのレシピ上の対応と差異は[DEVIATIONS](docs/DEVIATIONS.md)に記載する。

## ライセンス

[MIT License](LICENSE)
