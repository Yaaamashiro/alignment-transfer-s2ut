# 利用ガイド

各コマンドの実行環境は[README](../README.md#インストール)を参照。

実データの診断とstrongの3条件pilotをまとめて実行する場合は、[専用Colabと手順](PILOT.md)を参照。固定したtrain/dev部分集合、CUDA接続試験、実行ログ、GPU使用率、dev評価を用意する。

## 入力manifest

入力は1行1pairのJSONLです。最低限`pair_id, split, ja_text, en_text, ja_audio, en_audio`を持ちます。`split`は`train/dev/test`。文字列はstripやUnicode normalizationを行わず保持します。既存コーパスの`ja_wav_16k/en_wav_16k`もimport時に受け付けます。

```bash
python -m alignment_transfer_s2ut.prepare_manifest \
  --input /path/to/corpus/production/manifests/releases/accepted.jsonl \
  --corpus-root /path/to/corpus --output outputs/manifest.jsonl
```

canonical manifestを直接使う場合は`--corpus-root`を省略できます。全stageで`pair_id`を保持し、split間のID・対訳pair・音声pair重複を拒否します。既存splitの再分割は行いません。

Colabのpath設定例は[configs/data/colab.yaml](../configs/data/colab.yaml)を参照。コーパスと派生artifactの保存先は分離する。

## 前処理と学習

各コマンドはリポジトリ直下で実行する。`--max-updates 2`は接続確認用。本学習ではS2UTの4条件で共通の更新数を指定する。

```bash
# 1. Transformer構成の比較表を保存
python -m alignment_transfer_s2ut.compare_architectures \
  --t2tt-config configs/t2tt/fisher.yaml --s2ut-config configs/s2ut/fisher.yaml \
  --output outputs/architecture.json

# 2. train splitだけで文字語彙を作成し、T2TTを学習
python -m alignment_transfer_s2ut.train_t2tt \
  --manifest outputs/manifest.jsonl --config configs/t2tt/fisher.yaml \
  --output outputs/t2tt --device cuda --max-updates 2

# 3. teacher forcing、最終decoder層、全head平均（BOS/EOS/paddingは出力から除外）
python -m alignment_transfer_s2ut.extract_t2tt_attention \
  --manifest outputs/manifest.jsonl --checkpoint outputs/t2tt/checkpoint_best.pt \
  --output outputs/mt --device cuda

# 4. CTC Viterbi forced alignment。posteriorをdurationとして使わない
python -m alignment_transfer_s2ut.align_speech_text \
  --manifest outputs/manifest.jsonl --config configs/alignment/ctc.yaml \
  --output outputs/ctc --device cuda

# 5. 固定hashの公式KM100を取得し、HuBERT Base layer 6のunitを抽出
python -m alignment_transfer_s2ut.fetch_assets --config configs/units/hubert_km100.yaml
python -m alignment_transfer_s2ut.extract_units \
  --manifest outputs/manifest.jsonl --config configs/units/hubert_km100.yaml \
  --output outputs/units --device cuda

# 6. 時刻spanを保ったteacher packetを生成。Fへの投影はS2UTのforward時
python -m alignment_transfer_s2ut.build_teacher_alignment \
  --manifest outputs/manifest.jsonl --mt-attention outputs/mt --alignment outputs/ctc \
  --units outputs/units --config configs/s2ut/fisher.yaml --output outputs/teacher

# 7. 同一設定・同一初期化で4条件を実行
for method in baseline distillation guidance annealed_guidance; do
  python -m alignment_transfer_s2ut.train_s2ut \
    --manifest outputs/manifest.jsonl --config configs/s2ut/fisher.yaml \
    --units outputs/units --teacher outputs/teacher --method "$method" \
    --output "outputs/$method" --device cuda --max-updates 2
done
```

全4条件で同じseed、data、unit、batch、optimizer、scheduler、更新数を指定する。[実験条件](EXPERIMENTS.md)に各引数・再開契約を記載する。`--layer -1`が最終層、`--heads 0 2`などで0始まりのheadを選択できます。

```bash
python -m alignment_transfer_s2ut.compare_experiments \
  outputs/baseline outputs/distillation outputs/guidance outputs/annealed_guidance \
  --output outputs/fair-comparison.json
```

共通設定・初期化hash・train/dev pair・unit設定・teacher identity・予定更新数が一致しなければ失敗します。翻訳スコアを比較する際はvocoderと推論設定も同一にする。

## Alignment品質診断

teacher生成後、学習前にdevまたはtrainの100〜500件を確認する。

```bash
python -m alignment_transfer_s2ut.diagnose_alignment \
  --manifest outputs/manifest.jsonl --teacher outputs/teacher --units outputs/units \
  --config configs/s2ut/fisher.yaml --split dev --max-samples 500 --heatmaps 20 \
  --source-mapping frontend --prior-smoothing 0.05 --output outputs/alignment-audit
```

`summary.json`に有効source/target token率・unit row率・prior entropy、`pairs.jsonl`にsample別の分子・分母を保存する。`heatmaps/`のPNG/NPZはraw MT、投影後、平滑化後、単調prior。CTC coverageとMT語彙外によるmaskを区別して確認する。時刻精度の確認には元の音声も使う。指標の定義は[ALIGNMENT](ALIGNMENT.md#品質診断)を参照。再実行で置換する場合は`--overwrite`を付ける。

## 強いbaselineと単調priorの対照実験

`controlled`を既定とし、`--recipe strong`でbenchmark相当の補助taskと音響前処理を有効にする。以下は同じstrongレシピでBaseline / Annealed Monotonic / Annealed T2TTを比較する例。`UPDATES=20000`はpilot用で、本学習では比較する全条件を同じ更新数へ変更する（参照benchmarkの推奨は400000更新）。

```bash
UPDATES=20000
COMMON="--manifest outputs/manifest.jsonl --config configs/s2ut/fisher.yaml --units outputs/units --recipe strong --max-updates $UPDATES --validation-interval 1000 --save-interval-updates 1000 --device cuda"

python -m alignment_transfer_s2ut.train_s2ut $COMMON \
  --method baseline --output outputs/strong/baseline

python -m alignment_transfer_s2ut.train_s2ut $COMMON \
  --method annealed_guidance --prior-type monotonic --monotonic-sigma 0.1 \
  --source-mapping frontend --prior-smoothing 0.05 --anneal-steps 10000 \
  --output outputs/strong/annealed_monotonic

python -m alignment_transfer_s2ut.train_s2ut $COMMON \
  --method annealed_guidance --prior-type t2tt --teacher outputs/teacher \
  --source-mapping frontend --prior-smoothing 0.05 --anneal-steps 10000 \
  --output outputs/strong/annealed_t2tt

python -m alignment_transfer_s2ut.compare_experiments \
  outputs/strong/baseline outputs/strong/annealed_monotonic outputs/strong/annealed_t2tt \
  --output outputs/strong/fair-comparison.json
```

単調priorの学習には`--teacher`、T2TT checkpoint、CTC artifactは不要。同じ対照実験を`--recipe controlled`でも実行できる。recipeを変更するときは別のrun directoryを使う。strongの補助loss・語彙・CMVN/SpecAugmentの詳細は[EXPERIMENTS](EXPERIMENTS.md#strong-recipe)を参照。

## 転移係数

| 引数 | 既定値 |
|---|---|
| `--lambda-align` | 1.0 |
| `--beta` | 1.0 |
| `--beta-start` / `--beta-end` | 1.0 / 0.0 |
| `--anneal-steps` | 10000 |
| `--prior-type` | t2tt（対照実験はmonotonic） |
| `--prior-smoothing` | 0.05（0で無効） |
| `--source-mapping` | frontend（旧方式はuniform） |
| `--monotonic-sigma` | 0.1（音声長で正規化した時間座標） |

annealingは完了済みoptimizer更新数を基準とし、`step >= anneal_steps`で終端値になる。

## 推論と評価

推論は日本語音声のみを入力し、すべての方式で`beta=0`です。transcript、英語reference、T2TT、alignment、target speechを読みません。S2UT checkpointとvocoderだけで配備できます。

```bash
python -m alignment_transfer_s2ut.fetch_assets --config configs/s2ut/vocoder.yaml
python -m alignment_transfer_s2ut.infer --audio /path/to/ja.wav \
  --checkpoint outputs/annealed_guidance/checkpoint_best.pt \
  --vocoder-config configs/s2ut/vocoder.yaml --output outputs/predictions.jsonl --device cuda
```

複数pairには`--input speech_only.jsonl`を使用します。各行は`pair_id, ja_audio`のみ（`split`も任意）。referenceを含む学習manifestをそのまま渡すと拒否します。`--units-only`はvocoderなしのunit推論用です。

```bash
python -m alignment_transfer_s2ut.prepare_inference_input \
  --manifest outputs/manifest.jsonl --split test --output outputs/test-speech-only.jsonl
```

```bash
python -m alignment_transfer_s2ut.infer \
  --input outputs/test-speech-only.jsonl \
  --checkpoint outputs/annealed_guidance/checkpoint_best.pt \
  --vocoder-config configs/s2ut/vocoder.yaml \
  --output outputs/test/predictions.jsonl --device cuda
```

評価には選択splitと同じID集合の予測を渡します。評価用ASRはbenchmarkと同じ固定Whisper turboです。

```bash
python -m alignment_transfer_s2ut.evaluate --manifest outputs/manifest.jsonl \
  --predictions outputs/test/predictions.jsonl --split test --units outputs/units \
  --asr-config configs/experiments/evaluation.yaml --device cuda --output outputs/test/metrics.json

python -m alignment_transfer_s2ut.visualize_attention --manifest outputs/manifest.jsonl \
  --pair-id YOUR_PAIR_ID --checkpoint outputs/guidance/checkpoint_best.pt \
  --teacher outputs/teacher --units outputs/units --output outputs/figures --device cuda
```

可視化は同一sampleのT2TT、転移後、raw S2UT、guided S2UTの`.npz/.png`とmetadataを保存します。

## 再開

学習・Attention抽出・CTC・unit・teacher生成は`--resume`で既存artifactを検証して再開する。学習の`--resume-if-exists`はcheckpointの有無に応じて初回開始と再開を切り替える。

manifest・構成比較・推論・評価も`--resume`に対応する。推論はsampleごとのatomic cacheから再開する。既存出力の置換は`--overwrite`で明示する。`--resume`とは併用できない。

Colabセルは同一設定での再実行に対応する。保存形式と再開契約は[EXPERIMENTS](EXPERIMENTS.md)を参照。
