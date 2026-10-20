# Alignment Transfer for S2UT

[MIT License](LICENSE)

T2TTが学んだsoft alignment / routing patternを音声座標へ写像し、S2UTの学習を誘導する独立実装です。モデルの重み転移は行いません。Cross-Attentionを言語学的な正解アラインメントとは仮定しません。

```text
             T2TT                         S2UT
        Japanese Text                Japanese Speech
              ↓                            ↓
        Text Embedding              FBANK + Conv Frontend
              ↓                            ↓
     ┌─────────────────┐          ┌─────────────────┐
     │ Native fairseq  │          │ Native fairseq  │
     │ Transformer    │          │ Transformer    │
     │ Encoder/Decoder│          │ Encoder/Decoder│
     └─────────────────┘          └─────────────────┘
              ↓                            ↓
         English Text                English KM100 Units
                                           ↓
                                    Unit HiFi-GAN
```

同じFisher backbone設定から両モデルを構築します。T2TTはnative S2UT encoderのsubsamplerを文字embeddingに置き換え、同一のEncoder/Decoder層・Cross-Attention実装を利用します。巨大な事前学習MTは使いません。

## 動作環境

実験用はPython 3.10、PyTorch/torchaudio 2.7.1、fairseqの固定commitを使用します。Linux/Colabを推奨します。fairseq本体の編集、benchmarkパッケージのimport、benchmarkのcheckpointは不要です。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==23.3.2' 'setuptools<81' wheel
python -m pip install torch==2.7.1 torchaudio==2.7.1 numpy==1.26.4 Cython==3.2.9
python -m pip install --no-build-isolation -r requirements-fairseq.txt
python -m pip install -r requirements.txt
```

GPUは本学習に必要です。CLIの学習上限は明示的に変更しない限り2更新です。ColabのPythonが3.11以上の場合は[ノートブック](notebooks/colab/pipeline.ipynb)の独立したPython 3.10環境を使ってください。

## まず外部モデル不要で確認

```bash
python -m pytest -q
python -m alignment_transfer_s2ut.smoke --output outputs/smoke
```

smokeは小さなPyTorchモデル、人工CTC emission、人工unit、テスト音で全CLIの接続と数値計算を検証します。**実際のHuBERT/CTC/vocoderの動作・翻訳品質を証明する実験ではありません。** native fairseqテストは対応環境で実行されます。

## 保存先と入力

確認できた既存Colabの保存規則は以下です。これらは入力参照であり、新実装の必須依存ではありません。

| 対象 | 既存の保存規則 |
|---|---|
| コーパス | `/content/drive/MyDrive/ja-en-direct-s2st-corpus-data/production/manifests/releases/accepted.jsonl` |
| benchmark実験 | `/content/drive/MyDrive/ja-en-direct-s2st-benchmark-data/<MODE>/<EXPERIMENT>` |
| benchmarkのCACHE_ROOT | `<benchmark実験>/cache` |
| benchmarkのKM100 | `<CACHE_ROOT>/models/s2ut/hubert_base_l6_k100.bin` |
| benchmarkのunit vocoder | `<CACHE_ROOT>/models/vocoder/unit/{g_00500000,config.json}` |

新しい保存先の既定例は `/content/drive/MyDrive/alignment-transfer-s2ut-data` です。日英CTC forced-alignmentモデルの指定は既存リポジトリにないため、固定revision付きで[CTC設定](configs/alignment/ctc.yaml)を用意しています。新規モデルは`models/`へ保存でき、同じ公式モデルを独立に取得できます。

入力は1行1pairのJSONLです。最低限`pair_id, split, ja_text, en_text, ja_audio, en_audio`を持ちます。`split`は`train/dev/test`。文字列はstripやUnicode normalizationを行わず保持します。既存コーパスの`ja_wav_16k/en_wav_16k`もimport時に受け付けます。

```bash
python -m alignment_transfer_s2ut.prepare_manifest \
  --input /path/to/corpus/production/manifests/releases/accepted.jsonl \
  --corpus-root /path/to/corpus --output outputs/manifest.jsonl
```

canonical manifestを直接使う場合は`--corpus-root`を省略できます。全stageで`pair_id`を保持し、split間のID・対訳pair・音声pair重複を拒否します。既存splitの再分割は行いません。

## 実データpipeline

以下はリポジトリ直下で実行します。`--max-updates 2`は接続確認用なので、本学習時に共通の更新数へ変更してください。各方式のcheckpointは独立しています。

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

全4条件で同じseed、data、unit、batch、optimizer、scheduler、更新数を指定してください。[実験条件](docs/EXPERIMENTS.md)に各引数・再開契約を記載しています。`--layer -1`が最終層、`--heads 0 2`などで0始まりのheadを選択できます。

```bash
python -m alignment_transfer_s2ut.compare_experiments \
  outputs/baseline outputs/distillation outputs/guidance outputs/annealed_guidance \
  --output outputs/fair-comparison.json
```

共通設定・初期化hash・train/dev pair・unit設定・teacher identity・予定更新数が一致しなければ失敗します。翻訳スコアを比較する際はvocoderと推論設定も同一にしてください。

## 転移方法

`A_T = M_tgt @ A_MT @ M_src`を有効rowごとに正規化します。invalid rowは0のまま保持し、lossとguidanceの対象から除外します。

| method | 学習中の変更 |
|---|---|
| `baseline` | 通常のnative S2UT |
| `distillation` | `main_loss + lambda_align * KL(A_T || A_S)` |
| `guidance` | `softmax(S_S + beta * log(A_T + eps))`からcontextを計算 |
| `annealed_guidance` | 同上。`beta_start`から`beta_end`へlinear annealing |

既定値は`lambda_align=1, beta=1, beta_start=1, beta_end=0, anneal_steps=10000`です。annealingのstepは完了済みoptimizer更新数を基準とし、`step >= anneal_steps`で0になります。

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

## 再開と実装上の範囲

学習・Attention抽出・CTC・unit・teacher生成は`--resume`で再開できます。学習は既定で各更新後にmodel、Adam、sampler位置、Python/NumPy/Torch RNGをatomicに保存します。既存出力は明示的な`--overwrite`なしに置き換えません。`--overwrite`と`--resume`は併用できません。

学習の`--resume-if-exists`は初回開始と再開を同じコマンドで行います。Colabセルはこのオプションを利用します。manifest・構成比較・推論・評価も`--resume`で既存の入力/設定hashを検証して再利用できます。推論はsampleごとのatomic cacheを持ち、途中終了から再開できます。

[DEVIATIONS](docs/DEVIATIONS.md)を実験前に確認してください。S2UTのarchitectureはnative Fisherですが、独立trainerのレシピを採用し、既存benchmarkの補助CTC/文字decoderやSpecAugmentを含みません。既存benchmarkとのスコアの直接比較を保証するものではありません。4条件の内部比較では同じレシピを使います。

詳しい設計は[METHOD](docs/METHOD.md)、[ARCHITECTURE](docs/ARCHITECTURE.md)、[ALIGNMENT](docs/ALIGNMENT.md)、[EXPERIMENTS](docs/EXPERIMENTS.md)にあります。
