# 実データ診断とStrong S2UT Pilot

この少量pilotは独立trainer（`--trainer legacy`）による機能診断。既存benchmarkで学習したモデルを固定Baselineに使う比較は[BENCHMARK_COMPARISON](BENCHMARK_COMPARISON.md)と専用Colabを使用する。

[Colabで開く](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/pilot.ipynb)

実CTC・HuBERT・公式vocoderのCUDA接続確認、alignment品質診断、strongの3条件比較を行う。pilotには元のtrain/devだけを使い、最終testは手法・設定の確定後に別途評価する。

## Colabの設定

GPUランタイムを選択し、冒頭セルの`CORPUS`と`ROOT`を指定する。

| 設定 | 既定値 |
|---|---|
| accepted manifest | `CORPUS/production/manifests/releases/accepted.jsonl` |
| 出力先 | `/content/drive/MyDrive/alignment-transfer-s2ut-data/pilot-text-001` |
| train / dev上限 | 10000 / 200 pairs |
| seed | 1 |
| T2TT更新数 | 20000 |
| T2TT dev text評価 | dev全件、beam 5 / length penalty 1 / 最大1000出力step、SacreBLEU 13a |
| S2UT更新数 | 各条件10000 |
| annealing終了 | 5000 updates |
| warmup | 全条件1000 updates |
| validation / checkpoint間隔 | 500 updates |
| batch size / accumulation / max tokens | 8 / 4 / 20000 |
| CPU workers / prefetch | 2 / 2 nominal batches |
| local cache | `/content/alignment-transfer-cache`、上限8 GiB |
| recipe / smoothing / source mapping | strong / 0.05 / frontend |
| dev評価 | unit error rate（音声生成・ASR-BLEUは冒頭で有効化） |

更新数はT2TTとS2UTで独立する。S2UTを5000〜20000更新へ変更する際も、`ANNEAL_STEPS < S2UT_UPDATES`としてbeta=0の学習区間を残す。ここでのwarmupはpilot用設定で、[共通レシピ](EXPERIMENTS.md)の既定10000更新とは異なる。

初回セルはPython 3.10.22・固定バージョンの依存package・固定commitのfairseqを構築し、CUDA、torch/torchaudio、fairseq provenanceを検査する。CPU版PyTorchでは実行を停止する。

学習stageでは、少量の入力が揃うと学習を開始し、CPU workerが次を先読みする。未拡張のFBANK/CMVN・unit・teacherはVMのcacheへ必要時に保存し、3条件で共有する。[STREAMING](STREAMING.md)に上限・再開・検証を記載する。CTC/HuBERT/teacherの生成は学習前に完了させる。

## 実行順

1. **Pilot manifest生成**：元のsplit内でseed固定抽出し、選択した音声をhash検証する。T2TT/S2UTと全前処理で同一manifestを使う。
2. **CUDA接続試験**：train最大2件・dev1件で、実CTC/HuBERT、strongの3条件の2更新、teacher削除後の公式vocoder推論を試す。
3. **T2TT学習・text BLEUと前処理**：pilotのtrainでT2TTを学習する。最良dev lossのcheckpointから日本語だけを入力してdev全件を生成し、text BLEUと予測文を確認する。同じcheckpointでMT attentionを抽出し、日英CTC span、HuBERT/KM100、teacher packetを生成する。
4. **Train/dev診断**：各split最大500件・20 heatmapでcoverage・entropy・invalid rowを確認する。ノートブックには各split2例の原文・訳文・日英音声・heatmapを表示する。
5. **3条件pilot**：Baseline / Annealed Monotonic / Annealed T2TTを同じstrongレシピ・初期化で学習する。
6. **Dev評価と集計**：共通のteacher-free推論器でdev全件を評価し、比較契約、loss曲線、所要時間を保存する。

CLIは`subprocess`の引数配列で呼び出す。失敗したstageでセルを停止し、失敗・中断した試行も個別のログに残す。接続試験の2更新teacherと本pilotのteacherは別の保存先になる。

## データの固定と再開

`pilot prepare`は元のmanifest全体でID・対訳・既知の音声pairのsplit leakageを検査し、train/devの部分集合を選ぶ。音声ファイルを読み出すのは選択後で、未選択testの音声は要求しない。元のsplitを付け替えない。

`data/selection.json`に入力manifestのSHA256、抽出seed・上限、pair IDs、元のsplit別件数、実際の使用件数、canonical manifest hashを保存する。再開時には同一条件と音声内容を検証する。元のコーパスへ出力できない。

同じ設定でセルを再実行すると、前処理artifactと学習checkpointから再開する。データ、seed、recipe、batch設定、コードrevision、前処理に使ったT2TT teacherを変更する場合は新しい`ROOT`を使う。新しいColab runtimeでもノートブック内の固定commitを取得する。

## 診断結果の確認

指標の定義は[ALIGNMENT](ALIGNMENT.md#品質診断)を参照する。

- source/target token coverageを、CTCで有効な文字数と合わせて確認する。特にdevのOOVによるmaskを確認する。
- unit row coverageが低いsampleでは、CTC spanとunit runの時間的な重なりを音声と照合する。全有効row数が0ならpilotへ進まず停止する。
- raw priorと平滑化後のheatmap・entropyを確認する。全0のrowは平滑化後も0のまま。
- T2TTのdev/main_loss・dev text BLEU・raw attentionも確認する。翻訳自体の弱さとalignment転移の効果を区別するため、生成した英文も参照訳と照合する。T2TTの学習更新数だけではteacher品質を保証しない。

coverageとentropyは時刻精度そのものではない。少数例の聴取・可視化と、実データの数値集計を合わせて判断する。

T2TTのtext BLEUはteacher forcingを使わない自己回帰生成によるcorpus BLEU。文字tokenを英文へ戻し、未加工の英語参照訳とcase-sensitiveのSacreBLEU 2.5.1（13a・exp smoothing・effective order無効）で比較する。ASRや音声artifactは不要。BLEU署名、checkpoint hash、dev manifest hash、生成設定、予測文を保存する。`max-chars`はEOSを含む最大出力step数で、上限に達してもEOSを強制しない。EOS到達率とbrevity penaltyを併せて確認する。

このBLEUはdev lossで選んだteacherの診断値で、独立した最終testスコアではない。低いBLEUやEOS到達率に一律の合否閾値は設定しない。本学習へ進む前に翻訳品質とalignment診断を確認する。

## Pilot結果

| 保存先（ROOT配下） | 内容 |
|---|---|
| `data/` | 選択条件とcanonical manifest |
| `runtime.json` | Python・torch・CUDA・fairseq revisionとGPU機種/メモリ |
| `integration/attempt-*/` | CUDA接続試験の出力 |
| `t2tt/`, `mt/`, `ctc/`, `units/`, `teacher/` | 学習済みteacherと前処理artifact |
| `t2tt/dev-text-metrics.json`, `t2tt/dev-text-metrics-predictions.jsonl` | T2TT単体のdev text BLEU・署名・EOS到達率・生成設定と予測文 |
| `alignment-audit-train/`, `alignment-audit-dev/` | summary JSON・sample別JSONL・heatmap PNG/NPZ |
| `runs/{condition}/` | checkpoint・学習metrics・dev予測と評価 |
| `execution/` | 試行ごとのコマンド、終了状態、所要秒数、完全ログ、GPU使用率CSV |
| `pilot-plan.json` | 共通設定と3条件の実行コマンド |
| `fair-comparison.json` | recipe・初期化・データ・転移条件の比較検査 |
| `pilot-summary.json`, `pilot-summary.png` | 最良dev loss・更新数・beta・dev評価・記録時間・T2TT dev text評価とdev loss曲線 |

checkpoint選択・loss曲線は全条件ともbeta=0のdev main loss。補助lossを含むtrain total lossは目的関数が異なるため、条件間の順位付けに使わない。集計は再開で重複したstepについて最新のmetricsを採用する。

`pilot report --t2tt-evaluation`は、text BLEUを測定したT2TTがAnnealed T2TTのteacher checkpoint・dev data・backendと一致することを検査する。Colabではこの検査を有効にする。

所要時間は記録できた試行の合計で、失敗試行数も明示する。GPU CSVは5秒間隔のdevice全体の使用率・メモリ使用量であり、1更新ごとの所要時間やプロセス専用使用率ではない。低使用率の時間帯をログと照合し、FBANK/I/Oの処理が学習を待たせていないか確認する。

dev評価の既定はbeam 5・length penalty 1・max units 1000のunit推論。`GENERATE_WAVEFORMS=True`で固定vocoder、さらに`ASR_BLEU=True`で固定ASRを使う。3条件を同じ設定で評価し、設定変更時は新しい出力先を使う。

## CLIでの利用

```bash
python -m alignment_transfer_s2ut.pilot prepare \
  --input /path/to/corpus/production/manifests/releases/accepted.jsonl \
  --corpus-root /path/to/corpus --output outputs/pilot/data \
  --train-limit 10000 --dev-limit 200 --seed 1 --resume

# T2TT学習後、同じteacherのdev text BLEUを評価する。
python -m alignment_transfer_s2ut.evaluate_t2tt \
  --manifest outputs/pilot/data/manifest.jsonl \
  --checkpoint outputs/pilot/t2tt/checkpoint_best.pt --split dev \
  --beam-size 5 --length-penalty 1 --max-chars 1000 --device cuda \
  --output outputs/pilot/t2tt/dev-text-metrics.json --resume

# 前処理・診断後、実行予定を表示。planではGPU学習を開始しない。
python -m alignment_transfer_s2ut.pilot plan \
  --manifest outputs/pilot/data/manifest.jsonl --config configs/s2ut/fisher.yaml \
  --units outputs/pilot/units --teacher outputs/pilot/teacher \
  --output outputs/pilot/runs --logs outputs/pilot/execution

# 同じ引数でplanをtrainへ変えると、CUDA検査後に3条件を実行する。
python -m alignment_transfer_s2ut.pilot report \
  --runs outputs/pilot/runs --logs outputs/pilot/execution \
  --t2tt-evaluation outputs/pilot/t2tt/dev-text-metrics.json \
  --output outputs/pilot/pilot-summary.json
```

この手順のローカル検証では人工データを使う。実CTC/HuBERT/vocoderのCUDA接続、Colab上の実行、実コーパスでのpilot結果は、実行後に記録する。
