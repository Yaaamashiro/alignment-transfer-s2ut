# 学習済みbenchmark Baselineとの比較

[Colabで開く](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/colab/benchmark_comparison.ipynb)

Colabの通常運用は[COLAB_RUNTIME](COLAB_RUNTIME.md)を参照。入出力・ZIP保存・復元・S2UT performance hooksは固定revisionのbenchmark実装を再利用し、ROOTはVMローカル、PERSISTENTはDriveへ分離する。

別のリポジトリで学習したS2UT checkpointをBaselineとして登録する。転移モデルは同じ保存設定・preparedデータから新規学習し、両者を共通の推論・評価経路へ渡す。Baselineの再学習や、その重みを使った継続学習は必須にしない。

本比較の推奨条件は次のとおり。

| 位置付け | Condition | Colabのスイッチ |
|---|---|---|
| 主比較 | S2UT Baseline（登録済みモデル） | 再学習せず使用 |
| 主比較 | Annealed T2TT Guidance | `RUN_TRANSFER=True` |
| 任意のcontrol | Annealed Monotonic Guidance | `RUN_MONOTONIC=True` |

両スイッチの既定値はFalse。従来どおりteacher準備とT2TT側のplan確認まで行い、本学習を自動開始しない。両方Trueなら3条件を比較する。`RUN_MONOTONIC=True`・`RUN_TRANSFER=False`ではT2TT・CTC・teacher準備をスキップし、Monotonicだけを新規学習できる。完成済みT2TTがあれば評価へ再利用する。正式な条件名はAnnealed Monotonic Guidanceとし、native fairseq経路の実装名と区別する。

## 学習条件

本比較は公式fairseq commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`のTrainer、optimizer、scheduler、`speech_to_unit` criterion、multitask decoder、token-budget sampler、dataset/audio transformsを使う。独自PyTorch loopは`--trainer legacy`で明示する旧経路へ分離する。

Baseline checkpointの実際の`cfg`を再利用するため、FP16、Adam設定、学習率、warmup、label smoothing、勾配蓄積、batch上限、seed、モデル寸法、補助task設定をpresetの既定値から推測しない。preparedのTSV、文字辞書、CTC辞書、SentencePieceをそのまま使用する。benchmarkの文字inventoryがdev/testの文字を登録している場合も、その事実をdata lockに残して同じ辞書を使う。T2TTの文字語彙は引き続きtrainのみで作る。

対応するbenchmarkは単一processのfairseq S2UT。`paper_fisher_v1`のreport-only policyでは、保存された間隔で全devのteacher-free unit lossを測り、最良値のcheckpointを選ぶ。main lossはlabel-smoothed CEの和／main token数をbitsへ換算する。補助lossやalignment lossではbestを選ばない。古い独自early-stop policyは、同一性を保証できないためこの経路の登録時に拒否する。

ここでteacher-freeはT2TT priorを使わないことを指す。unit lossの計算はbenchmarkと同じ参照unitによるteacher forcingであり、音声生成による翻訳品質評価とは別。

FP16 overflowでは、optimizerへ入る前にloss scaleが下がった場合だけ、同じ論理batchとRNG/bufferを復元して再試行する。OOM・不明な更新失敗・optimizerに入った後の失敗は停止し、sampleを飛ばさない。確定済み更新数・optimizer/scaler・iterator・RNG・選択状態を保存し、同じrunのcheckpointだけを再開する。Baseline checkpointを転移側のrestore先へ指定しない。

Guidanceは選択した最終Cross-Attentionへpriorを加える。Distillationはnative lossへKLを加算し、勾配蓄積全体の有効alignment row数で正規化する。追加priorはFP32を保ち、native FP16 sample変換で丸めない。beta=0とvalidation/inferenceでは公式Attentionの経路を使う。

## 1. Snapshotを検証して別の場所へ復元

```bash
python -m alignment_transfer_s2ut.benchmark restore \
  --backup /path/to/benchmark/checkpoints/s2ut-training-run \
  --output outputs/comparison/baseline-snapshot --resume
```

`snapshot.json`が完成したsnapshotを更新数順に選び、ZIPのSHA256とmember一覧を検証する。best/last checkpoint、research metadata、recipe lockだけを別ディレクトリへ取り出し、各ファイルのhashを照合する。再実行では最初に選んだsnapshotを維持する。元のbackupは変更しない。

## 2. 外部Baselineを登録

```bash
python -m alignment_transfer_s2ut.benchmark register \
  --checkpoint outputs/comparison/baseline-snapshot/checkpoints/checkpoint_best.pt \
  --data /path/to/benchmark/data/s2ut/fairseq \
  --recipe-lock outputs/comparison/baseline-snapshot/s2ut-recipe-lock.json \
  --metadata outputs/comparison/baseline-snapshot/research-metadata.json \
  --output outputs/comparison/baseline.json --resume
```

benchmark自身が保存したrecipe fingerprintと、checkpoint内の`extra_state.s2st_recipe_fingerprint`、metadata、prepared inputのhashを照合する。対象は自分で学習した信頼済みのfairseq checkpointに限る。pickle形式でoptimizer等を含むため、見知らぬcheckpointを取得して読み込むための機能ではない。

`checkpoint_updates`は選んだbestの更新数、`training_budget`はそのrunが完了した更新数。予定400kの途中snapshotが365kなら、転移側も365k更新で比較する。400kの比較には400k完了snapshotを新しい出力先へ登録する。bestが早い更新で選ばれていても、bestの更新数だけへ学習budgetを短縮しない。

## 3. 同一pair・unitでalignmentを準備

```bash
python -m alignment_transfer_s2ut.benchmark_data manifest \
  --common /path/to/benchmark/data/common --prepared /path/to/benchmark/data/s2ut/fairseq \
  --output outputs/comparison/manifest.jsonl --resume

python -m alignment_transfer_s2ut.benchmark_data units \
  --manifest outputs/comparison/manifest.jsonl --benchmark-units /path/to/benchmark/data/s2ut/units \
  --prepared /path/to/benchmark/data/s2ut/fairseq --output outputs/comparison/units --resume
```

train/devの全pair・source pathをprepared TSVと照合する。音声内容hashを固定し、保存済みHuBERT原系列からrun spanを構築する。再計算したunitで学習targetを置き換えず、importしたreduced列がnative TSVのtargetと厳密一致することを検査する。testは既定でexportしない。

同じmanifestでT2TT学習、dev text BLEU、Attention抽出、CTC、teacher packet作成、alignment診断を実行する。Colabに一連のコマンドをまとめる。少量pilotのteacherや異なるtrain部分集合を本比較へ混ぜない。

Colabは登録Baselineの`backbone`から共通設定ファイルを生成し、T2TT・teacher作成・診断へ渡す。固定presetの寸法やdropoutを保存checkpointの設定と取り違えない。

### Attention抽出の処理速度と再開

6.2のAttention抽出は推論であり、T2TT/S2UTの学習batchとは別の設定を持つ。既定は最大8 pair/forward、padding後のsource+target token合計4096、`batch × target × source`のAttention要素数100万。各制約を満たす範囲でまとめ、FP32・teacher forcing・指定layer/head・未知文字mask・pairごとの保存形式を維持する。単体pairが予算を超える場合も黙って除外せず停止する。独立CLIでは`--batch-size`（1で単体推論）・`--max-tokens`・`--max-attention-cells`で制御できる。

checkpoint全体のSHA256とtokenizer/architectureのhashは開始時に1回計算する。checkpointのpath・size・mtimeが処理途中で変化したら停止し、同じcheckpoint名で異なるモデルの結果を混在させない。音声は読み込まず、manifest内の音声identityをそのまま含める。

進捗はおおむね1秒ごとと最後に表示する。分子はログ行数ではなく照合・保存済みpair数。`new`が新規抽出、`reused`がmetadata/payload照合済みの再利用、`new_pairs_per_second`が新規のforwardと保存にかかった速度。`eta_seconds`は残りをすべて新規抽出と仮定した目安で、初期化・再利用照合・Drive backup待ちを含む完了保証ではない。新規抽出前のETAはunknownとする。

従来の1件ずつのextractorとmetadataは共通で、`--resume`は完成済みpairを再利用する。batch推論と単体推論の間に浮動小数点の微小差が出る場合はあるが、意味を変える精度切り替えや再学習は行わない。既存stageの復旧identityを維持するため、Colabの既定commandにはbatch optionを追加しない。既存backup付きのcommandへ新たにoptionを追加すると、従来どおり別commandとして停止する。

旧版で6.1を完了して6.2を処理中の場合は、6.2を停止して子process終了を確認し、同じVM・ROOT・PERSISTENT・checkpoint・manifestでsourceだけを更新して6.2から再開する。取得済みデータや完成済みAttentionを削除しない。source更新後に旧T2TT runの6.1を再実行するとimplementation contractが異なるため拒否される。学習途中のrunを新sourceへ無条件に移す運用は行わない。

## 4. 転移モデルを新規学習

```bash
# planは設定を表示するだけで、学習を開始しない。
python -m alignment_transfer_s2ut.benchmark plan \
  --baseline outputs/comparison/baseline.json --output outputs/comparison/annealed-t2tt \
  --manifest outputs/comparison/manifest.jsonl --units outputs/comparison/units \
  --teacher outputs/comparison/teacher --method annealed_guidance --anneal-steps 200000

# 同じ引数でplanをtrainへ変更する。途中再開には--resumeを付ける。
```

転移モデルは保存済みseed・native model constructorから初期化する。Baselineの重み・optimizer stateはロードしない。`native-run.json`にBaselineの登録hash、学習設定、転移条件、前処理の識別、実装hashを保存する。annealing終了は完了済みbudgetより前に指定する。FP16の登録設定に対してCPUや別精度へ自動fallbackしない。

`train_s2ut`入口でも`--baseline-record`を指定できる。この場合、学習率・batch・更新数等をCLIで上書きする操作を拒否し、保存Baselineの条件を使用する。`--manifest/--units/--teacher`と転移係数は別途指定する。

任意のcontrolであるAnnealed Monotonic Guidanceは、別のrun directoryに保存する。

```bash
python -m alignment_transfer_s2ut.benchmark train \
  --baseline outputs/comparison/baseline.json --output outputs/comparison/annealed-monotonic \
  --manifest outputs/comparison/manifest.jsonl --units outputs/comparison/units \
  --method annealed_guidance --prior-type monotonic --anneal-steps 200000 --resume
```

Monotonicは音声時間からpriorを作り、`--teacher`を必要としない。T2TT側と同じBaseline登録・全train/dev・unit列・annealing終了を使用し、native fairseqの学習設定は変更しない。T2TT側は従来の`--teacher`付きコマンドを維持し、既定の`prior_type=t2tt`を使う。

## 5. 共通推論・評価・比較

```bash
python -m alignment_transfer_s2ut.benchmark register-transfer \
  --run outputs/comparison/annealed-t2tt \
  --checkpoint outputs/comparison/annealed-t2tt/checkpoints/checkpoint_best.pt \
  --output outputs/comparison/annealed-t2tt.json

python -m alignment_transfer_s2ut.benchmark infer \
  --record outputs/comparison/baseline.json --input outputs/comparison/dev-source.jsonl \
  --output outputs/comparison/baseline-dev-predictions.jsonl --units-only --device cuda --resume

python -m alignment_transfer_s2ut.benchmark compare \
  outputs/comparison/baseline.json outputs/comparison/annealed-t2tt.json \
  --output outputs/comparison/training-comparison.json
```

`dev-source.jsonl`は`prepare_inference_input --split dev`で作る。入力として受け付けるのは`pair_id/ja_audio/optional split`のみ。転移モデルの登録JSONでも同じ`benchmark infer`を呼ぶ。両モデルをstock S2UTとして読み込み、native fairseq SequenceGenerator（既定beam 10、max_len_a 1、max_len_b 200、length penalty 1、FP32推論）を共用する。生成時にtrain/dev/testのlabel TSVやteacherを参照しない。モデル構築用のprepared config・辞書は必要。

`--vocoder-config`で同一vocoderを指定できる。両方の予測を共通`evaluate` CLIで測り、`benchmark compare --evaluations BASELINE_METRICS TRANSFER_METRICS`へ渡す。生成設定・manifest・unit reference・ASR設定・vocoderが違う評価や、別checkpointのスコアを混ぜた比較は拒否／差分として報告する。未知の差分を一致扱いしない。

Colabでは実行した条件をBaseline → Annealed Monotonic Guidance → Annealed T2TT Guidanceの順で同じ推論・評価経路へ渡す。選択した条件のcheckpointが未完成なら比較を待つ。2条件／3条件のレポートは`ROOT/comparison/<条件の組み合わせ>/`へ分け、既存の2条件レポートに対するresumeと衝突させない。生成・評価cacheは各条件で継続利用する。

最後に次の形式のdev結果表を表示する。UERは比率。ASR-BLEUには冒頭から`GENERATE_WAVEFORMS=True`・`ASR_BLEU=True`を指定し、未実行の値は「—」とする。

| Condition | ASR-BLEU | UER |
|---|---:|---:|
| S2UT Baseline | — | — |
| Annealed Monotonic Guidance（任意） | — | — |
| Annealed T2TT Guidance | — | — |

## 範囲

モデルと登録・比較CLIは公式fairseqと保存artifactを使用する。ColabのI/O・recovery・S2UT performanceは固定revisionのbenchmark packageを直接利用する。原論文と同じスペイン語Fisher corpus／vocoder条件を再現する主張ではなく、ユーザーの既存日英Baselineと学習・評価条件を揃えるための機能。GPU・worker RNGや外部runtimeまで含めたbit単位の再現性は保証しない。既存の独立trainerによる機能診断は[旧pilot](PILOT.md)に残す。

Colabではbenchmarkのcache・worker制御・overflow retry・勾配監査を共用する。checkpoint保持数はbenchmarkのローカル容量対策に従い、科学的比較から除外するI/O項目とする。オンラインBLEU monitorは現時点では移植せず、共通の事後評価を使用する。`fair_comparison`は保存設定・データ・budget・選択規則、および指定された評価contractの照合結果。追加physical microbatch分割を使ったBaselineは登録時に拒否する。実checkpointとCUDAによる確認状況は[VALIDATION](VALIDATION.md)を参照。
