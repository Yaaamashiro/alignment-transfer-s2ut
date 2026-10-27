# Layer-wise Cross-Attention Transfer

主要比較は登録済みBaselineと、All-layer Distillation / All-layer Annealed Guidance / All-layer Guidance。T2TTのlearned soft alignmentを同じ番号のS2UT Decoder層へ転移する。最終層priorを全層へコピーしない。

## Colabでの実行

1. 最新の`benchmark_comparison.ipynb`で入力・Baselineを復元する。S2UT更新数はBaselineの`training_budget`を継承する。
2. `ALL_LAYERS=True`。T2TTの20k→200k延長・dev・固定8C 300件評価は既存手順を使う。T2TT自体のarchitectureや学習recipeは変更しない。
3. BLEU・翻訳例・曲線を確認し、`RUN_TEACHER_ARTIFACTS=True`で6.2へ進む。
4. 全層Attentionを抽出し、CTC・unitsからteacherを作る。Attention/teacher/auditは`-all-layers`付きの独立directory/stageへ保存する。CTCは再利用する。
5. セクション7で`RUN_DISTILLATION=True`・`RUN_TRANSFER=True`・`RUN_GUIDANCE=True`を指定する。各runは同じdata/teacher/seed/初期化方針/optimizer/scheduler/更新予算を使い、methodを変える。従来のfresh initializationを維持する。
6. セクション8で4条件を同じnative generator・dev評価へ渡す。全条件の推論はteacher-free、beta=0。

3つの学習スイッチは既定Falseで、本学習を自動開始しない。`RUN_TRANSFER`はAnnealed Guidanceのスイッチ。`RUN_MONOTONIC`も任意controlとして残し、全層時には時間だけの単調priorを層方向へbroadcastする。これはT2TT層のコピーではない。

## CLI・互換性

`train_t2tt`・`train_s2ut`・`extract_t2tt_attention`・`visualize_attention`と`benchmark plan/train`に`--all-layers`を追加。CLIの既定はFalse、Notebookの主要比較だけTrue。

`--all-layers`と明示した`--layer`は競合する。暗黙の既定値-1は競合しない。`ALL_LAYERS=False`で従来single-layerの保存先と計算へ戻せる。設定変更後は保存先セルも再実行する。

旧single-layer Attention/teacherはsingle-layer modeで使用できる。旧metadataに`all_layers`がない場合もsingleとみなす。all-layer modeへ使うと再抽出を要求する明示エラーになる。scope・layer indices・headsをteacher signatureと学習contractへ反映し、異なる設定へのresumeを拒否する。

全層teacherをsingle-layer modeで使う場合も、`--layer N`に対応する`mt_attention_layers[N]`を選ぶ。暗黙の最終層利用は既定`layer=-1`の場合だけ。legacy学習と可視化で同じ選択を使い、元teacherファイルは変更しない。native経路のsingle-layerは従来どおり最終層に固定する。

native benchmarkはteacher/studentとも全head平均を使う。`heads != None`が記録されたteacherはplan時に拒否し、`--heads`を省略した再抽出を要求する。head ablationを行うlegacy経路の選択機能は維持する。

single-layer teacherを使う場合も、記録されたteacher層とstudent層の対応を検査する。`-1`と実際の最終層番号は同じ層として扱う。別層のteacherを指定すると停止し、対応層での再抽出を要求する。legacyのhead選択もteacher/studentで同じ集合を要求し、全head平均（`None`）とsubsetの混在を拒否する。層・headの記録がない旧packetは従来の最終層・全head平均として扱い、中間層やsubsetへ暗黙流用しない。既存artifactを書き換えない。

単層診断は`diagnose_alignment --layer N`で学習と同じ層を指定する（既定`-1`）。全層teacherから対応層を選び、投影prior・coverage・heatmapをその層で計算し、summary/pair metadataへ層番号とteacherのhead選択を記録する。`--all-layers`と明示した`--layer`は競合する。診断自体にはstudentがいないため、headはteacherに記録された選択を使う。

## Scope ablationの比較

通常の比較はscopeを固定する。層の範囲を独立変数として変える場合だけ、比較コマンドへ`--allow-scope-ablation`を指定する。比較結果に独立変数と各条件のscopeを記録する。

```bash
python -m alignment_transfer_s2ut.compare_experiments \
  outputs/baseline outputs/single-distillation outputs/all-distillation \
  --allow-scope-ablation --output outputs/scope-ablation.json

python -m alignment_transfer_s2ut.benchmark compare \
  outputs/registrations/baseline.json outputs/registrations/single.json outputs/registrations/all.json \
  --allow-scope-ablation --output outputs/native-scope-ablation.json
```

同一の全層teacherを両scopeで使う。同じmethod/priorのままscopeを変え、recipe・更新予算・係数・初期化・teacher・評価に対する既存照合は維持する。異なるteacher signatureのsingle/all抽出結果を、このflagだけで同一扱いにはしない。legacyでは異なるsingle-layer番号も明示したscope条件として比較でき、nativeではfinal-layerとall-layerを比較する。学習のresume contractを緩めるflagではない。

nativeのablation用学習は、両方の`benchmark train`で同じ`--teacher`を指定し、一方だけに`--all-layers`を付ける。Notebookのscope変更で別のteacherを再抽出した結果を、比較flagで自動的に同一扱いにはしない。

ColabのGPU preflightでは、3転移方式ごとにtiny 2層FP16モデルを1更新→checkpoint保存・復元→2更新目まで検査し、実forwardのbetaとpeak allocated/reserved VRAMをJSONで出力する。ローカルCPUでは実測しておらず、これは本番6層モデルのVRAMや品質の測定でもない。

```bash
python -m alignment_transfer_s2ut.extract_t2tt_attention \
  --manifest outputs/manifest.jsonl --checkpoint outputs/t2tt-200k/checkpoint_best.pt \
  --output outputs/mt-all-layers --all-layers --device cuda --resume

python -m alignment_transfer_s2ut.benchmark plan \
  --baseline outputs/registrations/baseline.json --manifest outputs/manifest.jsonl \
  --units outputs/units --teacher outputs/teacher-all-layers \
  --output outputs/all-layer-distillation --method distillation --all-layers
```

`benchmark train`も同じ引数を使う。`--lambda-align`・`--beta`・`--smoothing`の初期値は1.0・1.0・0.05で、最適値ではない。Notebookでは`ALIGNMENT_LAMBDA`・`TRANSFER_BETA`から変更できる。自動grid searchは行わない。

## Shape・正規化

| 用途 | Single-layer | All-layer |
|---|---|---|
| 抽出Attention | `attention [T,S]` | `attention_layers [L,T,S]` |
| teacher packet | `mt_attention [T,S]` | `mt_attention_layers [L,T,S]` |
| 音声時刻へ投影したprior | `[U,F]` | `[L,U,F]` |
| collated prior / raw / guided | `[B,U,F]` | `[B,L,U,F]` |

全層のcapture/guidance設定後にDecoderを1回呼ぶ。T2TTを層数回forwardしない。source/target時刻mappingも1回作り、各層へ同じmappingを適用する。EOS・padding・未知文字・無効rowの扱いは従来通り。

互換表示用に最終層の`attention`/`mt_attention`も保存するが、全層学習は必ずlayer配列を使う。metadataはscope、ordered layer indices、decoder_layers、heads、shape、dtype、checkpoint/hashを持つ。抽出AttentionのFP32を維持し、float16へ落とさない。

optimizer update内の全microbatchについて、層lのKL和を有効row総数N_lで割り、その後L層の平均を取る。総row平均やL倍の単純加算にはしない。N_l=0の層は0として層平均へ含める。native Trainerがmain token数で割るため、criterionへ各層の係数`total_main_tokens / max(1,N_l) / L`を渡す。teacherはdetach。GuidanceのbetaはLで割らない。

Distillationはsoft alignment supervision、Annealed Guidanceはtemporary direct guidance、Guidanceはpersistent direct guidance。正解word alignmentとは主張しない。研究効果・最適なlambda/betaは実データpilotで評価する。

## Memory・可視化

Attention保持量はおおむねL倍（6層ならAttention部分で約6倍）。モデル全体のVRAMが一律6倍になるわけではない。抽出の`max_attention_cells`はB×L×T×Sで判定し、同じ上限ならbatchを抑える。単一pairでも超える場合は停止する。学習の論理batchやレシピを自動変更しない。

adapterのprior/capture/raw/guidedはfinallyで解放する。beta=0で同じraw/guidedを二重stackしない。loggingはdetachし、teacherへ勾配を流さない。NPZは全層に加えて互換用最終層も持つため、サイズは文字数・層数・圧縮率に依存する。実GPUのVRAM増加量は未測定。

legacyとnative転移checkpointの`visualize_attention --layer N`は、同じ層Nのteacher・prior・raw・guidedを表示する。native可視化では同じrunの`native-run.json`とpreparedデータを使い、checkpointとのcontractを照合する。auditは層別有効row数・entropyも保存し、互換heatmapは最終層であることをsummaryへ記録する。native研究モデルの診断は`alignment/kl_layer_N`・`alignment/kl_mean`を参照する。


## 変更ファイルと役割

| ファイル | 変更概要 |
|---|---|
| `s2ut/model.py`, `s2ut/fairseq_extension.py`, `s2ut/smoke.py` | 全層を1 forwardでcapture/guidance、beta=0のnative path、状態解放 |
| `training.py`, `guidance/attention.py` | 全層抽出・mask・memory budget・legacy学習・層別row正規化/logging |
| `alignment/layers.py`, `alignment/projection.py`, `preprocessing.py` | extraction signature、全層packet、共有時刻mapping、旧single互換 |
| `s2ut/benchmark_extension.py` | native dataset/collater/model/criterion/Trainerの全層対応 |
| `benchmark.py`, `cli.py`, `compare_experiments.py` | 新option、scope/teacher contract、native forwardingと比較判定 |
| `visualization/attention.py`, `diagnose_alignment.py` | 同じ層の可視化、native checkpoint対応、層別audit |
| `configs/experiments/conditions.yaml` | 3つのall-layer条件追加、旧条件保持 |
| `notebooks/colab/benchmark_comparison.ipynb` | 主要4条件・独立stage・GPU接続試験・共通評価 |
| `tests/test_all_layers.py`, `tests/test_all_layers_native.py`, `tests/test_all_layers_notebook.py` | shape/1 forward/mask/投影/勾配/正規化/旧artifact/native学習/推論/可視化/CLI/Notebook/GPU試験 |
| `tests/test_benchmark_native.py`, `tests/test_notebook_commands.py` | 旧テストを保ち、2層fixtureとscope選択を追加 |
| `README.md`, `docs/METHOD.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, `docs/USAGE.md`, `docs/BENCHMARK_COMPARISON.md`, `docs/T2TT_EXTENSION.md`, `docs/DEVIATIONS.md`, `docs/VALIDATION.md`, 本書 | 手順・数式・互換性・検証と実験範囲 |

file欄のPython pathは`src/alignment_transfer_s2ut/`からの相対パス。既知の未測定項目は実コーパスでの研究効果・GPUの数値/VRAMであり、テスト値をBLEUや研究結果として扱わない。学習途中の旧single-layer runを別implementationへ無条件resumeせず、同じrevisionを使う。T2TT延長は元sourceを隔離する既存のcontinuation処理を使う。
