# Verified T2TT 20k → 200k extension

保存済み20k T2TTの**同じ学習**を200kまで継続する。新しいモデルの初期化学習や20k stageの書換えは行わない。200kの学習結果・BLEU値は、実際にColabで実行した結果で判断する。

## Colabでの順番

`notebooks/colab/benchmark_comparison.ipynb` を最新版で開く。

1. Driveを接続し、学習時と同じ `ROOT`・`PERSISTENT`・`BENCHMARK` を設定する。
2. 設定と直下のT2TT保存先セルを実行する。既定は下表のとおり。
3. セクション2で環境検査、3〜5で元と同じ入力・Baseline metadata・manifest・backboneを準備する。
4. **6.1a** を実行する。初回はverified 20kからseedし、2回目以降は200k stage自身から再開する。
5. **6.1b** で200k完了を検証し、選択したbestのin-domain dev BLEUを測る。
6. **6.1c** の `RUN_EXTERNAL_TEXT=True` で、同じ保存済み8C 300件の外部BLEUと翻訳例を表示する。
7. **6.1d** でloss/LRの曲線と、保存済み20k・200k-bestの比較表を確認する。
8. 確認後、設定セルの `RUN_TEACHER_ARTIFACTS=True` を実行して **6.2** へ進む。
9. Attention・CTC・teacher alignment生成・診断を経て、必要ならS2UTへ進む。

既定の「すべて実行」は6.2で停止する。6.2は200k完了と、選択中のbestに一致するdev/外部評価結果も検査する。Monotonic-onlyの設定はこのteacher gateを通らず、従来どおりteacher処理をスキップする。

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `T2TT_BASE_UPDATES` | 20000 | sourceの完了予算。実際のlast stepも照合 |
| `T2TT_TARGET_UPDATES` | 200000 | 新stageの終了上限 |
| `RUN_T2TT_EXTENSION` | True | 6.1aの延長学習を実行 |
| `USE_EXTENDED_T2TT` | True | 評価・後続処理で200k stageのbestを選択 |
| `T2TT_SNAPSHOT_UPDATES` | 10000 | full Drive ZIPのupdate間隔 |
| `T2TT_KEEP_LAST` | 2 | 最新の保持個数。milestoneは別途保持 |
| `RUN_TEACHER_ARTIFACTS` | False | 評価確認後にAttentionへ進むための明示スイッチ |

`T2TT_UPDATES` は旧20k予算の互換aliasであり、200kの終了上限には使用しない。`USE_EXTENDED_T2TT` を切り替えたら、直下の保存先セルも再実行する。学習済み200kを評価するだけなら `RUN_T2TT_EXTENSION=False`、`USE_EXTENDED_T2TT=True` とする。20kの再学習は起動しない。

## 検証するsource

sourceは `PERSISTENT/stages/t2tt` のlatest **verified** ZIP。固定benchmark revisionの `latest/restore` をそのまま使い、ZIP SHA256・member一覧・各ファイルSHA256を確認する。

- snapshotのbenchmark revision、training module、manifest/configの論理パスと内容hash、output、planned max updatesを照合する。interpreterを含む元の入力identityも照合する。
- `checkpoint_last.pt`・`checkpoint_best.pt`・`experiment.json`・`metrics.jsonl` を必須とする。
- snapshotのupdatesとlastの実stepが20,000であることを確認する。bestは同じcontractで、stepがlast以下でなければならない。
- lastのmodel/Adam state、epoch、cursor、best loss、bad validations、Python/NumPy/Torch/CUDA RNGを必須とする。
- last・best・experimentのscientific contract本体とそのhashを照合する。元の学習ログはstep 1からlastまで連続したprefixでなければならない。

不足・破損・異なる設定を検出した場合は停止する。update 0から再学習したり、bestからresumeしたり、contract hashを書き換えたりしない。

## 元の学習コードを維持する理由

trainerのscientific contractは全Python sourceの `implementation_hash` を含む。現在のリポジトリへ評価・保存の機能を追加しただけでも、その値は20k時点と異なる。hashを上書きしてresume制約を解除すると、同一implementationの保証が失われる。

extensionはcheckpointのhashと一致する完全なsource treeを探す。現在のtreeが一致すればbyte単位でコピーし、異なる場合はGit履歴を照合して元revisionをexportする。隔離先は新stage内の `.training-runtime/`。一致しなければ停止する。

隔離した元のCLI/parser/model/trainerを別processで実行する。初回とresume時には、まず終了上限を現在の保存stepにした**optimizer更新0回のpreflight**を実行する。元trainer自身のcontract検査・optimizer/RNGロードを通し、target contractがsourceと等しいことを確認する。checkpointのbytesは変更しない。その後、同じlastを再ロードして本継続を開始する。

変更する科学的な実行条件は終了上限だけ。manifest、architecture、tokenizer、seed、LR/warmup、batch/update-frequency/max-tokens、clip、label smoothing、validation/save interval、patience、backend、train/dev、implementationは保持する。LRはglobal optimizer updateから続き、warmupはリセットしない。max updatesは従来どおりscientific contractに含めず、新stageのexperimentにだけ `planned_max_updates=200000` を保存する。

## 完全なcontinuationとbest選択

resume元は **`checkpoint_last.pt`**。model、optimizer、step、epoch/cursor、最良dev loss、bad validations、4種類のRNGを元trainerがそのまま復元する。

`checkpoint_best.pt` もそのままseedする。20k以後のdev/main_lossがさらに改善した場合だけ、元trainerが更新する。200kでの最終teacherは「200kまでに観測した最良dev lossのbest」であり、lastへ置き換えない。best stepが20kより前のままになる場合もある。

`extension-lineage.json` はsource stage/snapshot identity・ZIP SHA・last/best SHA・source/target契約hash・source/target予算を保存する。`extension-plan.json` は元command、隔離implementation、実行・保存policyを記録する。両方とmilestone記録・control scriptを各復旧ZIPにも含める。

## 保存頻度と保持

validation=1000、VM local checkpoint save=100は元のrecipeを変更しない。元trainerの**保存通知callbackだけ**を延長process内で包み、Driveへのfull ZIPを既定10000 updates、milestone、最終updateで公開する。local progressは従来のschemaで更新する。5000など、local save間隔の倍数へ変更できる。

sourceコードファイル・model/loss/optimizer処理は変更しない。queue・freeze・upload・checksum・復旧には固定benchmarkの実装を使う。元20k stageとS2UT stageのcallbackや保存頻度には影響しない。

新しいZIPのuploadとchecksum検証が成功した後に、**新stageだけ**を最新2個＋50k/100k/150k/200k milestoneへ整理する。20k sourceは一切pruneしない。不明なファイルを含むdirectoryは保持し、削除失敗は記録して継続する。

`extension-milestones.json` に20kおよび各milestoneの実best loss/best stepを記録する。各milestoneのZIPにはその時点のbestも入るため、後から同じ固定300件で再評価できる。

異常終了の最終backupでは、復旧checkpointより後の未確定metrics行を**freezeされたbackupコピーだけ**から除外する。VM上の原ログは保持し、再開後にstepの重複を作らない。曲線reportもupdateの欠落・重複を警告する。

## 独立した保存先

| 用途 | VMローカル | Drive stage |
|---|---|---|
| source 20k | `ROOT/t2tt`（extensionからread-only） | `PERSISTENT/stages/t2tt`（read-only） |
| extended training | `ROOT/t2tt-200k` | `PERSISTENT/stages/t2tt-200k` |
| extended dev | `ROOT/text-evaluation/t2tt-200k/dev-text.json` | `t2tt-dev-200k` |
| extended FLEURS | `ROOT/text-evaluation/t2tt-200k/flores-fleurs/flores-fleurs-text.json` | `t2tt-flores-fleurs-200k` |
| Attention | `ROOT/t2tt-attention-200k` | `t2tt-attention-200k` |
| CTC | `ROOT/speech-text-alignment-200k` | `speech-text-alignment-200k` |
| projected teacher | `ROOT/teacher-alignment-200k` | `teacher-alignment-200k` |
| alignment audit | `ROOT/audit-200k` | `diagnostics-200k` |

上表のstage略記はすべて `PERSISTENT/stages/` 以下。20kの従来の保存先は変更しない。FLEURSの入力は共通の `ROOT/external-text-input/fixed-8c.json` で、保存済み8Cの同じ300件を再利用する。

`T2TT_CHECKPOINT = T2TT_ACTIVE_ROOT / 'checkpoint_best.pt'` がdev・外部評価・Attentionの唯一の参照元。teacher作成とS2UTも選択中のAttention/teacher directoryを参照する。extended teacherを使うS2UT結果は `annealed-t2tt-200k` へ分離する。

## 再開

同じrevision・設定・入力パスでNotebookを再実行する。`t2tt-200k` 自身にverified snapshotがあれば、source stageを読み直さず、そこからlastを復元して続ける。必要な元implementationも保存したrevision/hashで再構築する。完了済み200kではoptimizer更新を再実行しない。

新stageにまだverified snapshotがなければ、sourceのverified 20kからseedする。未公開のローカル作業は元のbenchmark復元規則で `.interrupted-*` に保持する。新stageのZIPがすべて破損している場合は20kへの巻戻しを自動実行せず停止する。最後のdurable ZIP以降の更新は、VM消失時に再計算する。

## CLI

```bash
python -m alignment_transfer_s2ut.t2tt_extension extend \
  --source-work /local/work/t2tt \
  --source-backup /drive/experiment/stages/t2tt \
  --work /local/work/t2tt-200k \
  --backup /drive/experiment/stages/t2tt-200k \
  --manifest /local/work/manifest/manifest.jsonl \
  --config /local/work/backbone.json \
  --source-updates 20000 --target-updates 200000 \
  --snapshot-updates 10000 --keep-last 2
```

Colabでは6.1aの共通logger＋GPU監視経由で実行する。通常の `train_session()` に20k directoryのまま別budgetを渡す方法は使用しない。

```bash
python -m alignment_transfer_s2ut.t2tt_extension status \
  --work /local/work/t2tt-200k --target-updates 200000
python -m alignment_transfer_s2ut.t2tt_extension report \
  --work /local/work/t2tt-200k --output /local/work/t2tt-200k/diagnostics
```

reportはtrain/main_loss・dev/main_loss・train/learning_rateをoptimizer updateに対して描く。20k比較欄はVM上にある従来の保存JSONから読み、欠落した値は空欄とする。得点をハードコードしない。

## CPU継続試験の実行条件

縮小モデルの完全一致試験では、基準を学習するpytest processと、延長を学習する新しいPython processのCPUスレッド数をともに1へ揃える。`torch.set_num_threads(1)` は子processへ継承されないため、試験fixture内だけで `OMP_NUM_THREADS=1`・`MKL_NUM_THREADS=1` も設定し、終了時に元へ戻す。比較には引き続き `torch.equal` を使用する。本番のT2TT/S2UT学習・Notebook全体のスレッド数は変更しない。
