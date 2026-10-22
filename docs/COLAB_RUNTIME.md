# Colabの入出力・保存・再開

比較用の`benchmark_comparison.ipynb`は、動作実績のあるbenchmarkの実装を固定revisionで直接利用する。新たな学習レシピやDrive保存方式を独立に作り直さない。

参照元は[ja-en-direct-s2st-benchmark / 4361e55](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/tree/4361e55a99c7f4ea3a00c5c138f6820c34a07d06)。Colabが別のローカルcheckoutへ取得・editable installし、起動時にrevisionとtracked sourceの変更を検査する。benchmark checkoutや元データの編集は行わない。

| 項目 | 使用するbenchmarkの実装 |
|---|---|
| 音声ZIP・SHA256・ローカル展開・論理path mapping | `drive_staging.stage_audio` / `active_map` |
| 常駐学習process・単一upload worker・最大2個の待機snapshot | `session_runtime.run_continuous` / `train_runtime.checkpoint_saved` |
| snapshot用の完全checkpoint・best・選択状態の固定コピー | `recovery.freeze_files`＋alignment固有のmetadata名 |
| ZIP作成・容量予算・copy readback・最後に完了marker | `colab.publish_packed` |
| 完了済みsnapshotの検証・復元 | `colab.latest` / `restore` |
| S2UTのcache・epochごとのworker調整・確認済みFP16 overflow再試行・勾配監査 | `s2ut.performance.runtime_hooks` / `TrainingAudit` |

`ROOT`は`/content/alignment-transfer-work/benchmark-match-drive-safe-001`、`PERSISTENT`はDrive上の専用ディレクトリ。checkpoint・ログ・NPZ・音声の頻繁な書き込みはROOTへ行う。`PERSISTENT/stages/<stage名>/`へ検証済みZIPだけを保存する。旧Drive直接保存runはそのまま保持し、新しい保存先から開始する。旧checkpointを無条件に採用して学習条件を変えない。

## 通常の再開

GPUランタイムを選び、同じコードrevision・ROOT・PERSISTENT・学習設定で上から実行する。消えたPython環境を構築し、入力ZIPと保存stageをローカルへ復元する。学習では最後に検証済みZIPへ保存された更新数からoptimizer/scaler・iterator・RNG・best選択状態を復元する。ローカルの更新数とDriveへ保存済みの更新数は区別する。

学習backupの通常間隔はbenchmarkと同じ600秒。dev評価地点・最終更新では強制backupを行う。容量予算は1024 GBで、上限超過時は既存復旧点を保持して停止する。自動削除や無断の予算拡大を行わない。VM強制削除では未永続化のローカル進捗が失われ、直前の完了ZIPから再開する。Drive自体の障害や転送量制限をなくす保証ではない。

GPUで原本のWAVからZIPを作らない。benchmarkが作ったtrain/devの日英WAV packがあれば再利用する。不足する場合だけ、CPUランタイムで`PREPARE_AUDIO_PACKS=True`にして入力準備セルまで実行する。同じbenchmark関数で原本を一度読み、完成済みpackを再利用する。準備後はGPUへ切り替え、Falseへ戻して通常実行する。testの音声packは比較の前処理に要求しない。

## alignment固有の薄いadapter

元の音声・TSV・辞書の論理pathはデータ識別として保持し、物理的な読み込みだけを検証済みローカルコピーへ向ける。native fairseqでは辞書・multitask設定内の物理pathとTSV読み込みをscoped adapterで解決する。学習targetや文字辞書・補助lossを再構築しない。未stagingのDrive音声・unit payloadへ黙ってfallbackしない。unit importはbenchmarkのinline unit形式を要求する。

T2TTは既存のtext trainerを使うため、保存完了通知だけをbenchmarkのsnapshot protocolへ接続する。S2UTの本比較はbenchmarkのperformance hooksと公式fairseq Trainerを共用する。追加physical microbatch分割や別precisionは導入しない。

pair単位のAttention・CTC・unit timing・teacherにはbenchmarkの学習checkpointに対応物がないため、保存済みファイルの増分packだけをadapterで追加する。ZIP publisher・checksum・復元はbenchmarkのものを使用する。通常600秒ごととstage終了・エラー終了時に変更分をまとめ、既に保存したpairを次のpackへ複製しない。完成済みstageは入力file hashを照合して再利用する。新VMでは増分packを順に検証復元する。中断した未完了pairと途中ASR評価は再計算する場合がある。

旧`pilot.ipynb` / `pipeline.ipynb`は独立trainerの参考・機能診断用で、このDrive-safe運用の入口ではない。実際の比較と通常再開には`benchmark_comparison.ipynb`を使用する。
