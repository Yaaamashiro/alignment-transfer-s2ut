# Colabの入出力・保存・再開

全段階の画面表示・実行記録・無出力時の生存通知は[共通ログ仕様](LOGGING.md)を参照。

学習は [training.ipynb](../notebooks/colab/training.ipynb)、推論・比較は [evaluate.ipynb](../notebooks/colab/evaluate.ipynb) から実行する。動作実績のあるbenchmarkの実装を固定revisionで利用する。

参照元は[ja-en-direct-s2st-benchmark / 0c0024e](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/tree/0c0024ea888206149a3119bf096b86ba66d63b66)。Colabが別のローカルcheckoutへ取得・editable installし、起動時にrevisionとtracked sourceの変更を検査する。runtimeは旧4361e55へ音声ZIP処理だけを追加した固定コミットで、学習・評価実装を変更しない。既存snapshotのartifact/recipe lineageはBENCHMARK_REVISION=4361e55を維持し、実行するコードはBENCHMARK_RUNTIME_REVISION=0c0024eで検査する。インストール先checkoutや元データの編集は行わない。

ローカルで同じruntimeを導入する場合は、alignment用Python環境を有効にして次を実行する。配置先は既存のbenchmark作業checkoutとは別にする。

```bash
git clone --no-checkout https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git ../alignment-benchmark-runtime
git -C ../alignment-benchmark-runtime checkout --detach 0c0024ea888206149a3119bf096b86ba66d63b66
python -m pip install PyYAML==6.0.2 --no-deps -e ../alignment-benchmark-runtime
python -m alignment_transfer_s2ut.environment --native --benchmark --cuda
```

比較ノートブックも前処理より先にこの環境検査を実行する。Python 3.10、依存metadata、主要ライブラリのimport、公式fairseqのrevision、コンパイル済みbatch samplerの実行、benchmarkのrevision、CUDA接続を検査し、不足時は停止する。外部モデルの取得・実音声処理はこの検査に含まれない。

CLI subprocessの図はファイルへ出力するため、Notebookの設定セルで`MPLBACKEND=Agg`を指定する。Colab kernelの`module://matplotlib_inline.backend_inline`を別venvへ引き継ぐと、matplotlibのimportが失敗する場合がある。既に開いているNotebookでこのエラーが出た場合は、`import os; os.environ["MPLBACKEND"] = "Agg"`を実行して環境検査セルから再開する。コマンド失敗時は画面にも末尾のログを表示し、全文は`ROOT/execution/`へ保存する。

| 項目 | 使用するbenchmarkの実装 |
|---|---|
| 音声ZIP作成・索引・使用時の検証付き展開 | `drive_staging.stage_audio` / `active_map` / `stream_inputs.Downloader` |
| 常駐学習process・単一upload worker・最大2個の待機snapshot | `session_runtime.run_continuous` / `train_runtime.checkpoint_saved` |
| snapshot用の完全checkpoint・best・選択状態の固定コピー | `recovery.freeze_files`＋alignment固有のmetadata名 |
| ZIP作成・容量予算・copy readback・最後に完了marker | `colab.publish_packed` |
| 完了済みsnapshotの検証・復元 | `colab.latest` / `restore` |
| S2UTのcache・epochごとのworker調整・確認済みFP16 overflow再試行・勾配監査 | `s2ut.performance.runtime_hooks` / `TrainingAudit` |

`ROOT`は`/content/alignment-transfer-work/<EXPERIMENT_ID>`、`PERSISTENT`はDrive上の専用ディレクトリ。checkpoint・ログ・NPZ・音声の頻繁な書き込みはROOTへ行う。`PERSISTENT/stages/<stage名>/`へ検証済みZIPだけを保存する。旧Drive直接保存runはそのまま保持し、新しい保存先から開始する。旧checkpointを無条件に採用して学習条件を変えない。

## 通常の再開

GPUランタイムを選び、同じROOT・PERSISTENT・学習設定で上から実行する。Python環境を構築し、完了済みの前処理と最後の検証済みcheckpointを復元する。S2UTは保存されたoptimizer/scaler・iterator・RNG・best選択状態から続ける。音声ZIPは必要になったものから取得し、次のバッチの読み込みはfairseqのデータローダーが学習計算と並行して進める。

pair出力のNPZ本体を保存した後、外部checksumを書く前に切断された場合は、再開時に内部payloadとmetadataを検証して不足したchecksumだけを補う。元NPZと入力artifactは書き換えない。正常終了時にchecksum不足が残っていれば、完了ZIPの公開を拒否する。旧版で完了とされたAttention・CTC・unit・teacher stageは初回に全pairを再照合し、不足分だけを作成する。再検証済みの完了markerを新しい増分ZIPへ保存した後は通常どおりスキップする。旧ZIPとBaseline backupは変更しない。

学習backupの通常間隔はbenchmarkと同じ600秒。dev評価地点・最終更新では強制backupを行う。容量予算は1024 GBで、上限超過時は既存復旧点を保持して停止する。自動削除や無断の予算拡大を行わない。VM強制削除では未永続化のローカル進捗が失われ、直前の完了ZIPから再開する。Drive自体の障害や転送量制限をなくす保証ではない。

GPUで原本のWAVからZIPを作らない。benchmarkが作ったtrain/devの日英WAV packがあれば再利用する。不足する場合だけ、CPUランタイムで`PREPARE_AUDIO_PACKS=True`にして入力準備セルまで実行する。同じbenchmark関数で原本を一度読み、完成済みpackを再利用する。準備後はGPUへ切り替え、Falseへ戻して通常実行する。testの音声packは比較の前処理に要求しない。

GPUの「3. 音声と学習データ」では、manifestからZIPの索引を作る。Baselineの起動時は索引がprepared TSVの音声パスを網羅しているか確認し、音声ZIPと各WAVのSHA-256は初回読み込み時に検証する。同じキャッシュを使う学習ワーカーの同時転送は2本まで。同じVMでは検証済みの展開結果を再利用する。キャッシュは参照した分だけ増えるため、実験を通して使用する音声と派生物を格納する容量が必要。初回のunit抽出・音声検証は全入力を処理する。

`ROOT/execution/*.log`に転送開始・検証済みZIP数・再利用状況を保存する。開始時にログパスを表示し、JSON結果は別の`.stdout.log`へ記録する。切断時は最後にDriveへ保存できたcheckpointが再開点になる。

## alignment固有の薄いadapter

元の音声・TSV・辞書の論理pathはデータ識別として保持し、物理的な読み込みだけを検証済みローカルコピーへ向ける。native fairseqでは辞書・multitask設定内の物理pathとTSV読み込みをscoped adapterで解決する。学習targetや文字辞書・補助lossを再構築しない。未stagingのDrive音声・unit payloadへ黙ってfallbackしない。unit importはbenchmarkのinline unit形式を要求する。

T2TTは既存のtext trainerを使うため、保存完了通知だけをbenchmarkのsnapshot protocolへ接続する。S2UTの本比較はbenchmarkのperformance hooksと公式fairseq Trainerを共用する。追加physical microbatch分割や別precisionは導入しない。

pair単位のAttention・CTC・unit timing・teacherにはbenchmarkの学習checkpointに対応物がないため、保存済みファイルの増分packだけをadapterで追加する。ZIP publisher・checksum・復元はbenchmarkのものを使用する。通常600秒ごととstage終了・エラー終了時に変更分をまとめ、既に保存したpairを次のpackへ複製しない。完成済みstageは入力file hashを照合して再利用する。新VMでは増分packを順に検証復元する。中断した未完了pairと途中ASR評価は再計算する場合がある。

旧`pilot.ipynb` / `pipeline.ipynb`は独立trainerの参考・機能診断用で、このDrive-safe運用の入口ではない。通常再開にはtraining/evaluateを使用し、旧入口はnotebooks/legacyに保存する。

CPU音声ZIPの並列化・既存ZIPを使った再開・測定結果は [音声ZIPの並列化](AUDIO_PACKS.md) を参照。
