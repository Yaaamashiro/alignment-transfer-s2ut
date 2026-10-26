# 実行ログ

3つのColab Notebookでは、環境構築から前処理・学習・推論・評価まで、すべての外部コマンドを共通loggerへ渡す。学習を始める前のgit・uv・pip・テスト・設定取得も対象とする。

## 表示と記録

| 状態 | 表示する内容 |
|---|---|
| `started` | 処理名、全文ログの保存先 |
| `running` | 起動したPID、経過時間、子processの出力 |
| 無出力が10秒続いた場合 | `no_output`と生存通知。進捗が確認できたとは扱わない |
| `completed` | 終了コード0、経過時間、ログの保存先 |
| `failed` / `interrupted` | 終了コードまたは例外、経過時間、ログの保存先 |
| `skipped` | 無効化された条件やcheckpoint待ちなどの理由 |

子processの各行は間引かず、NotebookのPythonから明示的に表示する。同じ内容をローカルファイルへ逐次flushする。Python子processには`PYTHONUNBUFFERED=1`を渡す。CPUテストはテスト名が分かるverbose表示を使う。

生存通知だけでは、転送・最適化・検証が前進したと判断しない。実際の前進は、処理が出す`checked`・`zip`・pair数・学習update等を確認する。

## 保存先

| Notebook | `LOGS`の既定値 |
|---|---|
| benchmark_comparison | `ROOT/execution/` |
| pilot | `/content/alignment-transfer-logs/pilot/<ROOTの名前>/` |
| pipeline | `/content/alignment-transfer-logs/pipeline/<ROOTの名前>/<recipe>/` |

Drive接続前後の通知は`/content/alignment-transfer-logs/<Notebook名>/bootstrap/`へ保存する。頻繁なログ書き込み先はVMローカル。ログそのものはDrive snapshotへの保存を保証しないため、必要な記録はVM削除前に保持する。学習checkpointのZIP保存・復元は従来のbenchmark runtimeを使用する。

試行ごとに次のファイルを作る。

- `*.log`：開始・生存通知・子processの出力・終了状態を含む全文ログ。
- `*.json`：処理名、実行引数、開始時刻、PID、経過秒数、状態、終了コード、ログの場所。実行中も更新する。
- `*.stdout.log`：JSONを返すコマンドのstdout。人向けのstderrと分離する。
- `*-gpu.csv`：GPU監視を有効にした試行のdevice使用率・メモリ量。
- `notebook-events.log`：Drive接続・明示的なskip等の通知。

JSONを返す入力準備・runtime検査・planも、stderrを表示しながらstdoutだけを別ファイルで受け取る。結果がJSONとして読めない場合も失敗として記録する。

## 中断と再開

中断時はloggerが起動した子processを終了する。Colab/Linuxでは外側のloggerが専用process groupを作り、入れ子のlogger・学習workerが別のgroupへ逃げないようにする。既存の入力pack・checkpoint・Drive原本をloggerから削除しない。

実行中のセルはGitHub更新で変わらない。Notebookのコードと固定revisionを更新するときは、旧processの停止を確認してから再開し、同じ入力準備を同時に起動しない。学習開始後のrunは保存したコードrevisionを維持する。新しい実装を別の学習runへ無条件に適用しない。

## 実装と検査

共通実装は`src/alignment_transfer_s2ut/process_logging.py`。標準ライブラリのみで動くため、Colabのインストール前セルにも同じ関数を埋め込む。`pilot.run_logged`も同じ実装を利用する。

テストで、Notebook内の実装が共通sourceと一致すること、通常セルに直接のsubprocess呼び出しが残らないこと、完了前の出力・無出力通知・JSON分離・失敗と中断の扱いを確認する。モデル・loss・batch・学習budget・benchmarkの保存方式は変更しない。
