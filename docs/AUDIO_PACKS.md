# CPU音声ZIPの並列作成と再開

## Colabでの操作

1. 実行中のCPU作成セルを停止し、停止が完了するまで待つ。同じ保存先を処理するCPU VMは1つにする。
2. GitHubの最新 [training.ipynb](../notebooks/colab/training.ipynb) をColabで開く。既存のColabコピーには新設定が自動反映されない。
3. 元と同じEXPERIMENT_ID、T2TT設定、DATA_CONFIGまたはコーパス・manifestのOverrideを設定する。入力manifestの内容・行順・保存先を変えない。ZIP、plan、receiptは削除しない。
4. CPUランタイムで `PREPARE_S2UT=True`、`PREPARE_AUDIO_PACKS=True`、`AUDIO_PACK_WORKERS=2` にして「すべてのセルを実行」。必要ならworkersを1または4へ変更できる。
5. 音声ZIPの準備が終わったらGPUへ切り替え、同じ設定で `PREPARE_AUDIO_PACKS=False` にして再実行する。GPUでは作成済みの音声ZIPを必要になった分から取得する。 [Colabの入出力と再開](COLAB_RUNTIME.md)

完成済みZIPは検証して再利用し、残りだけを作る。旧4361e55実装のZIPも使用できる。既存の11個のreceiptと、12個目のZIP/planだけがある状態を再現し、対応する原本WAVを読めなくして4並列で再開した。12個とも内容・mtimeは変化せず、残り5個を作成して全17個が完成した。

pack_filesは128を維持し、workersを保存データのidentityへ追加しない。namespace、アーカイブ名、内部WAV名・順番、SHA-256、plan/receiptの形式は従来のまま。破損・出所不明の未完了ZIPは保持してエラーにする。

## 実装

1 ZIPを1 jobとして、ThreadPoolExecutorへ最大workers個だけ投入する。ZIP内は従来の順序でストリーミング読み込みし、原本のstampとSHA-256を照合する。workerごとにVM内の一時ディレクトリを使い、各workerの一時ZIP容量を合算して空き容量を検査する。ZIP_STORED、Driveコピー後のreadback SHA-256検査、receiptを最後に書く公開手順を維持する。

同一VM内の同じZIPへの二重書き込みはOS lockで拒否する。複数VMにまたがるDriveFSの分散lockはない。失敗時は新規投入を停止し、他workerへ中止を通知して終了を待つ。Python例外・中断時には所有する一時ファイルを片付け、公開済みZIPは残す。VM強制削除時の未公開ファイルを完了扱いにはしない。

進捗のchecked/completedは作成または再利用検証が完了した数。10秒間隔と開始・終了・失敗時に、workers、built/reused/failed、ZIP/s・MiB/s、ZIP作成・原本検証・Driveコピー・readback検証時間をstderrへ出す。Notebookの共通ロガーが画面とファイルへ保存する。原本検証時間はZIP作成時間に含まれ、各時間の合計はwall timeを意味しない。再利用時のMiB/sはアップロード速度ではない。

## 固定revision

- benchmark runtime: `0c0024ea888206149a3119bf096b86ba66d63b66`
- benchmark mainへの統合: `9f04d6c97948f9b371626aef487af65acbab57c5`

runtimeは旧固定4361e55を直接の親とし、音声ZIPの実装・検査・測定と文書だけを変更したコミット。最新mainの別の学習・推論変更は取り込まない。training/evaluateは新runtimeを固定SHAで取得し、runtime()とenvironment検査が実際のインストールrevisionを確認する。

artifact/recipe lineageを表すBENCHMARK_REVISIONは旧4361e55を維持する。これによりcommonなど既存snapshotのidentityを変更しない。実行コードの検査には新しいBENCHMARK_RUNTIME_REVISIONを使う。既存の学習checkpointのcontractを緩めたり書き換えたりはしない。旧版で学習中のrunは元のrevisionで継続する。旧T2TT実装を復元して新しく延長するときは、学習packageのhashとcheckpoint contractを保ったまま、controllerが独立して検証したruntimeへ保存通知を接続する。旧trainerのruntime検査を緩めたり、元のsource treeを修正したりしない。

## ローカル性能測定

Windows、16 kHz/mono/PCM16・1秒WAV 1,024個、128 WAV/ZIPで8 ZIP。各3回、実行順を回転した中央値。Driveではなくローカルの原本と保存先を使用した。

| workers | 秒 | ZIP/s | MiB/s | process CPU % | 一時ZIP peak MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 3.591 | 2.228 | 8.754 | 104.3 | 3.93 |
| 2 | 2.181 | 3.668 | 14.413 | 187.4 | 7.86 |
| 4 | 1.585 | 5.046 | 19.831 | 304.9 | 15.72 |

エラー0。CPUはprocess CPU秒/wall秒、temporary peakは10 ms間隔のsampling。DriveFS上の性能・アクセス制限への影響は未測定で、同じ改善率を保証しない。既定は安定性を考慮して2とする。

再測定用CLIはbenchmarkの `scripts/benchmark_audio_packs.py`。詳細とbenchmarkの検証記録は [benchmark文書](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/0c0024ea888206149a3119bf096b86ba66d63b66/docs/AUDIO_PACK_PARALLEL.md)。

## 変更ファイル

| Repository | ファイル |
|---|---|
| benchmark | `src/direct_s2st/drive_staging.py`、`src/direct_s2st/audio_pack_runtime.py` |
| benchmark | `tests/unit/test_drive_staging.py`、`tests/unit/test_audio_pack_parallel.py` |
| benchmark | `scripts/benchmark_audio_packs.py`、`docs/AUDIO_PACK_PARALLEL.md` |
| alignment-transfer | `src/alignment_transfer_s2ut/colab_runtime.py`、`src/alignment_transfer_s2ut/environment.py`、`src/alignment_transfer_s2ut/t2tt_extension.py` |
| alignment-transfer | `notebooks/colab/training.ipynb`、`notebooks/colab/evaluate.ipynb` |
| alignment-transfer | `tests/test_training_notebook.py`、`tests/test_benchmark_colab.py`、`tests/test_t2tt_extension.py`、`tests/test_pipeline.py` |
| alignment-transfer | `README.md`、`docs/COLAB_RUNTIME.md`、`docs/BENCHMARK_COMPARISON.md`、`docs/AUDIO_PACKS.md`、`docs/VALIDATION.md` |

検証結果は [VALIDATION.md](VALIDATION.md)。研究上のモデル・損失・校正・学習予算は変更しない。
