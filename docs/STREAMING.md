# 学習入力の先読みとローカルキャッシュ

必要な入力から読み出し、現在のbatchを学習している間にCPU workerが次の入力を準備する。全データをVMへコピーしてから学習を始める手順は不要。

```text
Drive上の音声・unit・teacher
          ↓ 必要なsampleだけ取得・検証
    CPU worker × 2 ── 未拡張FBANK/CMVN・artifactのlocal cache
          ↓ 順序を保つ、上限付きの先読みqueue
    現在のoptimizer更新用microbatches
          ↓ main threadでSpecAugment → pinned memory転送
        GPUで学習
          └─ この間にもCPU workerが次の入力を取得・準備
```

対象は、音声と前処理済みunit/teacherが元の保存先に存在する学習stage。CTC・HuBERT・teacherの生成は学習前に実行する。manifestの選択・hash生成や前処理CLIそのものを、GPU学習と同時実行する機能ではない。

## 設定

```bash
python -m alignment_transfer_s2ut.train_s2ut --trainer legacy \
  --manifest outputs/manifest.jsonl --config configs/s2ut/fisher.yaml \
  --units outputs/units --teacher outputs/teacher \
  --recipe strong --method annealed_guidance --max-updates 10000 --anneal-steps 5000 \
  --loader-workers 2 --prefetch-batches 2 \
  --cache-dir /content/alignment-transfer-cache --cache-gb 8 --cache-reserve-gb 1 \
  --pin-memory --device cuda --output outputs/annealed_t2tt
```

| 引数 | 意味 |
|---|---|
| `--loader-workers` | CPU読み込みthread数。通常CLIの既定0は同期処理、pilotの既定は2 |
| `--prefetch-batches` | 指定batch-size換算で保持する先読み量。既定2 |
| `--cache-dir` | VM上の専用cache directory。未指定なら永続cacheなし |
| `--cache-gb` | cache payloadの上限GiB。既定8 |
| `--cache-reserve-gb` | 書き込み時に残す空きディスクGiB。既定1 |
| `--pin-memory` | main threadからCUDAへpinned memory・non-blocking転送 |

batch size 8・prefetch 2では最大16 sample分を先読みする。これに加えて、現在のoptimizer更新に必要なupdate-frequency分のmicrobatchを保持する。実際のbyte数は音声・unit長に依存する。

[専用pilot Colab](../notebooks/colab/pilot.ipynb)は既定2 worker・2 batch先読み・8 GiB cacheを使用し、3条件で同じcacheを共有する。`LOADER_WORKERS`・`PREFETCH_BATCHES`・`LOCAL_CACHE`・`CACHE_GB`で設定する。T2TTの文字batchにも順序付き先読みを利用できる。

## キャッシュと検証

- 未拡張のsource特徴量（FBANKと任意のCMVN）、unit、teacherのnumeric payloadを必要時に保存する。SpecAugmentの結果は保存しない。
- 特徴量のキーには音声path/size/mtime・manifest SHA256・backend・特徴量設定・実装と数値runtimeの識別を含む。初回作成時に音声SHA256を検証する。
- unit/teacherは元artifactの内部payload hash・外部checksum・metadataを検証し、元のファイルhashも保存する。local cacheへ再格納してもteacherの`units_hash`は元artifactを指す。
- 学習では固定manifestの英語音声hashをunit/teacher metadataと照合する。source音声の特徴量は、検証したsourceから作成する。
- cache hitでもlocal payload hashを検証する。元fileのsize/mtimeやsidecarが変われば別entryとして検証し直す。入力は固定されたcanonical manifestとartifactを使用する。
- 完成したpayloadだけatomicに公開する。容量が足りない場合は、使用中のentryを避けて古いcache payloadを削除する。保存できない大きさや空き容量なら、そのsampleをcacheへ残さず処理する。
- cacheの削除対象は専用directory内の管理ファイルだけ。元の音声・unit・teacherは変更しない。空でない未知のdirectory、入力directory内、ColabのDrive配下はcacheとして拒否する。

cacheはVM上の一時領域として使用する。Colab再起動で消えても、必要なsampleから再作成する。3条件を順番に実行すると、同じVMのcacheを共有できる。

## 順序・再開・エラー

workerの完了順に関わらず、sampleは既定のseed/epoch permutation順に取り出す。`max_tokens`でbatchを縮めても、先に準備済みの残りsampleを捨てない。先読みしただけではcheckpointのepoch/cursorは進まない。

workerは読み込み・決定的な特徴量処理とprior投影だけを行う。NumPyのSpecAugment、モデルの乱数、CUDA操作は学習threadへ残す。CPU検証では、同期・先読み・cacheのcold/warm状態・worker数を変えた再開でparameterが一致する。

起動時は基準1件のunit/teacherでconfig identityを固定し、他のartifactは取り出す際にそのidentity・split・manifest・hashを検証する。必要な入力の欠損・checksum不一致・config混入は呼び出し元へ伝え、sampleを飛ばさず停止する。先読みで未消費のsampleは学習済みと数えない。

worker数・先読み数・cache位置などのI/O設定は`experiment.json`とcheckpointの`input_pipeline`へ記録する。これらは学習順や目的関数を変えず、同じコードのresumeでは変更できる。実験の速度を比較する場合は全条件で揃える。異なる実装コードのcheckpoint再開は既存のhash契約により拒否する。

## 計測

`metrics.jsonl`へ更新ごとに次を保存する。

- `input/data_wait_seconds`：必要なmicrobatchの取得待ちとCPU collateに実際に待ったwall time。background処理の総CPU時間ではない。
- `input/optimization_seconds`：CPU augmentation・転送・forward/backward・optimizerのwall time。CUDAでは更新終了を同期して測定する。
- `input/loader_workers`：その更新でのworker数。

validation・checkpoint保存・ログI/Oは上記のoptimization時間に含まない。pilot summaryにも待ち時間とoptimization時間の合計を記録し、GPU使用率CSVと照合できる。

実コーパス・Colabでの速度向上率は未計測。ローカルでは取得と消費の重なり、queue上限、cache再利用/破損検知/容量制限、学習結果の一致を検証する。
