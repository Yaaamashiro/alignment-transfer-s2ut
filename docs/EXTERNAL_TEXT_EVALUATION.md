# T2TTのFLEURS/FLORES外部テキスト評価

学習済みT2TTの翻訳品質を、以前のCascade/S2UT外部評価と同じ保存済み8C 300件で診断する。T2TTの再学習・語彙拡張・サンプル再選択は行わない。音声・CTC・HuBERT・vocoder・Hugging Faceからの取得は不要。

## Notebookで実行

`notebooks/colab/benchmark_comparison.ipynb` の **6.1.1 FLEURS/FLORES external text BLEU** を実行する。

1. 既存の環境・Drive接続・`run(...)` helperを準備する。6.1の学習を再実行する必要はない。
2. `ROOT`・`PERSISTENT`・`T2TT_UPDATES` を学習時と同じ値にする。VMにbest checkpointがなければ、6.1.1が `PERSISTENT/stages/t2tt` の学習完了済みZIPから復元する。
3. 6.1.1セルの `RUN_EXTERNAL_TEXT=True` にする。保存済み8Cが複数ある場合だけ `SOURCE_8C` に既存identityのフォルダ名を指定する。
4. このセルを実行する。固定300件の照合、生成、採点、ZIP保存、得点・固定seedの10例の表示まで行う。

最初に `restore-inputs` が固定benchmarkの `latest/restore` でZIP・member checksumを検証する。保存先・T2TT stage・固定benchmark revision・設定した更新予算の完了を照合し、**保存済み `checkpoint_best.pt` だけを評価に使用**する。既存のローカルbestは差し替えず、不完全なローカル学習ディレクトリは元runtimeの `.interrupted-*` 保持で残す。保存済みmodelがない・ZIPが壊れている・学習が未完了・bestが含まれない場合、学習やlast checkpointへの代替を行わず停止する。音声・元manifestの復元は不要。

`PERSISTENT/stages/t2tt-dev` に完了済みdev結果があれば、同じbest checkpointの評価であることを確認してmetricsとpredictionを復元する。増分ZIPの最終packから必要な2ファイルだけをローカルへ戻し、外部評価用サブディレクトリを含む親フォルダは置き換えない。異なるローカルdev結果は上書きしない。devの再生成は行わない。

dev結果がない場合も、既存 `evaluate_t2tt` の固定生成・BLEU設定で外部300件の評価を行う。devを再評価したりT2TTを学習したりせず、in-domain比較だけをスキップして明示する。devが存在する場合は、壊れた結果・欠落したprediction・異なるcheckpointや設定を拒否する。

完了済みZIPの復元で生成stageがスキップされた場合も、表示直前に読込専用の `verify` を実行する。checkpoint・固定セット・存在するdev評価とそのpredictionのハッシュ、外部predictionのID・ハッシュ・corpus BLEUを再照合し、変更があれば停止する。この確認は生成やファイルの更新を行わず、未完了の結果から勝手に生成を始めることもない。

表示するcheckpoint stepは**選択済みbestの実際の更新数**。学習予算20,000から推測して埋めない。devと外部のcheckpoint SHAが異なる場合は停止する。

## 保存済み8Cの取得と検証

`BENCHMARK/diagnostics/flores-fleurs-cascade/<identity>/` の次の小さな保存済みファイルを読む。元のbenchmark実験への書込みはしない。

- `identity.json`: `selected` に入っている日英テキスト、ID、テキストSHA256。
- `dataset-audit.json`: `selected_samples` と `selected=300`。
- `metrics.json`: 完了済みreport、`identity_sha256`、`metrics.samples=300`、埋込みper-sample結果。
- `per_sample.jsonl`: 実際に評価したID集合とテキスト。

固定runtimeの `direct_s2st.journal.digest` でidentityとフォルダ名を確認する。4箇所のID集合の完全一致、重複・欠落なし、テキストSHA、過去のBLEU実装との一致を確認する。順序は `sample_id` の昇順。元の順序が異なっても同じ300件・同じprediction順になる。

300件の完了済みreportがなければ、`No completed saved 8C FLORES/FLEURS 300-sample identity found...` と停止する。複数ある場合もmtimeで勝手に決めず、既存identityの明示指定を要求する。新規のデータ選択・不一致時の縮小・失敗サンプルの除外は行わない。

確認したテキスト証拠をVMの `ROOT/external-text-input/fixed-8c.json` に固定する。これは評価stageへのファイル引数となり、既存Drive-safe runtimeの入力ハッシュにも束縛される。元の8C evidenceが変わった場合、同じ出力先での再開は拒否する。

## 推論と指標

既存の `evaluation.text.generate_text` をそのまま使う。日本語を既存文字tokenizerで符号化し、source-onlyのautoregressive生成を行う。設定はdev評価の既定と同じbeam=5、max_chars=1000、length_penalty=1.0。外部評価固有の生成オプションはない。devの保存契約がこの設定と異なる場合は停止する。

生成loopへ渡すレコードは `sample_id` と `source_text` のみ。English referenceやその長さをdecoder・beam選択・終了条件へ渡さない。全300件の生成後に初めて参照文を結合して採点する。途中cacheにも参照文を入れない。生成文への追加post-processingはない。

| 出力 | 用途・処理 |
|---|---|
| `text_bleu` | devと同じSacreBLEU。13a、case-sensitive、exp smoothing、effective_order=False |
| `benchmark_normalized_bleu` | 固定benchmarkの `normalize_english` を仮説・参照に適用後、同じ `corpus_bleu` で採点。以前のCascade/S2UT外部評価との比較用の主指標 |
| `sentence_bleu` | 各例の診断用。短文にも対応するeffective_order=True。論文用の値はcorpus BLEU |

benchmarkの正規化はNFKC・小文字化・その実装が定める記号除去・空白整理を行う。**アポストロフィなど実装が残す文字もそのまま保持**する。独自の近似正規化に置き換えない。T2TTはテキスト出力を採点するため、Cascade/S2UTの最終ASR-BLEUとの差には音声生成・ASRの影響も含まれる。

未知文字は既存tokenizerのUNKのまま。Unicode codepoint数、UNK数・率、UNKを含む文数・割合、頻出未知文字を記録する。空白もtokenizer通り文字数に含む。語彙は変更しない。

## 保存と再開

VM上の独立ディレクトリ:

```text
ROOT/text-evaluation/flores-fleurs/
  flores-fleurs-text.json
  flores-fleurs-text-predictions.jsonl
  flores-fleurs-text-cache/*.json
```

Drive上の独立stage:

```text
PERSISTENT/stages/t2tt-flores-fleurs/<snapshot>/
  snapshot.json
  snapshot.zip
```

既存 `run(...)` が、固定benchmark runtimeの検証付き増分ZIPでcacheと結果を保存・復元する。個別JSONを繰り返しDriveへコピーしない。`t2tt`・`t2tt-dev` のstage、既存checkpoint、Cascade/S2UT結果は変更しない。

CLIで出力名を変えた場合も、metrics・prediction・cacheの全派生パスについて既存dev評価・prediction/cache・checkpoint・8C入力との重複を拒否する。たとえば `dev-text.jsonl` は、拡張子が異なっても既存devと同じprediction名になるため使えない。

同じVMなら6.1.1を再実行する。新しいVMなら同じrevision・パス・設定でセクション1・2の環境とDrive接続を準備後、6.1.1を実行する。必要なbestと保存済みdev結果は冒頭で復元する。devの有無とハッシュも再開契約に含む。検証済みcacheを再利用し、完了済み評価は生成しない。CLI単体の再開でも契約・predictionハッシュ・corpus BLEUの再計算を照合する。最後のZIP保存以降にVMから失われた生成分は再計算する。

既存学習の再開契約は全Pythonソースを束縛している。**未完了のT2TT/S2UT学習を、新しい評価コードへ更新したまま再開しない**。学習を元のrevisionで完了してから評価コードを使う。完成済みcheckpointの読込・外部評価には再学習は不要。

## CLI

固定benchmark runtimeを導入済みの環境で、保存済み入力の復元・固定セット取得・評価の順に実行する。

```bash
python -m alignment_transfer_s2ut.evaluate_t2tt_external restore-inputs \
  --work /local/work/t2tt --backup /drive/experiment/stages/t2tt \
  --expected-updates 20000 --dev-backup /drive/experiment/stages/t2tt-dev \
  --dev-output /local/work/text-evaluation/dev-text.json

python -m alignment_transfer_s2ut.evaluate_t2tt_external prepare \
  --benchmark /path/to/benchmark-experiment \
  --output /local/work/external-text-input/fixed-8c.json

python -m alignment_transfer_s2ut.evaluate_t2tt_external evaluate \
  --fixed-set /local/work/external-text-input/fixed-8c.json \
  --checkpoint /local/work/t2tt/checkpoint_best.pt \
  --dev-evaluation /local/work/text-evaluation/dev-text.json \
  --output /local/work/text-evaluation/flores-fleurs/flores-fleurs-text.json \
  --device cuda --resume
```

Colabでは上のCLIを直接起動せず6.1.1の共通logger・stage wrapper経由で実行する。
