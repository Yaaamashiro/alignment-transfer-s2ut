# Validation record — 2026-10-28

全Decoder層のlayer-wise transferを追加。最終状態の全体 **249 passed, 6 skipped**（369.07秒）、native checkpoint可視化追加後の関連検査は **25 passed, 2 skipped**（110.98秒）。2層モデルでT2TT/S2UTを各1 forwardでcaptureし、層別prior・未知文字mask・FP32 NPZ・共有mappingによる`[L,U,F]`投影・memory budgetのL倍計算を確認した。旧single-layer抽出/teacherの読込・無変更resume、all-layerへの誤利用拒否、層数・scope・headsの不一致拒否も検査した。

層ごとに有効row数を変え、全microbatchの層別row総数で正規化後に層平均するKLとその勾配が一括計算に一致した。native criterionでもTrainerのmain token正規化を含む損失差を実測照合。2層のstock fairseqに対しsingle/all両modeでbeta=0のlogits（atol=rtol=0）、loss、更新後parameterが厳密一致し、全層のq/kへ有限の非0勾配を確認した。native TrainerでBaseline/Distillation/Annealed Guidance/Guidanceを各2更新、beta減衰、teacher削除後の共通native生成を完走した。

legacyとnativeのcheckpointから同じ層番号のteacher/raw/guidedを表示し、NPZの内容とPNGを確認した。Notebookの4条件生成・別stage・共通評価・旧single/Monotonic-only/T2TT延長との互換性も検査。新しい全層CUDA抽出parity・FP16/FP32 prior接続試験をNotebookのGPU preflightへ追加した。6 skipはCUDA 5件とWindows上のPOSIX process group 1件であり、実GPUの学習性能・VRAM・実データBLEU/研究効果は未測定。固定benchmark/fairseq pin・既存Baseline・T2TT recipe・S2UT更新budgetは変更していない。全Notebookのschema/Python/HTML、69ローカル文書リンク、CLI help、pip check、wheel内の全63 Python sourceのbyte一致も確認した。

Colabの環境検査セル[10]では依存/CUDA検査が成功した後、延長学習のCPU完全一致試験3件が失敗した（223 passed）。fixtureが親processだけを `torch.set_num_threads(1)` にしており、新しい学習processに同じスレッド設定を渡していなかった。ローカルでも外部環境を `OMP_NUM_THREADS=2`・`MKL_NUM_THREADS=2` にするとsmoke/fairseqの2件で再現し、smokeのmodel parameter差は最大約3.53e-8だった。学習contractは一致していた。

試験fixtureに限定して親と子のCPUスレッド数を1へ揃え、終了時に元の設定へ戻すよう修正した。新しい子processの実スレッド数も回帰試験で検査する。意図的に外部環境を2スレッドにしたまま、関連 **26 passed**（159.00秒）。`torch.equal`によるmodel/optimizer/RNGの完全一致検査は維持した。本番training/runtime/Notebookのスレッド設定・checkpoint・scientific contractは変更していない。

同じ2スレッドの外部環境でColabと同じ `pytest -m 'not gpu'` を全実行し、**226 passed, 1 skipped, 3 deselected**（315.19秒）。skipはWindows上のPOSIX process group、deselectedはCUDA対象3件。これはローカルCPUでの再検証であり、修正後のColab/L4での結果はまだ取得していない。

verified T2TT 20k→200kの独立stage延長を追加し、全suiteは **225 passed, 4 skipped**（310.39秒）。縮小した実学習と固定benchmarkの実ZIPで、last/best・Adam・epoch/cursor・best loss/bad validations・各RNGの継承、連続学習とのbit一致、source work/backupの全ファイル不変、途中target snapshotからのVM消失後の再開を検証した。公式fairseq小型モデルでも更新後parameterが連続学習と一致した。破損・欠落・異なるrevision/step/manifest/config/recipeの拒否、metricsの連続性、best選択の継承を確認した。

checkpointのimplementation hashと一致する旧Git source（703b7e8）を実際にexportして学習・延長し、元trainer自身のcontract検証を通した。hashの書換えやmodel/trainerの変更は行っていない。validationとDrive保存間隔の分離、最新2個＋milestone保持、検証済みupload後だけの整理、削除失敗の非致命性も確認した。Windowsで進捗readerとatomic replaceが競合した場合に限る短い再試行を追加し、他のcheckpoint書込失敗はそのまま伝播することを検査した。

Notebookは20k/200k両モードのdev・外部評価・Attention・teacher・S2UT保存先と共通checkpoint変数、評価前のAttention停止、完了済み200kの評価専用実行、Monotonic-onlyの互換性を検証した。全Notebookのschema/Python構文/HTML変換、CLI help、`pip check`、wheel内の全Python source同梱を確認した。既存training/CLI/colab runtimeとbenchmark pinは変更していない。4 skipはCUDA 3件とWindows上のPOSIX process group 1件。実際の20k checkpointによるGPU延長・200k学習結果・dev/FLEURS BLEUは未実行・未測定であり、ここでの数値はテスト結果だけである。

新しいVMの6.1.1で `t2tt/checkpoint_best.pt` が欠落していたため、外部評価セルへ保存済み入力の復元を追加した。**189 passed, 4 skipped**（136.14秒）。`restore-inputs` は固定benchmarkの `latest/restore` で既存ZIPを検証し、同じrun path・T2TT stage・完了済み予算・bestの存在を確認して復元する。trainerやgeneratorは起動せず、best欠落をlastで代用しない。元manifest/音声の再取得も不要。devは同じcheckpointの完了済み最終packからmetrics/predictionだけを検証してVMへ復元し、外部評価の親フォルダや異なるローカル結果を置き換えない。

実benchmark ZIPを使う7件の追加試験で、19k best/20k完了の区別、未公開ローカルファイルと外部結果の保持、Driveバックアップのbytes不変、再実行時の再復元なし、保存なし・未完了・破損ZIP・異なるstage・bestなし・異なるローカルdevの拒否を確認した。Notebookの復元→外部評価→読込専用検証の配線も既存のCLI parserに通した。Notebook schema/Python/HTML、CLI help、`pip check`も成功。T2TT/S2UT学習・Attention・既存保存runtime source・benchmark pinは変更していない。CUDA 3件とPOSIX 1件は引き続きローカル未実行であり、利用者の実モデルでの外部BLEU値は未取得。

Colabから報告された6.1.1の失敗は、固定8C 300件の取得・照合成功後に `text-evaluation/dev-text.json` がなく停止したもの。指示書ではdev結果は存在時だけ比較する任意入力だったが、実装で必須化していたため修正した。未保存のdev結果を自動生成したりT2TTを再学習したりせず、既存CLIから取得した同じbeam=5/max_chars=1000/length_penalty=1.0とnative BLEU設定で外部評価を行う。devが存在する場合のcheckpoint・metric/generation契約・predictionハッシュ検証は維持し、不完全・壊れたdev結果を「不在」として扱わない。

関連 **46 passed**（62.80秒）。dev JSONだけ不在、JSON/predictionとも不在、引数省略、既存devの欠落prediction・不正JSON object拒否、devなしNotebook実行と比較skip表示、devなし完了結果の再検証・再利用を追加した。既存の参照非混入・BLEU再計算・実ZIP復元・公式fairseq/人工checkpoint各300件の生成と既存dev/Notebook互換性も通過。Notebook schema/構文/HTMLとCLI helpも確認した。今回の修正では学習・generator・保存runtime・benchmark pinは変更せず、全suiteの再実行は行っていない（直前の全suiteは174 passed/4 skipped）。実Colabでの外部BLEU値は未取得。

外部テキスト評価の再点検は **174 passed, 4 skipped**（125.23秒）。CLIのmetrics出力を `dev-text.jsonl` にすると、別拡張子でも既存devと同じprediction名になり上書きできる問題を回帰テストで再現し、全派生出力と保護対象のパス照合で生成前に拒否するよう修正した。Notebookは完了済みZIPをスキップした場合も、表示直前に読込専用 `verify` でdev入力・checkpoint・固定セット・外部prediction・corpus BLEUを再検証する。未完了時には生成せず停止し、検証でモデル読込やファイル更新を行わないこと、壊れたdev predictionを拒否することを確認した。

推論をstubにした既存試験に加え、有限の人工重みを保存したsmokeと公式fairseqの小型checkpointを実際のloader・tokenizer・既存beam-5 generatorで読み、各300件の外部生成・採点・検証を実行した。EOS/UNK集計と実際のcheckpoint stepも確認。これはCPU上の人工条件による配線検証であり、学習済み実teacherの翻訳性能評価ではない。`pip check`、Notebookのschema/Python/HTML、verify CLI、wheel buildと全Python source同梱も成功。T2TT/S2UT学習・Attention・既存runtime source・benchmark pinは変更していない。skipは従来通りCUDA 3件とPOSIX 1件。実データ/GPUの外部BLEU値は引き続き未測定。

保存済み8C 300件によるT2TT外部テキスト評価を追加し、全suiteは **170 passed, 4 skipped**（114.29秒）。追加16件では固定ID集合・決定的な順序・重複/欠落/改変の拒否、referenceを変えてもsource-only生成入力・設定が不変であること、native/benchmark正規化corpus BLEUの再計算、既存の正規化規則、UNK集計、完了済み・途中cacheの再利用と改変検出、best checkpoint以外や異なるdev生成設定の拒否を検査した。実際の固定benchmark ZIP publisher/restorerで、100件生成後の失敗→VM消失→残り200件だけ生成→完了後のVM消失→再生成なしの復元も成功。ニューラル計算はこの外部評価の統合試験ではstubを使い、既存generator自体の実モデル検証は従来のdev評価テストが引き続き通過している。

Notebook 6.1.1のCLI配線・独立stage名・dev値の読込・10例表示を検査し、schema/Python構文/HTML変換も成功。`pip check`、新CLIのhelp、56ローカルMarkdownリンク、wheel buildと全Python source同梱の照合も成功した。既存のT2TT/S2UT学習・Attention・CTC・Baseline登録・runtimeのPython source、benchmark revision pinは変更していない。4 skipは従来のCUDA 3件とPOSIX 1件。**実Driveの保存済み300件・学習済みT2TTを使うGPU評価は未実行であり、実際のFLEURS BLEU値は報告しない。**

同日の再点検は**154 passed, 4 skipped**。固定seedで長さ・空白・引用符・Unicode・未知文字の異なる40 pairを作り、2層decoderの途中層/最終層について、smokeと公式fairseqの4条件で単体/batch Attentionの一致を追加検証した。既存の保存・復元・中断・hash・native学習の検査も再実行し、新たな不具合は見つからなかった。`pip check`とenvironment検査、22 CLI、3 Notebookのschema/Python/HTML、53ローカルリンク、wheel内の全Python sourceの一致も成功。実行コードとNotebookの固定revision `107605f4f5bd7b53722d73ced3fa5a76560e7a9d`は変更せず、追加テストと本記録だけを更新した。CUDAの3件とPOSIXの1件は引き続きローカルでは未実行であり、実Colabの全コーパス完走を確認したという意味ではない。

2026-10-27の横断再点検は**150 passed, 4 skipped**。Attentionの全head平均・単一head・複数head、可変長padding・未知文字mask、単体抽出との数値一致、batch size 1の厳密一致、書き込み途中の中断と再利用、token/Attention予算超過の停止を確認した。ColabのCUDA接続試験へ単体/batch Attentionの一致検査を追加したが、ローカルのPyTorchはCPU buildのため実行していない。skipはCUDA必須の3件とPOSIX process group中断試験の1件。

NPZ保存後・外部checksum保存前の切断で、旧publisherがNPZを含まない完了ZIPを作る問題を再現して修正した。各前処理producerは内部payloadとmetadataを検証した出力だけに不足checksumを補い、元NPZのhash・mtimeを変えない。読込専用`get()`は入力を書き換えない。正常終了時に未完了NPZを残したstageの公開を拒否し、失敗時は完成pairだけを保存する。実benchmark ZIPを使い、旧完了ZIPから1 pairが漏れた新VM復元→不足分の作成→新しい増分ZIPからの再復元も成功した。既存Baseline・元コーパス・benchmark sourceは変更していない。

依存metadata/importのenvironment検査と`pip check`、22 CLIのhelp、3 Notebookのschema/Python構文/HTML変換、Markdownのローカルリンク、wheel buildと全Python sourceの同梱を確認した。STREAMING文書が独立trainer向けであることを明記し、本比較の事前ZIP取得と区別した。モデル・学習条件・precision・checkpoint選択・benchmark runtimeのpinは変更しない。今回の結果も実Colab/GPUでの全コーパス完走や翻訳品質を保証するものではない。

2026-10-27、Attention抽出がpairごとにcheckpoint全体と不変な語彙を再hashする問題を修正した。checkpoint SHA256は1回、forwardは最大8 pairずつとし、padding後のtoken/Attention要素数でbatchを制限する。公式fairseqとsmokeの可変長・空白・未知文字を含むデータで、旧単体抽出との数値一致（atol 2e-6 / rtol 2e-5）、旧metadataと保存済みpacketの再利用・既存file hash不変、checkpoint改変と混在packetの拒否、予算内batch、進捗と再利用表示を検査した。全suiteは**136 passed, 3 skipped**。3件はCUDA必須の2件とWindowsでは実行できないPOSIX process group中断試験。

CPUのtiny人工モデル・64 MiBのoptimizer相当payloadを加えたcheckpoint・128 pairでは、旧版9.80秒（checkpoint SHA 128回）から修正版2.78秒（SHA 1回・16 forward）になった。Attention最大絶対差は5.96e-8。ファイルをOS cacheから読める人工条件での約3.52倍であり、実GPU・実teacher・実コーパスの速度予測として扱わない。Colabでの実速度とCUDAでのbatch結果は未検証。

Colabのfull-manifestで共通文章と補助文字labelの不一致が報告された。alignment側のTSV readerがCSVの引用符処理を有効にしていたため、引用符を含むliteral fairseq labelを変形し、閉じていない引用符では後続rowまで取り込む問題を再現した。固定benchmarkと同じ`QUOTE_NONE`へ合わせ、header・column数・空/重複IDの検査を維持した。引用符4種類のmanifest生成とresume、引用符によるrow連結、破損TSV、実際に変更された文章の拒否を検査し、全suiteは**127 passed, 3 skipped**。元データ・label・語彙・学習条件の書き換えや照合の無効化は行わない。実データの不一致がこの読込不具合だけに起因するかは、Colabでのliteral TSV照合または修正版の再実行で確認する。

2026-10-26の全段階ログ統合後の再検証は**115 passed, 3 skipped**。隣接benchmark checkoutが別作業で更新されていたため、Colabが導入した固定runtimeをテストで優先するようfixtureを修正した。明示的な検証用sourceは`S2ST_BENCHMARK_TEST_ROOT`で指定でき、既存benchmark checkoutは変更しない。固定版sourceを別配置した全suiteと、環境からimportできる固定版を隣接checkoutより優先する保存・復元テストを実行した。モデル・学習条件・runtimeのpinは変更していない。

Colabから報告されたCPU preflightではPython 3.10.22・依存metadata・fairseq provenance・compiled batch sampler・benchmark pinが成功し、matplotlibだけが失敗していた。Notebook kernel由来の`MPLBACKEND=module://matplotlib_inline.backend_inline`を別venvへ継承する同じエラーをローカルで再現し、3つのNotebookで`MPLBACKEND=Agg`を設定した。各Notebookの実際の設定を適用した新規subprocessでmatplotlibのimportとPNG保存を検証した。短い失敗コマンドで省略されていたpreflight JSONも、失敗時の末尾ログ表示で読めることを確認した。

ColabのCPU test報告（95 passed / 1 failed）から、command引数をファイルとして調べる`identity_for`が長い`python -c`コードに対してLinuxの`ENAMETOOLONG`を受け取る問題を確認した。ファイルになり得ないこの場合だけを処理し、文字列全体はcommand identityへ引き続き含める。外部ファイルのhash照合とアクセス権エラーの停止は維持する。Linuxの例外を明示的に発生させる回帰テスト、権限エラー検査、実ZIPの中断復元を含む8件のruntimeテストと全suiteを検証した。

入力stagingの表示経路を修正し、関連するNotebook/CLIテスト**27 passed**。benchmarkがstderrへ出す進捗をNotebookから逐次表示してローカルへflushし、stdout JSONは別ファイルとして保持する。子processが終わる前に画面表示されたことを同期markerで検証し、childの非zero終了・ログ保存・中断後のchild終了も確認した。stagingそのもののalgorithm・pack配置・checksum・再利用条件は変更しない。NotebookのschemaとPython構文も確認した。

環境構築・検査・JSON取得・入力準備・前処理・学習・推論・評価を共通の標準ライブラリloggerへ統合した。3つのNotebookでgit/uv/pipを含めた直接subprocess呼び出しの残存を検出し、インストール前に埋め込む関数が共通sourceとAST一致することを検査した。無出力の子processが終了する前に生存通知と実行中metadataを読めること、速い途中行も画面とファイルへ保存されること、JSON stdoutの分離、起動失敗・異常終了・不正JSON・中断終了を確認した。既存のGPU CSV監視とbenchmarkのZIP/runtimeは維持する。

WindowsではPOSIXのprocess groupを実行できないため、入れ子のobserverがgroupを分離しない設定を検査し、実際のgroup中断試験はskipした。3件のskipはこのPOSIX中断試験とCUDA必須の2件。CPUでのnative Trainerの数値一致・checkpoint再開・実benchmarkによる保存/復元は全suite内で成功した。Notebookのschema/Python/HTML変換、文書リンク、wheel buildと共通loggerを含む全Python moduleの同梱も確認した。

独立したWindows / Python 3.10.22 / PyTorch 2.7.1 CPU環境で、`PYTHONUTF8=1 python -m pytest -q`は**115 passed, 3 skipped**。strongレシピのSentencePiece 0.2.0を含むoptional dependenciesを使用し、`pip check`も成功した。Windowsではnative TSV内の日本語を読むためUTF-8 modeを有効にする。

fairseqはbenchmark checkoutではなく、公式repositoryからcommit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`を別途取得した。Windowsのsymlink/C++ build制約のため、検証環境ではupstreamの`READTHEDOCS=1`経路でpure-Python packageをbuildした。モデルのPyTorch層とsourceは変更していない。source commit、tracked sourceのclean状態、installed Python sourceの内容一致を検証した。

今回の横断確認では、全CLIのヘルプ起動、主要ライブラリのimport、wheel buildと同梱ファイル、Python構文、README/docsのローカルリンクを確認した。3つのNotebookはnbformat 5.10.4の標準schema検証（警告もエラー扱い）に通過し、nbconvert 7.16.6でHTMLへ変換できた。ブラウザでの実表示はローカル接続制約により確認していない。外部モデルを取得する検査やColab上のインストール試験とは区別する。

依存関係は`pip check`に加え、新しい`environment` CLIでpreprocess/evaluation/dev extrasのバージョンとimportを検査する。CUDA・benchmarkの固定revision・native batch samplerは明示オプションで必須にできる。ローカルの通常preflightは成功し、`--native`では未buildの`fairseq.data.data_utils_fast`を検出して終了コード1になることを確認した。このWindows環境から本番native学習を開始できるという意味ではない。

ローカル環境のPyYAML 6.0.3・Cython 3.3.0・setuptools 84.2.0は、ColabのPyYAML 6.0.2・Cython 3.2.9・setuptools<81と異なる。benchmarkはテストfixtureから固定sourceを読み込んでおり、ローカルのpackage metadataへ導入していない。従って`pip check`の成功はColabの依存環境全体の再現を証明しない。ColabからのCPU preflight報告ではcompiled batch samplerまで成功しているが、Linuxでのnative CLI epoch iterator全体・実GPU更新・外部モデル込みの実行は引き続き別途検証が必要。古いOmegaConfのmetadataに対するpipのdeprecationと、upstream fairseqのimportlib/autocast deprecationは残っているが、今回のテストではエラーではない。

確認した内容:

- NotebookのT2TT/Monotonicの4通りのスイッチ、2条件/3条件・ASR有無の評価コマンドを実際のCLI parserへ渡して検証。Monotonicはteacher不要、評価は共通dev条件であることを確認。
- standalone Notebookの全CLIコマンドと5条件を検査。シェルescapeを引数listと`check=True`へ変更し、空白付きpathの保持と失敗時の停止を検証。学習レシピ・更新数・既存のコードpinは保持した。
- 環境preflightが依存不整合・optional extraの異なるversion・DLL import失敗・Python不一致・fairseq provenance不一致・native samplerの異常を検出することを確認。

- 固定revision `4361e55a99c7f4ea3a00c5c138f6820c34a07d06`のbenchmark runtimeを直接使用。実ZIP publisher・検証・復元を使い、pair処理の途中失敗→新VMを想定した復元→完了済みstageの再利用を検証。既存pairを後続増分packへ複製しない。
- benchmarkのCPU WAV packを作成後、原本WAVを参照不能にしてもGPU側staging関数で復元可能。native fairseqの辞書・設定・TSV・音声も原本payloadを参照不能にした状態でローカルコピーから読み込めることを確認。
- benchmarkの実常駐session/upload workerを通してT2TTを2更新し、ZIPへ保存。作業ディレクトリを失った状態から同じcheckpoint・optimizer・bestを復元し、完了済み更新を再学習しない。
- Windowsでは開いているファイルのatomic replaceが拒否されるため、T2TTの常駐session統合試験は最終更新で1 snapshotを作成する設定を使用。native hook側で周期保存を別途検証する。Colab/Linux向けbenchmark実装にWindows用の変更は加えていない。
- native S2UTにbenchmarkのperformance hooks・勾配監査を適用し、optimizer更新とsnapshot freeze protocolを検証。native contract・benchmark性能状態・optimizer・勾配監査を完全snapshotへ含める。
- 壊れた増分ZIPと保存容量予算超過を拒否し、既存復旧点を保持。最後のupload失敗でローカルに残る完了markerを永続化済みと誤認せず、再実行時に再保存する。ColabのローカルROOT、Drive PERSISTENT、runtime pin、通常600秒間隔・1024GB予算を静的検証。
- 別途学習した公式fairseq checkpointのrecipe/metadata/データhashを検証し、読み取り専用Baselineとして登録。snapshot ZIPのhash・member検査と、再開時のsnapshot固定を検証。
- 新経路で公式Trainerの実optimizer更新を実行。転移モデルの初期重みは同じseedの公式モデルとbit一致し、beta=0の更新後parameterも公式Trainerとbit一致。
- native optimizer/checkpointの保存・再開、完了runの再実行、teacher削除後のnative dev検証とstock SequenceGeneratorによるBaseline/転移モデルの共通推論を確認。
- report-onlyの全dev unit main lossによるbest選択とvalidation前後のRNG保持を検証。FP16の確認済みoverflowだけを再試行する制御は人工scalerで同一batch・RNG・buffer復元を検査。
- 元のfull train/dev・保存unit列のimportを検証し、prepared targetの変更・辞書やrecipeの改変・生成条件やcheckpointが異なる評価を検出。
- 新Colabは登録BaselineからT2TT/teacher用backbone設定を生成。native FP16接続試験、full train/dev前処理、共通推論・評価の構文と既定値を検証。
- native S2UT Baselineとlocal extensionのlogitが厳密一致。
- beta=0がBaselineに厳密一致。uniform priorも丸め誤差内で一致。
- biased priorが実際のcontext/outputを変更し、Cross-Attentionへ勾配が流れる。
- T2TT/S2UTのnative Encoder/Decoder/Cross-Attention layer型と共有parameter shapeが一致。
- 人工行列のprojection、row normalization、mask、invalid row、CTC繰返しlabel/Viterbi、unit reduction/timing、annealing endpoint。
- train-only文字語彙、split leakage guard、artifact hash/metadata、外部checksum保存中断からの再利用。
- 人工音声/CTC/unitを用い、nativeモデルのT2TT学習→Attention→projection→4条件のS2UT学習→可視化を実行。
- 外部モデル不要のsmoke pipelineで4条件を完走し、T2TT/MT/CTC/teacherを削除した後にもunitとテスト音を生成。
- checkpointと日本語音声だけをコピーした配置で推論成功。参照text付き推論入力は拒否。
- 中断再開と連続学習のCPU parameterがbit一致。変更したseed/推論条件のresumeを拒否。
- manifest/推論/評価のidempotent resumeと、source-only入力のexport。
- T2TT/S2UT構成比較は全共有項目MATCH。4条件の制御変数比較も`fair_comparison=true`。
- uniform混合でpadding・EOS・無効rowを復活させず、beta=0の結果が変わらない。
- 有効row数の異なるmicrobatchで、KLの和/全有効row数が一括計算と勾配一致。実際のtrainerでも一括batchとgradient accumulationのloss・更新勾配が一致。
- FBANK+Conv時刻supportが実際の畳み込みの勾配supportと一致し、端点paddingと出力長不一致を処理する。
- 単調priorでT2TT/CTC teacherなしに学習でき、dev診断とteacher-free推論に対応する。
- strongのsource/target文字補助decoderとdecoder CTCを学習し、重み付き補助lossがupstream fairseq MultitaskCriterionと数値一致。
- strongの16-bit FBANK scale・utterance CMVNがnative特徴抽出と一致。SpecAugment・補助decoder dropoutを含む連続学習/再開のCPU parameterがbit一致。
- strongのBaseline / Annealed Monotonic / Annealed T2TTで同じ初期重み・レシピを確認し、異なるsource mappingの比較を拒否。
- train-only補助文字語彙/SP modelをcheckpointへ内包。強いbaselineもcheckpointと日本語音声だけでunit推論可能。
- alignment診断で元の文字数に対するcoverage、unit row率、raw/平滑化entropy、PNG/NPZを生成。
- 既定strong構成（encoder 12 / decoder 6、dim 256、補助decoder各2層）でも人工音声・unitによるforward/backwardと有限勾配を確認。
- wheelにstrong YAMLと追加moduleを同梱。ColabのPython構文・CLI引数を静的検証（Colab上での実行とは別）。
- pilotのtrain/dev部分集合の再現性、元split保持、test音声の非読み出し、入力hash/seed/上限を変えたresumeの拒否を確認。
- corpus aliasと旧Colab音声pathの移設を扱い、除外するtest rowにある既知のsplit leakageも検出。
- strongの3条件コマンドとbeta=0の後半区間を検査。コマンド失敗時の終了コード・ログ・所要時間を保存。
- pilot集計で再開時のstep重複を扱い、条件間で異なるdev評価data/configの混在を拒否。専用notebookの構文と接続試験用JSONLを検証。
- ローカルのGTX 1650でnvidia-smiのCSV採取とサンプラ終了を確認（GPU学習は行っていない）。
- 先頭入力の消費と後続入力の取得が重なり、未完了の後続入力を待たずに先頭を使用できること、queue上限を確認。
- `max_tokens`による部分消費でも順序を保持し、未消費の先読みをcheckpoint位置に含めない。
- 5つのmethod/prior条件で、同期版・複数worker/cache版・worker数を変えたresumeのCPU parameter/RNGが一致。
- native strongでもSpecAugmentと補助decoder dropoutを含め、同期・先読み/cache・resumeのCPU parameterが一致。
- キャッシュ同時要求の重複生成防止、LRU容量制限、容量不足時の非保存、破損検知、元artifact hash保持、変更したsourceの拒否を確認。
- dev payloadの読み出し前に最初の学習forwardへ進み、後続sampleのunit config不一致は消費時に停止することを確認。
- pilotのdev上限は関数・CLI・Colabとも200件であることを確認。
- T2TTのsource-only自己回帰生成・EOS/出力上限・dev全件のcorpus BLEUを検証。既知の完全一致訳でBLEU=100となり、参照訳をdecoderへ渡さず、音声がなくても評価できる。
- native fairseq checkpointでもT2TT text生成/評価を実行。生成・評価の途中再開、変更した生成条件・dev data・改変した予測cacheの拒否を確認。
- pilot集計でT2TT dev text評価を添付し、異なるteacher checkpoint・dev data・test評価の混入を拒否。wheelに新しい評価CLI/moduleを同梱し、両Colabの構文を確認。

現在のCUDAに関する3件のskipは**native TrainerのFP16接続試験**、**T2TT Attentionの単体/batch一致試験**、**実CTC/HuBERT/公式vocoderを含むCUDA統合テスト**。加えてWindowsではPOSIXの入れ子process group中断試験をskipする。pure-Python fairseq環境にはCythonのbatch sampler拡張がないため、ローカルでは実Trainerのmicrobatch更新までを検証し、CLIのepoch iterator全体は未実行。Colabは通常のLinux buildを使う。

Drive上では`experiment-001`の`paper-fisher-v1` run、365000更新snapshot、対応するpreparedデータ・unitの保存先とJSONの学習設定を確認した。約1 GBの実checkpoint本体のローカル取得・登録、実コーパスの前処理、本学習、T2TT dev text BLEU、実音声品質、ASR-BLEUによる手法比較、Colab上の実行は未検証。人工smokeやnative Trainerテストの成功を研究仮説の検証結果として扱わない。

ローカルで準備した検証環境は`outputs/native-venv/`。ここから`python -m pytest -q`、各CLIを実行できる。`outputs/`内のPython、依存package、取得したupstream source、checkpoint、音声、画像はGit対象外。
