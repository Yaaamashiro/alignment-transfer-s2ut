# Validation record — 2026-10-23

独立したWindows / Python 3.10.22 / PyTorch 2.7.1 CPU環境で、`PYTHONUTF8=1 python -m pytest -q`は**96 passed, 2 skipped**。strongレシピのSentencePiece 0.2.0を含むoptional dependenciesを使用し、`pip check`も成功した。Windowsではnative TSV内の日本語を読むためUTF-8 modeを有効にする。

fairseqはbenchmark checkoutではなく、公式repositoryからcommit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`を別途取得した。Windowsのsymlink/C++ build制約のため、検証環境ではupstreamの`READTHEDOCS=1`経路でpure-Python packageをbuildした。モデルのPyTorch層とsourceは変更していない。source commit、tracked sourceのclean状態、installed Python sourceの内容一致を検証した。

今回の横断確認では、全CLIのヘルプ起動、主要ライブラリのimport、wheel buildと同梱ファイル、Python構文、README/docsのローカルリンクを確認した。3つのNotebookはnbformat 5.10.4の標準schema検証（警告もエラー扱い）に通過し、nbconvert 7.16.6でHTMLへ変換できた。ブラウザでの実表示はローカル接続制約により確認していない。外部モデルを取得する検査やColab上のインストール試験とは区別する。

依存関係は`pip check`に加え、新しい`environment` CLIでpreprocess/evaluation/dev extrasのバージョンとimportを検査する。CUDA・benchmarkの固定revision・native batch samplerは明示オプションで必須にできる。ローカルの通常preflightは成功し、`--native`では未buildの`fairseq.data.data_utils_fast`を検出して終了コード1になることを確認した。このWindows環境から本番native学習を開始できるという意味ではない。

ローカル環境のPyYAML 6.0.3・Cython 3.3.0・setuptools 84.2.0は、ColabのPyYAML 6.0.2・Cython 3.2.9・setuptools<81と異なる。benchmarkはテストfixtureから固定sourceを読み込んでおり、ローカルのpackage metadataへ導入していない。従って`pip check`の成功はColabの依存環境全体の再現を証明しない。Linuxでの通常fairseq buildとColabのbootstrapは別途検証が必要。古いOmegaConfのmetadataに対するpipのdeprecationと、upstream fairseqのimportlib/autocast deprecationは残っているが、今回のテストではエラーではない。

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

2件のskipは**native TrainerのFP16接続試験**と**実CTC/HuBERT/公式vocoderを含むCUDA統合テスト**。pure-Python fairseq環境にはCythonのbatch sampler拡張がないため、ローカルでは実Trainerのmicrobatch更新までを検証し、CLIのepoch iterator全体は未実行。Colabは通常のLinux buildを使う。

Drive上では`experiment-001`の`paper-fisher-v1` run、365000更新snapshot、対応するpreparedデータ・unitの保存先とJSONの学習設定を確認した。約1 GBの実checkpoint本体のローカル取得・登録、実コーパスの前処理、本学習、T2TT dev text BLEU、実音声品質、ASR-BLEUによる手法比較、Colab上の実行は未検証。人工smokeやnative Trainerテストの成功を研究仮説の検証結果として扱わない。

ローカルで準備した検証環境は`outputs/native-venv/`。ここから`python -m pytest -q`、各CLIを実行できる。`outputs/`内のPython、依存package、取得したupstream source、checkpoint、音声、画像はGit対象外。
