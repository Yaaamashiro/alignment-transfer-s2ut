# Validation record — 2026-10-22

独立したWindows / Python 3.10.22 / PyTorch 2.7.1 CPU環境で、`python -m pytest -q`は**54 passed, 1 skipped**。strongレシピのSentencePiece 0.2.0を含むoptional dependenciesを使用し、`pip check`も成功した。

fairseqはbenchmark checkoutではなく、公式repositoryからcommit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`を別途取得した。Windowsのsymlink/C++ build制約のため、検証環境ではupstreamの`READTHEDOCS=1`経路でpure-Python packageをbuildした。モデルのPyTorch層とsourceは変更していない。source commit、tracked sourceのclean状態、installed Python sourceの内容一致を検証した。

確認した内容:

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

1件のskipは**実CTC/HuBERT/公式vocoderを含むCUDA統合テスト**。実コーパスの前処理、本学習、T2TT dev text BLEU、実音声品質、ASR-BLEUによる手法比較、Colab上の実行は未検証。人工smokeやnative layerテストの成功を研究仮説の検証結果として扱わない。

ローカルで準備した検証環境は`outputs/native-venv/`。ここから`python -m pytest -q`、各CLIを実行できる。`outputs/`内のPython、依存package、取得したupstream source、checkpoint、音声、画像はGit対象外。
