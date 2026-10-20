# Validation record — 2026-10-20

独立したWindows / Python 3.10.22 / PyTorch 2.7.1 CPU環境で、`python -m pytest -q`は**21 passed, 1 skipped**（14.77秒）。宣言済みoptional dependenciesを含めてinstallし、`pip check`も成功した。

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

1件のskipは**実CTC/HuBERT/公式vocoderを含むCUDA統合テスト**。実コーパスの前処理、本学習、実音声品質、ASR-BLEUによる手法比較、Colab上の実行は未検証。人工smokeやnative layerテストの成功を研究仮説の検証結果として扱わない。

ローカルで準備した検証環境は`outputs/native-venv/`。ここから`python -m pytest -q`、各CLIを実行できる。`outputs/`内のPython、依存package、取得したupstream source、checkpoint、音声、画像はGit対象外。
