# Controlled Architecture

全層modeでも層数・parameter・state dictは増やさない。1 Decoder forward中に各層のCross-Attentionをcaptureし、T2TT層lのpriorだけをS2UT層lへ渡す。raw/guided/priorは`[B,L,T,S]`、single-layerは従来の`[B,T,S]`。例外時もfinallyでadapter状態を解放する。[全層転移](ALL_LAYER_TRANSFER.md)。

既存benchmarkとの比較では、公式fairseqの`s2ut_transformer_fisher`・`speech_to_speech` task・`speech_to_unit` criterionを基に、学習時だけ`benchmark_alignment_s2ut`・`benchmark_alignment_speech`・`speech_to_unit_alignment`を登録する。追加の学習parameterはなくstate dict keyを保つ。公式Trainerがoptimizer・scheduler・FP16・勾配蓄積を担当し、公式datasetが特徴抽出・SpecAugment・token batchを担当する。推論はstock modelへ戻して同一SequenceGeneratorを使用する。[比較経路の詳細](BENCHMARK_COMPARISON.md)を参照。

基準は公式fairseq commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`の`s2ut_transformer_fisher`。新規T2TTは同じS2UT model factoryから作成し、source subsamplerのみ`TextFrontend`へ置き換える。Encoder自体のposition生成、scale、dropout、layer normalization、Transformer層は共通コードを使う。decoderもnative `TransformerUnitDecoder`を共通利用し、T2TTのときは文字辞書を渡す。

| 項目 | T2TT | S2UT |
|---|---|---|
| Encoder layers | 12 | 12 |
| Decoder layers | 6 | 6 |
| Encoder/decoder dim | 256 / 256 | 256 / 256 |
| Encoder/decoder FFN dim | 2048 / 2048 | 2048 / 2048 |
| Encoder/decoder heads | 4 / 8 | 4 / 8 |
| Activation | ReLU | ReLU |
| Dropout / attention / activation | 0.1 / 0.1 / 0.1 | 0.1 / 0.1 / 0.1 |
| Layer norm | pre-norm | pre-norm |
| Position | native sinusoidal | native sinusoidal |
| Self-attention | native causal | native causal |
| Cross-attention | native MHA + local subclass | native MHA + local subclass |
| Source frontend | codepoint embedding | 80-bin FBANK, Conv1d 5/5, GLU, stride 2/2 |
| Target vocabulary | train English codepoints | KM100 + 4 specials |
| Decoder input/output embedding | shared | shared |

`alignment_transfer_s2ut_fisher`はrepository-localなregistered model。`TransferAttention`はnativeクラスと同じパラメータ・state dict keyを保持し、Baselineではnative forwardを呼ぶ。fairseq packageファイルは変更しない。classの変更によって追加の乱数初期化を行わない。

`compare_architectures`は設定の全共有項目を比較しJSON表を保存する。`tests/test_fairseq_native.py`は実際のlayer型・パラメータshape・native baseline出力一致を検証する。設定表だけで実装一致を証明したとは扱わない。

Tokenizerはabstract interfaceを持つ。v1は1 Unicode codepoint/token、U+0020のみ`<space>`。fairseq互換のID順は`<bos>=0, <pad>=1, <eos>=2, <unk>=3, <space>=4`。文字としての`<space>`入力は7 codepointとして扱い、特殊tokenへ解釈しない。sourceにBOS/EOSを加えず、target末尾にEOSを加える。decoder始端にはfairseqと同じEOSを使う。
