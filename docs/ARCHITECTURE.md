# Controlled Architecture

All-layer mode captures cross-attention for each existing decoder layer in a single forward pass. It matches T2TT layer `l` to S2UT layer `l`, with attention and prior tensors shaped `[B,L,T,S]` in all-layer mode and `[B,T,S]` in single-layer mode. A `finally` block releases adapter state even if the forward pass fails. [All-layer transfer](ALL_LAYER_TRANSFER.md)

The benchmark comparison builds on fairseq's `s2ut_transformer_fisher` model, `speech_to_speech` task, and `speech_to_unit` criterion. Training registers local alignment model, task, and criterion extensions. They preserve the model's parameter count and state dictionary keys. The fairseq Trainer handles the optimizer, scheduler, FP16, and gradient accumulation. The fairseq dataset handles feature extraction, SpecAugment, and token batching. Inference loads the stock model and uses the same `SequenceGenerator`. [Benchmark workflow](BENCHMARK_COMPARISON.md)

The reference is `s2ut_transformer_fisher` at fairseq commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`. T2TT uses the same S2UT model factory with `TextFrontend` in place of the source subsampler. Encoder positional embeddings, scaling, dropout, layer normalization, and Transformer layers share the same code. Both use `TransformerUnitDecoder`. The T2TT path supplies a text dictionary.

| Component | T2TT | S2UT |
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

`alignment_transfer_s2ut_fisher` is registered locally. `TransferAttention` uses the native parameters and state dictionary keys. Baseline calls the native forward path. The local extension keeps fairseq's source and parameter initialization intact.

`compare_architectures` compares all shared settings and writes a JSON table. `tests/test_fairseq_native.py` checks actual layer types, parameter shapes, and Baseline output parity.

The tokenizer uses an abstract interface. Character v1 uses one Unicode codepoint per token and maps only U+0020 to `<space>`. The fairseq-compatible ID order is `<bos>=0, <pad>=1, <eos>=2, <unk>=3, <space>=4`. A literal `<space>` in the input is seven codepoints. The target ends with EOS, the source has no BOS or EOS, and the decoder starts with fairseq's EOS.
