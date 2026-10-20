# Intentional Differences and Validation Limits

1. **Modality frontends/vocabularies:** T2TTはUnicode codepoint embedding/文字projection、S2UTはnative FBANK+Conv frontend/KM100 projection。辞書サイズに伴うparameter数の差は意図されたもの。共有Transformer層のarchitectureは同じ。
2. **独立trainer:** native fairseqモデルをPyTorch loopから学習する。`controlled`は補助loss・CMVN・SpecAugmentなし、`strong`はnative source/target character decoder、Unigram decoder CTC、utterance CMVNとSpecAugmentを追加する。補助lossの重みと参照層はbenchmarkに合わせるが、文字語彙は全条件でtrainのみ、文字列は暗黙にstrip/space collapseしない。オンラインBLEU monitor、fairseq trainerのdistributed/FP16、token bucket samplerは未実装。公開benchmarkとの厳密な学習再現やスコア同等性を主張しない。
3. **FBANK:** sourceは16 kHz mono、Kaldi FBANK 80 bins、25 ms/10 ms、dither 0。strongはfairseqの16-bit入力規約（waveform×32768）とutterance CMVNを使用し、trainだけにSpecAugmentを適用する。controlledは従来のwaveform scale 1を保持する。recipe間の混在比較は拒否する。
4. **source時刻投影:** 既定はFBANK窓とConvの局所受容野によるsupport、旧方式は`--source-mapping uniform`。いずれも実際のencoder長と照合する。frontend supportを語の正確な音響境界やTransformer後の受容野とは見なさない。
5. **target時刻投影:** HuBERT receptive-field中心を用い、重なる受容野そのものではなく1-hop幅でframe時間区間を作る。CTCのblank区間はtoken spanに自動的に埋め込まず、overlapしないunit rowはinvalidとして除外する。
6. **CTC alignment:** 既定モデルは外部pretrained ASRであり、MT teacherではない。英語の大文字化はCTC lookupだけの明示的変換。語彙外文字/punctuationは設定に従いmaskまたはerror。日本語読み正規化/G2Pで異なる文字列を暗黙生成しない。
7. **Teacher artifacts:** 前処理ではdense U×F行列を固定保存せず、MT attentionと時刻spanを持つpacketを保存する。実際のFに合わせてforward時に合成する。dense行列は可視化artifactへ保存できる。
8. **推論:** full-history beam searchを用い、native fairseq incremental generatorとの速度・tie-breaking・length正規化の厳密一致は保証しない。4条件では同じ推論器を使用する。全条件でbeta=0。
9. **Offline smoke:** 人工CTC emission、人工KM100 sequence、tiny PyTorchモデル、テスト音vocoderを使用する。外部モデルの再現・音声品質・研究仮説の検証とは別物。smoke checkpointはmetadataで区別し、本番teacherとの混合とsmokeのASR-BLEUを拒否する。
10. **実験未実行:** 本学習、実コーパスのCTC/HuBERT処理、公式vocoderによる音声品質評価、複数seedの翻訳性能比較は別途GPU環境で実行する。CLI実装やCPUテスト成功だけで研究効果を主張しない。
