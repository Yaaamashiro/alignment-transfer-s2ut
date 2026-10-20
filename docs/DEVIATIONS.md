# Intentional Differences and Validation Limits

1. **Modality frontends/vocabularies:** T2TTはUnicode codepoint embedding/文字projection、S2UTはnative FBANK+Conv frontend/KM100 projection。辞書サイズに伴うparameter数の差は意図されたもの。共有Transformer層のarchitectureは同じ。
2. **独立trainer:** native fairseqモデルをPyTorch loopから学習する。benchmark固有の補助source/target character decoder、Unigram CTC loss、SpecAugment、オンラインBLEU monitor、fairseq trainerのdistributed/FP16機能は実装していない。これは既存benchmarkのtraining recipeの厳密な再現ではない。4条件内部では同一レシピで比較する。公開benchmarkへの直接的なスコア同等性を主張しない。
3. **FBANK:** sourceは16 kHz mono、Kaldi FBANK 80 bins、25 ms/10 ms、dither 0。corpus全体のCMVNやSpecAugmentは既定で適用しない。
4. **source時刻投影:** 実際のencoder出力長に対する等間隔time binを使用する。Convの受容野中心を語ごとの正確な音響境界とは見なさない。将来のfrontendでも保存時刻spanを再利用できる。
5. **target時刻投影:** HuBERT receptive-field中心を用い、重なる受容野そのものではなく1-hop幅でframe時間区間を作る。CTCのblank区間はtoken spanに自動的に埋め込まず、overlapしないunit rowはinvalidとして除外する。
6. **CTC alignment:** 既定モデルは外部pretrained ASRであり、MT teacherではない。英語の大文字化はCTC lookupだけの明示的変換。語彙外文字/punctuationは設定に従いmaskまたはerror。日本語読み正規化/G2Pで異なる文字列を暗黙生成しない。
7. **Teacher artifacts:** 前処理ではdense U×F行列を固定保存せず、MT attentionと時刻spanを持つpacketを保存する。実際のFに合わせてforward時に合成する。dense行列は可視化artifactへ保存できる。
8. **推論:** full-history beam searchを用い、native fairseq incremental generatorとの速度・tie-breaking・length正規化の厳密一致は保証しない。4条件では同じ推論器を使用する。全条件でbeta=0。
9. **Offline smoke:** 人工CTC emission、人工KM100 sequence、tiny PyTorchモデル、テスト音vocoderを使用する。外部モデルの再現・音声品質・研究仮説の検証とは別物。smoke checkpointはmetadataで区別し、本番teacherとの混合とsmokeのASR-BLEUを拒否する。
10. **実験未実行:** 本学習、実コーパスのCTC/HuBERT処理、公式vocoderによる音声品質評価、複数seedの翻訳性能比較は別途GPU環境で実行する。CLI実装やCPUテスト成功だけで研究効果を主張しない。
