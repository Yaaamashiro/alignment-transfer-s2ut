# Intentional Differences and Validation Limits

1. **Modality frontends/vocabularies:** T2TTはUnicode codepoint embedding/文字projection、S2UTはnative FBANK+Conv frontend/KM100 projection。辞書サイズに伴うparameter数の差は意図されたもの。共有Transformer層のarchitectureは同じ。
2. **Trainerの範囲:** 既存Baselineとの[比較経路](BENCHMARK_COMPARISON.md)は公式fairseq Trainer、保存cfgのFP16・optimizer・scheduler・token-budget sampler、native dataset/criterion/multitaskを使う。prepared辞書・SentencePiece・ラベルを再利用し、完了済みbudgetとcheckpoint選択を揃える。旧独立PyTorch loopは`--trainer legacy`へ分離し、controlled/strongの機能診断に残す。旧経路の語彙・前処理・batch・精度はbenchmarkと同一ではなく、そこで学習したモデルをnative比較の一致条件へ混ぜない。
3. **FBANK:** sourceは16 kHz mono、Kaldi FBANK 80 bins、25 ms/10 ms、dither 0。strongはfairseqの16-bit入力規約（waveform×32768）とutterance CMVNを使用し、trainだけにSpecAugmentを適用する。controlledは従来のwaveform scale 1を保持する。recipe間の混在比較は拒否する。
4. **source時刻投影:** 既定はFBANK窓とConvの局所受容野によるsupport、旧方式は`--source-mapping uniform`。いずれも実際のencoder長と照合する。frontend supportを語の正確な音響境界やTransformer後の受容野とは見なさない。
5. **target時刻投影:** HuBERT receptive-field中心を用い、重なる受容野そのものではなく1-hop幅でframe時間区間を作る。CTCのblank区間はtoken spanに自動的に埋め込まず、overlapしないunit rowはinvalidとして除外する。
6. **CTC alignment:** 既定モデルは外部pretrained ASRであり、MT teacherではない。英語の大文字化はCTC lookupだけの明示的変換。語彙外文字/punctuationは設定に従いmaskまたはerror。日本語読み正規化/G2Pで異なる文字列を暗黙生成しない。
7. **Teacher artifacts:** 前処理ではdense U×F行列を固定保存せず、MT attentionと時刻spanを持つpacketを保存する。学習のCPU先読みで現在のfeature長に合わせて合成し、forwardの実測Fと照合する。dense行列は可視化artifactへ保存できる。
8. **推論:** native比較ではBaseline/転移モデルともstock S2UTとして構築し、fairseq SequenceGeneratorを共用する。既定beam 10、max_len_a 1、max_len_b 200、length penalty 1、FP32。旧standalone推論はfull-history beam searchを使用する。両経路ともbeta=0だが、異なる生成器間のスコアを同一条件とは扱わない。
9. **Offline smoke:** 人工CTC emission、人工KM100 sequence、tiny PyTorchモデル、テスト音vocoderを使用する。外部モデルの再現・音声品質・研究仮説の検証とは別物。smoke checkpointはmetadataで区別し、本番teacherとの混合とsmokeのASR-BLEUを拒否する。
10. **実験未実行:** 本学習、実コーパスのCTC/HuBERT処理、公式vocoderによる音声品質評価、複数seedの翻訳性能比較は別途GPU環境で実行する。CLI実装やCPUテスト成功だけで研究効果を主張しない。
11. **Runtimeと比較判定:** native経路は単一processの保存レシピに対応する。追加physical microbatch分割・旧early-stop policyは拒否する。ColabのZIP staging・保存/復元・常駐学習・cache/worker制御・overflow retry・勾配監査は[固定revisionのbenchmark実装](COLAB_RUNTIME.md)を共用する。alignment固有のpair artifact増分保存と読み込みpath解決だけをadapterとして追加する。オンラインBLEU monitorは未統合で共通の事後評価を使う。`fair_comparison`は保存された科学的設定・データ・完了budget・選択規則・指定された評価条件の照合結果であり、元runとのGPU演算・worker RNG・環境全体のbit一致や原論文スコアの再現を保証しない。
12. **外部テキスト評価:** T2TTの[保存済み8C外部評価](EXTERNAL_TEXT_EVALUATION.md)は、固定benchmarkのidentity digestとEnglish BLEUだけを追加adapterからimportする。新しいdataset loader・モデル・学習経路は取り込まない。保存済み300件の日英テキストとIDを照合し、音声の検証・取得は行わない。正規化BLEUは既存metricを直接再利用するが、T2TTテキスト出力の得点には音声生成・ASRの誤りが含まれない。
13. **T2TTの予算延長:** [20k→200k](T2TT_EXTENSION.md)は別stageへverified snapshotをseedし、checkpointのimplementation hashと一致する元source treeを隔離して同じtrainerを実行する。scientific contractを書換えず、max updatesだけを延長する。延長process内の保存通知callbackを包み、local save/validationとは別にDrive snapshot頻度・milestone・保持数を制御する。元20kや通常のT2TT/S2UT trainer・保存runtime sourceは変更しない。CPUの小型モデルでcontinuationの一致を検査し、実200k学習・GPUでの性能は別途検証する。

14. **層別転移:** [全層mode](ALL_LAYER_TRANSFER.md)は対応するT2TT/S2UT層のrouting priorを1 forwardで取得・適用する。層別のglobal valid-row正規化後に層平均し、betaは層数で割らない。nativeモデルのparameter/state key・Trainer・保存レシピは維持する。Attention/teacherは独立stageへ保存し、旧single-layerを全層へ複製しない。実GPUでの効果・VRAMは未測定。

15. **Scope ablationとhead制限:** single-layerで全層teacherを読むときも対応層を選び、native benchmarkでは記録されたselected-head teacherを拒否して全head平均を維持する。比較の`--allow-scope-ablation`はscopeだけを独立変数として明示する分析用optionであり、学習・resumeのcontractや他の照合を緩めない。GPU接続試験の3方式保存/復元・VRAM出力はtiny 2層用で、本番6層の実測とは区別する。

16. **単層・head ablationの整合性:** 旧single-layer teacherでも対応層を検査し、legacyではteacher/studentのhead選択を照合する。記録のない旧packetは最終層・全head平均に限り互換性を維持する。診断の`--layer`は同じ層のprior/coverage/heatmapを選び、teacherのhead選択を記録する。モデル・loss・学習予算・保存runtimeは変更しない。

17. **T2TT lossの観測:** 延長の親processだけでローカルmetricsを追記読みし、実lossを既存loggerへ出す。verified ZIP公開後にfreeze済みmetricsとbest checkpointから小さなサマリーを添付する。元trainer/control・科学的設定・snapshot頻度は維持する。サマリーは診断用で、ZIP/receiptによる復旧や本学習の成否を置き換えない。

18. **Tokenizer比較:** [SentencePiece T2TT](SENTENCEPIECE.md)はtrain-onlyの日英別Unigram/identity/8000目標。Transformer backbone・学習/評価条件・データ・CTC/unit/S2UTを維持し、語彙に依存するembedding/projection行数と系列長だけが変わる。元characterをseedせず独立stageで0から学習する。元codepoint spanを保存してCTC文字時刻をsubwordへ集約し、投影式は維持する。held-out UNKの完全round-tripは保証せず、元位置を検査・未知文字数を報告する。literal U+2581/NULは暗黙変換やデータ除外をせず停止する。実GPU200k・実コーパスの語彙数/長さ・性能は未測定。
