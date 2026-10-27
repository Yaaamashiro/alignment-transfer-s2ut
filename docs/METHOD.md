# Method

主要比較では`--all-layers`で対応層のlearned soft alignmentを転移する。

\[
A^{T2TT}_l \rightarrow A^{S2UT}_l,\qquad
L_{align}=\frac1L\sum_{l=1}^{L}\frac{\sum_m KLsum_{m,l}}{\max(1,N_l)}.
\]

投影済みteacherを`A^T_l`、S2UTのraw Attentionを`A^S_l`とすると、各層の有効row平均を用いた`L_align = (1/L) sum_l D_KL(A^T_l || A^S_l)`に相当する。`N_l`は同じoptimizer update内の全microbatchで数えた層別有効row数。層ごとのmeanを層平均し、betaをLで割らない。以下の選択層の説明はsingle-layer互換modeを指す。[全層の実行手順](ALL_LAYER_TRANSFER.md)。

T2TTとS2UTは同じ対訳pairとtrain splitから学習する。語彙もtrainのみから作成し、devは共通のcheckpoint選択、testは最終評価に使う。T2TTのパラメータをS2UTへコピーしない。

T2TTのteacher-forced cross-attentionを`A_MT[T,S]`とする。最終decoder層、head平均が既定値。出力側EOS、source/target padding、未知の文字を除外する。任意のdecoder層とhead集合を指定できる。

時刻spanから`M_src[S,F]`と`M_tgt[U,T]`を構成し、`A_T = M_tgt A_MT M_src`を計算する。各有効rowは確率分布である。invalid rowは全0とし、KLにもpriorにも寄与しない。

Distillationはhead平均後のraw S2UT attentionに`KL(A_T || A_S)`を適用する。teacherはdetachし、KLは有効unit rowで平均する。main lossはEOSを含む非padding target tokenのfairseq式label-smoothed CEである。

gradient accumulationでは、各microbatchのKLの**和**を、同じoptimizer更新内の全microbatchに含まれる有効unit row総数で割る。microbatchごとの平均を等重みで平均しない。全rowが無効ならKLは0。main CEは同じ更新内の非padding unit token総数で正規化する。

Guidanceは選択層の全headのlogitへ同じpriorを加え、そのsoftmaxから**実際のcontext**を計算する。raw/guidedを別々に保持する。`--heads`はalignment観測・KLのhead選択であり、priorは選択層全headへ作用する。`beta=0`はnative fairseq pathを呼ぶ。uniform priorはlogitに共通定数を加えるため元のattentionと一致する（丸め誤差を除く）。

Guidanceでは有効なacoustic位置上のuniform分布`U`を使い、`P' = (1-alpha) A_T + alpha U`としてから`softmax(Z + beta log(P' + 1e-8))`を計算する。`--prior-smoothing 0.05`が既定値。padding、EOS、全0の無効rowへ質量を与えない。Distillationの教師は平滑化前の`A_T`。`--prior-smoothing 0`で平滑化を無効化できる。

`--prior-type monotonic`は、各target unit runの中心時刻をtarget音声長で、source frontend supportの中心時刻をsource音声長で正規化し、距離にGaussian kernel（`--monotonic-sigma 0.1`）を適用する。行を正規化した同じ形のpriorをGuidance/KLへ渡す。unit時刻と音声長だけを使い、T2TT・transcript・CTC alignmentを参照しない。Baseline / Annealed Monotonic / Annealed T2TTを同じレシピで比較することで、単なる単調性の誘導と翻訳Attention由来の効果を区別する。

Annealed Guidanceは完了更新数`step`について`beta_start + min(step/anneal_steps,1)*(beta_end-beta_start)`。推論・devのcheckpoint選択は全方式で常に`beta=0`とする。通常Guidanceも推論時はteacher-freeであり、train/inference間にpriorの有無の差がある。

Cross-Attentionを正解アラインメントとは仮定せず、learned soft alignment / routing patternとして解釈する。効果の主張には複数seedによる実データの翻訳性能・収束速度・attention解析が必要である。
