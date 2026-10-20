# Method

T2TTとS2UTは同じ対訳pairとtrain splitから学習する。語彙もtrainのみから作成し、devは共通のcheckpoint選択、testは最終評価に使う。T2TTのパラメータをS2UTへコピーしない。

T2TTのteacher-forced cross-attentionを`A_MT[T,S]`とする。最終decoder層、head平均が既定値。出力側EOS、source/target padding、未知の文字を除外する。任意のdecoder層とhead集合を指定できる。

時刻spanから`M_src[S,F]`と`M_tgt[U,T]`を構成し、`A_T = M_tgt A_MT M_src`を計算する。各有効rowは確率分布である。invalid rowは全0とし、KLにもpriorにも寄与しない。

Distillationはhead平均後のraw S2UT attentionに`KL(A_T || A_S)`を適用する。teacherはdetachし、KLは有効unit rowで平均する。main lossはEOSを含む非padding target tokenのfairseq式label-smoothed CEである。

Guidanceは選択層の全headのlogitへ同じpriorを加え、そのsoftmaxから**実際のcontext**を計算する。raw/guidedを別々に保持する。`--heads`はalignment観測・KLのhead選択であり、priorは選択層全headへ作用する。`beta=0`はnative fairseq pathを呼ぶ。uniform priorはlogitに共通定数を加えるため元のattentionと一致する（丸め誤差を除く）。

Annealed Guidanceは完了更新数`step`について`beta_start + min(step/anneal_steps,1)*(beta_end-beta_start)`。推論・devのcheckpoint選択は全方式で常に`beta=0`とする。通常Guidanceも推論時はteacher-freeであり、train/inference間にpriorの有無の差がある。

Cross-Attentionを正解アラインメントとは仮定せず、learned soft alignment / routing patternとして解釈する。効果の主張には複数seedによる実データの翻訳性能・収束速度・attention解析が必要である。
