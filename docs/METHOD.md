# Method

T2TTのlearned cross-attention routing/alignment patternを、対応するS2UT Decoder層へlossとして転移する。Attentionをground-truth alignmentとは扱わない。比較順はBaseline、Distillation、Annealed Distillation、Mask Distillation。

## Teacherとforward

各層で全headを平均したT2TT Attentionを、既存のCTC文字時刻・HuBERT unit timing・source/target mappingで投影する。

\[
P_l=M_{tgt}A_l^{T2TT}M_{src}.
\]

対応はT2TT層l→S2UT層lで、最終層の複製ではない。teacherはdetachする。S2UT contextは常にstock fairseqのraw Attention `softmax(Z)`から計算する。teacherの内容・biasをforwardへ渡さない。native capture contextはshapeとscopeだけを持つ。validationとsource-only generationにはそのcontextもない。

## 4条件

Baselineは元のtask lossだけ。task lossにはnative `speech_to_unit`のunit lossと、元Baselineの補助lossを含める。

Distillationでは、teacher rowに確率massがあるtarget unitだけを使う。paddingとEOSは投影priorのzero rowとして除外する。

\[
D_l^{KL}=\frac{1}{N_l}\sum_{u\in V_l}D_{KL}(P_{l,u}\parallel A_{l,u}^{S2UT}).
\]

Annealed Distillationのforward・KL・初期λ・τはDistillationと同一。差はcosine scheduleだけ。

\[
T_a=0.8T,\quad \lambda_t=\lambda_{KL}\frac{1+\cos(\pi t/T_a)}2\ (0\le t<T_a),\qquad \lambda_t=0\ (t\ge T_a).
\]

`t`は現在までの完了update数（最初のforwardはt=0）。最後の20%はtask lossだけで更新する。teacherの参照は別途診断値を計算するために残り得るが、forwardへ注入せず、alignmentの勾配寄与は0になる。

Mask Distillationはteacherのsource確率を降順にstable sortし、累積massが明示選択したp以上になる最小集合Bを作る。同値なら元source indexの昇順。source paddingは除外し、有効rowは最低1 frame、無効rowは0 frame。teacher確率0のframeは選ばない。

\[
D_l^{mask}=1-\frac1{N_l}\sum_{u\in V_l}\sum_f B_{l,u,f}A_{l,u,f}^{S2UT}.
\]

teacher region内のstudent massが1なら0、0.7なら0.3になる。

## 層集約とnative gradient accumulation

\[
L_{align}=\tau[\operatorname{logsumexp}_l(D_l/\tau)-\log L],\qquad
w_l=\operatorname{softmax}_l(D_l/\tau).
\]

`torch.logsumexp`を使い、最大値で中心化して大きなlossでの勾配精度も保つ。層数はDecoderから取得する。全rowが無効の層はそのmicrobatchの集約から除外し、全層が無効なら停止する。

このLogSumExpは**criterionへ渡された現在のmicrobatchごと**に計算する。criterionは `task_sum + sample_size * lambda_t * L_align` を返し、stock Trainerが既存logical batchのtotal sample sizeで正規化する。従ってalignmentはsample-sizeで重み付けたmicrobatch LogSumExpの平均。logical batch全体の層lossを先に集めたLogSumExpとは異なる。全microbatch graph保持、二重forward、update_frequency変更は行わない。

## 学習前の校正

trainの固定順16 logical batches、元Baselineと同じfresh initialization seed・データ変換・precision・logical batchを使う。モデルはtrain modeでdropout条件を保存し、virtual batch indexに対しseed+indexを使って再生する。optimizer updateは0。最初のpassでτを求め、同じ初期化・入力・RNGへ戻して勾配を測る。calibrationだけの2 passであり、本学習のforward数は増やさない。

\[
\Delta=D_{max}-D_{median},\quad \tau=\operatorname{median}_{batch}\frac{\Delta}{\ln3}.
\]

偶数個のmedianはPyTorchと同じlower order statistic。near-zero差（明記した1e-8）、NaN、Infのbatchは除外し、全て無効なら停止する。手動τへのfallbackはない。KLとMaskを独立に校正する。τ推定のD_lは、各logical batch内のmicrobatch D_lをsample sizeで加重平均した代表値。λ測定には本学習と同じmicrobatch LogSumExpの勾配を使う。

Cross-Attentionのq/k/v/out projectionを含む全parameterについて、native logical task gradientと、上記microbatch集約後のalignment gradientを別々に`autograd.grad`で取得する。

\[
\lambda=\operatorname{median}_{batch}\left(0.5\frac{\|\nabla_{\theta_{CA}}L_{task}\|_2}{\|\nabla_{\theta_{CA}}L_{align}\|_2}\right).
\]

各microbatchの勾配だけを順に加算しgraphを保持しない。0/非finite normは停止条件。KLとMaskは別λ、Annealed Distillationは通常Distillationと同じλKL・τKLを必ず使う。

## Top-pとcontract

teacher-only分析はp=0.8/0.9/0.95についてcoverageのmean/median/p25/p75/p90、source長別・層別集計、valid row数、selected frame数、entropyとの相関、分布histogramを保存する。十分なteacher massを保ちつつregionが過度に広くならない候補をユーザーが選ぶ。学習は自動選択しない。

calibration JSONはimplementation、teacher、prepared data、Baseline登録、fresh model hash、seed、pair順/hash、batch数、各層loss、gradient norm、weight、top-p診断と選択pを保持する。training contractはファイルhashと内容hashを含める。異なるartifactでのresumeは拒否する。

実装は`alignment/losses.py`、`alignment/calibration.py`、`alignment_calibration.py`、`s2ut/benchmark_extension.py`。

## ログの集約

各updateのD_lとw_lは、microbatchの実測値をnative sample sizeで加重平均する。保存するworst layer indexはこの平均D_lのargmaxで、microbatchごとの最悪層の頻度も別fieldに保存する。selected frame数とvalid row数はmicrobatch間で合計する。これらの集計値を、本学習で計算していないlogical-batch LogSumExpと解釈しない。
