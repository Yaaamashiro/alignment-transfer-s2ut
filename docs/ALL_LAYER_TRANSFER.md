# All-layer Attention Transfer

T2TT Decoder層lの全head平均を、S2UT Decoder層lへ対応させる。全層抽出は1 decoder forward、保存は`[L,T,S]`。source/target modality mappingを共有し、各層で`M_tgt @ attention[l] @ M_src`を計算して`[L,U,F]`へ写す。6層をhard-codeしない。

既存teacher packetの`mt_attention_layers`・文字/subword span・CTC timing・unit timing・extraction signatureを再利用する。層数/順序/headsが一致しなければ停止する。既存single-layer形式は単層ablation/旧実験のために残るが、主要4条件のNotebookはall-layerに固定する。

新しい3条件はKL-based Attention Distillation、Annealed KL-based Attention Distillation、Attention-level Knowledge Distillation。層ごとのvalid-row meanをworst-layer LogSumExpで集約する。単純layer meanは主比較へ使わない。[式・microbatch定義・校正](METHOD.md)を参照。

native modelはstock Attentionをcaptureする。teacherの値はcriterionのlossだけが読み、forwardのcontextはshape/scopeだけを持つ。validation/inferenceではcapture contextもteacher dataset wrapperも使わない。

`alignment-train.jsonl`へ各updateの`alignment/D_l`・`alignment/w_l`・worst layer・loss・λ・τ・entropy・KL/Mask診断を保存する。別dev診断とAttention可視化も生成とは独立して行う。

数式はprojected teacherを\(\tilde A_l^{\mathrm{T2TT}}\)、層別lossを\(\ell_l^{\mathrm{KL}}\) / \(\ell_l^{\mathrm{mask}}\)と書く。内部packet名・`alignment/D_l`等のログIDは維持する。全体の目的関数は\(\mathcal L=\mathcal L_{\mathrm{S2UT}}+\lambda\mathcal L_{\mathrm{align}}\)。
