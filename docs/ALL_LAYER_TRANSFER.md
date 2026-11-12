# All-layer Attention Transfer

Match the attention averaged across all heads in T2TT decoder layer `l` to S2UT decoder layer `l`. Extract all layers in one decoder forward pass and store `[L,T,S]`. Share the source and target modality mappings across layers and compute `M_tgt @ attention[l] @ M_src` for each layer to obtain `[L,U,F]`. The implementation derives the layer count from the model.

Reuse `mt_attention_layers`, character and subword spans, CTC and unit timing, and the extraction signature from the teacher packet. The run stops when the layer count, order, or head configuration differs. Single-layer artifacts support ablations and historical runs. The primary notebook comparison uses all layers.

The three transfer conditions are KL-based Attention Distillation, Annealed KL-based Attention Distillation, and Attention-level Knowledge Distillation. Worst-layer LogSumExp aggregates the valid-row mean loss for each layer. [Equations, microbatch definition, and calibration](METHOD.md)

The native model captures stock attention. The criterion reads teacher values for the loss, and the forward context carries shape and scope. Validation and inference use the stock path without the capture context or the teacher dataset wrapper.

`alignment-train.jsonl` records `alignment/D_l`, `alignment/w_l`, the worst layer, loss, λ, τ, entropy, and KL and Mask diagnostics per update. Dev diagnostics and attention visualization run separately from generation.

The equations use \(\tilde A_l^{\mathrm{T2TT}}\) for the projected teacher and \(\ell_l^{\mathrm{KL}}\) or \(\ell_l^{\mathrm{mask}}\) for per-layer loss. The packet and log schema use identifiers such as `alignment/D_l`. The objective is \(\mathcal L=\mathcal L_{\mathrm{S2UT}}+\lambda\mathcal L_{\mathrm{align}}\).
