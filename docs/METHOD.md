# Method

Use learned T2TT cross-attention patterns as loss supervision for corresponding S2UT decoder layers. The comparison runs Baseline, KL-based Attention Distillation, Annealed KL-based Attention Distillation, and Attention-level Knowledge Distillation in that order.

## Teacher and forward pass

Average T2TT attention across heads in each layer, then project it through the source and target mappings using CTC character timing and HuBERT unit timing.

\[
\tilde A_l^{\mathrm{T2TT}}=M_{\mathrm{tgt}}A_l^{\mathrm{T2TT}}M_{\mathrm{src}}.
\]

T2TT layer `l` supervises S2UT layer `l`. Detach the teacher tensor. The S2UT context uses stock fairseq attention `softmax(Z)`. The native capture context carries shape and scope during training. Validation and source-only generation use the stock path.

## Four conditions

The objective is

\[
\boxed{\mathcal L=\mathcal L_{\mathrm{S2UT}}+\lambda\mathcal L_{\mathrm{align}}}.
\]

\(\mathcal L_{\mathrm{S2UT}}\) combines the native `speech_to_unit` unit loss and the Baseline auxiliary losses. The Baseline uses this objective alone. The annealed condition uses \(\lambda_t\) in place of \(\lambda\).

KL-based Attention Distillation uses target units whose projected teacher rows have positive probability mass. Padding and EOS have zero prior rows and are excluded.

\[
\ell_l^{\mathrm{KL}}=\frac{1}{N_l}\sum_{u\in V_l}\operatorname{KL}(\tilde A_{l,u}^{\mathrm{T2TT}}\parallel A_{l,u}^{\mathrm{S2UT}}).
\]

Annealed KL-based Attention Distillation uses the same forward pass, KL loss, initial λ, and τ as KL-based Attention Distillation, with the cosine schedule below.

\[
T_a=fT,\quad f=\text{policy[anneal\_fraction]}\ (\text{default }0.8),\quad \lambda_t=\lambda_{\mathrm{KL}}\frac{1+\cos(\pi t/T_a)}2\ (0\le t<T_a),\qquad \lambda_t=0\ (t\ge T_a).
\]

`t` counts completed updates, with `t=0` on the first forward pass. The default schedule uses only the S2UT task loss for the final 20% of updates. Diagnostics can still read teacher values during this task-only tail, while the alignment gradient contributes zero.

Attention-level Knowledge Distillation sorts teacher source probabilities in descending order with a stable sort and selects the smallest set B whose cumulative mass reaches the chosen p. Ties use the original source index order. The region excludes source padding and zero-probability frames. Valid rows contain at least one frame, and invalid rows contain zero.

\[
\ell_l^{\mathrm{mask}}=1-\frac1{N_l}\sum_{u\in V_l}\sum_f B_{l,u,f}A_{l,u,f}^{\mathrm{S2UT}}.
\]

Student mass of 1 inside the teacher region gives loss 0. Mass of 0.7 gives loss 0.3.

## Layer aggregation and native gradient accumulation

\[
\mathcal L_{\mathrm{align}}=\tau[\operatorname{logsumexp}_l(\ell_l/\tau)-\log L],\qquad
w_l=\operatorname{softmax}_l(\ell_l/\tau).
\]

Use `torch.logsumexp` with max centering to preserve gradient accuracy for large losses. The decoder determines the layer count. A microbatch excludes layers with zero valid rows and stops when all layers are invalid.

Compute LogSumExp for each microbatch passed to the criterion. The criterion returns `task_sum + sample_size * lambda_t * alignment_loss`, and the stock Trainer normalizes by the total sample size of the logical batch. This yields a sample-size-weighted mean of microbatch LogSumExp values. It preserves the native update frequency and uses a single forward pass per microbatch.

## Calibration before training

Use 16 fixed-order logical batches from training with the Baseline's fresh initialization seed, input transformations, precision, and batching. The model runs in training mode and replays dropout with the seed plus virtual batch index. The first pass estimates τ. The second restores the same initialization, inputs, and RNG to measure gradients. Calibration performs zero optimizer updates.

\[
\Delta_b=\ell_{\max}^{(b)}-\ell_{\mathrm{median}}^{(b)},\quad \tau=\operatorname{median}_{b\in\text{microbatches}}\frac{\Delta_b}{\ln3}.
\]

Calibrate τ at the same microbatch granularity as the training loss. Compute a candidate τ_b from the \(\ell_l\) values of each microbatch in the 16 logical batches and take the median. An update frequency of four gives up to 64 candidates.

Use only layers with positive counts in each microbatch. For even counts, use the lower order statistic, as in PyTorch. Reject gaps below `1e-8`, NaN, and Inf, and stop if every candidate is invalid. Calibrate KL and Mask independently. λ gradient norms use logical batches and accumulate the same microbatch LogSumExp with native sample-size weights.

Use `autograd.grad` to measure the native logical task gradient and the aggregated alignment gradient separately for all cross-attention parameters, including q, k, v, and output projections.

\[
\lambda=\operatorname{median}_{batch}\left(0.5\frac{\|\nabla_{\theta_{CA}}\mathcal L_{\mathrm{S2UT}}\|_2}{\|\nabla_{\theta_{CA}}\mathcal L_{\mathrm{align}}\|_2}\right).
\]

Accumulate microbatch gradients sequentially and release each graph. Zero or nonfinite norms stop calibration. KL and Mask use separate λ values. The two KL conditions share λKL and τKL.

The calibration report includes the measured task/alignment gradient cosine and the norm ratio after applying the final λ to each logical batch. It summarizes microbatch layer weights, worst-layer frequency, entropy, effective layer count, and the maximum-to-median weight ratio. The Pilot repeats the gradient observation on saved checkpoints in a separate process. [Native transfer pilot](PILOT.md)

## Top-p selection and the training contract

`tau_estimates.kl.rejected_microbatches` and `tau_estimates.mask.rejected_microbatches` store zero-based indices into the saved microbatch list. A candidate is rejected for missing valid layers, a near-zero gap, NaN, or Inf.

The teacher-only diagnostic records coverage mean, median, p25, p75, and p90 for p values 0.8, 0.9, and 0.95. It also records summaries by source length and layer, valid rows, selected frames, correlations with entropy, and histograms. Select a candidate that retains sufficient teacher mass with a suitably narrow region. Training requires the explicit choice.

The calibration JSON records the implementation, teacher, prepared data, Baseline registration, fresh model hash, seed, pair order and hash, batch count, per-layer losses, gradient norms, weights, top-p diagnostics, and selected p. The training contract binds the file and content hashes. A mismatched artifact prevents resume.

The implementation is in `alignment/losses.py`, `alignment/calibration.py`, `alignment_calibration.py`, and `s2ut/benchmark_extension.py`.

## Log aggregation

\(\ell_l\) maps to the existing `alignment/D_l` log ID. `task_sum` is the unnormalized S2UT loss sum inside the criterion.

Each update logs the microbatch measurements of \(\ell_l\) and \(w_l\) averaged by native sample size. The reported worst layer is the argmax of the averaged \(\ell_l\). A separate field records the frequency of each microbatch's worst layer. Selected frames and valid rows are summed across microbatches.

## Annealing settings and diagnostic layer masks

The plan loads `anneal_fraction` and `anneal_schedule` from `configs/experiments/conditions.yaml`. Both the loss coefficient and task-only progress logs use that policy. The schedule must be cosine and the fraction finite in (0,1]. The training contract includes the settings hash and rejects a mismatch on resume. The wheel bundles the canonical YAML at build time.

`alignment/kl_mean`, `alignment/kl_worst`, layer-averaged entropy, and independent dev KL mean and maximum use layers with valid rows. Logs retain zero per-layer values and valid-row counts for invalid layers. The task-only tail logs `alignment/task_only_start` and `alignment/task_only_phase`. Validation and inference are teacher-free.
