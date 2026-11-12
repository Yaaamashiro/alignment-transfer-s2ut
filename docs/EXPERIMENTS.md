# Experiments

The primary comparison uses the following order.

| Condition | Alignment supervision | Forward | Coefficients |
|---|---|---|---|
| Baseline | None | Raw attention | None |
| KL-based Attention Distillation | Per-layer KL and worst-layer LogSumExp | Raw attention | Calibrated λKL and τKL |
| Annealed KL-based Attention Distillation | Same KL loss and aggregation | Raw attention | Same λKL and τKL, with cosine decay through 0.8T |
| Attention-level Knowledge Distillation | Student mass in the selected top-p region and the same aggregation | Raw attention | Calibrated λmask and τmask |

The registered Baseline sets the fresh initialization policy, data, training budget, optimizer, scheduler, precision, logical batches, checkpoint selection, generation, and evaluation. Each transfer condition starts from a fresh initialization and uses the same all-head, all-layer teacher. The notebook reads the budget T from the registered Baseline.

1. Reuse or construct the all-layer teacher.
2. Display top-p diagnostics on a fixed training subset.
3. Select `MASK_TOP_P` explicitly.
4. Calibrate λ and τ from the same initialization and share the JSON across the three transfer conditions.
5. Train each condition in its own stage.
6. Run the same source-only native generation and dev evaluation.
7. Inspect per-layer KL, mask mass, weight distributions, and attention in a separate teacher-forced dev diagnostic.

Translation evaluation uses the shared ASR-BLEU and UER pipeline. Report TER when the existing evaluator provides it. Calibration, top-p selection, and checkpoint selection use training or dev data as appropriate. Connection tests establish that the workflow runs.

Historical Guidance and Annealed Guidance remain in standalone legacy workflows and validation records. The current native comparison uses the three loss-based transfer conditions above. [Workflow](BENCHMARK_COMPARISON.md) and [equations](METHOD.md)

The method IDs are `distillation`, `annealed_distillation`, and `mask_distillation` for KL-based Attention Distillation, Annealed KL-based Attention Distillation, and Attention-level Knowledge Distillation, respectively. Calibrate τ from the median of the microbatch estimates and λ from logical-batch gradient norms.
