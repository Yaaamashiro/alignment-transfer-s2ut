# Repository instructions

- Treat `ja-en-direct-s2st-benchmark` as the reference for established S2UT training and Colab operation. Prefer its pinned implementation and small adapters over independent replacements. Preserve working defaults and behavior where alignment transfer does not require a change.
- Keep scientific changes focused on alignment transfer and registration/comparison of separately trained baselines. Document necessary deviations and their validation in `docs/DEVIATIONS.md`.
- Use the fixed benchmark revision in `colab_runtime.py` for staging, performance and recovery. Do not silently follow its moving main branch or modify the benchmark/corpus repositories.
- Keep frequent writes, checkpoints, logs and per-pair artifacts on the Colab VM. Reuse benchmark ZIP staging, verified snapshot publication/restoration, bounded uploads and storage budgets for Drive persistence. Never silently fall back to repeated remote payload reads.
- Original corpus data and existing Baseline backups are read-only. Write derived archives outside `CORPUS_ROOT` and use dedicated local/output directories.
- Preserve native fairseq model/loss/optimizer/scheduler, precision, logical batches, completed budget and checkpoint selection for the matched comparison. Keep test data out of training and model selection.
- Integrate recovery into the normal notebook workflow. A new VM should use the same pinned code, settings, local paths and persistent directory; retain incomplete/unpublished work rather than deleting it or treating it as durable progress.
- Use Conventional Commits. Keep checkpoints, generated data, caches and large outputs out of Git.
