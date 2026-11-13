# Validation record

## 2026-11-14 — Calibration padding and teacher projection audit

- Compared identical teacher/student inputs with and without a source-padding mask before changing production code. With zero padding, KL, top-p region and mass, gradient norms and cosine, and λ and τ matched. Coverage by layer changed from `[.2500, .6250]` to `[.2917, .7083]`.
- With deliberately nonzero teacher padding, masking changed first-layer KL from 29.3471 to 1.1090 and top-p student region mass from .04988 to .17907. KL τ and λ changed from 19.9735 and 3.6096 to .25587 and 2.8299. Mask τ and λ changed from .57658 and 21.7486 to .45898 and 11.7326. Task gradient norm remained 2.4617. These are synthetic stress-test measurements.
- Verified that both calibration passes and the native criterion receive the same mask for a padded fairseq batch. Verified that the native teacher collater leaves source padding and target EOS/padding zero. The audit fixture measures .75 unmapped source mass, .50 valid unit row rate, and normalized valid projection rows. CPU regression suite passed 127 checks, followed by 23 focused checks on the final mask and projection cases.
- Passed 35 notebook, CLI, and calibration contract checks, including rejection of the previous implementation hash. Both pinned notebooks passed schema, Python syntax, and HTML checks. Three local CUDA checks passed on the first run. The Annealed FP16 recovery case encountered a Windows file-sharing error replacing the benchmark progress file and passed on an isolated rerun. It matched continuous training through update 5 after the update-2 and update-4 recovery checkpoints.
- Drive discovery found the experiment's common, prepared data, and unit extraction stages, plus the WAV packs. It did not find completed transfer manifest, imported Units, and Teacher stages under that experiment. Production projection coverage and T2TT link quality remain unmeasured. [Read-only Colab procedure](ALIGNMENT.md#run-on-saved-production-artifacts-in-colab)

## 2026-11-13 — Baseline revision reporting and shared notebook pin

- Passed 66 focused provenance, notebook, authentication, and logging checks. The POSIX process-group cancellation test was skipped on Windows. Verified both primary notebooks' schema, Python syntax, HTML rendering, and shared `5b398b5` pin.
- Verified that matching launch logs or metadata set `executed_revision_verified`, while a compatible notebook pin or missing evidence yields a warning on stderr and in the JSON report. The live Baseline revision and real-corpus Colab recovery still require verification with the original run evidence and GPU runtime.

## 2026-11-13 — Baseline provenance and Pilot recovery check

- Tested the historical `fcfbbfc` adapter against the exact Baseline runtime lock and checked successful and conflicting launch evidence, altered checkpoints, and missing revision evidence. The provenance check only reads the original Baseline artifacts.
- Four CUDA tests passed on a GTX 1650 Max-Q. The synthetic FP16 run published update 2, restored it into an empty run directory, passed the update-4 recovery check, and continued to update 5. The result matched uninterrupted training across the synthetic epoch boundaries, including model, optimizer, scheduler, RNG, sampler, scaler counters, and FP32 master weights.
- Forty focused provenance, notebook, and CLI checks passed. Verified the training notebook schema, Python cells, and HTML rendering. The actual Baseline launch logs, real corpus, Colab GPU, and DriveFS recovery remain to be checked in Colab.

## 2026-11-13 — Native transfer Pilot and diagnostics

- Passed 102 targeted CPU tests covering native training, calibration, ZIP recovery, Baseline registration, and the notebook commands. For all three transfer conditions, the tiny two-update run saved at update 1, rejected final registration while paused, restored a verified ZIP to the original path from a cold local directory, and continued to update 2. The resumed and uninterrupted runs matched model weights, optimizer state, scheduler state, iterator, RNG, and checkpoint selection. The KL case paused between regular saves and also produced a teacher-free dev UER score.
- Passed four CUDA tests on a GTX 1650 Max-Q with PyTorch 2.7.1+cu126. The Annealed KL Pilot also matched FP32 master weights and FP16 loss-scaler counters after recovery. The other three checks exercised the existing FP16 training and corpus integration paths.
- Passed four additional CPU checks, including an exact replay of the old `1c4ed71` calibration for λ, τ, per-layer losses and counts, and logical gradient norms. Verified the LF and CRLF Baseline-only compatibility rules and checked the wheel's new CLI and packaged source files.
- Passed 31 notebook and command tests after pinning training and evaluation to the tested implementation. The alternate transfer ID isolates transfer paths and shares the original Baseline and prepared data. Both notebooks passed schema, Python syntax, and HTML rendering checks.
- Verified Baseline registration and the unchanged native recipe on local fixtures. The active Colab session and its Drive artifacts were not accessed. Production corpus execution, Colab/Drive performance, and Pilot ASR-BLEU still require a Colab run.

## 2026-11-12 — Colab GitHub authentication and English documentation

- Verified anonymous clone, fetch, and checkout against the public GitHub repository at the exact revisions pinned in the training and evaluation notebooks. Checked that `origin` retained the plain HTTPS URL and that the public benchmark remote was reachable.
- Ran the authentication, logging, notebook, and artifact tests with 90 passed and one POSIX-only cancellation test skipped on Windows. Private access with valid, invalid, and missing credentials used a mock Git transport. Real private access and Colab Secrets require the manual check in [GitHub access from Colab](GITHUB_ACCESS.md).
- Validated the schema, Python syntax, and rendered HTML of all seven notebooks. Compared the executable code outside the bootstrap helper with the previous revision and checked the local documentation links.

## 2026-11-06 — Colab native FP16 preflight

- On a Windows GTX 1650 with PyTorch and Torchaudio 2.7.1+cu126, both pre-fix integration tests overflowed on the second update. fairseq reduced the loss scale from 128 to 64 and skipped the update.
- The integration Baseline now uses the pinned production runtime's FP16 retry. All three GPU preflight cases passed. Both release and canonical formats covered two Baseline updates, registration, and transfer save and resume.
- CPU checks passed 46 with 3 CUDA cases skipped. All eight pinned benchmark retry checks passed using an isolated fixture with the pinned fairseq sources. The full run on Colab L4 and real data remains untested.

## 2026-11-06 — S2UT audio streaming and cold-VM recovery

- A CPU test with the actual fairseq Trainer and loader built the ZIP index without fetching audio and completed update 1 while the second transfer was blocked. After publishing and restoring a verified snapshot into an empty audio cache, the next sample ID and features matched, and fetching one ZIP advanced update 1 to 2.
- On synthetic data, Baseline config and recipe checks extracted no WAVs at startup. The notebook GPU paths selected on-demand fetch and the CPU path used the existing ZIP builder.
- Full alignment suite **377 passed, 10 skipped** in 524.20 s. Legacy T2TT continuation passed. `stage-corpus` built an index with zero audio ZIP fetches, and the old S2UT implementation hashes matched for LF and CRLF.
- Passed 52 pinned benchmark streaming, staging, and parallel ZIP checks. Verified training and evaluation notebook schemas, Python and HTML, 77 wheel Python sources, compatibility data, and documentation links.
- Full-corpus execution, Colab CUDA, and concurrent DriveFS throughput remain untested. Initial audio preprocessing reads all input and accessed audio accumulates in the VM cache.

## 2026-11-05 — Bounded parallel CPU audio packs

- The benchmark runtime is pinned to `0c0024ea888206149a3119bf096b86ba66d63b66`, based on `4361e55` with audio ZIP changes and merged as `9f04d6c97948f9b371626aef487af65acbab57c5`. Artifact and recipe lineage use `4361e55`. Both primary notebooks fetch the verified runtime pin.
- Implemented one job per ZIP, two workers by default with an integer range of 1–4, bounded futures, separate worker directories, combined disk reservations, a VM OS lock, and cancellation and joining on failure. Verified 128 WAVs per ZIP, ZIP_STORED, namespace, order, plans, receipts, checksums, and file stamps. GPU recovery and training use the existing packs.
- Passed 36 staging and 105 benchmark regression checks covering progress, streaming, Colab, and preprocessing. With 11 receipts and a twelfth unpublished receipt from `4361e55`, four workers completed all 17 ZIPs with the corresponding 24 source WAVs unavailable. The 12 existing ZIPs kept their bytes and modification times.
- The controller uses its verified runtime to publish legacy T2TT saves while preserving the restored trainer hash, checkpoint contract, and optimizer. Updates 2 through 4 passed with the original `703b7e8` code and matching contract.
- The alignment suite initially had 375 passes, 10 skips, and one import allowlist failure in 478.94 s. Updating only the allowlist resolved the failing case. Final accounting is 376 passes and 10 skips across 386 cases, excluding the separate 35-case subset and standalone legacy check.
- Checked schema and Python for all seven notebooks, HTML for the two primary notebooks, 58 local documentation links, byte parity for 77 wheel Python sources, zero new lint warnings, and the actual runtime pin.
- Broad benchmark checks had 638 passes, 7 skips, 5 GPU exclusions, and 31 failures reproducible in the prior package. Causes included a missing fairseq fixture submodule, an existing Windows source hash overlay check, and mock timing of zero seconds. The Windows two-process Gloo test was excluded after waiting.
- With 1,024 local WAVs and eight ZIPs, median time over three runs at 1, 2, and 4 workers was 3.591, 2.181, and 1.585 s, or 2.228, 3.668, and 5.046 ZIP/s. DriveFS, Colab CUDA, and full-corpus processing remain untested. The Windows venv did not pass CLI checks for the compiled sampler and benchmark package metadata. [Measurement details and recovery](AUDIO_PACKS.md)

## 2026-11-05 — Baseline-only evaluation defaults

- The evaluation notebook now defaults transfer comparison switches to `False`. Tested Baseline-only restore, inference, and scoring, as well as comparison with three transfer conditions enabled. These paths perform evaluation.
- Related checks **12 passed** in 4.19 s. Also verified the two primary notebook schemas, Python, and HTML, 53 links, and wheel source parity. These counts are separate from the preceding full suite.
- Documented synthetic audio in GPU preflight. Full-corpus and Colab CUDA runs remain untested.

## 2026-11-05 — Corpus-independent inputs and experiment names

- Set public notebook IDs to `experiment-001` and `t2tt-001` and generalized stage, teacher, and runtime names. Corpus-specific Drive paths and counts are in an optional example. Configure the input root, manifest, and optional QC summary independently.
- Added an entry point for relative audio paths and absolute paths beneath the corpus root, alongside the original release format. Legacy CLI names remain available for existing runs. Resume with the original ID and revision.
- Synthetic release and canonical formats covered ZIP staging, unit shard reuse, pinned fairseq preparation, empty test, Baseline registration, three condition plans, native updates, save, and resume. HuBERT inference used a fixture.
- Full suite **368 passed, 10 skipped** in 472.77 s, including CUDA and Windows-specific skips. Verified seven notebook schemas and Python sources, HTML for the two primary notebooks, 53 links, 77 wheel sources, zero new lint warnings, and `pip check`.
- Colab CUDA, full-corpus processing, 400k production training, and evaluation remain untested.

## 2026-11-05 — Unified training and evaluation notebooks

- Consolidated training and preparation into `training.ipynb` and inference and comparison into `evaluate.ipynb`. Earlier notebooks moved to `notebooks/legacy/` and retain regression coverage.
- Ten notebook wiring checks passed. They covered CPU ZIP-only mode, T2TT-only execution, common inputs for three transfer conditions, evaluation without training calls, Baseline-only evaluation, and audio-free external T2TT evaluation.
- Related regressions passed 114 with two skips. A small native model restored and registered from a completed ZIP, and evaluation restoration completed without training.
- Verified schemas and Python for all seven notebooks, HTML for the two primary notebooks, document links, and the wheel. Production Colab GPU training and evaluation remain untested.

## 2026-11-05 — Kokoro Baseline registration with an empty test split

- Synthetic Kokoro data with two train, one dev, and zero test pairs covered preparation, `prepared_identity()`, two stock fairseq Baseline updates, registration, reread, and plans for three transfer conditions. Empty train or dev stops registration.
- With the same data, Annealed ran one update, saved, restored from a verified ZIP, and ran update two. The CUDA variant became part of notebook preflight.
- Related regressions passed 79 with five skips. Verified notebook schema, Python, and HTML, the wheel, and notebook registration preflight on small prepared data. Colab CUDA preflight and production training remain untested.

## 2026-11-05 — Kokoro JParaCrawl input and native Baseline

- Windows CPU ran 363 cases with 354 passes, eight skips, and one import-boundary failure. The isolated failure passed after allowing the new adapter.
- Synthetic PCM16 WAVs covered QC release import, empty test, ZIP restore with originals moved, unit shard reuse, pinned fairseq dictionaries and CTC labels, data locking, and the Baseline paper recipe lock. HuBERT used a fixture.
- Saved a small fairseq SentencePiece T2TT model to a ZIP and verified original train/dev manifest hashes, all-layer attention on the QC subset, and rejection of text or split mismatches.
- Verified five notebook schemas, Python and HTML, CLI arguments and the shared logger for both Kokoro notebooks, the wheel, and dependency resolution.
- Full-corpus verification, Colab CUDA HuBERT and CTC, 400,000-update Baseline training, transfer training, and evaluation remain untested.

## 2026-11-04 — Poster notation and calibration metadata

The benchmark workflow now displays Attention-level Knowledge Distillation. Tests verify that near-zero gaps, NaN, Inf, and invalid-layer microbatches are recorded under `rejected_microbatches` and that the native calibration JSON uses that key.

The README, method, projection, all-layer documents, and primary notebook use \(\tilde A_l^{\mathrm{T2TT}}\) for the projected teacher, \(\ell_l^{\mathrm{KL}}\) for per-layer KL, and \(\mathcal L=\mathcal L_{\mathrm{S2UT}}+\lambda\mathcal L_{\mathrm{align}}\) for the objective. S2UT loss includes the original unit and auxiliary losses. Existing computations, method IDs, and log IDs continue to apply.

Related checks **25 passed** in 35.46 s, including calibration reproducibility, update and resume for all four conditions, and the new JSON key. Verified zero lint findings, naming and notation, schema, Python and HTML for four notebooks, 44 links, and 75 byte-identical wheel sources. Counts are separate from the previous 350-case suite. Long training was outside this check.


## 2026-11-04 — Microbatch τ and policy-driven annealing

Full suite **350 passed, 8 skipped** in 489.24 s. Skips were seven CUDA cases and one POSIX process-group case on Windows. Related counts below are separate.

τ now takes the median of candidates from valid layers in each microbatch. The fixed 16 logical batches and logical-batch λ gradient calibration are retained. `alignment-calibration-v2` records `tau_calibration_unit=microbatch`, batch counts and indices, pair IDs, layer losses, and valid rows. It rejects v1 and preserves existing files.

Two microbatches with opposing worst layers showed that averaging could erase the gap, while individual candidates yield a worst-to-median weight ratio of 3. A tiny native test recomputed τ from two logical batches of two microbatches, checked the JSON, deterministic replay, zero optimizer updates, and update and resume for four conditions.

Canonical conditions YAML now drives the plan policy, loss, and logs. Changing the fraction to 0.7 affected the midpoint, end, and task-only logs and blocked resume with a conflicting policy. Validation rejects schedules other than cosine and zero, nonfinite, out-of-range, or Boolean fractions.

Invalid layers are excluded from KL mean and worst, entropy, and dev diagnostic means. The three public names are KL-based Attention Distillation, Annealed KL-based Attention Distillation, and Attention-level Knowledge Distillation. Internal IDs remain stable. The tail and its logs use `task_only`. Validation and inference are teacher-free.

Related checks **32 passed** in 43.70 s. Verified no lint findings in 14 changed Python files, source AST, four notebook schemas, Python and HTML, names, 44 links, `pip check`, and all 75 wheel sources. The wheel and source-archive rebuild bundled byte-identical canonical YAML and imported independently from the editable tree. The build generates the copy from one editable source.

The notebook writes new calibration and checkpoint stages under `research-v3-*`. Run fresh top-p and calibration there. The Baseline, corpus, T2TT attention, projected teacher, native recipe, and ZIP runtime remain reusable. CUDA calibration, FP16, six-layer VRAM, real-corpus quality, and long training remain unmeasured. The Windows tiny sampler fixture follows the prior test setup.


## 2026-11-03 — Calibrated loss-only layer-wise distillation

Full suite **337 passed, 8 skipped** in 494.94 s, with seven CUDA skips and one Windows POSIX process-group skip. The following results belong to this run.

Tested the Baseline and three loss conditions, detached KL teacher and valid rows, stable top-p with padding exclusion and at least one frame, mask loss 0 and 0.3 for mass 1 and 0.7, stable LogSumExp and its softmax gradient, and the cosine schedule at 0, 0.8T, and the tail. The native capture context carries shape and scope, and logits matched bitwise across different priors.

On pinned two-layer fairseq, ran top-p diagnostics, τ and λ calibration, two updates per condition, and save after update one and resume to update two. Calibration blocked Adam.step and preserved state, update zero, and empty accumulated gradients. Recalibration reproduced JSON. Both KL conditions shared coefficients. Changed calibration α or λ rejected resume. Near-zero gaps, nonfinite values, and zero gradients stopped calibration.

The native loss difference matched `sample_size * lambda * worst-layer LogSumExp`, and the zero-λ tail matched task loss. Updated tests for the microbatch definition. Zero supervision in single and all-layer modes matched stock logits, losses, and updates. Registration, FP16 overflow, verified ZIP, and source-only generation regressions remained covered. Historical Guidance is in the standalone path.

Ran separate teacher-forced dev diagnostics for per-layer KL, mask mass, and weights, plus loss and layer curves. The Mask checkpoint produced raw attention and teacher-region PNG and NPZ files. Native validation and forward worked after removing the teacher directory. The new update JSONL entered the existing freeze and ZIP path.

Ran notebook cells through CLI parsers and checked four-condition order, common inputs, calibration and budget, shared evaluation, default-disabled training and calibration, required top-p selection, and saved diagnostic and calibration recovery. Updated research notation and names.

Verified zero pyflakes findings for 20 changed Python files, source AST, CLI import and help, four notebook schemas, Python and HTML, 44 links, `pip check`, and 74 wheel source files. Lint tooling was isolated under `outputs`.

Windows tiny calibration tests use the same singleton batch sampler as the tiny Baseline fixture because the compiled sampler is unavailable there. Colab preflight requires the compiled sampler. CUDA calibration, FP16, VRAM, real-corpus top-p and coefficients, long training, and BLEU gains remain untested.


## External text-only T2TT

Full suite **342 passed, 8 skipped** in 459.38 s. Skips were seven CUDA cases and one POSIX process-group test on Windows. Subset results below are separate.

A four-field audio-free manifest trained for two updates with character and SentencePiece on smoke and pinned fairseq, generated dev output and BLEU, and extracted all-layer attention. Tests blocked audio feature, hash, path resolution, and existence calls. Adding audio paths and hashes preserved checkpoint, evaluation, and attention reuse. Text changes and ID or split leakage were rejected, and the legacy speech hash matched.

Added synthetic audio and CTC and unit fixtures and projected text-v1 attention into a speech-identity teacher. Audio-dependent stages reject text-only manifests, and CTC and units stop before model downloads. Legacy checkpoints use original audio metadata hashes and retain evaluation, extraction, and reuse coverage.

Both tokenizers resumed from pinned verified ZIPs after update two and VM-local loss, then reached update four. Added audio columns left canonical text bytes stable. Model, optimizer, RNG, epoch, cursor, and contract matched four uninterrupted updates exactly, and the durable summary reported four.

Notebook cells for both tokenizers passed through CLI parsers, covering dedicated stages, completion gates, optional attention, and logging. Four external evaluation checks now locate the BLEU table by content despite an earlier diagnostic table. The final external and notebook subset **40 passed** in 65.29 s.

Checked four notebook schemas, Python and HTML, 80 local links, `pip check`, and 70 byte-identical wheel sources. Real JParaCrawl, 200k GPU training, and real-corpus BLEU remain untested.

## 2026-10-29 — SentencePiece T2TT

After SentencePiece Unigram, the full suite **314 passed, 8 skipped** in 421.75 s. Checked separate train-only vocabularies, fixed order, one thread, identity normalization, whitespace, special IDs, serialization, hashes, codepoint spans, and UTF-8 offsets. Fullwidth, combining, emoji, tab, LF, CR, repeated spaces, and special-token-like literals round-tripped. Held-out codepoints retained UNK positions. Tests cover immutable piece-view owner lifetime.

Added protection for character stages with only best or last present. The related SentencePiece subset **19 passed** in 34.89 s. After notebook display wiring, **25 passed** in 11.53 s. The 314-pass full suite preceded the protection change and is separate from both subsets.

Small smoke and pinned fairseq models covered fresh SentencePiece training, same-tokenizer resume, dev BLEU, single and all-layer attention, character-to-subword CTC timing, teacher projection, legacy S2UT, and native dataset, criterion, and backward. Legacy character evaluation, extraction, resume and persistence passed. Rejected vocabulary, normalization, asset hash, tokenizer mix, and character resume or overwrite mismatches.

Fixed 300 evaluation used the actual benchmark loader, normalization, and BLEU, with neural generation replaced by a fixture. Checked checkpoint tokenizer restore and codepoint OOVs. Added length buckets, repeated 4-grams, empty and unique counts. Fresh SentencePiece restored from a verified ZIP at update two and matched uninterrupted update four for model, optimizer, RNG, epoch, cursor, and contract. Tokenizer assets and diagnostics restored, and completed checkpoints performed no new updates.

Checked Colab paths, CLI wiring, and completion gates for both tokenizers, 20k and 200k, and single and all-layer modes. Verified notebook schema, Python and HTML, 74 links, `pip check`, and 68 wheel sources. Protobuf 4.25.3 was pinned for model-setting checks. Skips were seven CUDA and one Windows POSIX test. Real GPU training, real-corpus vocabulary and lengths, and production BLEU remain unmeasured.

The synthetic two-train, one-dev fixture requested 96 pieces per language and produced source vocabulary 24 and target 19. Train means were 21 Japanese codepoints to 13 tokens and 21 English codepoints to 15 tokens. Dev had one unknown codepoint per language, so exact round trips failed as expected. Use the production run's `tokenization.json` for real-corpus measurements. The entries below are historical.

T2TT extension, loss monitoring, notebook, and logger checks **56 passed** in 266.56 s. The parent tails complete local JSONL rows and distinguishes train and dev measurement steps. Tested partial writes, monitoring errors, final output, and thread shutdown. Checked verified ZIP summaries against original metrics and checkpoints, history bounds, actual dev values, source hash stability, tamper rejection, checkpoint preservation after summary errors, and retention.

Resumed with byte-identical saved controller code and generated a missing summary for a completed run without updating checkpoints. Smoke and fairseq exact continuation, original Git source recovery, and interruption recovery passed. Verified notebook schema, Python and HTML, 70 links, `pip check`, and 64 wheel sources. GPU monitoring, production 200k summary publication, and real loss values remain untested. The following full-suite counts precede this 56-case subset.

After single-layer teacher, legacy head, and diagnostic layer fixes, the full suite **289 passed, 8 skipped** in 403.32 s. Range-check before normalizing positive and negative layer indices, accept equivalent layers, and reject invalid or different ones. Legacy requires matching head sets. Packets without metadata can reuse final-layer all-head defaults with identical artifact hashes. Native checks plans and delayed input reads.

Smoke and fairseq rejected mismatches and trained and plotted matching layers. Smoke also extracted a selected-head teacher and trained and plotted with that head. `--layer` diagnostics matched direct teacher, prior, and entropy projection and recorded layer and head metadata. Explicit `--all-layers --layer -1` conflicts were rejected. Checked all notebook schemas, Python and HTML, 70 links, CLI help, `pip check`, and 63 wheel sources.

Skips were seven CUDA cases and one Windows POSIX case. FP16 GPU training, recovery, and memory require notebook GPU preflight. The entries below are historical.

Full suite **264 passed, 8 skipped** in 387.28 s after review fixes. Single-layer use of an all-layer teacher selects the requested layer, including negative indices, consistently across training and visualization. Verified projection with distinct two-layer matrices, original artifact hashes, and intermediate-layer legacy training and PNG and NPZ output on smoke and fairseq.

Native plans reject selected-head teachers in both scopes and use all-head averages. `--allow-scope-ablation` records intentional scope changes in both comparison CLIs while checking method, prior, recipe, initialization, budget, coefficients, teacher, and signatures. Tested legacy single-layer pairs, native final and all-layer comparisons, CLI output, and rejection of other changes.

GPU preflight covers a tiny two-layer FP16 update, save and restore, and second update for Distillation, Annealed Guidance, and Guidance, with measured beta and peak allocated and reserved VRAM JSON shown with `-s`. Seven CUDA cases and one Windows POSIX case were skipped locally. Full six-layer training, VRAM, and BLEU require GPU data. Verified schema, Python and HTML, 70 links, CLI help, `pip check`, and 63 wheel sources.

All-layer decoder transfer full suite **249 passed, 6 skipped** in 369.07 s. The visualization subset **25 passed, 2 skipped** in 110.98 s. Two-layer tests checked one-pass T2TT and S2UT capture, per-layer priors, UNK masks, FP32 NPZ, shared `[L,U,F]` projection, L-scaled memory budget, single-layer reuse, and rejection of incompatible scope, counts, and heads.

Varying valid-row counts, gradients from per-layer normalization across microbatches and then layer averaging matched joint calculation. The native loss difference matched Trainer main-token normalization. With beta zero, single and all-layer modes matched stock fairseq logits exactly, loss, and post-update parameters. All layers had finite nonzero q and k gradients. The native Trainer completed two updates per historical condition, beta decay, and shared inference after removing the teacher.

Legacy and native visualization produced corresponding teacher, raw, and guided layer PNGs and NPZs. Notebook wiring covered four conditions, independent stages, common evaluation, and single-layer, Monotonic-only, and T2TT-extension compatibility. Added CUDA all-layer parity and FP16 with FP32 prior checks. Skips were five CUDA and one Windows POSIX. Production GPU, VRAM, and real-corpus BLEU remain unmeasured. Checked notebook schema, Python and HTML, 69 links, CLI help, `pip check`, and 63 wheel sources.

Colab cell [10] passed dependency and CUDA checks, then failed three exact CPU continuation cases with 223 passing. The fixture set `torch.set_num_threads(1)` in the parent only. Local `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` reproduced two smoke and fairseq cases, with smoke parameter difference up to about 3.53e-8 and matching contracts.

Set one CPU thread in parent and child within the fixture and restore settings afterward. Verify child thread count. With the outer environment still at two threads, related checks **26 passed** in 159.00 s. Model, optimizer, and RNG continue to be compared with `torch.equal`.

With that two-thread environment, `pytest -m 'not gpu'` gave **226 passed, 1 skipped, 3 deselected** in 315.19 s. The skip was a Windows POSIX process-group case and the deselected tests require CUDA. Colab L4 retest remains pending for this change.

The verified 20k-to-200k extension full suite **225 passed, 4 skipped** in 310.39 s. Small real training and benchmark ZIPs checked last and best, Adam, epoch and cursor, best loss and bad validations, RNG, exact continuation, unchanged source work and backups, and recovery after target VM loss. Small official fairseq parameters matched uninterrupted training. Verified rejection of corruption, missing files, revision, step, manifest, config, and recipe mismatches, as well as metric continuity and inherited best selection.

Exported matching old Git source `703b7e8` and used the original trainer's contract validation. Checked separate dev and Drive save intervals, two recent snapshots plus milestones, cleanup only after verified upload, and nonfatal cleanup errors. Windows progress-reader and atomic-replace contention retries briefly while unrelated checkpoint errors propagate.

Notebook checks covered 20k and 200k dev, external, attention, teacher, and S2UT paths, the shared checkpoint, evaluation gates, completed-run evaluation, and Monotonic-only compatibility. Verified schemas, Python and HTML, CLI help, `pip check`, and wheel packaging. Skips were three CUDA and one Windows POSIX. Real 20k GPU continuation, 200k results, and dev and FLEURS BLEU remain unmeasured.

Added external evaluation input restoration after a fresh VM lacked `t2tt/checkpoint_best.pt`. **189 passed, 4 skipped** in 136.14 s. `restore-inputs` checks verified benchmark ZIPs, original run path and stage, completed budget, and best checkpoint. Restore only the model and matching dev metrics and predictions, preserving incomplete local and external results.

Seven added real-ZIP tests covered 19k best in a completed 20k run, retained unpublished local files and external results, identical Drive bytes, reuse on rerun, and rejection of missing, incomplete, corrupt, mismatched-stage, best-less, and conflicting dev results. Checked restore, external evaluation, and read-only verification wiring, schema, Python and HTML, CLI help, and `pip check`. Three CUDA and one POSIX cases were skipped. Real-model external BLEU remains unmeasured.

Colab 6.1.1 loaded the fixed 8C 300 samples and then failed because optional `text-evaluation/dev-text.json` was missing. External evaluation now proceeds with beam 5, max chars 1000, length penalty 1.0, and native BLEU from the existing CLI. Existing dev data still requires checkpoint, generation, metric, and prediction checks, and partial or corrupt dev data raises an error.

Related checks **46 passed** in 62.80 s. Covered absent dev JSON or prediction, omitted arguments, invalid existing dev, notebook comparison skip, and read-only revalidation and reuse. Reference isolation, BLEU recalculation, real ZIP restoration, 300 predictions from official fairseq and synthetic checkpoints, and notebook compatibility passed. Verified schema, Python and HTML, and CLI help. The previous full suite was 174 passes and four skips. Colab external BLEU remains unmeasured.

External evaluation review **174 passed, 4 skipped** in 125.23 s. Reproduced a `dev-text.jsonl` derived prediction collision and now checks all output paths against protected inputs before generation. Restored results undergo read-only dev, checkpoint, fixed-set, prediction, and corpus BLEU verification before display. Incomplete evaluation stops and corrupt dev predictions are rejected.

Alongside stubbed tests, actual loaders, tokenizers, and beam-5 generation evaluated 300 cases each with smoke and official fairseq tiny synthetic checkpoints. Verified EOS, UNK, and actual checkpoint step, `pip check`, notebook schema, Python and HTML, verify CLI, and wheel source. Skips were three CUDA and one POSIX. These are CPU integration results. Real GPU external BLEU remains unmeasured.

Fixed 8C 300-sample evaluation full suite **170 passed, 4 skipped** in 114.29 s. Sixteen cases checked IDs and order, tamper rejection, reference-independent source-only generation, native and normalized BLEU, normalization, UNK, cache recovery, and checkpoint and dev settings. Real benchmark ZIPs restored after 100 predictions, generated only the remaining 200, and restored completion without regeneration. This integration used stubbed neural generation, with actual generation covered by dev tests.

Verified notebook 6.1.1 commands, separate stages, dev values, ten examples, schema, Python and HTML, `pip check`, CLI help, 56 local links, and wheel sources. Skips were three CUDA and one POSIX. GPU evaluation with the user's saved Drive set and trained checkpoint remains untested, so no actual FLEURS BLEU is reported.

Same-day rerun **154 passed, 4 skipped**. Forty seeded pairs with varied length, whitespace, quotes, Unicode, and unknown characters matched single and batched attention for intermediate and final layers on smoke and official fairseq. Save, restore, interruption, hash, and native training also passed, plus `pip check`, environment checks, 22 CLIs, three notebook schemas, Python and HTML, 53 links, and wheel sources. The source pin was `107605f4f5bd7b53722d73ced3fa5a76560e7a9d`. Three CUDA and one POSIX case were skipped.

The 2026-10-27 rerun **150 passed, 4 skipped**. Checked all-head, single-head, and multi-head attention, padding, UNK, single and batch parity, exact batch size one, partial writes and reuse, and budget limits. Added CUDA attention parity to notebook preflight. Local CPU skipped three CUDA cases and one POSIX process-group case.

Reproduced a disconnect after NPZ write and before its sidecar that left a completed ZIP missing one pair. Producers now verify internal payload and metadata before filling missing checksums. NPZ hashes and modification times remain stable. Read-only `get()` preserves inputs. Normal completion rejects unfinished NPZs, and failure saves finished pairs. Real benchmark ZIPs recovered the old missing pair and restored the new incremental pack.

Checked dependency metadata and imports, `pip check`, 22 CLI helps, three notebook schemas, Python and HTML, local links, and wheel sources. Clarified standalone prefetch versus the primary workflow's ZIP staging. Real-corpus Colab GPU training and translation quality require separate validation.

On 2026-10-27, attention extraction changed from per-pair checkpoint and vocabulary hashing to one checkpoint SHA-256 and up to eight pairs per forward pass, bounded by padded token and attention counts. Verified old single extraction parity (atol 2e-6, rtol 2e-5), legacy reuse and hashes, tamper and mixed-packet rejection, batch limits, progress, and reuse on smoke and official fairseq. Full suite **136 passed, 3 skipped**, including two CUDA and one Windows POSIX skip.

A tiny CPU fixture with 128 pairs and a 64 MiB optimizer-like checkpoint payload took 9.80 s with 128 hashes and 2.78 s with one hash and 16 forwards. Maximum attention difference was 5.96e-8. This is about 3.52× with locally cached files. GPU batch parity and Colab performance remain unmeasured.

Colab full-manifest reported a mismatch between common text and auxiliary labels. The TSV reader's CSV quoting changed literal labels and could merge rows after an unclosed quote. Matched the benchmark's `QUOTE_NONE` and preserved header, column, and ID validation. Tested four quote forms, resume, row merging, malformed TSV, and real text mismatches. Full suite **127 passed, 3 skipped**. Real-data attribution requires a literal TSV check or Colab rerun.

After unified logging on 2026-10-26, **115 passed, 3 skipped**. The fixture prioritizes the Colab-installed pin over an unrelated updated sibling benchmark checkout. `S2ST_BENCHMARK_TEST_ROOT` selects a separate verification checkout. Ran the full suite with pinned source and tested source precedence in save and restore.

Colab CPU preflight passed Python 3.10.22, dependencies, fairseq provenance, the compiled sampler, and benchmark pin, then failed Matplotlib. Reproduced the inherited `MPLBACKEND=module://matplotlib_inline.backend_inline` failure in a separate venv and set `MPLBACKEND=Agg` in three notebooks. Their settings passed new-process Matplotlib import and PNG save. Failure log tails expose preflight JSON.

A Colab CPU result of 95 passes and one failure exposed Linux `ENAMETOOLONG` when `identity_for` tried to treat long `python -c` input as a file. Handle that case while binding the complete command and preserving external file hash and permission checks. Verified Linux error simulation, permission errors, real ZIP interruption and restore, eight runtime tests, and the full suite.

Staging output changes passed **27 notebook and CLI tests**. Verified live stderr display and local flushing, separate stdout JSON, live output before child completion, failed exits, saved logs, and interruption cleanup. Also verified notebook schema and Python.

Unified the standard-library logger for setup, checks, JSON, preparation, training, inference, and evaluation. Verified AST parity of pre-install helpers and no direct subprocess calls in three notebooks. Covered liveness and running metadata, fast lines, JSON separation, launch and exit failure, invalid JSON, and interruption. GPU CSV and benchmark ZIP paths remain covered.

Windows checked nested observer process-group settings and skipped actual POSIX group interruption. The three skips were that case and two CUDA cases. Native Trainer parity, checkpoint resume, and real benchmark recovery passed on CPU. Verified notebook schema, Python and HTML, links, and wheel modules including the logger.

In an isolated Windows Python 3.10.22 and PyTorch 2.7.1 CPU environment, `PYTHONUTF8=1 python -m pytest -q` gave **115 passed, 3 skipped**. Installed optional dependencies including SentencePiece 0.2.0 and passed `pip check`. UTF-8 mode is required to read Japanese native TSVs on Windows.

Fetched official fairseq commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` separately. Used upstream `READTHEDOCS=1` to build a pure-Python package due to Windows symlink and C++ build constraints. Verified the clean commit, tracked sources, and installed Python source contents.

Verified CLI help, imports, wheel contents, Python syntax, and local Markdown links. Three notebooks passed nbformat 5.10.4 schema validation with warnings treated as errors and HTML conversion in nbconvert 7.16.6. Browser rendering, external checkpoints, and Colab installation were outside the local check.

`environment` verifies metadata and imports for preprocess, evaluation, and dev extras alongside `pip check`. Explicit flags require CUDA, benchmark pin, and the native sampler. Normal local preflight passed. `--native` found unbuilt `fairseq.data.data_utils_fast` and exited 1.

Local PyYAML 6.0.3, Cython 3.3.0, and setuptools 84.2.0 differ from the Colab pins 6.0.2, 3.2.9, and <81. The benchmark source was loaded from a test fixture outside package metadata. Colab CPU preflight reported a working compiled sampler. Linux native CLI epochs, GPU updates, and external models need separate verification. Existing OmegaConf metadata and fairseq importlib and autocast deprecations remained warnings.

Checks covered the following behavior.

- Passed all four T2TT and Monotonic switch combinations and two- and three-condition evaluation with and without ASR through real CLI parsers. Monotonic uses no teacher and shares dev settings.
- Checked all standalone notebook commands and five conditions. Argument lists with `check=True` preserve spaces in paths and stop on failure.
- Environment preflight detects dependency conflicts, wrong optional versions, DLL import failures, Python and fairseq mismatches, and native sampler errors.

- Used benchmark revision `4361e55a99c7f4ea3a00c5c138f6820c34a07d06` directly. Real ZIP publication, verification, and restore recovered interrupted per-pair processing on a fresh VM fixture and reused completed stages. Incremental packs omit already saved pairs.
- Restored GPU-side staging from CPU WAV packs with source WAVs unavailable. Native fairseq read dictionaries, config, TSVs, and audio from local copies.
- Ran two T2TT updates through the real resident session and upload worker. Restored checkpoint, optimizer, and best from ZIP after losing work. The completed updates were retained.
- Windows refuses atomic replacement of open files, so the resident T2TT test made one final snapshot. The native hook separately covered periodic saves.
- Tested native S2UT optimizer updates and snapshot freeze through benchmark performance hooks and gradient auditing. Full snapshots included contract, performance, optimizer, and audit state.
- Rejected corrupt incremental ZIPs and budget excess while retaining existing recovery points. An upload failure left local completion evidence that was republished on rerun. Checked Colab paths, pin, 600-second interval, and 1024 GB budget.
- Registered a separately trained official fairseq checkpoint read-only after verifying recipe, metadata, and data hashes. Verified snapshot ZIP and member hashes and pinned restore.
- Ran official Trainer updates. Transfer initial weights matched the same-seed official model bitwise, and beta-zero post-update parameters matched the official Trainer.
- Verified native optimizer and checkpoint save and resume, rerunning completed runs, dev validation after teacher removal, and stock SequenceGenerator inference shared by Baseline and transfer.
- Verified report-only best selection by full dev unit main loss and RNG preservation across validation. A synthetic scaler checked retries for confirmed FP16 overflow with the same batch, RNG, and buffers.
- Verified full train/dev and saved unit import and rejected changed prepared targets, dictionaries, recipes, generation settings, and checkpoints.
- The Colab workflow builds the T2TT and teacher backbone from Baseline registration. Checked native FP16 preflight, full train/dev preparation, and shared inference and evaluation defaults and syntax.
- Native S2UT Baseline and local extension logits matched exactly.
- Beta zero matched Baseline exactly. A uniform prior matched within rounding tolerance.
- A biased prior changed context and output and backpropagated into cross-attention.
- T2TT and S2UT matched native encoder, decoder, and cross-attention types and shared parameter shapes.
- Checked synthetic projection, row normalization, masks, invalid rows, CTC repeats and Viterbi, unit reduction and timing, and the annealing endpoint.
- Checked train-only character vocabulary, split leakage, artifact hash and metadata checks, and recovery after interrupted checksum writes.
- Ran native T2TT training, attention, projection, four S2UT conditions, and visualization with synthetic audio, CTC, and units.
- Ran four smoke conditions without external models. Generated units and test audio after removing T2TT, MT, CTC, and teacher data.
- Inference worked with only the checkpoint and Japanese audio. Inputs containing reference text were rejected.
- Interrupted and continuous CPU training matched parameters bitwise. Changed seeds or inference settings blocked resume.
- Verified idempotent manifest, inference, and evaluation resume and source-only export.
- All shared architecture settings were `MATCH`. Four-condition control comparison reported `fair_comparison=true`.
- Uniform mixing preserved zero padding, EOS, and invalid rows and beta-zero output.
- Across microbatches with varying valid rows, total KL divided by total valid rows matched joint gradients. The Trainer also matched full-batch and accumulated losses and gradients.
- FBANK and convolution time support matched convolution gradient support, including endpoint padding and output-length mismatch checks.
- Monotonic prior trained without T2TT or CTC and supported dev diagnostics and teacher-free inference.
- Strong source and target character decoders and decoder CTC trained, and weighted auxiliary losses matched upstream fairseq MultitaskCriterion.
- Strong 16-bit FBANK scaling and utterance CMVN matched native extraction. Continuous and resumed CPU runs matched parameters bitwise with SpecAugment and auxiliary decoder dropout.
- Strong Baseline, Annealed Monotonic, and Annealed T2TT shared initial weights and recipe. Comparison rejected different source mappings.
- Checkpoints embed train-only auxiliary character vocabularies and SentencePiece models. Strong Baseline infers units from the checkpoint and Japanese audio.
- Alignment diagnostics produced original-character coverage, unit row rate, raw and smoothed entropy, and PNG and NPZ files.
- The default strong model (12 encoder layers, 6 decoder layers, dimension 256, and two layers per auxiliary decoder) produced finite gradients on synthetic audio and units.
- The wheel included the strong YAML and added modules. Checked Colab syntax and CLI arguments statically.
- Pilot train/dev selection was reproducible, preserved splits, skipped test audio reads, and rejected changed input hashes, seed, or limits on resume.
- Handled corpus aliases and moved Colab audio paths and detected known split leakage in excluded test rows.
- Checked three strong commands and the beta-zero tail. Logged exit codes, full output, and elapsed time on command failure.
- Pilot reporting deduplicated resumed steps and rejected mismatched dev data or config. Verified dedicated notebook syntax and integration JSONL.
- Verified `nvidia-smi` CSV collection and sampler shutdown on a local GTX 1650.
- Initial inputs were consumed while later inputs were being fetched, and the queue stayed bounded.
- Partial consumption under `max_tokens` preserved order and kept unconsumed prefetch out of checkpoint position.
- Across five method and prior conditions, synchronous, cached multiworker, and resumed worker-count variants matched CPU parameters and RNG.
- Strong native parameters matched on CPU across synchronous, cached, and resumed loading with SpecAugment and auxiliary dropout.
- Checked concurrent cache deduplication, LRU limits, no caching with insufficient space, corruption detection, original artifact hashes, and changed-source rejection.
- The first training forward ran before dev payload reads. A later sample's unit config mismatch stopped when that sample was consumed.
- Pilot dev limit was 200 in the function, CLI, and Colab.
- Verified source-only T2TT autoregressive generation, EOS and output limits, and corpus BLEU on all dev samples. Exact translations scored 100, references stayed out of decoding, and evaluation worked without audio.
- Native fairseq checkpoint text generation and evaluation passed. Tested interruption recovery and rejection of changed generation settings, dev data, and prediction caches.
- Pilot summaries attached T2TT dev evaluation and rejected different teachers, dev data, or test evaluations. Checked new evaluation CLI and module in the wheel and syntax of both Colab notebooks.

Three CUDA skips cover native FP16 Trainer connectivity, single and batched T2TT attention parity, and full CTC, HuBERT, and official vocoder integration. Windows also skips nested POSIX process-group interruption. The pure-Python fairseq build covers Trainer microbatch updates. Colab's Linux build supplies the compiled sampler for the CLI epoch iterator.

Drive records confirmed `experiment-001`, recipe `paper-fisher-v1`, a 365000-update snapshot, prepared data and unit paths, and training config JSON. Download and registration of the approximately 1 GB checkpoint, real-corpus preparation and training, dev T2TT BLEU, audio quality, ASR-BLEU comparisons, and Colab execution remain unverified. Smoke and Trainer checks establish implementation behavior.

The local verification environment is `outputs/native-venv/`. Use it for `python -m pytest -q` and CLI commands. Git excludes the downloaded Python and dependencies, upstream sources, checkpoints, audio, and images under `outputs/`.
