# Transfer Metric Schema v3

Schema v3 measures three fresh transfer runs beside the existing 400k Baseline. The transfer profile uses `experiment-001/transfer-fixed-v2` in `direct-s2st-v3` under `yaaamashiro-national-institute-of-technology-okinawa-college`. The Baseline Run is `5db658f8baac28cbe211f3b8` and belongs to the Benchmark importer. The original transfer stage, BLEU directories and W&B project keep their existing identities. [Benchmark schema at ad2577e](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/ad2577e995d4406a0cbc29d0365ce510a9bcf9d3/docs/WANDB_METRIC_SCHEMA_V3.md) · [Benchmark CPU Reporter at 6d242da](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/6d242da09902bc15c0cff85f6eb331e5274d0731/src/direct_s2st/baseline_wandb_v3.py) · [Shared fixture](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/ad2577e995d4406a0cbc29d0365ce510a9bcf9d3/tests/fixtures/wandb_metric_schema_v3.json)

The training runtime remains pinned at `0c0024ea888206149a3119bf096b86ba66d63b66`. The recipe remains `4361e55a99c7f4ea3a00c5c138f6820c34a07d06` and Fairseq `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`. The Benchmark Reporter SHA is a specification reference for this repository.

## Comparison series

For microbatch \(i\), \(C_i\) is the raw label-smoothed Unit CE sum, \(N_i\) is the main sample size, \(A_{i,k}\) and \(M_{i,k}\) are auxiliary loss sum and sample size for task \(k\), and \(B_i\) is the actual weighted alignment contribution added to the objective. Sizes are summed over all gathered microbatches. Training records one successful logical update and Dev records a complete teacher-forced pass of 1,453 examples.

| Metric | Formula and source | Unit and cadence |
| --- | --- | --- |
| `train/main_loss`, `dev/main_loss` | \(\sum C_i / \sum N_i\), raw main criterion numerator, label smoothing 0.2 | Nats per target unit, Train every 100 and Dev every 1,000 |
| `train/task_loss`, `dev/total_loss` | Actual task criterion sum divided by \(\sum N_i\). Checked against \(\sum(C_i+8A_{i,source}+8A_{i,target}+1.6A_{i,ctc})/\sum N_i\) | Nats per main target unit, Train every 100 and Dev every 1,000 |
| `train/total_loss` | Actual returned objective sum divided by \(\sum N_i\). Checked against Task plus \(\sum B_i/\sum N_i\) | Objective value per main target unit, every 100 |
| `train/aux/source_letter_loss`, `train/aux/target_letter_loss`, `train/aux/decoder_target_ctc_loss` and the corresponding `dev/aux/*` | \(\sum A_{i,k}/\sum M_{i,k}\) for each unweighted auxiliary | Nats per task sample, Train every 100 and Dev every 1,000 |
| `train/learning_rate` | Scheduler value recorded after the successful optimizer update | Learning rate, every 100 |
| `eval/aux_bleu`, `eval/aux_failures`, `eval/aux_samples` | Qualified target-letter decoding result, recognized decoder failures, fixed Dev scoring population | Corpus BLEU points and counts, 500 examples at 1k/5k/10k/25k/50k/100k/200k/400k |
| `eval/asr_bleu`, `eval/asr_failures`, `eval/asr_samples` | Qualified unit-to-vocoder-to-ASR result, recognized decoder failures, fixed Dev scoring population | Corpus BLEU points and counts, 100 examples at the same milestones |

The recorder saves raw sums and denominators in `loss_sums` and the sampled population in `sample_count`. Raw criterion losses are in nats. Baseline's importer converts normalized stored bits using \(\ln 2\). V1/v2 reuse the name `dev/main_loss` for bits under the legacy profile, which is why v3 publication has a distinct Run binding.

At 1,000 updates the comparison grid is 10 Train points per metric, one Dev point per metric, and one of each BLEU triplet. At 1,100 the Train grid has 11 points and the earlier Dev/BLEU point remains. At 400k, the expectations are 4,000 per Train metric, 400 per Dev metric, and eight per BLEU metric. Baseline Aux-BLEU remains unavailable under the Benchmark evidence rules. Update 1 is displayed separately as a diagnostic in the audit.

### Diagnostic series

- The existing logical `alignment/update/*` values retain their criterion weights and meaning. Required at the comparison grid for all methods are `loss`, `weighted_loss`, `D_0…5`, `w_0…5`, `kl_layer_0…5`, `kl_mean`, `kl_worst`, `teacher_entropy`, `student_entropy`, `valid_rows_0…5`, `worst_layer`, `worst_layer_fraction_0…5`, `lambda`, and `tau`. The original JSONL is also checked each successful update. `alignment/update/task_loss` is an alias of the same measured `train/task_loss`.
- Mask adds required `top_p`, `region_mass_0…5`, `coverage_0…5`, and `selected_frames_0…5`. Annealed KL adds required `annealing_progress` and `task_only_phase`, with `task_only_start` once the pre-update counter reaches 320,000. The first successful update in that phase is 320,001. Diagnostics continue in the task-only tail and the applied weighted alignment loss is zero. KL and entropy use natural logs. Counts, weights, probabilities and Mask losses retain their native scale.
- For these diagnostics, `D_l` is the mean over valid teacher rows in layer `l`, using row KL or `1 − Student region mass`. The per-microbatch aggregate `loss` is `τ × log(mean(exp(D_l/τ)))` over valid layers, and `w_l` is `softmax(D_l/τ)`. The reducer weights scalar diagnostics by main sample size across microbatches and sums `valid_rows_l` and `selected_frames_l`. Region mass sums Student probabilities inside the teacher Top-p set, while coverage divides the selected positions by allowed source positions for each row. The worst layer maximizes the reduced `D_l` over eligible layers, and its microbatch frequency is sample-size weighted. [Pooled denominators and window diagnostics](WANDB_METRIC_AUDIT.md#metric-definitions)
- Logical training diagnostics are `train/unit_nll` (raw NLL sum / target tokens in nats), `train/gradient_norm` (uncapped L2), `train/clip_applied` (binary when clipping applies), `train/loss_scale` (actual FP16 scaler), `train/overflow_retries` (confirmed attempts before one successful optimizer update, from the pinned scaler counter), and `train/seconds` (cumulative training wall-clock seconds). These are separate from the common four-run comparison. Missing diagnostics are omitted.
- Optional Fairseq series retain their existing scopes. `train/window/*` uses Fairseq's meter window and original precision, with CE in nats, optimization/NLL and auxiliary losses in bits, recipe weights, reported optimizer statistics, and clip percentage converted to a fraction. `alignment/window/*` uses the existing Fairseq window aggregate of alignment diagnostics. `dev/full/*` preserves the optional Fairseq log series at its original precision, including loss and auxiliary values in bits and the CE diagnostic in nats. The primary Dev values come from the raw recorder. [Full window and legacy key definitions](WANDB_METRIC_AUDIT.md#metric-definitions)

### BLEU evidence

The existing evaluator uses the original Dev references, seed-1 SHA-256 pair selection, source-only decoding, greedy auxiliary decoding with EOS by token 256, and Unit beam 10 with max length source length plus 200 and required EOS. It uses the LJSpeech Unit vocoder, Whisper large-v3-turbo at `60be3615a4d667e1258e8ad29130467587c489aa`, the pinned English normalization, and SacreBLEU 2.5.1 with 13a, exponential smoothing and effective order off. The ASR subset is the first 100 of the fixed 500. Recognized generation failures retain their empty hypotheses and journal entries. Result qualification checks the exact milestone receipt, published hash, policy, scoring signatures, fixed populations and failure classification. Infrastructure failures remain in the failure journal. [Evaluation and saved scores](TRANSFER_BLEU.md)

## Science and reuse evidence

The 400k Baseline registration, FP16 Calibration and Top-p diagnostic ZIPs and their members were checked against their Drive receipts. The Baseline record hash is `60f6a6935699aa0b73961336b51ec025f3356364827a5485324882c329ba0e57`. Calibration binds that registration, its prepared data and scientific configuration, all six layers and the shared teacher artifacts. Its member SHA-256 is `e02033c393db82f5204ce9226720f711072b80983fa375d0d3ca5817232d4e48`. The production artifact reports 16 qualified batches, FP32 accumulation at scale 1024 and runtime Python 3.10.22, torch 2.7.1+cu126 and NumPy 1.26.4. The separately published Top-p diagnostic also equals the report embedded in the Calibration. [Baseline registration receipt](https://drive.google.com/file/d/1yScLRLbCFMuonZBhNsgCBDiImSTdNHJj/view) · [Calibration receipt](https://drive.google.com/file/d/1BufCIcuujhO09b5GbYtqEvXgdSRN4ApL/view) · [Top-p diagnostic receipt](https://drive.google.com/file/d/1n6vYDlvchgivy03zS7-o_Nf6taU0vClF/view)

Rebuilding the old plans from the verified registration, Calibration binding and observed invocation produced byte-identical plan SHA-256 values to all three published receipts. Rebuilding for `transfer-fixed-v2` changed the output/log identities and their contract digests. The scientific configuration, initialization, data, Teacher, Calibration and conditions matched. Seed 1, four accumulated batches, 20k maximum batch tokens, 10k warmup, the 400k budget, label smoothing 0.2, FP16, and full-Dev selection at 1k remain in the plan. New transfers initialize from the same fresh seeded model. KL uses λ=1 and τ≈0.553894, Annealed KL uses the same λ and τ with cosine decay to zero at pre-update 320k, and Mask uses λ=10, τ≈0.044954 and top-p 0.8. [KL receipt](https://drive.google.com/file/d/1worIl5MsQKT7tX_zppsmKEL9SJKG8WJy/view) · [Annealed KL receipt](https://drive.google.com/file/d/1U6ZDU4YS4qaGW35ZJvNCAa5e2uuB-zsu/view) · [Mask receipt](https://drive.google.com/file/d/1WAXTq5tr0Qcngk9osVf9KYEowyfT74Ej/view)

The old v3 Calibration acceptance test exposed an ordering bug. Its coefficient-era guard checked raw source files before the existing reviewed telemetry alias had restored the canonical fingerprints. The guard now checks the canonical fingerprints. Exact LF and CRLF allowlist entries cover the reviewed recorder and validation edits. Unmatched source changes still alter the computed identity. The actual Linux production Calibration hash `8f49ebdfbe97e299c715c6893a9a1d79d7cc2f34e52ad62c45839bbb17b82a8b` matches the corrected verifier's reviewed coefficient predecessor. Offline reconstruction verified binding and hashing using the recorded runtime identity. Colab still verifies the real installed runtime, staged files, receipt-matched old plan, and regenerated plan before training.

The observer runs after the successful native reduction. It gathers a small vector of detached scalars per device, without another model forward or retained model activations. It keeps native checkpoint selection in bits and publishes the v3 Dev values separately in nats. A runtime may incur an extra scalar transfer and larger JSONL files, compressed into the usual verified snapshots. Throughput and GPU memory overhead on L4 remain measurements for the first 1k run.

## Audit, recovery and publication

`--audit-v3=local` uses local JSONL through the newest published update. `snapshot` verifies the archived plan and declared JSONL against their receipt SHA-256. `wandb` also reads uncached, unsampled history on the contract-bound Run. The audit checks fixed update sets, successful train rows, duplicates, recorded sample counts, finite values, raw ratios, required phase/condition fields, result hashes and the Summary. It reports unavailable originals, waiting or failed BLEU, eligible unsent W&B cells, acknowledgement/indexing delays, and conflicting or duplicate cells separately. `--final` requires all BLEU milestones through the audited update.

The same original-row validator runs in the trainer on every successful optimizer update and every scheduled full-Dev result. It checks τ against the registered Calibration, nonnegative integral row and frame counts, normalized eligible layer weights and worst-layer fractions, the worst layer against the reduced layer losses, and the actual applied `lambda × loss`. Mask coverage and region mass must be probabilities. Small FP32 rounding differences are permitted. The validator uses the existing sample-size-weighted statistics and does not recompute weights from averaged layer losses.

### Stop and resume after an original-measurement failure

1. Read the training log and the local run folder's `metric-stop.json` for the attempted update and validation error. The trainer exits before saving that update or replacing the best checkpoint. The benchmark publisher checks the frozen copy against the checkpoint update, including the fallback freeze after a child exits. The capture of normal earlier updates can still produce a valid durable point.
2. Keep the existing run, backup and W&B identities. Run **Recover W&B writer and sync saved metrics** with `AUDIT_WANDB_METRICS_ONLY=True` and `V3_AUDIT_SCOPE='snapshot'` to identify the durable point. Turn `V3_REQUIRE_SCHEDULED_BLEU` off when checking originals after a failed or interrupted evaluation. A rejected Snapshot must not be used as a recovery source. Keep its files for diagnosis.
3. Resolve the reported measurement issue before restarting. Run the same training cell with the same inputs and intended stop. The normal recovery path restores the published state, retains a displaced local work folder, and checks every original row through the restored update before training. The current guard prevents a failed observation from becoming a new published resume point. If an older release already published incomplete originals, the resume guard stops and reports that Snapshot for investigation.
4. Each successful segment runs `--audit-v3 snapshot --originals-only --expected-update <step>` automatically. A failed audit ends the notebook loop before the next segment. This audit is independent of W&B and BLEU. When a Snapshot passes and W&B is unavailable, the reporter can recover from the saved originals without restarting training.

The W&B Run ID is the first 24 hexadecimal digits of the SHA-256 of compact UTF-8 JSON `["transfer",3,entity,project,group,method,scientific_contract]`. Config verifies that identity and records `metric_schema=3`. W&B's internal step increases naturally, and every chart uses `update`. `train/target_update`, `train/max_update`, and `train/durable_update` are Summary fields. Existing writers reopen with `resume="must"`. An uncreated v3 Run uses the same derived ID with create-only resume semantics. Sync uses the existing ownership/FileLock mechanism, rejects source and remote conflicts before opening a writer, and waits for acknowledgement if a known watermark covers missing history. For manual recovery without GCS, stop all other writers to the Run across all VMs. A Run already bound to GCS requires the same bucket. [Writer recovery](WANDB_RECOVERY.md#restore-writer-ownership-on-a-cpu-runtime)

### Repair repeatedly observed W&B gaps

`missing` identifies eligible cells without an acknowledgement watermark. `ack_pending` means a cell is absent in the current history scan even though the watermark covers it. A single scan cannot distinguish late indexing from a lost cell. Normal sync never resends a cell in `ack_pending`. Duplicates, value conflicts, or a change in Run, writer, or Snapshot identity block gap repair.

With all writers stopped, use the same v3 W&B command and SDK environment as in **Recover W&B writer and sync saved metrics**. In a copy of that cell, use its `recovery_command`, `python` and `env` for the following three **separate** invocations. The observation files are local to this CPU runtime. The observation command can return a nonzero status while the audit still reports incomplete delivery.

```python
first = Path('/content/v3-gap-first.json')
second = Path('/content/v3-gap-second.json')
execute(python, *recovery_command, '--audit-v3', 'wandb', '--record-v3-gaps', first, env=env)
```

Wait at least 10 minutes with writers stopped. Use a new filename if repeating an observation.

```python
execute(python, *recovery_command, '--audit-v3', 'wandb', '--record-v3-gaps', second,
        '--compare-v3-gaps', first, env=env)
```

Inspect `repeatedly_observed_gaps`. After confirming the points and that all writers remain stopped, authorize repair explicitly.

```python
execute(python, *recovery_command, '--sync-metrics-only', '--recover-writer',
        '--confirm-writers-stopped', '--repair-v3-gaps', first, second, env=env)
```

Repair revalidates the published originals, scans again immediately before opening the existing Run, and appends only verified cells missing in both observations and the fresh scan. The final scan reports gaps still pending or any conflicts. If indexing is still pending, wait and rerun the read-only audit. A completed repair can be rerun without adding duplicate history.

## Colab workflow

1. Open **Preparation and training**. Set `TRANSFER_METRIC_PROFILE='v3'` in **Experiment and training stages**. Leave the original corpus and teacher inputs selected. Use the saved Baseline, T2TT, Teacher, Top-p and Calibration. Keep their creation/training switches off. The profile sets `transfer-fixed-v2`, the original Calibration ID, the v3 W&B project and full mode, and fills in `MASK_TOP_P=0.8` if it is unset. For the first run, enable only `RUN_DISTILLATION` and set `FULL_STOP_AFTER_UPDATE=1000`.
2. Run the notebook through **Train the selected transfer conditions** on L4. Plan verification compares the existing Drive plan and the next plan before launching the trainer. The run stops at 1,000 after native Dev validation, verified checkpoint publication and its scheduled BLEU. The recipe and checkpoint still target 400,000.
3. In **Recover W&B writer and sync saved metrics**, choose that method and set `RECOVER_WANDB_METRICS=True`, `AUDIT_WANDB_METRICS_ONLY=True`, and `V3_REQUIRE_SCHEDULED_BLEU=True`. Use `V3_AUDIT_SCOPE='local'`, then `'snapshot'`, then `'wandb'`. The first two work on CPU without a W&B SDK. Check `VERIFIED`, 1,000 original rows, 10 official Train points per metric, one Dev point and both BLEU triplets. An unsettled delivery can be audited again. If genuinely unsent, stop all writers and turn audit off with `CONFIRM_WANDB_WRITERS_STOPPED=True` to reconcile from verified originals.
4. In a fresh L4 runtime use the same profile and inputs, with `FULL_STOP_AFTER_UPDATE=1100`. Re-execute through the training cell. Confirm that it restored update 1,000 and reused the saved BLEU if the milestone is visited during resume. Audit 1,100 original rows and 11 official Train points, the same Run ID, and the same Dev/BLEU update set.
5. Run only `RUN_ANNEALED_DISTILLATION`, then only `RUN_MASK_DISTILLATION`, each with a 1,000 stop. Audit each method with the same three scopes and its required schedule or Mask fields.
6. After the checks pass, set `FULL_STOP_AFTER_UPDATE=0` to continue the selected runs to 400k. Keep the profile and inputs identical. **Evaluation and comparison** can backfill or sync a missing saved milestone using the same v3 profile. When a successful evaluation already exists, the evaluator verifies and reuses its result and the reporter checks the existing history before publishing.
