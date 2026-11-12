# Real-data diagnostics and the strong S2UT pilot

This page covers the historical standalone workflow. See the [native workflow](BENCHMARK_COMPARISON.md) for the current comparison.

This small pilot checks the standalone trainer (`--trainer legacy`). The matched comparison uses a separately trained Baseline from the benchmark. [Benchmark workflow](BENCHMARK_COMPARISON.md)

[Open in Colab](https://colab.research.google.com/github/Yaaamashiro/alignment-transfer-s2ut/blob/main/notebooks/legacy/pilot.ipynb)

The pilot checks CUDA with real CTC, HuBERT, and the official vocoder, diagnoses alignment quality, and compares three strong conditions. It uses the original train and dev splits. Evaluate test separately after choosing the method and settings.

## Colab settings

Select a GPU runtime and set `CORPUS` and `ROOT` in the opening cell.

| Setting | Default |
|---|---|
| accepted manifest | `CORPUS/production/manifests/releases/accepted.jsonl` |
| Output | `/content/drive/MyDrive/alignment-transfer-s2ut-data/pilot-text-001` |
| Train / dev limit | 10000 / 200 pairs |
| seed | 1 |
| T2TT updates | 20000 |
| T2TT dev text evaluation | All dev pairs, beam 5, length penalty 1, at most 1000 output steps, SacreBLEU 13a |
| S2UT updates | 10000 per condition |
| Annealing end | 5000 updates |
| Warmup | 1000 updates for each condition |
| Validation / checkpoint interval | 500 updates |
| batch size / accumulation / max tokens | 8 / 4 / 20000 |
| CPU workers / prefetch | 2 / 2 nominal batches |
| Local cache | `/content/alignment-transfer-cache`, maximum 8 GiB |
| recipe / smoothing / source mapping | strong / 0.05 / frontend |
| Dev evaluation | Unit error rate. Enable waveform generation and ASR-BLEU in the opening cell |

T2TT and S2UT have independent update budgets. If setting S2UT to 5000–20000 updates, keep `ANNEAL_STEPS < S2UT_UPDATES` for a beta-zero training interval. The pilot warmup is shorter than the 10000 updates in the [shared recipe](EXPERIMENTS.md).

The first cell installs Python 3.10.22, pinned dependencies, and the fixed fairseq commit. It checks CUDA, torch and torchaudio, and fairseq provenance and stops for CPU-only PyTorch.

Training begins once the initial inputs are ready and CPU workers prefetch later inputs. Unaugmented FBANK/CMVN, units, and teacher artifacts are cached on demand on the VM and shared across three conditions. Complete CTC, HuBERT, and teacher construction before training. [Prefetch limits, recovery, and checks](STREAMING.md)

## Execution order

1. **Prepare the pilot manifest.** Sample within the original splits with a fixed seed and verify selected audio hashes. T2TT, S2UT, and preprocessing share this manifest.
2. **Run CUDA integration.** With up to two train pairs and one dev pair, run real CTC and HuBERT, two updates in each strong condition, and official vocoder inference after removing the teacher.
3. **Train T2TT, evaluate text, and preprocess.** Train on pilot train and use the best dev-loss checkpoint to generate all dev translations from Japanese only. Inspect text BLEU and predictions. Use the same checkpoint to extract MT attention and generate Japanese and English CTC spans, HuBERT/KM100 units, and teacher packets.
4. **Diagnose train and dev.** Check coverage, entropy, and invalid rows on up to 500 pairs and 20 heatmaps per split. The notebook displays source, translation, both audio files, and heatmaps for two examples per split.
5. **Train three pilot conditions.** Baseline, Annealed Monotonic, and Annealed T2TT share the strong recipe and initialization.
6. **Evaluate dev and report.** Evaluate all dev pairs with the common teacher-free generator and save the comparison contract, loss curves, and timings.

The notebook invokes each CLI with a subprocess argument list. A failed stage stops the cell and retains its own failure or interruption log. The two-update integration teacher and the full pilot teacher have separate destinations.

## Data selection and recovery

`pilot prepare` checks the full source manifest for ID, text pair, and known audio pair leakage, then selects train and dev subsets. It reads audio after selection, so unselected test audio is optional. It preserves the original splits.

`data/selection.json` records the input manifest SHA-256, seed, limits, pair IDs, source split counts, selected counts, and canonical manifest hash. Resume checks the same settings and audio contents. Output goes outside the source corpus.

Rerun with the same settings to resume from preprocessing artifacts and training checkpoints. Use a new `ROOT` when changing data, seed, recipe, batching, code revision, or the T2TT teacher. Fresh Colab runtimes fetch the notebook's pinned commit.

## Inspect diagnostics

[Metric definitions](ALIGNMENT.md#alignment-quality-diagnostics)

- Inspect source and target token coverage alongside valid CTC character counts, especially dev masks caused by out-of-vocabulary characters.
- For low unit-row coverage, listen to the sample and compare CTC spans with unit runs. Zero total valid rows stops the pilot.
- Compare raw and smoothed priors with heatmaps and entropy. Smoothing preserves zero rows.
- Check T2TT dev main loss, dev text BLEU, and raw attention. Compare generated English with references to assess translation quality separately from alignment transfer.

Coverage and entropy describe the distribution. Combine quantitative results with listening and visual inspection to assess timing.

T2TT text BLEU uses autoregressive corpus generation. Decode character tokens and score against the original English references with case-sensitive SacreBLEU 2.5.1, 13a, exp smoothing, and effective order disabled. This stage uses text only. It saves the BLEU signature, checkpoint and dev hashes, generation settings, and predictions. `max-chars` counts output steps including EOS. At the limit, generation can end without EOS. Inspect EOS rate and brevity penalty together.

The score diagnoses the teacher selected by dev loss. Check translation quality and alignment together before the main experiment. The pilot uses no universal BLEU or EOS threshold.

## Pilot outputs

| Path under `ROOT` | Contents |
|---|---|
| `data/` | Selection settings and canonical manifest |
| `runtime.json` | Python, torch, CUDA, fairseq revision, and GPU model and memory |
| `integration/attempt-*/` | CUDA integration outputs |
| `t2tt/`, `mt/`, `ctc/`, `units/`, `teacher/` | Trained teacher and preprocessing artifacts |
| `t2tt/dev-text-metrics.json`, `t2tt/dev-text-metrics-predictions.jsonl` | T2TT dev text BLEU, signature, EOS rate, generation settings, and predictions |
| `alignment-audit-train/`, `alignment-audit-dev/` | Summary JSON, per-sample JSONL, and heatmap PNG and NPZ files |
| `runs/{condition}/` | Checkpoints, training metrics, dev predictions, and evaluation |
| `LOGS` on the local Colab VM | Commands, exit status, elapsed time, full logs, and GPU utilization CSV per attempt. [Logging](LOGGING.md) |
| `pilot-plan.json` | Shared settings and commands for three conditions |
| `fair-comparison.json` | Checks for recipe, initialization, data, and transfer settings |
| `pilot-summary.json`, `pilot-summary.png` | Best dev loss, updates, beta, evaluation, timings, T2TT text evaluation, and dev curves |

All conditions select checkpoints and plot curves with beta-zero dev main loss. Training total losses include condition-specific auxiliary objectives. For duplicate steps after resume, reporting uses the latest metrics.

`pilot report --t2tt-evaluation` checks that text BLEU used the same checkpoint, dev data, and backend as the Annealed T2TT teacher. Colab enables this check.

Timing sums recorded attempts and reports failure counts. The GPU CSV samples device-wide utilization and memory every five seconds. Match low-utilization intervals to logs to inspect FBANK and I/O stalls.

Dev unit inference defaults to beam 5, length penalty 1, and at most 1000 units. `GENERATE_WAVEFORMS=True` uses the fixed vocoder, and `ASR_BLEU=True` uses the fixed ASR. Evaluate all three conditions with the same settings and use a new destination when changing them.

## CLI

```bash
python -m alignment_transfer_s2ut.pilot prepare \
  --input /path/to/corpus/production/manifests/releases/accepted.jsonl \
  --corpus-root /path/to/corpus --output outputs/pilot/data \
  --train-limit 10000 --dev-limit 200 --seed 1 --resume

# After training T2TT, evaluate the same teacher on dev text.
python -m alignment_transfer_s2ut.evaluate_t2tt \
  --manifest outputs/pilot/data/manifest.jsonl \
  --checkpoint outputs/pilot/t2tt/checkpoint_best.pt --split dev \
  --beam-size 5 --length-penalty 1 --max-chars 1000 --device cuda \
  --output outputs/pilot/t2tt/dev-text-metrics.json --resume

# After preprocessing and diagnostics, print the plan without starting GPU training.
python -m alignment_transfer_s2ut.pilot plan \
  --manifest outputs/pilot/data/manifest.jsonl --config configs/s2ut/fisher.yaml \
  --units outputs/pilot/units --teacher outputs/pilot/teacher \
  --output outputs/pilot/runs --logs outputs/pilot/execution

# Replace plan with train using the same arguments to run the three conditions after CUDA checks.
python -m alignment_transfer_s2ut.pilot report \
  --runs outputs/pilot/runs --logs outputs/pilot/execution \
  --t2tt-evaluation outputs/pilot/t2tt/dev-text-metrics.json \
  --output outputs/pilot/pilot-summary.json
```

Local workflow tests use synthetic data. Record CUDA integration with real CTC, HuBERT, and the vocoder, Colab execution, and real-corpus pilot results separately.
