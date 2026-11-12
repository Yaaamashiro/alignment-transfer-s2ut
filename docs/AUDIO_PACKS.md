# Parallel CPU audio ZIP preparation and recovery

## Colab workflow

1. Stop the current CPU preparation cell and wait for it to finish stopping. Use one CPU VM per destination.
2. Open the latest notebook from GitHub in Colab. Existing Colab copies retain their own settings. [training.ipynb](../notebooks/colab/training.ipynb)
3. Reuse the original `EXPERIMENT_ID`, T2TT settings, `DATA_CONFIG` or corpus and manifest overrides, manifest contents and row order, and output paths. Keep existing ZIPs, plans, and receipts.
4. In a CPU runtime, set `PREPARE_S2UT=True`, `PREPARE_AUDIO_PACKS=True`, and `AUDIO_PACK_WORKERS=2`, then run all cells. You can set the worker count to another integer from 1 to 4.
5. After preparing the ZIPs, switch to GPU, set `PREPARE_AUDIO_PACKS=False`, and rerun with the same settings. The GPU runtime fetches existing ZIPs as needed. [Colab I/O and recovery](COLAB_RUNTIME.md)

The workflow verifies and reuses completed ZIPs, including those from `4361e55`, and builds the remaining packs. In a recovery test with 11 receipts and a twelfth ZIP and plan, four workers resumed with the corresponding source WAVs unavailable. The 12 existing ZIPs retained their contents and modification times, and the remaining five brought the total to 17.

Each pack contains 128 files. Worker count is excluded from the stored data identity. Namespaces, archive and WAV member names, member order, SHA-256 hashes, plans, and receipts use their existing formats. The workflow preserves incomplete ZIPs with missing provenance or corruption and reports an error.

## Implementation

Submit one job per ZIP to a `ThreadPoolExecutor`, with at most `workers` jobs in flight. Each job streams WAVs in member order and verifies source file stamps and SHA-256 hashes. Workers use separate local temporary directories and check free space against their combined temporary ZIP reservations. `ZIP_STORED`, readback SHA-256 after copying to Drive, and publishing the receipt last are part of the archive protocol.

An OS lock prevents concurrent writes to the same ZIP on one VM. Keep each destination assigned to one VM because DriveFS has no cross-VM lock in this workflow. On failure, stop submitting jobs, signal other workers, and wait for them to finish. Python exceptions and interruptions clean up owned temporary files and preserve published ZIPs. Unpublished files left by forced VM deletion remain incomplete.

The `checked` and `completed` counters count verified builds and reuses. At startup, shutdown, failure, and every 10 seconds, stderr reports workers, build and reuse counts, failures, ZIP/s, MiB/s, and build, source verification, Drive copy, and readback times. The common logger shows and saves the output. Build time includes source verification, and overlapping stages have separate timings. For reuse, MiB/s measures verification throughput.

## Pinned revisions

- benchmark runtime: `0c0024ea888206149a3119bf096b86ba66d63b66`
- Merge into benchmark main `9f04d6c97948f9b371626aef487af65acbab57c5`

The runtime commit is based directly on `4361e55` and changes audio ZIP implementation, tests, measurement, and documentation. Training and evaluation fetch that exact SHA. `runtime()` and the environment check verify the installed revision.

`BENCHMARK_REVISION` identifies artifact and recipe lineage as `4361e55`, so existing snapshots such as `common` retain their identity. `BENCHMARK_RUNTIME_REVISION` identifies the executable code. Resume an in-progress run with its original revision and checkpoint contract. When extending a recovered T2TT implementation, the controller uses its separately verified runtime for save notifications while keeping the original trainer source, package hash, and checkpoint contract.

## Local performance measurements

The Windows fixture used 1,024 one-second 16 kHz mono PCM16 WAVs and eight ZIPs of 128 WAVs each. Each row reports the median of three runs with rotated execution order and local source and output directories.

| Workers | Seconds | ZIP/s | MiB/s | Process CPU % | Peak temporary ZIP MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 3.591 | 2.228 | 8.754 | 104.3 | 3.93 |
| 2 | 2.181 | 3.668 | 14.413 | 187.4 | 7.86 |
| 4 | 1.585 | 5.046 | 19.831 | 304.9 | 15.72 |

All runs completed without errors. CPU percentage divides process CPU seconds by wall-clock seconds. The temporary peak was sampled every 10 ms. DriveFS throughput and quota effects still require measurement. The default is two workers.

Use `scripts/benchmark_audio_packs.py` in the benchmark repository to reproduce the measurement. [Benchmark documentation](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/0c0024ea888206149a3119bf096b86ba66d63b66/docs/AUDIO_PACK_PARALLEL.md)

## Changed files

| Repository | File |
|---|---|
| benchmark | `src/direct_s2st/drive_staging.py`、`src/direct_s2st/audio_pack_runtime.py` |
| benchmark | `tests/unit/test_drive_staging.py`、`tests/unit/test_audio_pack_parallel.py` |
| benchmark | `scripts/benchmark_audio_packs.py`、`docs/AUDIO_PACK_PARALLEL.md` |
| alignment-transfer | `src/alignment_transfer_s2ut/colab_runtime.py`、`src/alignment_transfer_s2ut/environment.py`、`src/alignment_transfer_s2ut/t2tt_extension.py` |
| alignment-transfer | `notebooks/colab/training.ipynb`、`notebooks/colab/evaluate.ipynb` |
| alignment-transfer | `tests/test_training_notebook.py`、`tests/test_benchmark_colab.py`、`tests/test_t2tt_extension.py`、`tests/test_pipeline.py` |
| alignment-transfer | `README.md`、`docs/COLAB_RUNTIME.md`、`docs/BENCHMARK_COMPARISON.md`、`docs/AUDIO_PACKS.md`、`docs/VALIDATION.md` |

[Validation results](VALIDATION.md)
