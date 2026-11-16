# Colab I/O, storage, and recovery

[Shared process logging](LOGGING.md) covers live output, execution records, and liveness notices for every stage.

Run training in [training.ipynb](../notebooks/colab/training.ipynb) and inference and comparison in [evaluate.ipynb](../notebooks/colab/evaluate.ipynb). Both use a pinned benchmark revision.

Colab fetches the benchmark into a separate local checkout, installs it in editable mode, and checks the revision and tracked source at startup. This pin adds audio ZIP processing to `4361e55` and keeps its training and evaluation implementation. Existing snapshots use `BENCHMARK_REVISION=4361e55` for artifact and recipe lineage. The executable code is verified against `BENCHMARK_RUNTIME_REVISION=0c0024e`. [Pinned benchmark revision](https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/tree/0c0024ea888206149a3119bf096b86ba66d63b66)

For local installation, activate the alignment Python environment and run the following commands in a location separate from an existing benchmark checkout.

```bash
git clone --no-checkout https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git ../alignment-benchmark-runtime
git -C ../alignment-benchmark-runtime checkout --detach 0c0024ea888206149a3119bf096b86ba66d63b66
python -m pip install PyYAML==6.0.2 --no-deps -e ../alignment-benchmark-runtime
python -m alignment_transfer_s2ut.environment --native --benchmark --cuda
```

The notebooks run the environment check before preprocessing. It requires Python 3.10, compatible dependency metadata and imports, the pinned fairseq and benchmark revisions, a working compiled batch sampler, and CUDA. The check uses local runtime probes. External checkpoints and real audio are checked in their own stages.

Set `MPLBACKEND=Agg` in the notebook settings so CLI subprocesses can write figures. A separate virtual environment may fail to import Matplotlib when it inherits Colab's `module://matplotlib_inline.backend_inline`. In an open notebook, set `os.environ["MPLBACKEND"] = "Agg"` and restart from the environment check. Command failures display the log tail and preserve the full output under `ROOT/execution/`.

| Operation | Benchmark implementation |
|---|---|
| Audio ZIP creation, indexing, and verified extraction on demand | `drive_staging.stage_audio` / `active_map` / `stream_inputs.Downloader` |
| Resident trainer, one upload worker, and up to two queued snapshots | `session_runtime.run_continuous` / `train_runtime.checkpoint_saved` |
| Frozen complete checkpoint, best checkpoint, and selection state | `recovery.freeze_files` and alignment metadata |
| ZIP creation, storage budget, readback verification, and final completion marker | `colab.publish_packed` |
| Verify and restore completed snapshots | `colab.latest` / `restore` |
| S2UT cache, epoch-level worker tuning, verified FP16 overflow retries, and gradient auditing | `s2ut.performance.runtime_hooks` / `TrainingAudit` |

`ROOT` is `/content/alignment-transfer-work/<EXPERIMENT_ID>`. `PERSISTENT` is the dedicated Drive directory. Checkpoints, logs, NPZ files, and audio writes go to `ROOT`. Verified ZIPs go to `PERSISTENT/stages/<stage name>/`. Preserve old runs that wrote directly to Drive, and use a separate destination for a new experiment.

## Notebook and source versions

The two primary notebooks share [release.json](../notebooks/colab/release.json). `NOTEBOOK_VERSION` identifies the manually released notebook form and bootstrap. `REPO_REVISION` and `BENCHMARK_RUNTIME_REVISION` identify the exact source commits used by the experiment. The source commit may precede the notebook release commit. This allows both notebooks to pin an already tested source tree.

Before installing dependencies, **Check and record notebook and Git versions** checks each detached HEAD against its pin and lists tracked and untracked worktree changes. A checkout error, mismatched HEAD, or dirty worktree stops setup. The check uses the recorded pins, including when a historical release is behind the current main branch. For review, each execution writes its version receipt under `ROOT/execution/` and `PERSISTENT/execution-provenance/`. Subsequent notebook command receipts carry the same version information and run ID.

To publish a notebook release, update the manifest and run `python tools/colab_release.py`. Run `python tools/colab_release.py --check` to verify both embedded copies. Use a saved notebook release and its original pins to reproduce or resume a previous experiment.

## Resuming a run

Select a GPU runtime and rerun from the top with the same `ROOT`, `PERSISTENT`, and training settings. The notebook installs the environment and restores completed preprocessing and the latest verified checkpoint. S2UT resumes the saved optimizer, scaler, iterator, RNG, and best-checkpoint selection. It fetches audio ZIPs on demand, and the fairseq data loader fetches the next batch while training proceeds.

If a disconnect happens after writing an NPZ but before writing its checksum, resume verifies the internal payload and metadata and adds the missing checksum. A stage with missing checksums at normal exit cannot publish a completion ZIP. On the first run under the newer verifier, older completed attention, CTC, unit, and teacher stages recheck each pair and fill missing artifacts. Later runs skip these stages after a verified completion marker appears in an incremental ZIP. The original NPZ files, inputs, ZIPs, and Baseline backups remain available.

Training uses the benchmark's 600-second backup interval and forces backups at dev evaluation points and the final update. The 1,024 GB storage budget stops publication at the limit and preserves existing recovery points. If Colab deletes the VM, resume from the last completed ZIP. Drive outages and transfer limits can still affect recovery.

GPU runs reuse train and dev Japanese and English WAV packs created by the benchmark. If packs are missing, use a CPU runtime with `PREPARE_AUDIO_PACKS=True` through the input-preparation cell. The benchmark reads original WAVs when creating missing packs and reuses completed packs. Then switch to GPU, set `PREPARE_AUDIO_PACKS=False`, and run the workflow. Comparison preprocessing uses train and dev packs.

On GPU, the speech and training data stage indexes ZIPs from the manifest. Baseline startup verifies that the index covers the prepared TSV paths. The first read verifies the audio ZIP and each WAV by SHA-256. At most two workers sharing a cache transfer data concurrently. The VM reuses verified extractions and needs space for the accessed audio and derivatives. Initial unit extraction and audio validation process the full input.

`ROOT/execution/*.log` records transfer starts, verified ZIP counts, and reuse. Commands display their log paths at startup. JSON results go to a separate `.stdout.log`. The last checkpoint published to Drive is the recovery point after a disconnect.

## Alignment adapters

The adapter preserves original logical paths for audio, TSVs, and dictionaries as data identifiers and redirects physical reads to verified local copies. It resolves dictionary paths in the native fairseq multitask config and TSV reads. It uses the registered targets, dictionaries, and auxiliary losses. Staged audio and unit payloads are required, and unit import requires the benchmark's inline format.

T2TT uses the existing text trainer and sends completed-save notifications to the benchmark snapshot protocol. The main S2UT comparison uses the benchmark performance hooks and official fairseq Trainer with the registered physical batches and precision.

For per-pair attention, CTC, unit timing, and teacher artifacts, the adapter packages incremental files using the benchmark publisher, checksums, and restore code. It packages changed files every 600 seconds and at stage completion or failure, with each saved pair included once. Completed stages are reused after input hash verification. A fresh VM verifies and restores packs in order. Interrupted pairs and partial ASR evaluation may be recomputed.

Use the current training and evaluation notebooks for Drive recovery. The `pilot.ipynb` and `pipeline.ipynb` files under `notebooks/legacy` document and test standalone trainers.

[Parallel CPU ZIP preparation, recovery, and measurements](AUDIO_PACKS.md)
