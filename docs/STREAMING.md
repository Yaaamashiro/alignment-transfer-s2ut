# Training input prefetch and local caching

This page describes the historical standalone workflow. See the [native workflow](BENCHMARK_COMPARISON.md) for the current comparison.

The standalone trainer (`--trainer legacy`) and T2TT use the prefetch pipeline below. Current Colab S2UT uses the fairseq data loader to fetch and verify ZIPs through the pinned benchmark runtime. [Colab I/O and recovery](COLAB_RUNTIME.md)

Workers fetch inputs as needed and prepare the next batch while the GPU trains on the current batch.

```text
Audio, units, and teacher artifacts in Drive
          ↓ Fetch and verify required samples
    Two CPU workers and a local cache for unaugmented FBANK/CMVN and artifacts
          ↓ Ordered, bounded prefetch queue
    Microbatches for the current optimizer update
          ↓ SpecAugment on the main thread, then pinned-memory transfer
        GPU training
          └─ CPU workers continue fetching and preparing inputs
```

This pipeline applies to training stages with existing audio, unit, and teacher artifacts. Run CTC, HuBERT, and teacher construction before training. Manifest selection and hashing are also preparation steps.

## Configuration

```bash
python -m alignment_transfer_s2ut.train_s2ut --trainer legacy \
  --manifest outputs/manifest.jsonl --config configs/s2ut/fisher.yaml \
  --units outputs/units --teacher outputs/teacher \
  --recipe strong --method annealed_guidance --max-updates 10000 --anneal-steps 5000 \
  --loader-workers 2 --prefetch-batches 2 \
  --cache-dir /content/alignment-transfer-cache --cache-gb 8 --cache-reserve-gb 1 \
  --pin-memory --device cuda --output outputs/annealed_t2tt
```

| Argument | Meaning |
|---|---|
| `--loader-workers` | CPU loader threads. The CLI defaults to 0 for synchronous loading, and the pilot defaults to 2 |
| `--prefetch-batches` | Prefetch capacity in units of the configured batch size. Default 2 |
| `--cache-dir` | Dedicated local VM directory. Omit it to run without a persistent cache |
| `--cache-gb` | Maximum cache payload in GiB. Default 8 |
| `--cache-reserve-gb` | Free disk space to retain when writing, in GiB. Default 1 |
| `--pin-memory` | Pinned memory and nonblocking transfer from the main thread to CUDA |

Batch size 8 and prefetch depth 2 prepare up to 16 samples, in addition to the microbatches needed for the current optimizer update. Actual bytes depend on audio and unit lengths.

The legacy pilot uses two workers, two prefetched batches, and an 8 GiB cache shared across three conditions. Configure them with `LOADER_WORKERS`, `PREFETCH_BATCHES`, `LOCAL_CACHE`, and `CACHE_GB`. T2TT character batches also support ordered prefetch. [Pilot notebook](../notebooks/legacy/pilot.ipynb)

## Cache and verification

- Cache unaugmented source features (FBANK and optional CMVN), units, and numeric teacher payloads on demand. Apply SpecAugment after reading cached features.
- Feature keys bind the audio path, size, and modification time, manifest SHA-256, backend, feature settings, implementation, and numeric runtime. Verify the audio SHA-256 when creating an entry.
- Verify unit and teacher metadata, internal payload hashes, and external checksums, and retain the original file hashes. Teacher `units_hash` always refers to the original artifact.
- Training checks the canonical manifest's English audio hash against unit and teacher metadata. Source features come from verified source audio.
- Verify the local payload hash on cache hits. A changed source size, modification time, or sidecar is checked as a separate entry. Use fixed canonical inputs.
- Publish complete payloads atomically. When space is limited, evict older entries that are out of use. Process oversized samples or samples that exceed available space without caching them.
- Evict only managed files inside the dedicated cache. The cache rejects nonempty unknown directories, input directories, and locations on Colab Drive.

The cache is temporary VM storage and is rebuilt on demand after a runtime reset. Sequential runs can share the same VM cache.

## Ordering, recovery, and errors

Consume samples in the original seed and epoch permutation regardless of worker completion order. Keep remaining prefetched samples if `max_tokens` shrinks a batch. Advance the checkpoint epoch and cursor only as training consumes samples.

Workers read inputs, compute deterministic features, and project priors. The training thread handles NumPy SpecAugment, model RNG, and CUDA. CPU tests found identical parameters across synchronous and prefetched loading, cold and warm caches, and worker changes on resume.

At startup, bind the configuration identity from one reference unit and teacher artifact. Check identity, split, manifest, and hashes when reading subsequent artifacts. Missing inputs, checksum errors, and incompatible settings stop the run at the affected sample. Prefetched samples count only when consumed.

`experiment.json` and the checkpoint's `input_pipeline` record worker count, prefetch depth, cache path, and other I/O settings. You can change these when resuming with the same code because they preserve sample order and the objective. Match them for speed comparisons. The implementation hash controls checkpoint compatibility.

## Timing

Record these values per update in `metrics.jsonl`.

- `input/data_wait_seconds` measures wall-clock waiting for required microbatches and CPU collation.
- `input/optimization_seconds` measures wall time for CPU augmentation, transfer, forward and backward passes, and the optimizer. On CUDA, synchronize at the end of the update.
- `input/loader_workers` gives the worker count for the update.

Optimization timing excludes validation, checkpoint saves, and log I/O. The pilot summary also records total waiting and optimization time for comparison with the GPU utilization CSV.

Local tests cover overlapping fetch and consumption, queue bounds, cache reuse, corruption detection, storage limits, and matching training results. Speedups on a real corpus in Colab remain unmeasured.
