# Process logs

Colab notebooks send external commands to the shared logger throughout setup, preprocessing, training, inference, and evaluation. This includes Git, uv, pip, preflight tests, and configuration commands before training.

## Display and persistence

| Status | Display |
|---|---|
| `started` | Stage name and full log path |
| `running` | Child PID, elapsed time, and child output |
| 10 seconds without output | `no_output` and a liveness notice |
| `completed` | Exit code 0, elapsed time, and log path |
| `failed` / `interrupted` | Exit code or exception, elapsed time, and log path |
| `skipped` | Reason, such as a disabled condition or a missing checkpoint |

The notebook displays child output line by line and flushes the same lines to a local file. Python child processes receive `PYTHONUNBUFFERED=1`. CPU tests use verbose output to show test names.

A liveness notice confirms the process is alive. Use stage counters such as `checked`, `zip`, pair counts, and optimizer updates to confirm progress.

## Log paths

| Notebook | Default `LOGS` |
|---|---|
| benchmark_comparison | `ROOT/execution/` |
| pilot | `/content/alignment-transfer-logs/pilot/<ROOT name>/` |
| pipeline | `/content/alignment-transfer-logs/pipeline/<ROOT name>/<recipe>/` |

Drive connection notices go to `/content/alignment-transfer-logs/<notebook name>/bootstrap/`. Frequent writes stay local to the VM. Retain any needed logs before deleting the VM, since snapshots cover training artifacts. The benchmark runtime handles checkpoint ZIP publication and recovery.

Each attempt writes the following files.

- `*.log` contains the full log, including startup, liveness notices, child output, and the final status.
- `*.json` records the stage name, arguments, start time, PID, elapsed time, status, exit code, and log paths. The logger updates it during execution.
- `*.stdout.log` holds stdout for JSON-producing commands separately from human-readable stderr.
- `*-gpu.csv` records device utilization and memory for monitored attempts.
- `notebook-events.log` records Drive connection and explicit skips.

For input preparation, runtime checks, and plans that return JSON, the notebook shows stderr and captures stdout separately. Invalid JSON is recorded as a failure.

## Interruption and recovery

On interruption, the logger terminates the child processes it started. On Colab and Linux, the outer logger creates a process group shared by nested loggers and training workers. Input packs, checkpoints, and original Drive files are outside logger cleanup.

A running cell continues with its loaded code. When updating the notebook or pinned revision, wait for the old process to stop before rerunning a preparation stage. Resume started runs with their saved source revision.

## Implementation and tests

`src/alignment_transfer_s2ut/process_logging.py` is the shared implementation. The standard-library functions are embedded in pre-install Colab cells and also used by `pilot.run_logged`.

Tests check that the embedded logger matches the shared source, notebook cells route subprocesses through it, and it handles live output, silence, JSON separation, failure, and interruption.
