"""Standard-library-only process logging, also embedded in Colab before install."""


def log_event(name, state, logs, detail=""):
    from datetime import datetime, timezone
    from pathlib import Path
    root = Path(logs)
    root.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} [{name}] {state} {detail}\n"
    with (root / "notebook-events.log").open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
    print(line, end="", flush=True)


def run_process(command, logs, *, name="command", capture_json=False, env=None, cwd=None,
                heartbeat_seconds=10., extra_record=None):
    """Stream every child line; silence triggers liveness, never fake progress.

    stdout JSON is kept apart from stderr. Nested loggers share an owned POSIX
    process group so interruption cannot strand a second training process.
    """
    from datetime import datetime, timezone
    import hashlib
    import json
    import os
    from pathlib import Path
    import queue
    import signal
    import subprocess
    import tempfile
    import threading
    import time

    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    root = Path(logs)
    root.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    key = hashlib.sha256(json.dumps(name, ensure_ascii=False).encode()).hexdigest()[:12] + "-" + str(time.time_ns())
    output, result_path, receipt = root / (key + ".log"), root / (key + ".stdout.log"), root / (key + ".json")
    launch_env = dict(os.environ if env is None else env)
    launch_env["PYTHONUNBUFFERED"] = "1"
    owns_group = os.name == "posix" and launch_env.get("AT_LOG_MANAGED_GROUP") != "1"
    if owns_group:
        launch_env["AT_LOG_MANAGED_GROUP"] = "1"
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    record = dict(name=name, command=command, started_utc=started,
        host_utc_offset_seconds=int(datetime.now().astimezone().utcoffset().total_seconds()),
        elapsed_seconds=0., returncode=None, status="running", exception=None,
        pid=None, log=str(output.resolve()), stdout_log=str(result_path.resolve()) if capture_json else None,
        gpu_csv=None, gpu_monitor_error=None, cwd=str(Path(cwd or Path.cwd()).resolve()))
    lock, stop, done = threading.Lock(), threading.Event(), threading.Event()
    messages, reader_errors = queue.Queue(maxsize=256), []
    process = reader = result_stream = None
    stream = output.open("w", encoding="utf-8")

    def save_record():
        record["elapsed_seconds"] = time.perf_counter() - clock
        fd, temporary = tempfile.mkstemp(dir=root, prefix=".log-record-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(record, target, ensure_ascii=False, indent=2, allow_nan=False)
                target.write("\n")
                target.flush()
            os.replace(temporary, receipt)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def notice(state, detail=""):
        line = f"[{name}] {state} pid={record['pid']} elapsed={time.perf_counter()-clock:.1f}s {detail}\n"
        with lock:
            stream.write(line)
            stream.flush()
        print(line, end="", flush=True)

    def read_lines(pipe):
        try:
            for line in pipe:
                with lock:
                    stream.write(line)
                    stream.flush()
                while not stop.is_set():
                    try:
                        messages.put(line, timeout=.1)
                        break
                    except queue.Full:
                        pass
        except Exception as error:
            reader_errors.append(error)
        finally:
            done.set()

    def terminate_child():
        if process is None:
            return
        if owns_group:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if owns_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        if owns_group:
            # A nested child may outlive its exited leader after cancellation.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        notice("started", f"full log: {output.resolve()}")
        save_record()
        if capture_json:
            result_stream = result_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=result_stream if capture_json else subprocess.PIPE,
            stderr=subprocess.PIPE if capture_json else subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1, env=launch_env, cwd=cwd,
            start_new_session=owns_group)
        record["pid"] = process.pid
        notice("running", "waiting for child output")
        save_record()
        pipe = process.stderr if capture_json else process.stdout
        reader = threading.Thread(target=read_lines, args=(pipe,), daemon=True, name="command-log-reader")
        reader.start()
        last_output = last_notice = time.perf_counter()
        while not (done.is_set() and messages.empty() and process.poll() is not None):
            try:
                line = messages.get(timeout=min(.1, heartbeat_seconds))
            except queue.Empty:
                line = None
            if line is not None:
                print(line, end="", flush=True)
                if line.strip():
                    last_output = time.perf_counter()
            if reader_errors:
                raise reader_errors[0]
            now = time.perf_counter()
            if now - max(last_output, last_notice) >= heartbeat_seconds:
                notice("running", f"no_output={now-last_output:.1f}s (liveness only; progress not confirmed)")
                save_record()
                last_notice = now
        code = process.wait()
        if reader_errors:
            raise reader_errors[0]
        if capture_json and code == 0:
            json.loads(result_path.read_text(encoding="utf-8"))
        record.update(returncode=code, status="succeeded" if code == 0 else "failed_or_interrupted")
        if code:
            raise subprocess.CalledProcessError(code, command)
        notice("completed", f"exit={code}; full log: {output.resolve()}")
    except BaseException as error:
        record.update(status="failed_or_interrupted", exception=type(error).__name__ + ": " + str(error))
        terminate_child()
        if process is not None:
            record["returncode"] = process.returncode
        notice("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
               f"exit={record['returncode']} {type(error).__name__}; full log: {output.resolve()}")
        raise
    finally:
        stop.set()
        if reader is not None:
            reader.join(timeout=2)
        if process is not None and (reader is None or not reader.is_alive()):
            pipe = process.stderr if capture_json else process.stdout
            if pipe is not None:
                pipe.close()
        if result_stream is not None:
            result_stream.close()
        if extra_record is not None:
            record.update(extra_record() if callable(extra_record) else extra_record)
        save_record()
        stream.close()
    result = subprocess.CompletedProcess(command, record["returncode"],
        result_path.read_text(encoding="utf-8") if capture_json else None)
    result.record = record
    return result
