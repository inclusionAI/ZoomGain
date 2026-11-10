"""Opt-in stage logging for ZwZ test runs; heartbeat is not a progress estimate."""
import os
import time
import threading
from contextlib import contextmanager


def enabled():
    return os.environ.get("ZWZ_PHASE_LOGS") == "1"


def emit(message):
    print(f"[ZwZ progress] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def report_step(step, total, state):
    if enabled():
        emit(f"step={step}/{total} {state}")


def wrap_timer(timer):
    @contextmanager
    def logged_timer(name, *args, **kwargs):
        if not enabled():
            with timer(name, *args, **kwargs) as result:
                yield result
            return
        try:
            interval = max(0.01, float(os.environ.get("ZWZ_PHASE_LOG_INTERVAL", "30")))
            if not interval < float("inf"):
                interval = 30.0
        except ValueError:
            interval = 30.0
        start = time.monotonic()
        stopped = threading.Event()
        def heartbeat():
            while not stopped.wait(interval):
                emit(f"stage={name} waiting elapsed={time.monotonic() - start:.1f}s")
        emit(f"stage={name} start")
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        state = "complete"
        try:
            with timer(name, *args, **kwargs) as result:
                yield result
        except BaseException:
            state = "failed"
            raise
        finally:
            stopped.set()
            thread.join(timeout=1)
            emit(f"stage={name} {state} elapsed={time.monotonic() - start:.1f}s")
    return logged_timer
