"""Per-PID metrics sampler for the CPU stress harness.

Spawns a background thread that polls ``psutil.Process(pid)`` at a fixed
cadence (default 0.5s) for each of the three stress processes (driver,
session_server subprocess, mock_sglang subprocess). RSS and CPU% are
recorded as a time-series, written to a CSV at end-of-run, and summarised
(peak RSS, mean CPU%) into the cell's ``summary.json``.

Deliberately minimal — Round 3 wires the existing psutil API and does not
attempt anything more invasive (no /proc parsing, no GC pressure probing).
The richer profiler hooks live in ``_server_proc.py``'s SIGUSR1
tracemalloc handler.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil


@dataclass
class PerPidSample:
    timestamp: float
    rss_bytes: int
    cpu_percent: float


@dataclass
class PerPidSeries:
    pid: int
    label: str
    samples: list[PerPidSample] = field(default_factory=list)

    @property
    def peak_rss_bytes(self) -> int:
        return max((s.rss_bytes for s in self.samples), default=0)

    @property
    def mean_cpu_percent(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.cpu_percent for s in self.samples) / len(self.samples)


class PerPidSampler:
    """Background thread sampling RSS and CPU% per PID at a fixed interval.

    Usage::

        with PerPidSampler({"driver": os.getpid(), "session": p1, "mock": p2}) as s:
            # ... drive load ...
        s.write_csv(Path("rss.csv"))
        summary["per_pid_metrics"] = s.summary()

    Processes that disappear mid-run (subprocess exit, OOM kill, etc.) are
    silently dropped from subsequent sampling iterations rather than
    causing the sampler thread to crash.
    """

    def __init__(self, pids: dict[str, int], interval: float = 0.5):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.series: dict[str, PerPidSeries] = {
            label: PerPidSeries(pid=pid, label=label) for label, pid in pids.items()
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._procs: dict[str, psutil.Process | None] = {}
        for label, pid in pids.items():
            try:
                proc = psutil.Process(pid)
                # Prime so the first cpu_percent call returns a meaningful delta.
                proc.cpu_percent(interval=None)
                self._procs[label] = proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._procs[label] = None

    def __enter__(self) -> PerPidSampler:
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="PerPidSampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        # sample-then-sleep, not sleep-then-sample, so short-lived cells still
        # get at least one data point before the sampler is stopped.
        self._sample_once()
        while not self._stop.wait(self.interval):
            self._sample_once()

    def _sample_once(self) -> None:
        now = time.time()
        for label, proc in list(self._procs.items()):
            if proc is None:
                continue
            try:
                with proc.oneshot():
                    rss = proc.memory_info().rss
                    cpu = proc.cpu_percent(interval=None)
                self.series[label].samples.append(PerPidSample(now, rss, cpu))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Subprocess died; stop sampling it but keep the recorded series.
                self._procs[label] = None

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write("timestamp,label,pid,rss_bytes,cpu_percent\n")
            for label, ser in self.series.items():
                for s in ser.samples:
                    f.write(f"{s.timestamp:.3f},{label},{ser.pid},{s.rss_bytes},{s.cpu_percent:.2f}\n")

    def summary(self) -> dict:
        return {
            label: {
                "pid": ser.pid,
                "samples": len(ser.samples),
                "peak_rss_bytes": ser.peak_rss_bytes,
                "mean_cpu_percent": round(ser.mean_cpu_percent, 2),
            }
            for label, ser in self.series.items()
        }


def compute_latency_percentiles(latencies_seconds: list[float]) -> dict:
    """Driver-side latency aggregate. Returns p50/p95/p99/max in milliseconds,
    plus the raw count. Returns an empty dict when given no samples."""
    if not latencies_seconds:
        return {}
    ms = sorted(x * 1000.0 for x in latencies_seconds)
    n = len(ms)

    def _q(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ms[idx]

    return {
        "count": n,
        "p50_ms": _q(0.50),
        "p95_ms": _q(0.95),
        "p99_ms": _q(0.99),
        "max_ms": ms[-1],
    }


def estimate_cell_bytes(
    num_sessions: int, trajectory_length: int, r3_num_layers: int, r3_topk: int, inject_r3: bool
) -> int:
    """Conservative byte estimate for a sweep cell, used by capacity preflight.

    Models the dominant footprint: ``trajectory_length`` tokens per session,
    each carrying ``num_layers * topk * 4`` bytes of R3 + 4 bytes of logp +
    4 bytes of token_id + ~32 bytes of JSON / Python list overhead.
    """
    per_token = 0
    if inject_r3:
        per_token += r3_num_layers * r3_topk * 4
    per_token += 4 + 4 + 32
    return num_sessions * trajectory_length * per_token
