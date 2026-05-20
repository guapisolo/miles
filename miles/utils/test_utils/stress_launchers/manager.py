"""Three-process orchestrator: driver + mock_sglang subprocess + session_server subprocess.

The driver runs in the calling Python process; this manager handles the
two subprocess children. ``StressProcessTrio`` is the only public surface.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressR3Spec:
    """Per-token R3 shape config passed through to both subprocesses."""

    inject: bool = False
    num_layers: int = 0
    topk: int = 0

    def expected_int32_per_token(self) -> int:
        return self.num_layers * self.topk

    def validate(self) -> None:
        if self.inject and (self.num_layers <= 0 or self.topk <= 0):
            raise ValueError(
                f"R3 injection requires num_layers > 0 and topk > 0; got "
                f"num_layers={self.num_layers}, topk={self.topk}"
            )


def _allocate_localhost_port() -> int:
    """Bind ephemeral, return the assigned port, immediately release.

    There is an inherent TOCTOU window between release and the subprocess's
    bind, but the surface is tiny on a localhost-only test host. Using
    SO_REUSEADDR keeps the subprocess able to rebind even if the kernel
    keeps the socket in TIME_WAIT briefly.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _poll_health(url: str, timeout: float) -> None:
    """Block until ``url`` returns HTTP 200, or raise ``TimeoutError``."""
    deadline = time.monotonic() + timeout
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=1.0)
            if resp.status_code == 200:
                return
            last_err = RuntimeError(f"non-200 status: {resp.status_code}")
        except (requests.RequestException, OSError) as exc:
            last_err = exc
        time.sleep(0.2)
    raise TimeoutError(f"readiness probe timed out for {url}: {last_err}")


class StressProcessTrio:
    """Spawn mock_sglang and session_server subprocesses, wire driver to them.

    Usage::

        with StressProcessTrio(output_tokens=2048, r3=StressR3Spec(inject=True, num_layers=28, topk=8)) as trio:
            # drive trio.session_url with httpx ...

    The context manager ensures both subprocesses are terminated on exit,
    even on driver exception or abrupt interpreter shutdown (via ``atexit``).
    """

    def __init__(
        self,
        *,
        hf_checkpoint: str = "Qwen/Qwen3-0.6B",
        output_tokens: int,
        r3: StressR3Spec | None = None,
        readiness_timeout: float = 60.0,
        log_dir: str | None = None,
        tracemalloc_dump_dir: str | None = None,
    ):
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        self.hf_checkpoint = hf_checkpoint
        self.output_tokens = output_tokens
        self.r3 = r3 or StressR3Spec()
        self.r3.validate()
        self.readiness_timeout = readiness_timeout
        self.log_dir = log_dir
        self.tracemalloc_dump_dir = tracemalloc_dump_dir

        self.mock_port: int | None = None
        self.session_port: int | None = None
        self.mock_proc: subprocess.Popen | None = None
        self.session_proc: subprocess.Popen | None = None

        self._atexit_registered = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def session_url(self) -> str:
        if self.session_port is None:
            raise RuntimeError("trio not started")
        return f"http://127.0.0.1:{self.session_port}"

    @property
    def mock_url(self) -> str:
        if self.mock_port is None:
            raise RuntimeError("trio not started")
        return f"http://127.0.0.1:{self.mock_port}"

    def start(self) -> None:
        self.mock_port = _allocate_localhost_port()
        self.session_port = _allocate_localhost_port()

        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

        self.mock_proc = subprocess.Popen(
            self._mock_command(),
            env=env,
            stdout=self._stream_for("mock", "stdout"),
            stderr=self._stream_for("mock", "stderr"),
        )
        try:
            _poll_health(f"{self.mock_url}/health", self.readiness_timeout)
        except Exception:
            self._stop_one("mock")
            raise

        self.session_proc = subprocess.Popen(
            self._session_command(),
            env=env,
            stdout=self._stream_for("session", "stdout"),
            stderr=self._stream_for("session", "stderr"),
        )
        try:
            _poll_health(f"{self.session_url}/health", self.readiness_timeout)
        except Exception:
            self._stop_one("session")
            self._stop_one("mock")
            raise

    def stop(self) -> None:
        self._stop_one("session")
        self._stop_one("mock")

    def __enter__(self) -> StressProcessTrio:
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _stream_for(self, role: str, channel: str):
        if self.log_dir is None:
            return subprocess.DEVNULL
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"{role}.{channel}.log")
        return open(path, "ab")  # noqa: SIM115 - subprocess takes ownership of the fd

    def _mock_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "miles.utils.test_utils.stress_launchers._mock_proc",
            "--port",
            str(self.mock_port),
            "--hf-checkpoint",
            self.hf_checkpoint,
            "--output-tokens",
            str(self.output_tokens),
            "--r3-num-layers",
            str(self.r3.num_layers),
            "--r3-topk",
            str(self.r3.topk),
        ]
        if self.r3.inject:
            cmd.append("--inject-r3")
        return cmd

    def _session_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "miles.utils.test_utils.stress_launchers._server_proc",
            "--port",
            str(self.session_port),
            "--backend-url",
            self.mock_url,
            "--hf-checkpoint",
            self.hf_checkpoint,
        ]
        if self.r3.inject:
            cmd.append("--enable-r3")
        if self.tracemalloc_dump_dir:
            cmd.extend(["--tracemalloc-dump-dir", self.tracemalloc_dump_dir])
        return cmd

    def signal_session_tracemalloc(self) -> None:
        """Send SIGUSR1 to the session_server subprocess (no-op if dead).

        The handler installed by ``_server_proc._install_sigusr1_tracemalloc``
        takes a tracemalloc snapshot and dumps top-N allocations to
        ``tracemalloc_dump_dir``. Call this mid-cell to capture an
        in-flight snapshot."""
        if self.session_proc is None or self.session_proc.poll() is not None:
            return
        try:
            self.session_proc.send_signal(signal.SIGUSR1)
        except ProcessLookupError:
            pass

    def _stop_one(self, role: str) -> None:
        proc = self.mock_proc if role == "mock" else self.session_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            logger.warning("[%s] subprocess did not exit on SIGTERM, sending SIGKILL", role)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)

    def _atexit_cleanup(self) -> None:
        # Defensive: in case the user forgot to call stop() in a non-context
        # invocation, atexit kills any survivors.
        self.stop()
