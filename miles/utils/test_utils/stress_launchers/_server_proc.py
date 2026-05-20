"""Subprocess entry point: real :class:`SessionServer` over uvicorn.

Spawned by :class:`miles.utils.test_utils.stress_launchers.StressProcessTrio`.
Builds the same ``args``-shaped namespace ``ray.rollout._start_session_server``
uses in production, constructs a ``SessionServer``, and serves it via
``uvicorn.run`` (blocking call).

On startup the process installs a SIGUSR1 handler that takes a tracemalloc
snapshot and dumps the top-N allocations to a timestamped file under
``--tracemalloc-dump-dir`` (when supplied). The driver sends SIGUSR1
periodically during a cell so the harness can attribute peak server-side
RSS back to Python objects — required for the AC-11 / AC-12 root-cause
story when a cell crashes.

Invoke via ``python -m miles.utils.test_utils.stress_launchers._server_proc``.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import uvicorn  # noqa: E402

from miles.rollout.session.session_server import SessionServer  # noqa: E402


TRACEMALLOC_TOP_N = 25


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stress-mode SessionServer subprocess entry point.")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--backend-url", required=True)
    p.add_argument("--hf-checkpoint", default="Qwen/Qwen3-0.6B")
    p.add_argument("--enable-r3", action="store_true")
    p.add_argument("--allowed-append-roles", default="tool")
    p.add_argument("--router-timeout", type=float, default=600.0)
    p.add_argument("--log-level", default="warning")
    p.add_argument(
        "--tracemalloc-dump-dir",
        default=None,
        help="If set, start tracemalloc at startup and dump top-N allocations to this directory on every SIGUSR1.",
    )
    return p


def _install_sigusr1_tracemalloc(dump_dir: str | None) -> None:
    """Install a SIGUSR1 handler that snapshots tracemalloc and writes the
    top-N statistics to ``dump_dir``. If ``dump_dir`` is None, the handler
    logs to stderr but does not crash on signal.

    Calling ``tracemalloc.start()`` here is safe: it is a no-op if already
    started, raises if started with a smaller frame count, so we catch
    that defensively.
    """
    if dump_dir:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        try:
            tracemalloc.start(25)
        except RuntimeError:
            # Already started with a different frame count; keep whatever
            # frame depth the existing instance has.
            pass

    def _handler(*_):
        try:
            snapshot = tracemalloc.take_snapshot()
        except RuntimeError:
            sys.stderr.write("[stress-session-server] tracemalloc not started; SIGUSR1 ignored\n")
            sys.stderr.flush()
            return
        if dump_dir is None:
            sys.stderr.write(
                "[stress-session-server] SIGUSR1: tracemalloc snapshot taken but no --tracemalloc-dump-dir; "
                "snapshot discarded\n"
            )
            sys.stderr.flush()
            return
        ts = int(time.time() * 1000)
        path = Path(dump_dir) / f"tracemalloc.{ts}.txt"
        with path.open("w") as f:
            f.write(f"=== tracemalloc snapshot pid={os.getpid()} ts_ms={ts} ===\n")
            for stat in snapshot.statistics("lineno")[:TRACEMALLOC_TOP_N]:
                f.write(f"{stat}\n")
        sys.stderr.write(f"[stress-session-server] SIGUSR1 dumped tracemalloc -> {path}\n")
        sys.stderr.flush()

    signal.signal(signal.SIGUSR1, _handler)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _install_sigusr1_tracemalloc(args.tracemalloc_dump_dir)

    server_args = SimpleNamespace(
        miles_router_timeout=args.router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        tito_allowed_append_roles=[role.strip() for role in args.allowed_append_roles.split(",") if role.strip()],
        trajectory_manager="linear_trajectory",
        session_server_instance_id=uuid.uuid4().hex,
        use_rollout_routing_replay=args.enable_r3,
    )
    server_obj = SessionServer(server_args, backend_url=args.backend_url)

    sys.stdout.write(f"[stress-session-server] starting on http://127.0.0.1:{args.port}\n")
    sys.stdout.flush()

    uvicorn.run(
        server_obj.app,
        host="127.0.0.1",
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
