"""Subprocess entry point: real :class:`SessionServer` over uvicorn.

Spawned by :class:`miles.utils.test_utils.stress_launchers.StressProcessTrio`.
Builds the same ``args``-shaped namespace ``ray.rollout._start_session_server``
uses in production, constructs a ``SessionServer``, and serves it via
``uvicorn.run`` (blocking call). A ``SIGUSR1`` no-op handler is installed so
the driver can probe / poke the process without accidentally killing it; the
Round 3 tracemalloc hook will replace this stub.

Invoke via ``python -m miles.utils.test_utils.stress_launchers._server_proc``.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import signal  # noqa: E402
import sys  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import uvicorn  # noqa: E402

from miles.rollout.session.session_server import SessionServer  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stress-mode SessionServer subprocess entry point.")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--backend-url", required=True)
    p.add_argument("--hf-checkpoint", default="Qwen/Qwen3-0.6B")
    p.add_argument("--enable-r3", action="store_true")
    p.add_argument("--allowed-append-roles", default="tool")
    p.add_argument("--router-timeout", type=float, default=600.0)
    p.add_argument("--log-level", default="warning")
    return p


def _install_sigusr1_stub() -> None:
    """Reserve SIGUSR1 for Round 3's tracemalloc dump hook. Without an
    explicit handler Python's default action for SIGUSR1 is to terminate
    the process, which would defeat any later probe attempt; the stub just
    logs the receipt and continues."""

    def _handler(*_):
        sys.stderr.write("[stress-session-server] SIGUSR1 received (Round 2 stub; tracemalloc dump deferred)\n")
        sys.stderr.flush()

    signal.signal(signal.SIGUSR1, _handler)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _install_sigusr1_stub()

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
