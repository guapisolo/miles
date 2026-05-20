"""Subprocess entry point: stress-mode :class:`MockSGLangServer`.

Spawned by :class:`miles.utils.test_utils.stress_launchers.StressProcessTrio`.
Reads its configuration from CLI args, starts a stress-mode
``MockSGLangServer`` (which runs uvicorn in a background thread), then
blocks on ``SIGTERM`` / ``SIGINT`` so the parent driver can shut it down
deterministically.

Invoke via ``python -m miles.utils.test_utils.stress_launchers._mock_proc``.
"""

from __future__ import annotations

import argparse
import os

# Force CPU-only before any miles imports — the harness's CPU-only contract
# (AC-1 negative test) is enforced at this layer.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, StressMockConfig  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stress-mode MockSGLangServer subprocess entry point.")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--hf-checkpoint", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--r3-num-layers", type=int, default=0)
    p.add_argument("--r3-topk", type=int, default=0)
    p.add_argument("--inject-r3", action="store_true")
    p.add_argument("--no-canonical-json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    config = StressMockConfig(
        enabled=True,
        output_tokens=args.output_tokens,
        inject_routed_experts=args.inject_r3,
        r3_num_layers=args.r3_num_layers,
        r3_topk=args.r3_topk,
        echo_signature=True,
        canonical_json=not args.no_canonical_json,
    )

    backend = MockSGLangServer(
        model_name=args.hf_checkpoint,
        process_fn=lambda _prompt: None,  # never called in stress mode
        host="127.0.0.1",
        port=args.port,
        latency=0.0,
        stress_config=config,
    )
    backend.start()
    sys.stdout.write(f"[stress-mock] ready on http://127.0.0.1:{args.port}\n")
    sys.stdout.flush()

    shutdown = threading.Event()

    def _on_signal(*_):
        shutdown.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    shutdown.wait()
    backend.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
