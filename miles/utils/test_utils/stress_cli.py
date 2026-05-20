"""Importable entry point for the CPU-only stress harness CLI.

The user-facing script lives at ``scripts/tools/stress_session_server_cpu.py``
as a thin wrapper around :func:`main` here; the logic lives in this module so
that pytest tests can ``from miles.utils.test_utils.stress_cli import main``,
monkey-patch ``verify_session``, and exercise the driver's exit-code path
without re-implementing subprocess plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402

from miles.utils.test_utils.r3_codec import format_signature  # noqa: E402
from miles.utils.test_utils.stress_launchers import StressProcessTrio, StressR3Spec  # noqa: E402
from miles.utils.test_utils.stress_verifier import verify_session  # noqa: E402

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DriverConfig:
    num_sessions: int
    num_turns: int
    output_tokens: int
    r3: StressR3Spec
    hf_checkpoint: str
    output_dir: Path
    request_timeout: float
    readiness_timeout: float
    log_subprocesses: bool


SMOKE_DEFAULTS = DriverConfig(
    num_sessions=4,
    num_turns=1,
    output_tokens=1024,
    r3=StressR3Spec(inject=True, num_layers=28, topk=8),
    hf_checkpoint="Qwen/Qwen3-0.6B",
    output_dir=Path("outputs/cpu-stress/smoke"),
    request_timeout=60.0,
    readiness_timeout=120.0,
    log_subprocesses=False,
)


def parse_args(argv: list[str] | None = None) -> DriverConfig:
    p = argparse.ArgumentParser(description="CPU-only stress driver for the Miles session server (single-cell mode).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use plan-standard smoke defaults (4 sessions x 1 turn x 1024 tokens, R3 on).",
    )
    p.add_argument("--num-sessions", type=int)
    p.add_argument("--num-turns", type=int)
    p.add_argument("--output-tokens", type=int)
    p.add_argument("--r3-num-layers", type=int)
    p.add_argument("--r3-topk", type=int)
    p.add_argument("--inject-r3", dest="inject_r3", action="store_true", default=None)
    p.add_argument("--no-inject-r3", dest="inject_r3", action="store_false")
    p.add_argument("--hf-checkpoint")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--request-timeout", type=float)
    p.add_argument("--readiness-timeout", type=float)
    p.add_argument("--log-subprocesses", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    base = (
        SMOKE_DEFAULTS
        if args.smoke
        else dataclasses.replace(
            SMOKE_DEFAULTS,
            num_sessions=1,
            num_turns=1,
            output_tokens=32,
            r3=StressR3Spec(inject=False, num_layers=0, topk=0),
            output_dir=Path("outputs/cpu-stress/ad-hoc"),
        )
    )

    cfg = dataclasses.replace(
        base,
        num_sessions=args.num_sessions if args.num_sessions is not None else base.num_sessions,
        num_turns=args.num_turns if args.num_turns is not None else base.num_turns,
        output_tokens=args.output_tokens if args.output_tokens is not None else base.output_tokens,
        r3=StressR3Spec(
            inject=(args.inject_r3 if args.inject_r3 is not None else base.r3.inject),
            num_layers=(args.r3_num_layers if args.r3_num_layers is not None else base.r3.num_layers),
            topk=(args.r3_topk if args.r3_topk is not None else base.r3.topk),
        ),
        hf_checkpoint=args.hf_checkpoint or base.hf_checkpoint,
        output_dir=args.output_dir or base.output_dir,
        request_timeout=args.request_timeout if args.request_timeout is not None else base.request_timeout,
        readiness_timeout=args.readiness_timeout if args.readiness_timeout is not None else base.readiness_timeout,
        log_subprocesses=args.log_subprocesses or base.log_subprocesses,
    )

    if cfg.num_sessions <= 0:
        raise SystemExit("--num-sessions must be positive")
    if cfg.num_turns <= 0:
        raise SystemExit("--num-turns must be positive")
    if cfg.output_tokens <= 0:
        raise SystemExit("--output-tokens must be positive")
    cfg.r3.validate()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return cfg


async def _create_session(client: httpx.AsyncClient, base_url: str) -> str:
    r = await client.post(f"{base_url}/sessions")
    r.raise_for_status()
    return r.json()["session_id"]


async def _post_one_turn(
    client: httpx.AsyncClient,
    base_url: str,
    sid: str,
    messages: list[dict],
) -> dict:
    payload = {"messages": messages, "model": "mock-stress"}
    r = await client.post(f"{base_url}/sessions/{sid}/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()


async def _run_one_session(
    client: httpx.AsyncClient,
    base_url: str,
    session_idx: int,
    num_turns: int,
) -> dict:
    """Walk one session through ``num_turns`` chat-completions calls.

    Pattern: turn 0 sends ``[user]``; turn k>0 sends
    ``[user, assistant_0, tool_0, ..., assistant_{k-1}, tool_{k-1}]``,
    where each tool message carries the new turn's signature. Both the
    session server's TITO append-only validator and the verifier's per-record
    signature extractor key on the LAST message of the request.
    """
    sid = await _create_session(client, base_url)
    expected_signatures: list[str] = []
    messages: list[dict] = []

    for turn in range(num_turns):
        sig = format_signature(sid, turn)
        if turn == 0:
            messages = [{"role": "user", "content": f"{sig} session-{session_idx} start"}]
        else:
            messages.append(
                {
                    "role": "tool",
                    "content": f"{sig} session-{session_idx} continue",
                    "tool_call_id": f"call-{turn:05d}",
                }
            )
        body = await _post_one_turn(client, base_url, sid, messages)
        assistant_msg = body["choices"][0]["message"]
        messages.append(assistant_msg)
        expected_signatures.append(sig)

    return {
        "sid": sid,
        "session_idx": session_idx,
        "num_turns": num_turns,
        "signatures": expected_signatures,
    }


async def _drive(cfg: DriverConfig, trio: StressProcessTrio) -> dict:
    base_url = trio.session_url
    limits = httpx.Limits(max_connections=cfg.num_sessions + 16)
    timeout = httpx.Timeout(cfg.request_timeout)

    t0 = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        session_results = await asyncio.gather(
            *[_run_one_session(client, base_url, i, cfg.num_turns) for i in range(cfg.num_sessions)],
            return_exceptions=True,
        )
        load_seconds = time.perf_counter() - t0

        per_session_dumps = []
        for r in session_results:
            if isinstance(r, BaseException):
                per_session_dumps.append({"error": repr(r)})
                continue
            dump_resp = await client.get(f"{base_url}/sessions/{r['sid']}")
            dump_resp.raise_for_status()
            per_session_dumps.append({"info": r, "dump": dump_resp.json()})

    return {"load_seconds": load_seconds, "per_session": per_session_dumps}


def _run_verifier(stress_result: dict, cfg: DriverConfig) -> dict:
    """Run the batch verifier against every session. The driver wrapping this
    exits nonzero if any session fails any invariant."""
    all_pass = True
    verifier_reports = []
    for entry in stress_result["per_session"]:
        if "error" in entry:
            all_pass = False
            verifier_reports.append({"error": entry["error"]})
            continue
        info = entry["info"]
        dump = entry["dump"]
        report = verify_session(
            records=dump["records"],
            expected_sid=info["sid"],
            expected_turn_count=info["num_turns"],
            accumulated_token_ids=dump.get("metadata", {}).get("accumulated_token_ids", []),
            r3_num_layers=cfg.r3.num_layers if cfg.r3.inject else 0,
            r3_topk=cfg.r3.topk if cfg.r3.inject else 0,
        )
        if not report.overall_pass:
            all_pass = False
        verifier_reports.append(
            {
                "sid": info["sid"],
                "session_idx": info["session_idx"],
                "num_turns": info["num_turns"],
                "overall_pass": report.overall_pass,
                "by_invariant": {
                    name: {"passed": r.passed, "mismatches": r.mismatches[:5]}
                    for name, r in report.by_invariant.items()
                },
            }
        )
    return {"overall_pass": all_pass, "reports": verifier_reports}


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(cfg.output_dir / "subprocess-logs") if cfg.log_subprocesses else None

    summary = {
        "config": {
            "num_sessions": cfg.num_sessions,
            "num_turns": cfg.num_turns,
            "output_tokens": cfg.output_tokens,
            "r3": dataclasses.asdict(cfg.r3),
            "hf_checkpoint": cfg.hf_checkpoint,
        },
        "started_at": time.time(),
    }

    exit_code = 0
    try:
        with StressProcessTrio(
            hf_checkpoint=cfg.hf_checkpoint,
            output_tokens=cfg.output_tokens,
            r3=cfg.r3,
            readiness_timeout=cfg.readiness_timeout,
            log_dir=log_dir,
        ) as trio:
            summary["mock_url"] = trio.mock_url
            summary["session_url"] = trio.session_url
            summary["mock_pid"] = trio.mock_proc.pid if trio.mock_proc else None
            summary["session_pid"] = trio.session_proc.pid if trio.session_proc else None
            summary["driver_pid"] = os.getpid()

            stress_result = asyncio.run(_drive(cfg, trio))
            summary["load_seconds"] = stress_result["load_seconds"]

            verifier_result = _run_verifier(stress_result, cfg)
            summary["verifier"] = verifier_result
            if not verifier_result["overall_pass"]:
                exit_code = 2
                logger.error("Verifier failed: see %s/summary.json", cfg.output_dir)
    except Exception as exc:
        summary["fatal_error"] = repr(exc)
        exit_code = 3
        logger.exception("Stress driver hit a fatal error")
    finally:
        summary["ended_at"] = time.time()
        summary["exit_code"] = exit_code
        (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if exit_code == 0:
        logger.info(
            "Stress run PASS in %.2fs across %d sessions x %d turns (R3 inject=%s)",
            summary.get("load_seconds", 0.0),
            cfg.num_sessions,
            cfg.num_turns,
            cfg.r3.inject,
        )

    return exit_code
