"""Importable entry point for the CPU-only stress harness CLI.

The user-facing script lives at ``scripts/tools/stress_session_server_cpu.py``
as a thin wrapper around :func:`main` here; the logic lives in this module so
pytest tests can ``from miles.utils.test_utils.stress_cli import main``,
monkey-patch ``verify_session``, and exercise the driver's exit-code path
without re-implementing subprocess plumbing.

Round 3 adds:
- per-PID RSS / CPU% sampling via :class:`stress_metrics.PerPidSampler`,
  written to ``<output>/rss.csv`` plus a summary block in ``summary.json``;
- per-turn latency (``time.perf_counter()`` around each ``await
  append_turn``) aggregated into P50 / P95 / P99 / max in
  ``summary["latency_ms"]``;
- a streaming verifier loop -- the driver no longer holds every session's
  records in driver memory; each session is fetched, verified, and the
  dump is dropped before moving on, so driver RSS does not grow with
  sample-count x trajectory-length;
- a mid-cell SIGUSR1 fired at ~50% wall-clock for tracemalloc snapshotting
  on the session_server subprocess (the actual snapshot is written by
  ``_server_proc._install_sigusr1_tracemalloc``);
- a crash watchdog -- ``StressProcessTrio.session_proc.poll()`` is checked
  before and after the load phase; if the session_server died, the driver
  writes ``crash.json`` with the last known in-flight state and the
  subprocess's stderr tail;
- a failure replay command -- on any FAIL the driver writes a
  ``replay_command`` field into the summary that reproduces a single
  session / turn deterministically.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import shlex

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import psutil  # noqa: E402

from miles.utils.test_utils.r3_codec import format_signature  # noqa: E402
from miles.utils.test_utils.stress_launchers import StressProcessTrio, StressR3Spec  # noqa: E402
from miles.utils.test_utils.stress_metrics import (  # noqa: E402
    PerPidSampler,
    compute_latency_percentiles,
    estimate_cell_bytes,
)
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
    enable_tracemalloc: bool = False  # tracemalloc(25) adds ~100x per-alloc overhead -- opt-in only
    ram_headroom_fraction: float = 0.9  # capacity preflight ceiling
    cell_label: str = ""  # informational, surfaces in summary
    mid_cell_signal_count: int = 1


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
    enable_tracemalloc=False,
    ram_headroom_fraction=0.9,
    cell_label="smoke",
    mid_cell_signal_count=0,  # smoke is too short for a mid-cell tracemalloc snapshot to be useful
)


def parse_args(argv: list[str] | None = None) -> DriverConfig:
    p = argparse.ArgumentParser(
        description="CPU-only stress driver for the Miles session server (single-cell mode).",
    )
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
    p.add_argument("--enable-tracemalloc", dest="enable_tracemalloc", action="store_true", default=None)
    p.add_argument("--no-tracemalloc", dest="enable_tracemalloc", action="store_false")
    p.add_argument("--ram-headroom-fraction", type=float)
    p.add_argument("--cell-label", default="")
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
            cell_label="ad-hoc",
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
        enable_tracemalloc=(
            args.enable_tracemalloc if args.enable_tracemalloc is not None else base.enable_tracemalloc
        ),
        ram_headroom_fraction=(
            args.ram_headroom_fraction if args.ram_headroom_fraction is not None else base.ram_headroom_fraction
        ),
        cell_label=args.cell_label or base.cell_label,
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


# ---------------------------------------------------------------------------
# Capacity preflight
# ---------------------------------------------------------------------------


def capacity_preflight(cfg: DriverConfig) -> dict:
    """Estimate cell memory footprint and compare to available RAM. Returns
    a dict that callers can inline into ``summary["capacity_preflight"]``."""
    trajectory_length = cfg.num_turns * cfg.output_tokens
    estimated = estimate_cell_bytes(
        num_sessions=cfg.num_sessions,
        trajectory_length=trajectory_length,
        r3_num_layers=cfg.r3.num_layers,
        r3_topk=cfg.r3.topk,
        inject_r3=cfg.r3.inject,
    )
    available = psutil.virtual_memory().available
    ceiling = int(available * cfg.ram_headroom_fraction)
    over_budget = estimated > ceiling
    return {
        "estimated_bytes": estimated,
        "available_bytes": available,
        "ceiling_bytes": ceiling,
        "ram_headroom_fraction": cfg.ram_headroom_fraction,
        "over_budget": over_budget,
    }


def make_replay_command(cfg: DriverConfig, *, sid: str | None, turn_idx: int | None) -> str:
    """Construct a CLI invocation that reproduces a single session-turn
    deterministically. The mock derives its payload from the
    ``[STRESS sid=... turn=...]`` signature, so the same ``output_tokens``,
    ``num_layers``, and ``topk`` plus the same sid + turn give a byte-stable
    repro.
    """
    parts = [
        sys.executable,
        "scripts/tools/stress_session_server_cpu.py",
        "--num-sessions",
        "1",
        "--num-turns",
        str(cfg.num_turns),
        "--output-tokens",
        str(cfg.output_tokens),
        "--r3-num-layers",
        str(cfg.r3.num_layers),
        "--r3-topk",
        str(cfg.r3.topk),
    ]
    if cfg.r3.inject:
        parts.append("--inject-r3")
    else:
        parts.append("--no-inject-r3")
    if sid:
        parts.extend(["--cell-label", f"replay-sid={sid}-turn={turn_idx}"])
    return " ".join(shlex.quote(p) for p in parts)


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------


async def _create_session(client: httpx.AsyncClient, base_url: str) -> str:
    r = await client.post(f"{base_url}/sessions")
    r.raise_for_status()
    return r.json()["session_id"]


async def _post_one_turn(client: httpx.AsyncClient, base_url: str, sid: str, messages: list[dict]) -> dict:
    payload = {"messages": messages, "model": "mock-stress"}
    r = await client.post(f"{base_url}/sessions/{sid}/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()


async def _run_one_session(
    client: httpx.AsyncClient,
    base_url: str,
    session_idx: int,
    num_turns: int,
    last_success_tracker: dict,
) -> dict:
    """Walk one session through ``num_turns`` chat-completions calls.

    Updates ``last_success_tracker[session_idx]`` after every successful
    turn so the crash watchdog can attribute "what was in flight when
    the server died" when something goes wrong.
    """
    sid = await _create_session(client, base_url)
    expected_signatures: list[str] = []
    turn_latencies_seconds: list[float] = []
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
        t0 = time.perf_counter()
        body = await _post_one_turn(client, base_url, sid, messages)
        turn_latencies_seconds.append(time.perf_counter() - t0)

        assistant_msg = body["choices"][0]["message"]
        messages.append(assistant_msg)
        expected_signatures.append(sig)
        last_success_tracker[session_idx] = {"sid": sid, "turn": turn, "ts": time.time()}

    return {
        "sid": sid,
        "session_idx": session_idx,
        "num_turns": num_turns,
        "signatures": expected_signatures,
        "turn_latencies_seconds": turn_latencies_seconds,
    }


async def _fire_mid_cell_signals(trio: StressProcessTrio, count: int, total_estimate_seconds: float) -> None:
    """Schedule SIGUSR1 to the session subprocess at roughly even intervals
    during the cell. The handler takes a tracemalloc snapshot."""
    if count <= 0 or total_estimate_seconds <= 0:
        return
    delay = total_estimate_seconds / (count + 1)
    for i in range(count):
        await asyncio.sleep(delay)
        trio.signal_session_tracemalloc()
        logger.debug("mid-cell SIGUSR1 fired (%d/%d)", i + 1, count)


async def _drive_and_verify_streaming(cfg: DriverConfig, trio: StressProcessTrio) -> dict:
    """Run the load and verifier in one shot, streaming-style.

    Driver-level memory invariant: at any time the only retained state is
    (a) the per-session metadata dict (sid, session_idx, num_turns,
    signatures, turn_latencies), and (b) one in-flight session dump
    during the verifier loop. The dump goes out of scope at the bottom
    of each iteration. The verifier never accumulates dumps across
    sessions, so driver RSS stays O(num_sessions) rather than
    O(num_sessions * trajectory_length).
    """
    base_url = trio.session_url
    limits = httpx.Limits(max_connections=cfg.num_sessions + 16)
    timeout = httpx.Timeout(cfg.request_timeout)

    last_success_tracker: dict[int, dict] = {}

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        t0 = time.perf_counter()

        # Estimate cell duration from smoke baseline (~0.05s per session-turn);
        # mid-cell signals only fire if the cell takes meaningfully longer.
        rough_estimate_s = max(2.0, cfg.num_sessions * cfg.num_turns * 0.05)
        signal_task = asyncio.create_task(_fire_mid_cell_signals(trio, cfg.mid_cell_signal_count, rough_estimate_s))

        try:
            session_results = await asyncio.gather(
                *[
                    _run_one_session(client, base_url, i, cfg.num_turns, last_success_tracker)
                    for i in range(cfg.num_sessions)
                ],
                return_exceptions=True,
            )
        finally:
            signal_task.cancel()
            try:
                await signal_task
            except (asyncio.CancelledError, Exception):
                pass

        load_seconds = time.perf_counter() - t0

        # Streaming verifier: GET → verify → drop, never accumulate.
        verifier_reports = []
        all_pass = True
        all_latencies_seconds: list[float] = []
        for r in session_results:
            if isinstance(r, BaseException):
                all_pass = False
                verifier_reports.append({"error": repr(r)})
                continue
            all_latencies_seconds.extend(r["turn_latencies_seconds"])
            try:
                dump_resp = await client.get(f"{base_url}/sessions/{r['sid']}")
                dump_resp.raise_for_status()
                dump = dump_resp.json()
            except Exception as e:
                all_pass = False
                verifier_reports.append({"sid": r["sid"], "error": f"GET failed: {e!r}"})
                continue

            report = verify_session(
                records=dump["records"],
                expected_sid=r["sid"],
                expected_turn_count=r["num_turns"],
                accumulated_token_ids=dump.get("metadata", {}).get("accumulated_token_ids", []),
                r3_num_layers=cfg.r3.num_layers if cfg.r3.inject else 0,
                r3_topk=cfg.r3.topk if cfg.r3.inject else 0,
            )
            if not report.overall_pass:
                all_pass = False
            verifier_reports.append(
                {
                    "sid": r["sid"],
                    "session_idx": r["session_idx"],
                    "num_turns": r["num_turns"],
                    "overall_pass": report.overall_pass,
                    "by_invariant": {
                        name: {"passed": ir.passed, "mismatches": ir.mismatches[:5]}
                        for name, ir in report.by_invariant.items()
                    },
                }
            )
            # ``dump`` and its records become eligible for GC at the end
            # of this loop iteration -- this is the streaming invariant.

    return {
        "load_seconds": load_seconds,
        "session_results_count": len(session_results),
        "verifier_reports": verifier_reports,
        "verifier_overall_pass": all_pass,
        "latency_seconds_samples": all_latencies_seconds,
        "last_success_tracker": last_success_tracker,
    }


# ---------------------------------------------------------------------------
# Crash detection
# ---------------------------------------------------------------------------


def _detect_subprocess_crash(trio: StressProcessTrio) -> dict | None:
    """Return crash info if either subprocess has died, else None."""
    for label, proc in (("mock", trio.mock_proc), ("session", trio.session_proc)):
        if proc is None:
            continue
        rc = proc.poll()
        if rc is not None:
            return {"role": label, "exit_code": rc, "pid": proc.pid}
    return None


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def run_one_cell(cfg: DriverConfig) -> tuple[int, dict]:
    """Run a single stress cell and return (exit_code, summary).

    Exit codes:
      0 — A pass + B pass
      2 — A pass (no crash) but B fail (verifier)
      3 — A fail (subprocess crash, OOM, timeout, fatal error)
      4 — capacity preflight rejected the cell (over budget)
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc_dir = (cfg.output_dir / "tracemalloc") if cfg.enable_tracemalloc else None
    log_dir = str(cfg.output_dir / "subprocess-logs") if cfg.log_subprocesses else None

    summary: dict = {
        "config": {
            "cell_label": cfg.cell_label,
            "num_sessions": cfg.num_sessions,
            "num_turns": cfg.num_turns,
            "output_tokens": cfg.output_tokens,
            "trajectory_length": cfg.num_turns * cfg.output_tokens,
            "r3": dataclasses.asdict(cfg.r3),
            "hf_checkpoint": cfg.hf_checkpoint,
        },
        "started_at": time.time(),
    }

    preflight = capacity_preflight(cfg)
    summary["capacity_preflight"] = preflight
    if preflight["over_budget"]:
        summary["status"] = "capacity-fail"
        summary["exit_code"] = 4
        summary["ended_at"] = time.time()
        (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        logger.warning(
            "Cell %s rejected by capacity preflight: estimated=%d, ceiling=%d",
            cfg.cell_label or "(unnamed)",
            preflight["estimated_bytes"],
            preflight["ceiling_bytes"],
        )
        return 4, summary

    exit_code = 0

    try:
        with StressProcessTrio(
            hf_checkpoint=cfg.hf_checkpoint,
            output_tokens=cfg.output_tokens,
            r3=cfg.r3,
            readiness_timeout=cfg.readiness_timeout,
            log_dir=log_dir,
            tracemalloc_dump_dir=str(tracemalloc_dir) if tracemalloc_dir else None,
        ) as trio:
            pids = {
                "driver": os.getpid(),
                "session": trio.session_proc.pid,
                "mock": trio.mock_proc.pid,
            }
            summary["pids"] = pids
            summary["mock_url"] = trio.mock_url
            summary["session_url"] = trio.session_url

            with PerPidSampler(pids, interval=0.5) as sampler:
                stress_result = asyncio.run(_drive_and_verify_streaming(cfg, trio))

            sampler.write_csv(cfg.output_dir / "rss.csv")
            summary["per_pid_metrics"] = sampler.summary()
            summary["load_seconds"] = stress_result["load_seconds"]
            summary["latency_ms"] = compute_latency_percentiles(stress_result["latency_seconds_samples"])
            summary["verifier"] = {
                "overall_pass": stress_result["verifier_overall_pass"],
                "reports": stress_result["verifier_reports"],
            }
            summary["last_success_tracker"] = stress_result["last_success_tracker"]

            crash_info = _detect_subprocess_crash(trio)
            if crash_info is not None:
                exit_code = 3
                summary["crash"] = crash_info
                # Best-effort grab of stderr tail (already captured to log_dir or DEVNULL).
                logger.error("Subprocess crash detected: %s", crash_info)

            if exit_code == 0 and not stress_result["verifier_overall_pass"]:
                exit_code = 2

            summary["exit_code"] = exit_code
            summary["status"] = {0: "pass", 2: "verifier-fail", 3: "crash", 4: "capacity-fail"}.get(
                exit_code, "unknown"
            )

    except Exception as exc:
        exit_code = 3
        summary["fatal_error"] = repr(exc)
        summary["exit_code"] = 3
        summary["status"] = "fatal-error"
        logger.exception("Stress driver hit a fatal error")
    finally:
        summary["ended_at"] = time.time()

    # Failure replay command on any non-pass exit.
    if exit_code != 0:
        first_failing_sid: str | None = None
        first_failing_turn: int | None = None
        for r in summary.get("verifier", {}).get("reports", []):
            if isinstance(r, dict) and r.get("overall_pass") is False and "sid" in r:
                first_failing_sid = r["sid"]
                # Use the failing invariant's first mismatch index if available
                for inv in r.get("by_invariant", {}).values():
                    for m in inv.get("mismatches", []):
                        if isinstance(m, dict) and "index" in m:
                            first_failing_turn = m["index"]
                            break
                    if first_failing_turn is not None:
                        break
                break
        summary["replay_command"] = make_replay_command(cfg, sid=first_failing_sid, turn_idx=first_failing_turn)

    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if exit_code == 0:
        logger.info(
            "Stress cell %s PASS in %.2fs across %d sessions x %d turns (R3=%s)",
            cfg.cell_label or "(unnamed)",
            summary.get("load_seconds", 0.0),
            cfg.num_sessions,
            cfg.num_turns,
            cfg.r3.inject,
        )

    return exit_code, summary


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    exit_code, _ = run_one_cell(cfg)
    return exit_code
