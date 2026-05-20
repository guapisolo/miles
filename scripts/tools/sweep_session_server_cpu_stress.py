#!/usr/bin/env python3
"""Sweep harness for the CPU-only session-server stress test.

Wraps :func:`miles.utils.test_utils.stress_cli.run_one_cell` to iterate
across a matrix of (num_sessions, trajectory_length) cells, write a
per-cell ``summary.json`` plus a top-level ``REPORT.md`` aggregating the
findings (status, peak RSS per PID, latency P50/P99, observed backend
concurrency, failure modes).

Sweep semantics:

- Cells run in (num_sessions ASC, trajectory_length ASC) order. The
  smallest cell runs first to establish a smoke baseline.
- Each cell starts a fresh ``StressProcessTrio`` (so ``SessionRegistry``
  starts empty per cell).
- Capacity preflight skips physically impossible cells before they spawn
  any subprocess (avoids OOM-killing the host).
- Stop conditions: ``error_rate > threshold`` / ``RSS over budget`` /
  ``wall_time > max_cell_walltime`` / verifier fail. On any cell failure,
  remaining cells at the same ``num_sessions`` row with larger
  ``trajectory_length`` are skipped (the boundary has been bracketed).
- Wave-batch fallback: when the largest configured cell A-fails (crash
  / OOM), the harness automatically runs the same total token volume in
  ``num_sessions // wave_size`` waves and labels the result
  ``equivalent_concurrency_pass=False`` (degraded mode).

Default matrix is intentionally small so the harness can demo on a
typical dev host (≤ 64 sessions × 4096 tokens). The plan's headline
5×3 matrix ({64, 128, 256, 512, 1024} × {5k, 20k, 50k}) targets a
dedicated stress host with ≥ 150 GB RAM; pass ``--matrix-config`` to
override.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

from miles.utils.test_utils.stress_cli import DriverConfig, run_one_cell  # noqa: E402
from miles.utils.test_utils.stress_launchers import StressR3Spec  # noqa: E402

logger = logging.getLogger(__name__)


DEFAULT_MATRIX_NS = [4, 16, 32, 64]
DEFAULT_MATRIX_LS = [256, 1024, 4096]
PLAN_MATRIX_NS = [64, 128, 256, 512, 1024]
PLAN_MATRIX_LS = [5_000, 20_000, 50_000]


@dataclasses.dataclass
class SweepConfig:
    matrix_ns: list[int]
    matrix_ls: list[int]
    tokens_per_turn: int
    r3: StressR3Spec
    hf_checkpoint: str
    output_dir: Path
    request_timeout: float
    readiness_timeout: float
    max_cell_walltime: float
    error_rate_threshold: float
    ram_headroom_fraction: float
    enable_tracemalloc: bool
    wave_batch_at_max: bool
    wave_size: int


def _derive_cell_shape(trajectory_length: int, tokens_per_turn: int) -> tuple[int, int]:
    """Returns (num_turns, output_tokens). For Round 3 simplicity all turns
    share ``output_tokens=tokens_per_turn`` and the effective trajectory
    length is ``num_turns * tokens_per_turn`` (so we ceiling the requested
    L to the nearest turn multiple).
    """
    if trajectory_length <= 0 or tokens_per_turn <= 0:
        raise ValueError("trajectory_length and tokens_per_turn must be positive")
    num_turns = max(1, -(-trajectory_length // tokens_per_turn))  # ceil division
    return num_turns, tokens_per_turn


def _cell_label(n: int, L_: int) -> str:
    return f"N{n}_L{L_}"


def _build_cell_cfg(
    sweep_cfg: SweepConfig, n: int, L_: int, *, cell_dir: Path, override_label: str | None = None
) -> DriverConfig:
    num_turns, output_tokens = _derive_cell_shape(L_, sweep_cfg.tokens_per_turn)
    return DriverConfig(
        num_sessions=n,
        num_turns=num_turns,
        output_tokens=output_tokens,
        r3=sweep_cfg.r3,
        hf_checkpoint=sweep_cfg.hf_checkpoint,
        output_dir=cell_dir,
        request_timeout=sweep_cfg.request_timeout,
        readiness_timeout=sweep_cfg.readiness_timeout,
        log_subprocesses=False,
        enable_tracemalloc=sweep_cfg.enable_tracemalloc,
        ram_headroom_fraction=sweep_cfg.ram_headroom_fraction,
        cell_label=override_label or _cell_label(n, L_),
        mid_cell_signal_count=1 if sweep_cfg.enable_tracemalloc else 0,
    )


def _maybe_run_wave_batch(sweep_cfg: SweepConfig, n: int, L_: int, root: Path) -> dict | None:
    """When the (max N, max L) cell A-fails, replay the same total token
    volume in ``num_sessions // wave_size`` waves. Each wave is a full
    cell with ``num_sessions=wave_size``; the union of waves equals the
    original cell's total tokens. Returned dict goes into the REPORT
    under "wave_batch_fallback".
    """
    waves = max(1, n // sweep_cfg.wave_size)
    cell_dir = root / f"wave-batch-N{n}_L{L_}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    wave_results: list[dict] = []
    aggregate_pass = True
    for wave_idx in range(waves):
        wave_dir = cell_dir / f"wave{wave_idx:02d}"
        cell_cfg = _build_cell_cfg(
            sweep_cfg,
            n=sweep_cfg.wave_size,
            L_=L_,
            cell_dir=wave_dir,
            override_label=f"wave-batch-N{n}_L{L_}-wave{wave_idx:02d}",
        )
        exit_code, summary = run_one_cell(cell_cfg)
        if exit_code != 0:
            aggregate_pass = False
        wave_results.append(
            {
                "wave_idx": wave_idx,
                "exit_code": exit_code,
                "status": summary.get("status"),
                "load_seconds": summary.get("load_seconds"),
                "latency_ms": summary.get("latency_ms"),
            }
        )
        # If wave-batch also fails, keep collecting waves so the REPORT
        # can show whether the failure is concurrency-driven or
        # data-volume-driven.
    return {
        "mode": "wave-batch",
        "equivalent_concurrency_pass": False,  # never equivalent to live N concurrency
        "wave_size": sweep_cfg.wave_size,
        "waves": waves,
        "aggregate_pass": aggregate_pass,
        "per_wave": wave_results,
    }


def run_sweep(sweep_cfg: SweepConfig) -> dict:
    sweep_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_report: dict = {
        "sweep_config": {
            "matrix_ns": sweep_cfg.matrix_ns,
            "matrix_ls": sweep_cfg.matrix_ls,
            "tokens_per_turn": sweep_cfg.tokens_per_turn,
            "r3": dataclasses.asdict(sweep_cfg.r3),
            "hf_checkpoint": sweep_cfg.hf_checkpoint,
            "enable_tracemalloc": sweep_cfg.enable_tracemalloc,
            "wave_batch_at_max": sweep_cfg.wave_batch_at_max,
        },
        "started_at": time.time(),
        "cells": [],
        "wave_batch_fallback": None,
    }

    # Track per-row early termination: once a (N, L) cell fails, skip
    # remaining cells at the same N with larger L.
    skip_rows: set[int] = set()

    smoke_p99_ms: float | None = None

    for n in sweep_cfg.matrix_ns:
        for L_ in sweep_cfg.matrix_ls:
            if n in skip_rows:
                sweep_report["cells"].append(
                    {
                        "label": _cell_label(n, L_),
                        "num_sessions": n,
                        "trajectory_length": L_,
                        "status": "skipped-by-stop-condition",
                    }
                )
                continue
            cell_dir = sweep_cfg.output_dir / _cell_label(n, L_)
            cell_cfg = _build_cell_cfg(sweep_cfg, n, L_, cell_dir=cell_dir)
            cell_t0 = time.perf_counter()
            exit_code, summary = run_one_cell(cell_cfg)
            cell_wall = time.perf_counter() - cell_t0

            latency = summary.get("latency_ms", {})
            cell_record = {
                "label": _cell_label(n, L_),
                "num_sessions": n,
                "trajectory_length": L_,
                "num_turns": cell_cfg.num_turns,
                "output_tokens": cell_cfg.output_tokens,
                "exit_code": exit_code,
                "status": summary.get("status"),
                "load_seconds": summary.get("load_seconds"),
                "wall_seconds": cell_wall,
                "latency_ms": latency,
                "per_pid_metrics": summary.get("per_pid_metrics"),
                "capacity_preflight": summary.get("capacity_preflight"),
                "verifier_overall_pass": summary.get("verifier", {}).get("overall_pass"),
                "crash": summary.get("crash"),
                "replay_command": summary.get("replay_command"),
                "latency_degraded": False,
            }

            # Soft latency guardrail: smoke baseline P99 × 100 → "latency-degraded".
            if smoke_p99_ms is None and exit_code == 0 and latency.get("p99_ms"):
                smoke_p99_ms = float(latency["p99_ms"])
            if (
                smoke_p99_ms is not None
                and latency.get("p99_ms") is not None
                and exit_code == 0
                and float(latency["p99_ms"]) > smoke_p99_ms * 100
            ):
                cell_record["latency_degraded"] = True

            sweep_report["cells"].append(cell_record)

            # Stop conditions: any non-pass → skip the rest of this N row.
            if exit_code != 0:
                logger.warning(
                    "Cell %s exited %d (%s); skipping larger L at N=%d",
                    cell_cfg.cell_label,
                    exit_code,
                    summary.get("status"),
                    n,
                )
                skip_rows.add(n)
            elif cell_wall > sweep_cfg.max_cell_walltime:
                logger.warning(
                    "Cell %s exceeded max wall time (%.1fs > %.1fs); skipping larger L at N=%d",
                    cell_cfg.cell_label,
                    cell_wall,
                    sweep_cfg.max_cell_walltime,
                    n,
                )
                cell_record["status"] = "walltime-exceeded"
                skip_rows.add(n)

    # Wave-batch fallback: only triggered when the (max N, max L) cell
    # was a crash/A-fail (status in crash / fatal-error), not a verifier
    # fail (which is a different invariant).
    if sweep_cfg.wave_batch_at_max:
        max_n = max(sweep_cfg.matrix_ns)
        max_l = max(sweep_cfg.matrix_ls)
        target_cell = next(
            (c for c in sweep_report["cells"] if c["num_sessions"] == max_n and c["trajectory_length"] == max_l),
            None,
        )
        if target_cell and target_cell.get("status") in {"crash", "fatal-error", "capacity-fail"}:
            sweep_report["wave_batch_fallback"] = _maybe_run_wave_batch(sweep_cfg, max_n, max_l, sweep_cfg.output_dir)

    sweep_report["ended_at"] = time.time()
    (sweep_cfg.output_dir / "sweep_report.json").write_text(
        json.dumps(sweep_report, indent=2, ensure_ascii=False, default=str)
    )
    _write_report_md(sweep_cfg, sweep_report)
    return sweep_report


def _format_bytes(b: int | None) -> str:
    if not b:
        return "—"
    if b < 1024:
        return f"{b}B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    val = float(b) / 1024
    for unit in units:
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PiB"


def _write_report_md(sweep_cfg: SweepConfig, sweep_report: dict) -> None:
    path = sweep_cfg.output_dir / "REPORT.md"
    lines: list[str] = []
    lines.append("# CPU Stress Sweep Report")
    lines.append("")
    cfg = sweep_report["sweep_config"]
    lines.append(f"- Matrix: N ∈ {cfg['matrix_ns']} × trajectory_length ∈ {cfg['matrix_ls']}")
    lines.append(f"- Tokens per turn: {cfg['tokens_per_turn']}")
    lines.append(
        f"- R3 inject: {cfg['r3']['inject']} "
        + (f"(num_layers={cfg['r3']['num_layers']}, topk={cfg['r3']['topk']})" if cfg["r3"]["inject"] else "")
    )
    lines.append(f"- HF checkpoint: `{cfg['hf_checkpoint']}`")
    lines.append(f"- Tracemalloc: {cfg['enable_tracemalloc']}")
    lines.append("")

    lines.append("## Cell Summary")
    lines.append("")
    lines.append(
        "| Cell | Status | Load (s) | Wall (s) | P50 (ms) | P99 (ms) | Driver RSS | Session RSS | Mock RSS | Replay |"
    )
    lines.append(
        "|------|--------|----------|----------|----------|----------|-----------|-------------|----------|--------|"
    )
    for c in sweep_report["cells"]:
        pid_metrics = c.get("per_pid_metrics") or {}
        latency = c.get("latency_ms") or {}
        replay_short = "yes" if c.get("replay_command") else "—"
        lines.append(
            "| {label} | {status} | {load:.2f} | {wall:.2f} | {p50} | {p99} | {drv} | {sess} | {mck} | {rep} |".format(
                label=c["label"],
                status=c.get("status", "?"),
                load=c.get("load_seconds") or 0.0,
                wall=c.get("wall_seconds") or 0.0,
                p50=f"{latency['p50_ms']:.1f}" if latency.get("p50_ms") is not None else "—",
                p99=f"{latency['p99_ms']:.1f}" if latency.get("p99_ms") is not None else "—",
                drv=_format_bytes((pid_metrics.get("driver") or {}).get("peak_rss_bytes")),
                sess=_format_bytes((pid_metrics.get("session") or {}).get("peak_rss_bytes")),
                mck=_format_bytes((pid_metrics.get("mock") or {}).get("peak_rss_bytes")),
                rep=replay_short,
            )
        )
    lines.append("")

    fb = sweep_report.get("wave_batch_fallback")
    if fb:
        lines.append("## Wave-Batch Fallback")
        lines.append("")
        lines.append(
            f"Triggered at (N={max(cfg['matrix_ns'])}, L={max(cfg['matrix_ls'])}); "
            f"split into {fb['waves']} waves of N={fb['wave_size']} each. "
            f"`equivalent_concurrency_pass = {fb['equivalent_concurrency_pass']}` "
            f"(wave-batch is **never** equivalent to a live high-concurrency pass)."
        )
        lines.append("")
        lines.append("| Wave | Status | Load (s) | P99 (ms) |")
        lines.append("|------|--------|----------|----------|")
        for w in fb["per_wave"]:
            lat = w.get("latency_ms") or {}
            lines.append(
                "| {} | {} | {:.2f} | {} |".format(
                    w["wave_idx"],
                    w.get("status", "?"),
                    w.get("load_seconds") or 0.0,
                    f"{lat['p99_ms']:.1f}" if lat.get("p99_ms") is not None else "—",
                )
            )
        lines.append("")

    lines.append("## Failure Analysis")
    lines.append("")
    failing_cells = [c for c in sweep_report["cells"] if c.get("status") not in (None, "pass")]
    if not failing_cells:
        lines.append("All cells passed; no failure analysis required.")
        lines.append("")
    else:
        for c in failing_cells:
            lines.append(f"### {c['label']} — {c.get('status')}")
            if c.get("crash"):
                lines.append(f"- crash: {c['crash']}")
            if c.get("capacity_preflight", {}).get("over_budget"):
                lines.append(
                    f"- capacity preflight: estimated {_format_bytes(c['capacity_preflight']['estimated_bytes'])} "
                    f"> ceiling {_format_bytes(c['capacity_preflight']['ceiling_bytes'])}"
                )
            if c.get("verifier_overall_pass") is False:
                lines.append(
                    "- verifier reported invariant failure (see per-cell summary.json for invariant breakdown)"
                )
            if c.get("replay_command"):
                lines.append(f"- replay: `{c['replay_command']}`")
            lines.append("")

    lines.append("## Methodology Notes")
    lines.append("")
    lines.append(
        "- Each cell starts a fresh `StressProcessTrio`; `SessionRegistry` is empty per cell.\n"
        "- Verifier is always on and runs in streaming mode (per-session GET → verify → drop) "
        "so driver RSS is O(num_sessions) rather than O(num_sessions × trajectory_length).\n"
        "- Latency P50/P99 is driver-side `time.perf_counter()` around `await append_turn`. "
        "No py-spy / flamegraph / async profiler is installed; AC-11.1 mandates rough numbers only.\n"
        "- The plan's headline 5×3 matrix ({64,128,256,512,1024} × {5k,20k,50k}) requires "
        "≥ 150 GB host RAM (R3 alone ≈ 50 GB at 1024×50k×28×8 int32 plus FastAPI buffering); "
        "run on a dedicated host via `--matrix-config`.\n"
    )

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> SweepConfig:
    p = argparse.ArgumentParser(description="Sweep harness for the CPU-only stress test.")
    p.add_argument("--matrix-ns", default=",".join(str(n) for n in DEFAULT_MATRIX_NS))
    p.add_argument("--matrix-ls", default=",".join(str(L_) for L_ in DEFAULT_MATRIX_LS))
    p.add_argument("--plan-matrix", action="store_true", help="Use the plan's 5×3 matrix (needs ≥150GB host RAM).")
    p.add_argument("--tokens-per-turn", type=int, default=2048)
    p.add_argument("--r3-num-layers", type=int, default=28)
    p.add_argument("--r3-topk", type=int, default=8)
    p.add_argument("--no-inject-r3", dest="inject_r3", action="store_false", default=True)
    p.add_argument("--hf-checkpoint", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cpu-stress/sweep"))
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--readiness-timeout", type=float, default=120.0)
    p.add_argument("--max-cell-walltime", type=float, default=600.0)
    p.add_argument("--error-rate-threshold", type=float, default=0.05)
    p.add_argument("--ram-headroom-fraction", type=float, default=0.9)
    p.add_argument("--enable-tracemalloc", action="store_true")
    p.add_argument("--no-wave-batch-at-max", dest="wave_batch_at_max", action="store_false", default=True)
    p.add_argument("--wave-size", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.plan_matrix:
        ns = list(PLAN_MATRIX_NS)
        ls = list(PLAN_MATRIX_LS)
    else:
        ns = _parse_int_list(args.matrix_ns)
        ls = _parse_int_list(args.matrix_ls)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    return SweepConfig(
        matrix_ns=ns,
        matrix_ls=ls,
        tokens_per_turn=args.tokens_per_turn,
        r3=StressR3Spec(inject=args.inject_r3, num_layers=args.r3_num_layers, topk=args.r3_topk),
        hf_checkpoint=args.hf_checkpoint,
        output_dir=args.output_dir,
        request_timeout=args.request_timeout,
        readiness_timeout=args.readiness_timeout,
        max_cell_walltime=args.max_cell_walltime,
        error_rate_threshold=args.error_rate_threshold,
        ram_headroom_fraction=args.ram_headroom_fraction,
        enable_tracemalloc=args.enable_tracemalloc,
        wave_batch_at_max=args.wave_batch_at_max,
        wave_size=args.wave_size,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    report = run_sweep(cfg)
    all_pass = all(c.get("status") == "pass" for c in report["cells"])
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
