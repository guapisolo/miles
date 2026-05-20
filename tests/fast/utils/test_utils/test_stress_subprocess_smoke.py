"""Integration tests for the three-process stress harness.

Covers Round 2 AC-1 (standalone runnable), AC-2 (three-process deployment),
AC-5 / AC-6 happy path through real session-server multi-turn, AC-9
(verifier always on; nonzero exit on failure), and launcher cleanup
guarantees (no orphan subprocesses).

These tests spawn real subprocesses through ``StressProcessTrio``; total
runtime is dominated by tokenizer load in each subprocess (Qwen3-0.6B is
~1GB on disk, ~3s import + load). Registered as stage-a-fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Force CPU-only before any miles imports.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import psutil  # noqa: E402
from tests.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=120, suite="stage-a-fast")


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER_SCRIPT = REPO_ROOT / "scripts" / "tools" / "stress_session_server_cpu.py"


def _pid_alive(pid: int) -> bool:
    """psutil-based liveness check that tolerates the zombie-then-reaped race."""
    if not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        return proc.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def test_subprocess_smoke_exits_zero(tmp_path: Path):
    """End-to-end: invoke the CLI as a subprocess (the way a real user
    would), confirm exit 0 and a non-empty summary.json with verifier pass.
    This is the AC-1 "standalone runnable" gate."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER_SCRIPT),
            "--smoke",
            "--output-dir",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"smoke exit={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["exit_code"] == 0
    assert summary["verifier"]["overall_pass"] is True
    assert all(r["overall_pass"] for r in summary["verifier"]["reports"])


def test_in_process_smoke_has_three_distinct_pids_and_cleans_up(tmp_path: Path):
    """In-process invocation of main(): inspect summary.json for AC-2's
    three distinct PIDs. Then re-import StressProcessTrio with a tiny
    config to spawn-and-stop and confirm both subprocess PIDs are dead.
    """
    from miles.utils.test_utils.stress_cli import main

    exit_code = main(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    pids = {summary["driver_pid"], summary["session_pid"], summary["mock_pid"]}
    assert len(pids) == 3, f"expected three distinct PIDs, got {pids}"
    assert summary["driver_pid"] == os.getpid()

    # Cleanup probe: with-statement on a fresh trio, then assert no orphan.
    from miles.utils.test_utils.stress_launchers import StressProcessTrio, StressR3Spec

    with StressProcessTrio(
        hf_checkpoint="Qwen/Qwen3-0.6B",
        output_tokens=8,
        r3=StressR3Spec(inject=False),
    ) as trio:
        mock_pid = trio.mock_proc.pid
        session_pid = trio.session_proc.pid
        assert _pid_alive(mock_pid)
        assert _pid_alive(session_pid)

    # Allow OS a tick to reap zombies after __exit__.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_alive(mock_pid) and not _pid_alive(session_pid):
            break
        time.sleep(0.1)
    assert not _pid_alive(mock_pid), f"mock subprocess still alive after cleanup, pid={mock_pid}"
    assert not _pid_alive(session_pid), f"session subprocess still alive after cleanup, pid={session_pid}"


def test_multi_turn_real_chain_passes_all_invariants(tmp_path: Path):
    """N=2 sessions x 3 turns x 64 output tokens through the real
    session-server, with R3 injected. This is the AC-5 / AC-6 real-chain
    coverage Codex Round 1 review asked for.

    Multi-turn uses the [user → assistant → tool → assistant → tool → ...]
    pattern; the session server's TITO append-only validator (allowed_append_roles=tool)
    accepts each new tool message, and the verifier extracts the signature
    from messages[-1] for each record.
    """
    from miles.utils.test_utils.stress_cli import main

    exit_code = main(
        [
            "--num-sessions",
            "2",
            "--num-turns",
            "3",
            "--output-tokens",
            "64",
            "--inject-r3",
            "--r3-num-layers",
            "4",
            "--r3-topk",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verifier"]["overall_pass"] is True
    assert len(summary["verifier"]["reports"]) == 2
    for report in summary["verifier"]["reports"]:
        assert report["overall_pass"] is True
        assert report["num_turns"] == 3
        for inv_name, inv in report["by_invariant"].items():
            assert inv["passed"], f"{inv_name} unexpectedly failed: {inv['mismatches']}"


def test_verifier_failure_forces_nonzero_exit(tmp_path: Path, monkeypatch):
    """Monkey-patch ``verify_session`` so it always returns a failing report,
    then run the smoke flow. The driver MUST exit nonzero — verifier is
    always on; there is no opt-out path."""
    from miles.utils.test_utils import stress_cli
    from miles.utils.test_utils.stress_verifier import InvariantResult, VerifyReport

    def fake_verify(*_args, **_kwargs):
        return VerifyReport(
            overall_pass=False,
            by_invariant={
                "FORCED_TEST_FAIL": InvariantResult(
                    name="FORCED_TEST_FAIL",
                    passed=False,
                    mismatches=[{"injected_by": "test_verifier_failure_forces_nonzero_exit"}],
                ),
            },
        )

    monkeypatch.setattr(stress_cli, "verify_session", fake_verify)

    exit_code = stress_cli.main(
        [
            "--num-sessions",
            "1",
            "--num-turns",
            "1",
            "--output-tokens",
            "8",
            "--no-inject-r3",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 2, f"expected exit_code=2 for verifier failure, got {exit_code}"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["exit_code"] == 2
    assert summary["verifier"]["overall_pass"] is False


def test_cli_rejects_no_verify_flag(tmp_path: Path):
    """Grep-level check that --no-verify (or similar opt-out) is not exposed
    on the CLI. AC-9 says verifier is always on."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, str(DRIVER_SCRIPT), "--no-verify"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert (
        "unrecognized" in combined or "no such" in combined or "error" in combined
    ), f"expected argparse to reject --no-verify, got:\nstdout: {result.stdout}\nstderr: {result.stderr}"
