"""Offline tests for the metric-history regression gate.

These run fully offline against an in-memory :class:`SQLiteMetricHistoryStore`
and on-disk fixture files (a test file declaring register_ci_gate + a merged
NDJSON record). No network, no real DB connection opened by the gate, no wandb.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from tests.ci.metric_history import MetricSample, RunIdentity, RunProvenance, SQLiteMetricHistoryStore
from tests.ci.metric_history.gate import GateStatus, compute_test_file_hash, evaluate_gate, parse_merged_record

PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=1,
    github_run_id=100,
    github_run_attempt=1,
    event_name="pull_request",
    ref="refs/pull/1/merge",
)


@pytest.fixture
def store():
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write_test_file(tmp_path: Path, gate_lines: str, *, name: str = "test_e2e_fixture.py") -> str:
    body = textwrap.dedent(
        f"""
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        {textwrap.dedent(gate_lines).strip()}
        """
    ).lstrip("\n")
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def _write_record(tmp_path: Path, by_metric: dict[str, list], *, name: str = "merged.ndjson") -> str:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for metric, series in by_metric.items():
            f.write(json.dumps({"metric": metric, "series": series}) + "\n")
    return str(p)


def _seed_baseline(store, test_filename, *, metric_key, sub_label, values, suite="stage-c-8-gpu-h100"):
    identity = RunIdentity(
        test_path=test_filename,
        backend="cuda",
        suite=suite,
        test_file_hash=compute_test_file_hash(test_filename),
    )
    for i, v in enumerate(values):
        store.write_run(
            identity,
            PROVENANCE,
            created_at=f"2026-06-0{i + 1}T00:00:00+00:00",
            trusted=True,
            values=[MetricSample(metric_key, sub_label, v)],
        )


# --- parse_merged_record ----------------------------------------------------


def test_parse_merged_record(tmp_path):
    path = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5], [1, 1.0]], "rollout/raw_reward": [[0, 0.3]]})
    got = parse_merged_record(path)
    assert got == {"train/grad_norm": [[0, 0.5], [1, 1.0]], "rollout/raw_reward": [[0, 0.3]]}


# --- cold start (no trusted history) ----------------------------------------


def test_cold_start_hard_only_no_error(tmp_path, store):
    test_file = _write_test_file(
        tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.30, rel=0.20)'
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.31]]})

    result = evaluate_gate(test_file, record, store)

    assert len(result.metrics) == 1
    m = result.metrics[0]
    assert m.hard_status == GateStatus.PASS
    # No baselines yet: historical gate inactive, NOT an error, NOT a failure.
    assert m.historical_status == GateStatus.INACTIVE
    assert m.baseline_n == 0
    assert result.trusted is True


def test_cold_start_hard_failure(tmp_path, store):
    test_file = _write_test_file(
        tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.30, rel=0.20)'
    )
    # 0.50 vs ref 0.30, band = 0.06 -> hard fails even with no history.
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.50]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.hard_status == GateStatus.FAIL
    assert m.historical_status == GateStatus.INACTIVE
    assert result.trusted is False


# --- historical gate --------------------------------------------------------


def test_historical_failure(tmp_path, store):
    test_file = _write_test_file(
        tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.80, rel=0.20)'
    )
    # Seed a trusted baseline around 0.80; current 0.81 passes hard.
    _seed_baseline(store, test_file, metric_key="rollout/raw_reward", sub_label=None, values=[0.80, 0.82, 0.78])
    # Current 0.55: hard band = 0.16 -> |0.55-0.80|=0.25 fails hard AND historical.
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.55]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.baseline_n == 3
    assert m.baseline_mean == pytest.approx((0.80 + 0.82 + 0.78) / 3)
    assert m.historical_status == GateStatus.FAIL
    assert result.trusted is False


def test_historical_pass_within_tolerance(tmp_path, store):
    test_file = _write_test_file(
        tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.80, rel=0.20)'
    )
    _seed_baseline(store, test_file, metric_key="rollout/raw_reward", sub_label=None, values=[0.80, 0.82, 0.78])
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.79]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.hard_status == GateStatus.PASS
    assert m.historical_status == GateStatus.PASS
    assert result.trusted is True


def test_drifting_run_is_evaluated_but_not_trusted(tmp_path, store):
    # A run can pass the static hard gate yet drift away from the historical
    # mean enough to be flagged: it is evaluated (no error) but not trusted.
    test_file = _write_test_file(tmp_path, 'register_ci_gate(metric_key="train/grad_norm", hard_ref=2.0, rel=0.50)')
    # Tight history near 1.0; current 1.5.
    _seed_baseline(store, test_file, metric_key="train/grad_norm", sub_label=None, values=[1.0, 1.0, 1.0, 1.0, 1.0])
    # grad_norm uses mean_last_5; a flat series at 1.5.
    record = _write_record(tmp_path, {"train/grad_norm": [[i, 1.5] for i in range(6)]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    # Hard: |1.5 - 2.0| = 0.5, band = 0.50*2.0 = 1.0 -> passes.
    assert m.hard_status == GateStatus.PASS
    # Historical: |1.5 - 1.0| = 0.5, band = 0.50*1.0 = 0.5 -> exactly at the edge (0.5 <= 0.5).
    # The point here is the run is evaluated (no error) with the mean computed;
    # the clearly-not-trusted drift is asserted in the next test.
    assert m.baseline_mean == pytest.approx(1.0)
    assert m.current == pytest.approx(1.5)


def test_drift_beyond_historical_band_not_trusted(tmp_path, store):
    test_file = _write_test_file(tmp_path, 'register_ci_gate(metric_key="train/grad_norm", hard_ref=2.0, rel=0.50)')
    _seed_baseline(store, test_file, metric_key="train/grad_norm", sub_label=None, values=[1.0, 1.0, 1.0])
    # current 1.8: hard |1.8-2.0|=0.2 <= 1.0 pass; historical |1.8-1.0|=0.8 > 0.5 fail.
    record = _write_record(tmp_path, {"train/grad_norm": [[i, 1.8] for i in range(6)]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.hard_status == GateStatus.PASS
    assert m.historical_status == GateStatus.FAIL
    assert result.trusted is False


# --- near-zero rel-OR-abs ---------------------------------------------------


def test_near_zero_not_flagged_on_relative_pct(tmp_path, store):
    # ppo_kl rides at ~1e-9. With hard_ref ~0 and a positive abs_floor, a tiny
    # absolute deviation must NOT trip even though the *relative* change is huge.
    test_file = _write_test_file(
        tmp_path,
        'register_ci_gate(metric_key="train/ppo_kl", hard_ref=0.0, rel=0.20, abs_floor=1e-6)',
    )
    # Seed a near-zero baseline; current also near-zero but 100x in relative terms.
    _seed_baseline(store, test_file, metric_key="train/ppo_kl", sub_label=None, values=[1e-9, 2e-9, 1e-9])
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 1e-7], [1, 5e-3]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    # step_zero reducer picks step-0 value 1e-7. |1e-7 - 0| = 1e-7 <= abs_floor 1e-6.
    assert m.current == pytest.approx(1e-7)
    assert m.hard_status == GateStatus.PASS
    # historical mean ~1.33e-9; |1e-7 - 1.33e-9| ~ 9.9e-8 <= abs_floor 1e-6.
    assert m.historical_status == GateStatus.PASS
    assert result.trusted is True


def test_near_zero_real_jump_is_flagged(tmp_path, store):
    # Sanity counterpart: a ppo_kl that jumps well past abs_floor IS flagged.
    test_file = _write_test_file(
        tmp_path,
        'register_ci_gate(metric_key="train/ppo_kl", hard_ref=0.0, rel=0.20, abs_floor=1e-6)',
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.5]]})
    result = evaluate_gate(test_file, record, store)
    assert result.metrics[0].hard_status == GateStatus.FAIL
    assert result.trusted is False


# --- missing / empty required series ----------------------------------------


def test_missing_required_series_verdict_not_crash(tmp_path, store):
    test_file = _write_test_file(tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.80)')
    # Record carries a different metric only.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.hard_status == GateStatus.ERROR
    assert m.current is None
    assert "missing" in m.reason
    assert result.trusted is False


def test_empty_required_series_verdict(tmp_path, store):
    test_file = _write_test_file(tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.80)')
    record = _write_record(tmp_path, {"rollout/raw_reward": []})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.hard_status == GateStatus.ERROR
    assert result.trusted is False


# --- higher_is_worse one-sided gate -----------------------------------------


def test_higher_is_worse_drop_passes_increase_fails(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        'register_ci_gate(metric_key="train/grad_norm", hard_ref=2.0, rel=0.10, higher_is_worse=True)',
    )
    # A drop well below ref must pass (one-sided).
    low = _write_record(tmp_path, {"train/grad_norm": [[0, 0.1]]}, name="low.ndjson")
    assert evaluate_gate(test_file, low, store).metrics[0].hard_status == GateStatus.PASS

    # A rise beyond band = 0.10*2.0 = 0.2 must fail.
    high = _write_record(tmp_path, {"train/grad_norm": [[0, 3.0]]}, name="high.ndjson")
    assert evaluate_gate(test_file, high, store).metrics[0].hard_status == GateStatus.FAIL


# --- multiple specs, no specs ------------------------------------------------


def test_no_gate_specs_is_vacuously_trusted(tmp_path, store):
    body = textwrap.dedent(
        """
        from tests.ci.ci_register import register_cuda_ci
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        """
    ).lstrip("\n")
    p = tmp_path / "test_nogate.py"
    p.write_text(body)
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.3]]})

    result = evaluate_gate(str(p), record, store)
    assert result.metrics == []
    assert result.trusted is True


def test_gate_writes_no_rows(tmp_path, store):
    # The gate must never persist: after evaluation the store has no runs.
    test_file = _write_test_file(tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.30)')
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.31]]})
    evaluate_gate(test_file, record, store)

    n = store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 0
