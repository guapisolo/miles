"""Offline unit tests for the per-metric reducers and register_ci_gate parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, parse_ci_gate_specs, ut_parse_one_file
from tests.ci.metric_reducers import (
    METRIC_SPECS,
    ReducerError,
    default_reducer_name,
    reduce_last,
    reduce_mean_last_5,
    reduce_series,
    reduce_step_zero,
)


# --- Reducers ---------------------------------------------------------------


def test_grad_norm_mean_of_last_5():
    series = [[0, 1.0], [1, 2.0], [2, 3.0], [3, 4.0], [4, 5.0], [5, 6.0], [6, 7.0]]
    # Last 5 are 3..7 -> mean 5.0.
    assert reduce_mean_last_5(series) == pytest.approx(5.0)


def test_grad_norm_mean_with_fewer_than_5():
    assert reduce_mean_last_5([[0, 2.0], [1, 4.0]]) == pytest.approx(3.0)


def test_ppo_kl_step_zero():
    series = [[0, 0.001], [1, 0.5], [2, 0.9]]
    assert reduce_step_zero(series) == pytest.approx(0.001)


def test_step_zero_all_null_steps_falls_back_to_first():
    series = [[None, 0.7], [None, 0.9]]
    assert reduce_step_zero(series) == pytest.approx(0.7)


def test_step_zero_missing_step_zero_with_explicit_steps_errors():
    # Explicit steps present but none is 0 -> a step-0 reducer cannot answer.
    with pytest.raises(ReducerError):
        reduce_step_zero([[1, 0.5], [2, 0.9]])


def test_last_value():
    assert reduce_last([[0, 0.1], [1, 0.2], [2, 0.35]]) == pytest.approx(0.35)


def test_reducers_skip_non_numeric_and_bool():
    # bool sneaks through isinstance(int); it and a None value must be dropped.
    series = [[0, True], [1, None], [2, 1.5], [3, 2.5]]
    assert reduce_last(series) == pytest.approx(2.5)
    assert reduce_mean_last_5(series) == pytest.approx(2.0)


def test_empty_series_errors_clearly():
    for fn in (reduce_mean_last_5, reduce_step_zero, reduce_last):
        with pytest.raises(ReducerError):
            fn([])


def test_default_reducer_per_metric():
    assert default_reducer_name("train/grad_norm") == "mean_last_5"
    assert default_reducer_name("train/ppo_kl") == "step_zero"
    assert default_reducer_name("train/train_rollout_logprob_abs_diff") == "last"
    assert default_reducer_name("train/train_rollout_kl") == "last"
    assert default_reducer_name("rollout/raw_reward") == "last"
    # Unknown metric falls back to the conservative last-value rule.
    assert default_reducer_name("train/unknown") == "last"


def test_ppo_kl_carries_abs_floor():
    assert METRIC_SPECS["train/ppo_kl"].abs_floor > 0.0
    assert METRIC_SPECS["train/grad_norm"].abs_floor == 0.0


def test_reduce_series_unknown_reducer_errors():
    with pytest.raises(ReducerError):
        reduce_series([[0, 1.0]], "no_such_reducer")


# --- register_ci_gate parsing -----------------------------------------------


def _make_fixture(body: str, tmp_path: Path, name: str = "test_gatefix.py") -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(p)


def test_parse_single_spec_with_defaults(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci, register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.5)
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.metric_key == "train/grad_norm"
    assert s.hard_ref == pytest.approx(1.5)
    assert s.rel == pytest.approx(0.20)
    assert s.abs_floor == pytest.approx(0.0)
    assert s.reducer is None
    assert s.sub_label is None
    assert s.higher_is_worse is False
    assert s.enforce is False
    assert s.allowlist_reason is None
    assert s.filename == path


def test_parse_all_fields(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl",
            hard_ref=0.0,
            rel=0.5,
            abs_floor=1e-6,
            reducer="step_zero",
            sub_label="shard-0",
            higher_is_worse=True,
            enforce=True,
            allowlist_reason="known noisy",
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.metric_key == "train/ppo_kl"
    assert s.hard_ref == pytest.approx(0.0)
    assert s.rel == pytest.approx(0.5)
    assert s.abs_floor == pytest.approx(1e-6)
    assert s.reducer == "step_zero"
    assert s.sub_label == "shard-0"
    assert s.higher_is_worse is True
    assert s.enforce is True
    assert s.allowlist_reason == "known noisy"


def test_parse_multiple_specs(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.0)
        register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.8, higher_is_worse=False)
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert [s.metric_key for s in specs] == ["train/grad_norm", "rollout/raw_reward"]


def test_unknown_kwarg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.0, bogus=3)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown argument 'bogus'"):
        parse_ci_gate_specs(path)


def test_non_literal_arg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        X = 1.0
        register_ci_gate(metric_key="train/grad_norm", hard_ref=X)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal constant"):
        parse_ci_gate_specs(path)


def test_missing_required_metric_key_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate(hard_ref=1.0)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="metric_key is required"):
        parse_ci_gate_specs(path)


def test_missing_required_hard_ref_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm")
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="hard_ref is required"):
        parse_ci_gate_specs(path)


def test_positional_arg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate("train/grad_norm", 1.0)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="only keyword arguments"):
        parse_ci_gate_specs(path)


def test_non_bool_higher_is_worse_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.0, higher_is_worse=1)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="higher_is_worse.*must be a boolean"):
        parse_ci_gate_specs(path)


def test_register_ci_gate_does_not_disturb_suite_parsing(tmp_path):
    # The suite RegistryVisitor must still find exactly the register_cuda_ci
    # call and ignore the register_ci_gate calls beside it.
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci, register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"])
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.5)
        register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.8)
        """,
        tmp_path,
    )
    registries = ut_parse_one_file(path)
    assert len(registries) == 1
    assert isinstance(registries[0], CIRegistry)
    assert registries[0].backend == HWBackend.CUDA
    assert registries[0].suite == "stage-c-8-gpu-h100"


def test_register_ci_gate_runtime_is_noop():
    from tests.ci.ci_register import register_ci_gate

    assert register_ci_gate(metric_key="train/grad_norm", hard_ref=1.0) is None
