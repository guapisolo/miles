"""Offline unit tests for extractors, constraints, coordinate encoding, and
register_ci_gate parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, ut_parse_one_file
from tests.ci.metric_history.constraints import ConstraintError, evaluate_constraint
from tests.ci.metric_history.extractors import ExtractorError, encode_coordinate, extract
from tests.ci.metric_history.register import parse_ci_gate_specs

# --- extractors ---------------------------------------------------------------


def test_last_picks_last_numeric_point():
    got = extract([[0, 0.1], [1, 0.2], [2, 0.35]], {"name": "last"})
    assert len(got) == 1
    assert got[0].coord == "last"
    assert got[0].step == 2
    assert got[0].value == pytest.approx(0.35)


def test_extractors_skip_bool_none_nan_inf():
    # bool sneaks through isinstance(int); NaN/Inf must never reach a comparison.
    series = [[0, True], [1, None], [2, float("nan")], [3, float("inf")], [4, 2.5]]
    got = extract(series, {"name": "last"})
    assert got[0].value == pytest.approx(2.5)


def test_empty_series_errors_clearly():
    for extractor in ({"name": "last"}, {"name": "per_step"}, {"name": "steps", "steps": [0]}):
        with pytest.raises(ExtractorError):
            extract([], extractor)


def test_per_step_fans_out_one_extraction_per_step():
    got = extract([[0, 1.0], [1, 2.0], [2, 3.0]], {"name": "per_step"})
    assert [(e.coord, e.step, e.value) for e in got] == [
        ("step=0", 0, 1.0),
        ("step=1", 1, 2.0),
        ("step=2", 2, 3.0),
    ]


def test_per_step_null_step_errors():
    with pytest.raises(ExtractorError, match="no step index"):
        extract([[0, 1.0], [None, 2.0]], {"name": "per_step"})


def test_per_step_duplicate_step_errors():
    with pytest.raises(ExtractorError, match="duplicate step"):
        extract([[0, 1.0], [0, 2.0]], {"name": "per_step"})


def test_steps_picks_named_steps():
    got = extract([[0, 0.001], [1, 0.5], [2, 0.9]], {"name": "steps", "steps": [0, 2]})
    assert [(e.coord, e.value) for e in got] == [("step=0", 0.001), ("step=2", 0.9)]


def test_steps_missing_named_step_errors():
    with pytest.raises(ExtractorError, match="required step 3 missing"):
        extract([[0, 0.001], [1, 0.5]], {"name": "steps", "steps": [3]})


# --- constraints --------------------------------------------------------------


def test_rel_two_sided():
    c = {"name": "rel", "rel": 0.20, "direction": "two_sided"}
    assert evaluate_constraint(c, 1.1, 1.0).ok  # band 0.2
    assert not evaluate_constraint(c, 1.3, 1.0).ok
    assert not evaluate_constraint(c, 0.7, 1.0).ok


def test_abs_floor_covers_near_zero():
    # ref ~0 makes rel*|ref| vanish; abs_floor is the only tolerance left.
    c = {"name": "abs", "abs_floor": 1e-6, "rel": 0.20, "direction": "two_sided"}
    assert evaluate_constraint(c, 1e-7, 0.0).ok
    assert not evaluate_constraint(c, 0.5, 0.0).ok


def test_abs_band_is_max_of_rel_and_floor():
    c = {"name": "abs", "abs_floor": 0.1, "rel": 0.20, "direction": "two_sided"}
    # ref=1.0: rel band 0.2 > floor 0.1 -> band 0.2.
    assert evaluate_constraint(c, 1.19, 1.0).ok
    assert not evaluate_constraint(c, 1.21, 1.0).ok


def test_higher_is_worse_one_sided():
    c = {"name": "rel", "rel": 0.10, "direction": "higher_is_worse"}
    assert evaluate_constraint(c, 0.1, 2.0).ok  # any drop passes
    assert not evaluate_constraint(c, 2.3, 2.0).ok  # rise beyond band fails


def test_lower_is_worse_one_sided():
    c = {"name": "rel", "rel": 0.10, "direction": "lower_is_worse"}
    assert evaluate_constraint(c, 3.0, 2.0).ok  # any rise passes
    assert not evaluate_constraint(c, 1.7, 2.0).ok  # drop beyond band fails


def test_unknown_constraint_errors():
    with pytest.raises(ConstraintError, match="unknown constraint"):
        evaluate_constraint({"name": "bogus"}, 1.0, 1.0)


# --- coordinate encoding ------------------------------------------------------


def test_encode_coordinate_wire_format():
    # Pins the exact wire format: any accidental change strands baselines.
    assert encode_coordinate("last", None) == "v1|last"
    assert encode_coordinate("step=0", None) == "v1|step=0"
    assert encode_coordinate("step=3", "shard-0") == "v1|step=3|lbl=shard-0"


# --- register_ci_gate parsing -----------------------------------------------


def _make_fixture(body: str, tmp_path: Path, name: str = "test_gatefix.py") -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(p)


def test_parse_single_spec_with_defaults(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        register_ci_gate(
            metric_key="train/grad_norm",
            hard_ref=1.5,
            extractor={"name": "per_step"},
            constraint={"name": "rel", "rel": 0.20},
        )
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.metric_key == "train/grad_norm"
    assert s.hard_ref == pytest.approx(1.5)
    assert s.extractor == {"name": "per_step"}
    # direction defaults to two_sided.
    assert s.constraint == {"name": "rel", "rel": 0.20, "direction": "two_sided"}
    assert s.sub_label is None
    assert s.enforce is False
    assert s.allowlist_reason is None
    assert s.filename == path


def test_parse_all_fields(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl",
            hard_ref=0.0,
            extractor={"name": "steps", "steps": [0, 1]},
            constraint={"name": "abs", "abs_floor": 1e-6, "rel": 0.5, "direction": "higher_is_worse"},
            sub_label="shard-0",
            enforce=True,
            allowlist_reason="known noisy",
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.extractor == {"name": "steps", "steps": [0, 1]}
    assert s.constraint == {"name": "abs", "abs_floor": 1e-6, "rel": 0.5, "direction": "higher_is_worse"}
    assert s.sub_label == "shard-0"
    assert s.enforce is True
    assert s.allowlist_reason == "known noisy"


def test_abs_optional_rel_defaults_to_zero(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl", hard_ref=0.0,
            extractor={"name": "last"},
            constraint={"name": "abs", "abs_floor": 1e-6},
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.constraint == {"name": "abs", "abs_floor": 1e-6, "rel": 0.0, "direction": "two_sided"}


def test_parse_multiple_specs(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.0,
                         extractor={"name": "per_step"}, constraint={"name": "rel", "rel": 0.2})
        register_ci_gate(metric_key="rollout/raw_reward", hard_ref=0.8,
                         extractor={"name": "last"},
                         constraint={"name": "rel", "rel": 0.2, "direction": "lower_is_worse"})
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert [s.metric_key for s in specs] == ["train/grad_norm", "rollout/raw_reward"]


def test_negative_hard_ref_parses(tmp_path):
    # -1.5 is an ast.UnaryOp, not a bare Constant; the parser must accept it.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=-1.5,
                         extractor={"name": "last"}, constraint={"name": "rel", "rel": 0.2})
        """,
        tmp_path,
    )
    assert parse_ci_gate_specs(path)[0].hard_ref == pytest.approx(-1.5)


def test_unknown_kwarg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last"}, constraint={"name": "rel", "rel": 0.2}, bogus=3)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown argument 'bogus'"):
        parse_ci_gate_specs(path)


def test_non_literal_arg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        X = 1.0
        register_ci_gate(metric_key="train/x", hard_ref=X,
                         extractor={"name": "last"}, constraint={"name": "rel", "rel": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(path)


def test_non_literal_inside_dict_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        NAME = "last"
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": NAME}, constraint={"name": "rel", "rel": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize("missing", ["metric_key", "hard_ref", "extractor", "constraint"])
def test_missing_required_field_rejected(tmp_path, missing):
    fields = {
        "metric_key": 'metric_key="train/x"',
        "hard_ref": "hard_ref=1.0",
        "extractor": 'extractor={"name": "last"}',
        "constraint": 'constraint={"name": "rel", "rel": 0.2}',
    }
    del fields[missing]
    call = f"register_ci_gate({', '.join(fields.values())})"
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        {call}
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match=f"{missing} is required"):
        parse_ci_gate_specs(path)


def test_positional_arg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate("train/x", 1.0)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="only keyword arguments"):
        parse_ci_gate_specs(path)


def test_unknown_extractor_name_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "mean_last_9000"}, constraint={"name": "rel", "rel": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown extractor name 'mean_last_9000'"):
        parse_ci_gate_specs(path)


def test_unknown_constraint_name_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last"}, constraint={"name": "bogus", "rel": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown constraint name 'bogus'"):
        parse_ci_gate_specs(path)


def test_unknown_key_for_extractor_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last", "steps": [0]}, constraint={"name": "rel", "rel": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown key 'steps' for extractor 'last'"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize(
    "steps_literal",
    ["[]", "[1.5]", "[True]", "[-1]", "[0, 0]"],
    ids=["empty", "float", "bool", "negative", "duplicate"],
)
def test_bad_steps_list_rejected(tmp_path, steps_literal):
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={{"name": "steps", "steps": {steps_literal}}},
                         constraint={{"name": "rel", "rel": 0.2}})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="param 'steps'"):
        parse_ci_gate_specs(path)


def test_bad_direction_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last"},
                         constraint={"name": "rel", "rel": 0.2, "direction": "up_only"})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="param 'direction'"):
        parse_ci_gate_specs(path)


def test_negative_rel_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last"}, constraint={"name": "rel", "rel": -0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="param 'rel'"):
        parse_ci_gate_specs(path)


def test_duplicate_dict_key_rejected(tmp_path):
    # A plain dict would silently keep the last value; the parser must reject.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last"},
                         constraint={"name": "rel", "rel": 0.2, "rel": 0.3})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="duplicate key 'rel'"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize("label", ["shard|0", "a=b"])
def test_sub_label_reserved_char_rejected(tmp_path, label):
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={{"name": "last"}}, constraint={{"name": "rel", "rel": 0.2}},
                         sub_label="{label}")
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="reserved character"):
        parse_ci_gate_specs(path)


def test_non_bool_enforce_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x", hard_ref=1.0,
                         extractor={"name": "last"}, constraint={"name": "rel", "rel": 0.2}, enforce=1)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="enforce must be a boolean"):
        parse_ci_gate_specs(path)


def test_register_ci_gate_does_not_disturb_suite_parsing(tmp_path):
    # The suite RegistryVisitor must still find exactly the register_cuda_ci
    # call and ignore the register_ci_gate calls beside it.
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"])
        register_ci_gate(metric_key="train/grad_norm", hard_ref=1.5,
                         extractor={"name": "per_step"}, constraint={"name": "rel", "rel": 0.2})
        """,
        tmp_path,
    )
    registries = ut_parse_one_file(path)
    assert len(registries) == 1
    assert isinstance(registries[0], CIRegistry)
    assert registries[0].backend == HWBackend.CUDA
    assert registries[0].suite == "stage-c-8-gpu-h100"


def test_register_ci_gate_runtime_is_noop():
    from tests.ci.metric_history import register_ci_gate

    assert (
        register_ci_gate(
            metric_key="train/grad_norm",
            hard_ref=1.0,
            extractor={"name": "per_step"},
            constraint={"name": "rel", "rel": 0.2},
        )
        is None
    )
