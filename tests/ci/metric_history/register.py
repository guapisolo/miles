# doc-dev: docs/ci/03-metric-history-gate.md
"""Declare and parse metric-history regression gates.

``register_ci_gate(...)`` is the marker a test file uses to declare a gate. Like
``register_cuda_ci`` it is a runtime no-op, parsed out of the file's AST rather
than executed. ``parse_ci_gate_specs`` extracts every such declaration as a
:class:`CiGateSpec`, which the gate evaluator consumes.
"""

import ast
from dataclasses import dataclass


def register_ci_gate(
    *,
    metric_key: str,
    hard_ref: float,
    rel: float = 0.20,
    abs_floor: float = 0.0,
    reducer: str | None = None,
    sub_label: str | None = None,
    higher_is_worse: bool = False,
    enforce: bool = False,
    allowlist_reason: str | None = None,
):
    """Declare one history-gate spec for the test file it sits in.

    Parsed via AST (like ``register_cuda_ci``); a runtime no-op. Every argument
    is keyword-only and must be a literal constant. ``metric_key`` names the
    target metric, ``hard_ref`` the always-on absolute reference, ``rel`` the
    relative tolerance and ``abs_floor`` the near-zero absolute tolerance applied
    as ``max(rel*|ref|, abs_floor)``. ``reducer`` overrides the metric's default
    series reducer; ``sub_label`` selects a labeled measurement. When
    ``higher_is_worse`` the hard gate is one-sided (only an increase fails).
    ``enforce`` and ``allowlist_reason`` are policy metadata the gate carries
    through without acting on (the run verdict is informational this round).
    """
    return None


# Field schema for register_ci_gate(): name -> (accepted python types, default).
# `_GATE_REQUIRED` carries no default and must appear in the call.
_GATE_REQUIRED = object()
_CI_GATE_FIELDS: dict[str, tuple[tuple[type, ...], object]] = {
    "metric_key": ((str,), _GATE_REQUIRED),
    "hard_ref": ((int, float), _GATE_REQUIRED),
    "rel": ((int, float), 0.20),
    "abs_floor": ((int, float), 0.0),
    "reducer": ((str, type(None)), None),
    "sub_label": ((str, type(None)), None),
    "higher_is_worse": ((bool,), False),
    "enforce": ((bool,), False),
    "allowlist_reason": ((str, type(None)), None),
}

_CI_GATE_REGISTER_NAME = "register_ci_gate"


@dataclass(frozen=True)
class CiGateSpec:
    """One parsed ``register_ci_gate`` declaration.

    ``filename`` is the test file the spec governs; the gate derives identity
    (test_path/backend/suite) from the file's CIRegistry and value identity from
    (metric_key, sub_label).
    """

    filename: str
    metric_key: str
    hard_ref: float
    rel: float = 0.20
    abs_floor: float = 0.0
    reducer: str | None = None
    sub_label: str | None = None
    higher_is_worse: bool = False
    enforce: bool = False
    allowlist_reason: str | None = None


def _parse_ci_gate_call(func_call: ast.Call, filename: str) -> CiGateSpec:
    if func_call.args:
        raise ValueError(
            f"{filename}: {_CI_GATE_REGISTER_NAME}() takes only keyword arguments "
            f"(got {len(func_call.args)} positional)"
        )

    parsed: dict[str, object] = {}
    for kw in func_call.keywords:
        if kw.arg is None:
            raise ValueError(f"{filename}: **kwargs are not supported in {_CI_GATE_REGISTER_NAME}()")
        if kw.arg not in _CI_GATE_FIELDS:
            valid = ", ".join(sorted(_CI_GATE_FIELDS))
            raise ValueError(
                f"{filename}: unknown argument '{kw.arg}' in {_CI_GATE_REGISTER_NAME}(); valid: [{valid}]"
            )
        if kw.arg in parsed:
            raise ValueError(f"{filename}: duplicated argument '{kw.arg}' in {_CI_GATE_REGISTER_NAME}()")
        # Only a bare literal is accepted; anything else (name, expression) is rejected.
        if not isinstance(kw.value, ast.Constant):
            raise ValueError(f"{filename}: {kw.arg} in {_CI_GATE_REGISTER_NAME}() must be a literal constant")
        parsed[kw.arg] = kw.value.value

    resolved: dict[str, object] = {}
    for name, (types, default) in _CI_GATE_FIELDS.items():
        if name in parsed:
            value = parsed[name]
            # bool is an int subclass; a bool slipping into a numeric field (or a
            # number into a bool field) is a mistake, so check bool-ness exactly.
            if bool in types:
                if not isinstance(value, bool):
                    raise ValueError(f"{filename}: {name} in {_CI_GATE_REGISTER_NAME}() must be a boolean")
            elif isinstance(value, bool) or not isinstance(value, types):
                allowed = " or ".join(t.__name__ for t in types)
                raise ValueError(f"{filename}: {name} in {_CI_GATE_REGISTER_NAME}() must be {allowed}")
            resolved[name] = value
        elif default is _GATE_REQUIRED:
            raise ValueError(f"{filename}: {name} is required in {_CI_GATE_REGISTER_NAME}()")
        else:
            resolved[name] = default

    return CiGateSpec(
        filename=filename,
        metric_key=resolved["metric_key"],
        hard_ref=float(resolved["hard_ref"]),
        rel=float(resolved["rel"]),
        abs_floor=float(resolved["abs_floor"]),
        reducer=resolved["reducer"],
        sub_label=resolved["sub_label"],
        higher_is_worse=resolved["higher_is_worse"],
        enforce=resolved["enforce"],
        allowlist_reason=resolved["allowlist_reason"],
    )


def parse_ci_gate_specs(filename: str) -> list[CiGateSpec]:
    """Return every ``register_ci_gate`` spec declared at top level in ``filename``.

    Parsed the same way as ``register_cuda_ci``: top-level ``Expr(Call)`` whose
    callee is the bare name ``register_ci_gate``. Non-literal or unknown kwargs
    raise ValueError with the file and field named.
    """
    with open(filename) as f:
        tree = ast.parse(f.read(), filename=filename)
    specs: list[CiGateSpec] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Name) or call.func.id != _CI_GATE_REGISTER_NAME:
            continue
        specs.append(_parse_ci_gate_call(call, filename))
    return specs
