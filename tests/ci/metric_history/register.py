# doc-dev: docs/ci/03-metric-history-gate.md
"""Declare and parse metric-history regression gates.

* ``register_ci_gate(...)`` is the marker a test file uses to declare a gate.
* Like ``register_cuda_ci`` it is a runtime no-op, parsed out of the file's AST
  rather than executed.
* A gate composes an ``extractor`` (which value(s) to pull from a metric's
  series) and a ``constraint`` (the pass/fail rule).
* Extractor and constraint are each a literal dict ``{"name": ..., <params>}``,
  validated against the per-name schemas in :mod:`extractors` /
  :mod:`constraints`.
* ``parse_ci_gate_specs`` extracts every declaration as a :class:`CiGateSpec`.

Caveats:

* ``register_ci_gate``'s Python signature does NOT validate calls -- the parser
  here does, at parse time.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from tests.ci.metric_history.constraints import CONSTRAINT_SCHEMAS, DIRECTIONS
from tests.ci.metric_history.extractors import COORD_RESERVED, EXTRACTOR_SCHEMAS


def register_ci_gate(
    *,
    metric_key: str,
    hard_ref: float,
    extractor: dict,
    constraint: dict,
    sub_label: str | None = None,
    enforce: bool = False,
    allowlist_reason: str | None = None,
):
    """Declare one history-gate spec for the test file it sits in.

    Parsed via AST (like ``register_cuda_ci``); a runtime no-op. Every argument
    is keyword-only and must be a literal. ``metric_key`` names the target metric
    and ``hard_ref`` the always-on absolute reference. ``extractor`` and
    ``constraint`` are literal dicts ``{"name": ..., <params>}`` -- see
    :data:`extractors.EXTRACTOR_SCHEMAS` / :data:`constraints.CONSTRAINT_SCHEMAS`
    for the valid names and params. ``sub_label`` labels a measurement (e.g. a
    shard); ``enforce`` and ``allowlist_reason`` are policy metadata the gate
    carries without acting on (the verdict is informational this round).
    """
    return None


_REGISTER_NAME = "register_ci_gate"
_REQUIRED = object()

# Top-level register_ci_gate fields: name -> (required, default).
_FIELDS: dict[str, tuple[bool, object]] = {
    "metric_key": (True, _REQUIRED),
    "hard_ref": (True, _REQUIRED),
    "extractor": (True, _REQUIRED),
    "constraint": (True, _REQUIRED),
    "sub_label": (False, None),
    "enforce": (False, False),
    "allowlist_reason": (False, None),
}


@dataclass(frozen=True)
class CiGateSpec:
    """One parsed ``register_ci_gate`` declaration.

    ``extractor`` / ``constraint`` are normalized dicts (name + validated params
    + filled defaults). ``filename`` is the test file the spec governs; the gate
    derives identity from its CIRegistry and value identity from the extractor
    coordinate + ``sub_label``.
    """

    filename: str
    metric_key: str
    hard_ref: float
    extractor: dict
    constraint: dict
    sub_label: str | None = None
    enforce: bool = False
    allowlist_reason: str | None = None


class _ParseError(Exception):
    """Internal: a bare message the caller wraps with file + field context."""


def _literal(node: ast.AST) -> object:
    """A Python literal from an AST node: a constant, a negative number, or a
    list / dict of the same. Rejects any non-literal (name, call, expression).

    Negative numbers matter because ``-1.0`` is an ``ast.UnaryOp``, not an
    ``ast.Constant``; a plain constant check would wrongly reject them. Dict keys
    must be string literals and duplicates are rejected (a plain dict would
    silently keep the last).
    """
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        out: dict = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                raise _ParseError("dict unpacking (**) is not allowed")
            key = _literal(key_node)
            if not isinstance(key, str):
                raise _ParseError("dict keys must be string literals")
            if key in out:
                raise _ParseError(f"duplicate key {key!r}")
            out[key] = _literal(value_node)
        return out
    raise _ParseError(f"must be a literal (got {type(node).__name__})")


def _validate_param(validator: str, value: object) -> object:
    """Validate one extractor/constraint param; return it (normalized)."""
    if validator == "float_nonneg":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _ParseError("must be a number")
        if not math.isfinite(value) or value < 0:
            raise _ParseError("must be a finite number >= 0")
        return float(value)
    if validator == "direction":
        if value not in DIRECTIONS:
            raise _ParseError(f"must be one of {list(DIRECTIONS)}")
        return value
    if validator == "step_list":
        if not isinstance(value, list) or not value:
            raise _ParseError("must be a non-empty list of step indices")
        seen: set[int] = set()
        for s in value:
            if isinstance(s, bool) or not isinstance(s, int):
                raise _ParseError("step indices must be integers")
            if s < 0:
                raise _ParseError("step indices must be >= 0")
            if s in seen:
                raise _ParseError(f"duplicate step {s}")
            seen.add(s)
        return list(value)
    raise _ParseError(f"internal: unknown validator {validator!r}")


def _normalize_axis(axis: str, raw: object, schemas: dict) -> dict:
    """Validate an extractor/constraint dict against its per-name schema and
    return a normalized dict (name + validated params + filled defaults)."""
    if not isinstance(raw, dict):
        raise _ParseError(f"{axis} must be a dict")
    if "name" not in raw:
        raise _ParseError(f"{axis} dict must have a 'name'")
    name = raw["name"]
    if not isinstance(name, str):
        raise _ParseError(f"{axis} 'name' must be a string")
    schema = schemas.get(name)
    if schema is None:
        raise _ParseError(f"unknown {axis} name {name!r}; known: {sorted(schemas)}")
    for key in raw:
        if key != "name" and key not in schema:
            raise _ParseError(f"unknown key {key!r} for {axis} {name!r}; valid: {sorted(schema)}")
    normalized: dict = {"name": name}
    for param, (validator, required, default) in schema.items():
        if param in raw:
            try:
                normalized[param] = _validate_param(validator, raw[param])
            except _ParseError as e:
                raise _ParseError(f"{axis} {name!r} param {param!r}: {e}") from None
        elif required:
            raise _ParseError(f"{axis} {name!r} requires {param!r}")
        else:
            normalized[param] = default
    return normalized


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _ParseError(f"{field} must be a string")
    return value


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ParseError(f"{field} must be a number")
    if not math.isfinite(value):
        raise _ParseError(f"{field} must be finite")
    return float(value)


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise _ParseError(f"{field} must be a boolean")
    return value


def _require_opt_str(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise _ParseError(f"{field} must be a string or None")
    return value


def _require_sub_label(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _ParseError("sub_label must be a string or None")
    for ch in COORD_RESERVED:
        if ch in value:
            raise _ParseError(f"sub_label must not contain reserved character {ch!r}")
    return value


def _parse_ci_gate_call(call: ast.Call, filename: str) -> CiGateSpec:
    prefix = f"{filename}: {_REGISTER_NAME}()"
    if call.args:
        raise ValueError(f"{prefix} takes only keyword arguments (got {len(call.args)} positional)")

    raw: dict[str, object] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError(f"{prefix}: **kwargs are not supported")
        if kw.arg not in _FIELDS:
            raise ValueError(f"{prefix}: unknown argument {kw.arg!r}; valid: {sorted(_FIELDS)}")
        if kw.arg in raw:
            raise ValueError(f"{prefix}: duplicated argument {kw.arg!r}")
        try:
            raw[kw.arg] = _literal(kw.value)
        except _ParseError as e:
            raise ValueError(f"{prefix}: {kw.arg} {e}") from None

    for field, (required, default) in _FIELDS.items():
        if field not in raw:
            if required:
                raise ValueError(f"{prefix}: {field} is required")
            raw[field] = default

    try:
        return CiGateSpec(
            filename=filename,
            metric_key=_require_str(raw["metric_key"], "metric_key"),
            hard_ref=_require_number(raw["hard_ref"], "hard_ref"),
            extractor=_normalize_axis("extractor", raw["extractor"], EXTRACTOR_SCHEMAS),
            constraint=_normalize_axis("constraint", raw["constraint"], CONSTRAINT_SCHEMAS),
            sub_label=_require_sub_label(raw["sub_label"]),
            enforce=_require_bool(raw["enforce"], "enforce"),
            allowlist_reason=_require_opt_str(raw["allowlist_reason"], "allowlist_reason"),
        )
    except _ParseError as e:
        raise ValueError(f"{prefix}: {e}") from None


def parse_ci_gate_specs(filename: str) -> list[CiGateSpec]:
    """Return every ``register_ci_gate`` spec declared at top level in ``filename``.

    Parsed the same way as ``register_cuda_ci``: top-level ``Expr(Call)`` whose
    callee is the bare name ``register_ci_gate``. Non-literal / invalid args raise
    ValueError naming the file and field.

    Note for the future writer: two specs may map to the same baseline coordinate
    (same extractor + sub_label, different constraint) -- that is intentional
    (shared baseline, different rule). The writer must dedupe metric_values by
    coordinate so one run contributes one row per coordinate.
    """
    with open(filename) as f:
        tree = ast.parse(f.read(), filename=filename)
    specs: list[CiGateSpec] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Name) or call.func.id != _REGISTER_NAME:
            continue
        specs.append(_parse_ci_gate_call(call, filename))
    return specs
