# doc-dev: docs/ci/03-metric-history-gate.md
"""Constraint functions for the CI regression gate.

A constraint decides whether one extracted scalar ``cur`` passes against a
reference ``ref`` -- the static ``hard_ref`` for the hard gate, or the mean of
the trusted baseline for the historical gate (the *same* constraint is applied
to both). Constraints are a pluggable, name-keyed registry: ``register_ci_gate``
declares one as a literal dict ``{"name": ..., <params>}`` and the parser
validates it against :data:`CONSTRAINT_SCHEMAS` before the gate ever runs. The
constraint is never part of the baseline coordinate, so tightening or loosening
a rule does not reset history.

Both constraints today are tolerance bands with a 3-way ``direction``:

* ``rel`` -- band ``= rel * |ref|`` (a relative percentage).
* ``abs`` -- band ``= max(rel * |ref|, abs_floor)``; ``abs_floor`` keeps a metric
  riding near zero (where ``rel * |ref|`` vanishes) from flagging on a
  meaningless relative percentage. ``rel`` defaults to 0, so a bare ``abs`` is a
  pure absolute band.

``direction`` narrows what counts as a failure:

* ``two_sided``       -- any deviation beyond the band fails.
* ``higher_is_worse`` -- only an increase beyond the band fails (a drop passes).
* ``lower_is_worse``  -- only a decrease beyond the band fails (a rise passes).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

DIRECTIONS = ("two_sided", "higher_is_worse", "lower_is_worse")


class ConstraintError(ValueError):
    """A gate named an unknown constraint."""


@dataclass(frozen=True)
class ConstraintOutcome:
    """Whether ``cur`` passed, and the tolerance band that was applied."""

    ok: bool
    band: float


def _within(cur: float, ref: float, band: float, direction: str) -> bool:
    if direction == "higher_is_worse":
        return (cur - ref) <= band
    if direction == "lower_is_worse":
        return (ref - cur) <= band
    return abs(cur - ref) <= band


def _rel(cur: float, ref: float, params: dict) -> ConstraintOutcome:
    band = params["rel"] * abs(ref)
    return ConstraintOutcome(_within(cur, ref, band, params["direction"]), band)


def _abs(cur: float, ref: float, params: dict) -> ConstraintOutcome:
    band = max(params["rel"] * abs(ref), params["abs_floor"])
    return ConstraintOutcome(_within(cur, ref, band, params["direction"]), band)


# name -> constraint function.
CONSTRAINTS: dict[str, Callable[[float, float, dict], ConstraintOutcome]] = {
    "rel": _rel,
    "abs": _abs,
}

# Parse-time param schema for each constraint name, consumed by register.py.
# Each entry: param -> (validator_key, required, default). "name" is implicit.
CONSTRAINT_SCHEMAS: dict[str, dict[str, tuple[str, bool, object]]] = {
    "rel": {
        "rel": ("float_nonneg", True, None),
        "direction": ("direction", False, "two_sided"),
    },
    "abs": {
        "abs_floor": ("float_nonneg", True, None),
        "rel": ("float_nonneg", False, 0.0),
        "direction": ("direction", False, "two_sided"),
    },
}


def evaluate_constraint(constraint: dict, cur: float, ref: float) -> ConstraintOutcome:
    """Apply a normalized constraint dict to ``cur`` vs ``ref``."""
    fn = CONSTRAINTS.get(constraint["name"])
    if fn is None:
        raise ConstraintError(f"unknown constraint {constraint['name']!r}; known: {sorted(CONSTRAINTS)}")
    return fn(cur, ref, constraint)
