"""Per-metric reducers for the CI regression gate.

A merged per-run record is one NDJSON file whose lines are
``{"metric": <key>, "series": [[step, value], ...]}`` (step may be null; the
series is concatenated across processes and sorted by step, null steps last).
The gate compares a *scalar* per metric against a reference and against history,
so each target metric needs a rule that collapses its series to one number.

This module owns those rules. A reducer is pure: it takes the parsed series for
one metric (``list[(step, value)]``) and returns a float, or raises
:class:`ReducerError` when a required series is missing or carries no numeric
point. Raising (rather than returning a sentinel) lets the gate turn the failure
into a clear per-metric verdict instead of crashing on ``None`` arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# A single (step, value) point. ``step`` may be None; ``value`` is numeric.
Point = Sequence  # [step, value]


class ReducerError(ValueError):
    """A required metric series is absent or has no usable numeric point."""


def _numeric_values(series: Sequence[Point]) -> list[float]:
    """Return the numeric ``value`` of each point, in series order.

    Points are ``[step, value]``; ``step`` is ignored here. A bool is not a
    number for our purposes (it sneaks through ``isinstance(x, int)``), so it is
    dropped along with any non-numeric value.
    """
    out: list[float] = []
    for point in series:
        if len(point) < 2:
            continue
        value = point[1]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out.append(float(value))
    return out


def reduce_mean_last_5(series: Sequence[Point]) -> float:
    """Mean of the last 5 numeric points (fewer than 5 -> mean of what exists)."""
    values = _numeric_values(series)
    if not values:
        raise ReducerError("series has no numeric point")
    tail = values[-5:]
    return sum(tail) / len(tail)


def reduce_step_zero(series: Sequence[Point]) -> float:
    """Value at step 0.

    Falls back to the first numeric point only when no point carries an explicit
    step (every step is null) -- the collection backend writes step alongside
    each point, so a real step-0 sample is matched exactly; an all-null series
    still yields its single early value rather than erroring.
    """
    explicit_step_zero: float | None = None
    first_numeric: float | None = None
    saw_explicit_step = False
    for point in series:
        if len(point) < 2:
            continue
        step, value = point[0], point[1]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if first_numeric is None:
            first_numeric = value
        if step is not None:
            saw_explicit_step = True
            if step == 0 and explicit_step_zero is None:
                explicit_step_zero = value
    if explicit_step_zero is not None:
        return explicit_step_zero
    if not saw_explicit_step and first_numeric is not None:
        return first_numeric
    if first_numeric is None:
        raise ReducerError("series has no numeric point")
    raise ReducerError("series has no step-0 point")


def reduce_last(series: Sequence[Point]) -> float:
    """Value of the last numeric point."""
    values = _numeric_values(series)
    if not values:
        raise ReducerError("series has no numeric point")
    return values[-1]


# Registry of reducer functions by name. ``register_ci_gate(reducer=...)`` names
# one of these; a spec that omits ``reducer`` falls back to the per-metric
# default in METRIC_SPECS.
REDUCERS: dict[str, Callable[[Sequence[Point]], float]] = {
    "mean_last_5": reduce_mean_last_5,
    "step_zero": reduce_step_zero,
    "last": reduce_last,
}


@dataclass(frozen=True)
class MetricReducerSpec:
    """How a target metric collapses to a scalar.

    ``reducer`` names an entry in :data:`REDUCERS`. ``abs_floor`` is the metric's
    natural near-zero absolute tolerance: it seeds the gate's ``abs_floor`` for
    metrics (notably ``train/ppo_kl``) that ride at ~1e-9, where a relative
    percentage on a near-zero reference is meaningless.
    """

    reducer: str
    abs_floor: float = 0.0


# Authoritative per-metric reduction rules for the history gate's target keys.
METRIC_SPECS: dict[str, MetricReducerSpec] = {
    "train/grad_norm": MetricReducerSpec(reducer="mean_last_5"),
    "train/ppo_kl": MetricReducerSpec(reducer="step_zero", abs_floor=1e-6),
    "train/train_rollout_logprob_abs_diff": MetricReducerSpec(reducer="last"),
    "train/train_rollout_kl": MetricReducerSpec(reducer="last"),
    "rollout/raw_reward": MetricReducerSpec(reducer="last"),
}


def default_reducer_name(metric_key: str) -> str:
    """The reducer a metric uses when a gate spec names none.

    Unknown metric keys fall back to ``last`` so a gate may target a metric not
    in :data:`METRIC_SPECS` by naming a reducer explicitly; if it also omits the
    reducer, the conservative last-value rule applies.
    """
    spec = METRIC_SPECS.get(metric_key)
    return spec.reducer if spec is not None else "last"


def reduce_series(series: Sequence[Point], reducer_name: str) -> float:
    """Apply the named reducer to one metric's series."""
    func = REDUCERS.get(reducer_name)
    if func is None:
        raise ReducerError(f"unknown reducer {reducer_name!r}; known: {sorted(REDUCERS)}")
    return func(series)
