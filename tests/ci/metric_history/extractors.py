# doc-dev: docs/ci/03-metric-history-gate.md
"""Extractors + coordinate encoding for the CI regression gate.

* An extractor pulls the comparison value(s) out of one metric's per-run
  series `[[step, value], ...]`; step may be None.
* `register_ci_gate` declares one as a literal dict `{"name": ..., <params>}`;
  the parser validates it against :data:`EXTRACTOR_SCHEMAS`.
* Extractors are pure and return a list of :class:`Extraction` -- one entry per
  comparison coordinate.
* `last` -- the last numeric point (1 coordinate).
* `per_step` -- every step present in the series, fanned out (N coordinates).
* `steps` -- the named steps, fanned out (`len(steps)` coordinates); a named
  step missing from the series is an error, not a silent skip.
* A fanned coordinate is identified by its step, so this run's step-0 value is
  compared only against past runs' step-0 values.
* :func:`encode_coordinate` turns an extraction's coordinate token + the
  author's `sub_label` into the single `sub_label` string the store keys on;
  the constraint is never part of it.
* Raising :class:`ExtractorError` (rather than returning a sentinel) lets the
  gate turn the failure into a clear per-coordinate verdict.

Caveats:

* A non-finite value (NaN/±Inf) at a coordinate the extractor selects is an
  ExtractorError -- judged, never silently dropped. Points whose value is not a
  number at all (bool/None/...) are ignored, as are points no extractor selects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Point = Sequence  # [step, value]


class ExtractorError(ValueError):
    """A required series is absent, empty, or ill-formed for this extractor."""


@dataclass(frozen=True)
class Extraction:
    """One comparison value pulled from a series.

    `coord` is the extractor-identity token within the metric (`"last"` or
    `"step=<k>"`); `step` is the step index the value came from (None for a
    positional extractor) and is carried for reporting.
    """

    coord: str
    step: int | None
    value: float


def _is_number(value: object) -> bool:
    """A real int/float (not bool — it sneaks through `isinstance(x, int)`).

    Finiteness is deliberately not checked here: non-finite values stay in the
    series and error at selection time, never silently dropped.
    """
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _valid_step(step: object) -> bool:
    return isinstance(step, int) and not isinstance(step, bool)


def _numeric_points(series: Sequence[Point]) -> list[tuple[int | None, float]]:
    """(step, value) for each point whose value is a number.

    A non-int (or bool) step is normalized to None; the value is kept —
    including non-finite floats, which the extractors reject if selected.
    Points with a non-numeric value are dropped.
    """
    out: list[tuple[int | None, float]] = []
    for point in series:
        if len(point) < 2:
            continue
        step, value = point[0], point[1]
        if not _is_number(value):
            continue
        out.append((step if _valid_step(step) else None, float(value)))
    return out


def _extract_last(series: Sequence[Point]) -> list[Extraction]:
    points = _numeric_points(series)
    if not points:
        raise ExtractorError("series has no numeric point")
    step, value = points[-1]
    if not math.isfinite(value):
        raise ExtractorError(f"last: non-finite value {value!r} at the last point (step {step})")
    return [Extraction(coord="last", step=step, value=value)]


def _extract_per_step(series: Sequence[Point]) -> list[Extraction]:
    points = _numeric_points(series)
    if not points:
        raise ExtractorError("series has no numeric point")
    out: list[Extraction] = []
    seen: set[int] = set()
    for step, value in points:
        if step is None:
            raise ExtractorError("per_step: a numeric point carries no step index")
        if step in seen:
            raise ExtractorError(f"per_step: duplicate step {step} in series")
        if not math.isfinite(value):
            raise ExtractorError(f"per_step: non-finite value {value!r} at step {step}")
        seen.add(step)
        out.append(Extraction(coord=f"step={step}", step=step, value=value))
    return out


def _extract_steps(series: Sequence[Point], steps: Sequence[int]) -> list[Extraction]:
    by_step: dict[int, float] = {}
    for step, value in _numeric_points(series):
        if step is None:
            continue
        if step in by_step:
            raise ExtractorError(f"steps: duplicate step {step} in series")
        by_step[step] = value
    out: list[Extraction] = []
    for k in steps:
        if k not in by_step:
            raise ExtractorError(f"steps: required step {k} missing from series")
        if not math.isfinite(by_step[k]):
            raise ExtractorError(f"steps: non-finite value {by_step[k]!r} at required step {k}")
        out.append(Extraction(coord=f"step={k}", step=k, value=by_step[k]))
    return out


# Parse-time param schema for each extractor name, consumed by register.py.
# Each entry: param -> (validator_key, required, default). "name" is implicit.
EXTRACTOR_SCHEMAS: dict[str, dict[str, tuple[str, bool, object]]] = {
    "last": {},
    "per_step": {},
    "steps": {"steps": ("step_list", True, None)},
}


def extract(series: Sequence[Point], extractor: dict) -> list[Extraction]:
    """Apply a normalized extractor dict to a series."""
    name = extractor["name"]
    if name == "last":
        return _extract_last(series)
    if name == "per_step":
        return _extract_per_step(series)
    if name == "steps":
        return _extract_steps(series, extractor["steps"])
    raise ExtractorError(f"unknown extractor {name!r}; known: {sorted(EXTRACTOR_SCHEMAS)}")


# --- coordinate encoding ----------------------------------------------------

_COORD_VERSION = "v1"
# Characters the encoding uses as delimiters; an author sub_label may not contain
# them (enforced at parse time in register.py).
COORD_RESERVED = ("|", "=")


def encode_coordinate(coord: str, author_sub_label: str | None) -> str:
    """The store `sub_label` for one comparison coordinate.

    Combines the extractor-identity token (`coord`) with the author's optional
    `sub_label` under a versioned, deterministic format. The constraint is
    never encoded here: two gates that share an extractor share this coordinate
    (and thus one baseline), differing only in the pass/fail rule. Because the
    token is the extractor identity (`"step=<k>"`, not the extractor name), a
    `per_step` step-k value and an explicit `steps: [k]` value share the same
    coordinate by construction.
    """
    segments = [coord]
    if author_sub_label is not None:
        segments.append(f"lbl={author_sub_label}")
    return f"{_COORD_VERSION}|" + "|".join(segments)
