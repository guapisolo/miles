# doc-dev: docs/ci/03-metric-history-gate.md
"""Offline regression gate for the CI metric-history system.

The gate consumes one already-merged per-run NDJSON record (the passed attempt's
record; a later round picks which attempt) and a set of ``register_ci_gate``
specs declared in the test file, and decides whether the run is *trusted*.

Two checks run per spec, both using the same rel-OR-abs tolerance
``max(rel*|ref|, abs_floor)``:

* HARD gate -- always active. Compares the current scalar against the static
  ``hard_ref`` declared in the spec. Two-sided by default; one-sided (only an
  increase fails) when ``higher_is_worse``.
* HISTORICAL gate -- active only when the store returns >=1 trusted baseline
  value for this (identity, metric, sub_label). Compares the current scalar
  against the mean of those values. With zero trusted values the historical gate
  is INACTIVE -- a cold start, not a failure.

The run is trusted iff every *active* gate passed for every value. The gate is
pure: it takes a :class:`MetricHistoryStore` by dependency injection, opens no
connection, reads no wandb, and writes no rows. It only calls
``store.recent_trusted_values``; persistence is a later round.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from tests.ci.ci_register import CiGateSpec, CIRegistry, HWBackend, parse_ci_gate_specs, ut_parse_one_file
from tests.ci.metric_history import MetricHistoryStore
from tests.ci.metric_reducers import ReducerError, default_reducer_name, reduce_series

# Maps the parsed HWBackend enum to the lowercase backend string the store keys
# on. The store's identity tuple is all strings; CIRegistry.backend is the enum.
_BACKEND_STR: dict[HWBackend, str] = {
    HWBackend.CPU: "cpu",
    HWBackend.CUDA: "cuda",
    HWBackend.ROCM: "rocm",
}


class GateStatus(Enum):
    """Outcome of one check (hard or historical) for one value."""

    PASS = "pass"
    FAIL = "fail"
    INACTIVE = "inactive"  # historical gate with no trusted baseline (cold start)
    ERROR = "error"  # the metric could not be reduced (missing/empty series, bad reducer)


@dataclass(frozen=True)
class MetricGateResult:
    """Per-(metric_key, sub_label) verdict.

    ``current`` is the reduced scalar, or None when reduction errored.
    ``baseline_mean`` is the mean of trusted history when the historical gate is
    active, else None. ``trusted`` is True iff every active check here passed.
    """

    metric_key: str
    sub_label: str | None
    current: float | None
    hard_status: GateStatus
    historical_status: GateStatus
    baseline_n: int
    baseline_mean: float | None
    reason: str

    @property
    def trusted(self) -> bool:
        return self.hard_status == GateStatus.PASS and self.historical_status in (
            GateStatus.PASS,
            GateStatus.INACTIVE,
        )


@dataclass(frozen=True)
class GateResult:
    """Run-level verdict over every gate spec for one test file."""

    test_path: str
    backend: str
    suite: str
    test_file_hash: str
    metrics: list[MetricGateResult] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        """The run is trusted iff every per-metric verdict is trusted.

        An empty metrics list (no gate specs) is vacuously trusted: a file that
        declares no gate cannot regress.
        """
        return all(m.trusted for m in self.metrics)


def compute_test_file_hash(filename: str) -> str:
    """sha256 of the test file's raw bytes -- the store's ``test_file_hash``."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _tolerance(ref: float, rel: float, abs_floor: float) -> float:
    """The rel-OR-abs band: ``max(rel*|ref|, abs_floor)``.

    A near-zero ``ref`` makes ``rel*|ref|`` vanish, so ``abs_floor`` is what
    keeps a metric riding at ~1e-9 from flagging on a meaningless relative
    percentage.
    """
    return max(rel * abs(ref), abs_floor)


def _check_against(cur: float, ref: float, rel: float, abs_floor: float, higher_is_worse: bool) -> bool:
    """True when ``cur`` is within tolerance of ``ref``.

    Two-sided unless ``higher_is_worse``, where only an increase beyond
    tolerance fails (a drop is always fine).
    """
    band = _tolerance(ref, rel, abs_floor)
    if higher_is_worse:
        return (cur - ref) <= band
    return abs(cur - ref) <= band


def parse_merged_record(record_path: str) -> dict[str, list]:
    """Read a merged NDJSON record into ``{metric_key: series}``.

    Each line is ``{"metric": key, "series": [[step, value], ...]}``. A repeated
    metric key (should not happen post-merge) keeps the last line's series.
    """
    by_metric: dict[str, list] = {}
    with open(record_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_metric[rec["metric"]] = rec["series"]
    return by_metric


def _registry_for(filename: str) -> CIRegistry:
    """The single CIRegistry governing this test file.

    A gate needs exactly one (backend, suite) identity. A file with no register
    call, or more than one, is an authoring error the gate refuses rather than
    guessing which suite a metric belongs to.
    """
    registries = ut_parse_one_file(filename)
    if not registries:
        raise ValueError(f"{filename}: no register_*_ci() call; gate identity is undefined")
    if len(registries) > 1:
        raise ValueError(f"{filename}: {len(registries)} register_*_ci() calls; gate identity is ambiguous")
    return registries[0]


def _evaluate_spec(
    spec: CiGateSpec,
    by_metric: dict[str, list],
    store: MetricHistoryStore,
    *,
    test_path: str,
    backend: str,
    suite: str,
    test_file_hash: str,
    history_limit: int,
) -> MetricGateResult:
    reducer_name = spec.reducer or default_reducer_name(spec.metric_key)
    series = by_metric.get(spec.metric_key)

    if series is None:
        return MetricGateResult(
            metric_key=spec.metric_key,
            sub_label=spec.sub_label,
            current=None,
            hard_status=GateStatus.ERROR,
            historical_status=GateStatus.INACTIVE,
            baseline_n=0,
            baseline_mean=None,
            reason=f"required metric {spec.metric_key!r} missing from record",
        )

    try:
        cur = reduce_series(series, reducer_name)
    except ReducerError as e:
        return MetricGateResult(
            metric_key=spec.metric_key,
            sub_label=spec.sub_label,
            current=None,
            hard_status=GateStatus.ERROR,
            historical_status=GateStatus.INACTIVE,
            baseline_n=0,
            baseline_mean=None,
            reason=f"metric {spec.metric_key!r} ({reducer_name}): {e}",
        )

    hard_ok = _check_against(cur, spec.hard_ref, spec.rel, spec.abs_floor, spec.higher_is_worse)
    hard_status = GateStatus.PASS if hard_ok else GateStatus.FAIL
    reasons: list[str] = []
    if not hard_ok:
        band = _tolerance(spec.hard_ref, spec.rel, spec.abs_floor)
        reasons.append(f"hard: cur={cur:.6g} vs ref={spec.hard_ref:.6g} exceeds band={band:.6g}")

    trusted_values = store.recent_trusted_values(
        test_path,
        backend,
        suite,
        spec.metric_key,
        spec.sub_label,
        test_file_hash,
        history_limit,
    )
    if not trusted_values:
        historical_status = GateStatus.INACTIVE
        baseline_mean = None
        reasons.append("historical: cold start (0 trusted baselines)")
    else:
        baseline_mean = sum(trusted_values) / len(trusted_values)
        hist_ok = _check_against(cur, baseline_mean, spec.rel, spec.abs_floor, spec.higher_is_worse)
        historical_status = GateStatus.PASS if hist_ok else GateStatus.FAIL
        if not hist_ok:
            band = _tolerance(baseline_mean, spec.rel, spec.abs_floor)
            reasons.append(
                f"historical: cur={cur:.6g} vs mean={baseline_mean:.6g} "
                f"(n={len(trusted_values)}) exceeds band={band:.6g}"
            )

    if hard_status == GateStatus.PASS and historical_status in (GateStatus.PASS, GateStatus.INACTIVE):
        reasons.insert(0, "ok")

    return MetricGateResult(
        metric_key=spec.metric_key,
        sub_label=spec.sub_label,
        current=cur,
        hard_status=hard_status,
        historical_status=historical_status,
        baseline_n=len(trusted_values),
        baseline_mean=baseline_mean,
        reason="; ".join(reasons),
    )


def evaluate_gate(
    test_filename: str,
    merged_record_path: str,
    store: MetricHistoryStore,
    *,
    history_limit: int = 20,
) -> GateResult:
    """Evaluate every ``register_ci_gate`` spec in ``test_filename`` against a record.

    ``test_filename`` is the repo-relative test path; its CIRegistry supplies the
    (backend, suite) identity and its contents the ``test_file_hash``.
    ``merged_record_path`` is the merged per-run NDJSON of the passed attempt --
    the gate never globs a base directory to find it. ``store`` answers the
    baseline query and nothing else (no writes, no connection opened here).
    """
    specs = parse_ci_gate_specs(test_filename)
    registry = _registry_for(test_filename)
    backend = _BACKEND_STR[registry.backend]
    test_file_hash = compute_test_file_hash(test_filename)
    by_metric = parse_merged_record(merged_record_path)

    results = [
        _evaluate_spec(
            spec,
            by_metric,
            store,
            test_path=registry.filename,
            backend=backend,
            suite=registry.suite,
            test_file_hash=test_file_hash,
            history_limit=history_limit,
        )
        for spec in specs
    ]

    return GateResult(
        test_path=registry.filename,
        backend=backend,
        suite=registry.suite,
        test_file_hash=test_file_hash,
        metrics=results,
    )
