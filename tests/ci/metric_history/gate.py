# doc-dev: docs/ci/03-metric-history-gate.md
"""Offline regression gate for the CI metric-history system.

The gate consumes one already-merged per-run NDJSON record (the passed attempt's
record; a later round picks which attempt) and a set of ``register_ci_gate``
specs declared in the test file, and decides whether the run is *trusted*.

Each spec pairs an EXTRACTOR (which value(s) to pull from the metric's series)
with a CONSTRAINT (the pass/fail rule). The extractor may fan out: ``per_step``
and ``steps`` yield one comparison coordinate per step, so one spec produces N
per-step verdicts, each compared only against the same step's history. Two
checks run per coordinate, both using the spec's constraint:

* HARD gate -- always active. Compares the current scalar against the static
  ``hard_ref`` declared in the spec.
* HISTORICAL gate -- active only when the store returns >=1 trusted baseline
  value for this (identity, coordinate). Compares against the mean of those
  values. With zero trusted values the historical gate is INACTIVE -- a cold
  start, not a failure.

The run is trusted iff every *active* gate passed for every coordinate. The gate
is pure: it takes a :class:`MetricHistoryStore` by dependency injection, opens no
connection, reads no wandb, and writes no rows. It only calls
``store.recent_trusted_values``; persistence is a later round.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from tests.ci.ci_register import CIRegistry, HWBackend, ut_parse_one_file
from tests.ci.metric_history.constraints import evaluate_constraint
from tests.ci.metric_history.extractors import ExtractorError, encode_coordinate, extract
from tests.ci.metric_history.register import CiGateSpec, parse_ci_gate_specs
from tests.ci.metric_history.storage import MetricHistoryStore

# Maps the parsed HWBackend enum to the lowercase backend string the store keys
# on. The store's identity tuple is all strings; CIRegistry.backend is the enum.
_BACKEND_STR: dict[HWBackend, str] = {
    HWBackend.CPU: "cpu",
    HWBackend.CUDA: "cuda",
    HWBackend.ROCM: "rocm",
}


class GateStatus(Enum):
    """Outcome of one check (hard or historical) for one coordinate."""

    PASS = "pass"
    FAIL = "fail"
    INACTIVE = "inactive"  # historical gate with no trusted baseline (cold start)
    ERROR = "error"  # the metric could not be extracted (missing/empty series, bad step)


@dataclass(frozen=True)
class MetricGateResult:
    """Per-coordinate verdict.

    ``sub_label`` is the encoded baseline coordinate (extractor identity + step +
    author label); ``step`` is the step this coordinate came from (None for a
    positional extractor like ``last``, or for an extraction error). ``current``
    is the extracted scalar, or None when extraction errored. ``baseline_mean``
    is the mean of trusted history when the historical gate is active, else None.
    ``trusted`` is True iff every active check here passed.
    """

    metric_key: str
    sub_label: str | None
    step: int | None
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
        """The run is trusted iff every per-coordinate verdict is trusted.

        An empty metrics list (no gate specs) is vacuously trusted: a file that
        declares no gate cannot regress. One failing step of a fanned-out spec
        untrusts the whole run.
        """
        return all(m.trusted for m in self.metrics)


def compute_test_file_hash(filename: str) -> str:
    """sha256 of the test file's raw bytes -- the store's ``test_file_hash``."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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


def _error_result(
    spec: CiGateSpec,
    reason: str,
    *,
    sub_label: str | None = None,
    step: int | None = None,
    current: float | None = None,
) -> MetricGateResult:
    return MetricGateResult(
        metric_key=spec.metric_key,
        sub_label=sub_label if sub_label is not None else spec.sub_label,
        step=step,
        current=current,
        hard_status=GateStatus.ERROR,
        historical_status=GateStatus.INACTIVE,
        baseline_n=0,
        baseline_mean=None,
        reason=reason,
    )


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
) -> list[MetricGateResult]:
    series = by_metric.get(spec.metric_key)
    if series is None:
        return [_error_result(spec, f"required metric {spec.metric_key!r} missing from record")]

    try:
        extractions = extract(series, spec.extractor)
    except ExtractorError as e:
        return [_error_result(spec, f"metric {spec.metric_key!r} ({spec.extractor['name']}): {e}")]

    results: list[MetricGateResult] = []
    for ex in extractions:
        coord_sub_label = encode_coordinate(ex.coord, spec.sub_label)

        hard = evaluate_constraint(spec.constraint, ex.value, spec.hard_ref)
        hard_status = GateStatus.PASS if hard.ok else GateStatus.FAIL
        reasons: list[str] = []
        if not hard.ok:
            reasons.append(f"hard: cur={ex.value:.6g} vs ref={spec.hard_ref:.6g} exceeds band={hard.band:.6g}")

        trusted_values = store.recent_trusted_values(
            test_path,
            backend,
            suite,
            spec.metric_key,
            coord_sub_label,
            test_file_hash,
            history_limit,
        )
        if not trusted_values:
            historical_status = GateStatus.INACTIVE
            baseline_mean = None
            reasons.append("historical: cold start (0 trusted baselines)")
        else:
            baseline_mean = sum(trusted_values) / len(trusted_values)
            hist = evaluate_constraint(spec.constraint, ex.value, baseline_mean)
            historical_status = GateStatus.PASS if hist.ok else GateStatus.FAIL
            if not hist.ok:
                reasons.append(
                    f"historical: cur={ex.value:.6g} vs mean={baseline_mean:.6g} "
                    f"(n={len(trusted_values)}) exceeds band={hist.band:.6g}"
                )

        if hard_status == GateStatus.PASS and historical_status in (GateStatus.PASS, GateStatus.INACTIVE):
            reasons.insert(0, "ok")

        results.append(
            MetricGateResult(
                metric_key=spec.metric_key,
                sub_label=coord_sub_label,
                step=ex.step,
                current=ex.value,
                hard_status=hard_status,
                historical_status=historical_status,
                baseline_n=len(trusted_values),
                baseline_mean=baseline_mean,
                reason="; ".join(reasons),
            )
        )
    return results


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
    baseline query and nothing else (no writes, no connection opened here). A
    fanned-out spec contributes one MetricGateResult per step.
    """
    specs = parse_ci_gate_specs(test_filename)
    registry = _registry_for(test_filename)
    backend = _BACKEND_STR[registry.backend]
    test_file_hash = compute_test_file_hash(test_filename)
    by_metric = parse_merged_record(merged_record_path)

    results: list[MetricGateResult] = []
    for spec in specs:
        results.extend(
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
        )

    return GateResult(
        test_path=registry.filename,
        backend=backend,
        suite=registry.suite,
        test_file_hash=test_file_hash,
        metrics=results,
    )
