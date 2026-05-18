"""Unit tests for `run_suite.py` label handling (M2 / AC-9 / AC-10).

These cover the Python-side label pipeline:

* `strip_run_ci_prefix`: empty input, prefix stripping, ignoring inputs
  without the `run-ci-` prefix (warning only).
* `filter_tests`: the six AC-9 scenarios around `--labels` and
  `--match-all-labels`.
* `PER_COMMIT_SUITES`: locked to the new 5-suite taxonomy (AC-10).

We build `CIRegistry` instances directly via a small factory rather than
parsing fixture files -- the AST-side validation lives in
`test_ci_register.py`; this module exercises the runtime filter.
"""

import warnings

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend
from tests.ci.run_suite import PER_COMMIT_SUITES, filter_tests, strip_run_ci_prefix


def _make(
    filename: str,
    *,
    backend: HWBackend = HWBackend.CUDA,
    suite: str = "stage-c-8-gpu-h100",
    labels: list[str] | None = None,
    always_on: bool = False,
    est_time: float = 60.0,
    nightly: bool = False,
    disabled: str | None = None,
) -> CIRegistry:
    """Minimal `CIRegistry` factory for filter tests."""
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=est_time,
        suite=suite,
        labels=list(labels) if labels is not None else [],
        always_on=always_on,
        nightly=nightly,
        disabled=disabled,
    )


# --- AC-10: PER_COMMIT_SUITES locked to the new taxonomy --------------------


class TestPerCommitSuites:
    def test_cpu_suites_exact(self):
        assert PER_COMMIT_SUITES[HWBackend.CPU] == ["stage-a-cpu", "stage-b-cpu"]

    def test_cuda_suites_exact(self):
        assert PER_COMMIT_SUITES[HWBackend.CUDA] == [
            "stage-c-8-gpu-h100",
            "stage-c-4-gpu-h200",
            "stage-c-glm5-8-gpu",
        ]

    def test_no_legacy_suite_names_remain(self):
        legacy = {
            "stage-a-fast",
            "stage-b-fast-1-gpu",
            "stage-b-fast-gpu",
            "stage-b-short-8-gpu",
            "stage-b-sglang-8-gpu",
            "stage-c-fsdp-8-gpu",
            "stage-c-megatron-8-gpu",
            "stage-c-precision-8-gpu",
            "stage-c-ckpt-8-gpu",
            "stage-c-long-8-gpu",
            "stage-c-lora-8-gpu",
            "stage-c-all",
        }
        all_suites = {s for suites in PER_COMMIT_SUITES.values() for s in suites}
        assert legacy.isdisjoint(all_suites), f"Legacy suite name(s) still present: {legacy & all_suites}"


# --- `strip_run_ci_prefix` direct tests -------------------------------------


class TestStripRunCiPrefix:
    def test_empty_input_yields_empty_set(self):
        assert strip_run_ci_prefix([]) == set()

    def test_single_prefixed_label_stripped(self):
        assert strip_run_ci_prefix(["run-ci-megatron"]) == {"megatron"}

    def test_multiple_prefixed_labels_stripped(self):
        assert strip_run_ci_prefix(["run-ci-megatron", "run-ci-fsdp"]) == {"megatron", "fsdp"}

    def test_duplicate_inputs_deduplicate(self):
        # Set semantics: identical inputs collapse.
        assert strip_run_ci_prefix(["run-ci-megatron", "run-ci-megatron"]) == {"megatron"}

    def test_non_prefixed_input_warns_and_is_skipped(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["megatron"])
        assert result == set(), "non-prefixed entries must be dropped, not silently included"
        assert len(caught) == 1
        assert "missing" in str(caught[0].message)
        assert "run-ci-" in str(caught[0].message)

    def test_mixed_inputs_keep_only_prefixed(self):
        # Prefixed entries survive; bare entries are warned + dropped. Mixed input
        # is the realistic case where one workflow string slipped through wrong.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["run-ci-megatron", "fsdp", "run-ci-short"])
        assert result == {"megatron", "short"}
        assert len(caught) == 1  # only the bare `fsdp` warns

    def test_empty_string_entries_skipped_without_warning(self):
        # argparse can hand us empty strings if someone writes `--labels ""`;
        # treat as no-op rather than warning, since they carry no information.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["", "run-ci-megatron"])
        assert result == {"megatron"}
        assert len(caught) == 0


# --- AC-9: `filter_tests` six scenarios -------------------------------------


@pytest.fixture
def cuda_h100_tests():
    """A representative `stage-c-8-gpu-h100` registry used across scenarios.

    Composition:
    * 2 `always_on=True` tests (no domain label)
    * 1 megatron-only test
    * 1 fsdp-only test
    * 1 megatron+sglang test (multi-label, exercises OR semantics)
    * 1 disabled test (must always be classified as skipped)
    """
    return [
        _make("tests/e2e/fast1.py", labels=[], always_on=True),
        _make("tests/e2e/fast2.py", labels=[], always_on=True),
        _make("tests/e2e/megatron/m1.py", labels=["megatron"]),
        _make("tests/e2e/fsdp/f1.py", labels=["fsdp"]),
        _make("tests/e2e/megatron/m_or_s.py", labels=["megatron", "sglang"]),
        _make("tests/e2e/megatron/disabled.py", labels=["megatron"], disabled="known flaky"),
    ]


def _names(tests: list[CIRegistry]) -> set[str]:
    return {t.filename for t in tests}


class TestFilterTestsLabels:
    def test_case1_no_labels_keeps_only_always_on(self, cuda_h100_tests):
        # AC-9 (i): empty --labels (after stripping) -> only always_on=True
        # tests survive. Disabled test goes to skipped bucket regardless of
        # always_on status; here our disabled test is not always_on, so it
        # falls out of the labels filter first (no overlap, not always_on)
        # and never makes it to the disabled split -- which is exactly what
        # the predicate should do.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(),
        )
        assert _names(enabled) == {"tests/e2e/fast1.py", "tests/e2e/fast2.py"}
        assert skipped == []

    def test_case2_single_domain_label(self, cuda_h100_tests):
        # AC-9 (ii): `run-ci-megatron` -> always_on + megatron-labeled tests.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron"},
        )
        assert _names(enabled) == {
            "tests/e2e/fast1.py",
            "tests/e2e/fast2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        # `disabled.py` matches the megatron label but is disabled, so it
        # belongs to the skipped bucket.
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case3_multiple_domain_labels_or_semantics(self, cuda_h100_tests):
        # AC-9 (iii): {megatron, fsdp} -> union (OR) of matches.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron", "fsdp"},
        )
        assert _names(enabled) == {
            "tests/e2e/fast1.py",
            "tests/e2e/fast2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }

    def test_case4_match_all_labels_runs_everything_in_suite(self, cuda_h100_tests):
        # AC-9 (iv): --match-all-labels ignores labels and always_on; every
        # enabled hw/suite/nightly-matching test runs.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(),
            match_all_labels=True,
        )
        assert _names(enabled) == {
            "tests/e2e/fast1.py",
            "tests/e2e/fast2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case5_unknown_pr_side_label_is_silent_noop(self, cuda_h100_tests):
        # AC-9 (v): unknown PR-side label (e.g. `run-ci-foo`) -- after
        # stripping, `foo` simply produces an empty intersection with every
        # test. No error; only always_on tests survive.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"foo"},
        )
        assert _names(enabled) == {"tests/e2e/fast1.py", "tests/e2e/fast2.py"}

    def test_case6_match_all_labels_wins_over_labels(self, cuda_h100_tests):
        # AC-9 (vi): both flags passed -> match_all_labels takes precedence.
        # Compare against case4: same result regardless of `labels` value.
        enabled_with_labels, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron"},
            match_all_labels=True,
        )
        enabled_without_labels, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(),
            match_all_labels=True,
        )
        assert _names(enabled_with_labels) == _names(enabled_without_labels)


# --- filter_tests: hw/suite/nightly partitioning still works ----------------


class TestFilterTestsBaseDimensions:
    def test_cross_suite_isolation(self):
        # A test registered to stage-c-glm5-8-gpu must not surface in
        # stage-c-8-gpu-h100, even with match_all_labels=True.
        tests = [
            _make("tests/e2e/h100/t.py", suite="stage-c-8-gpu-h100", always_on=True),
            _make("tests/e2e/glm5/t.py", suite="stage-c-glm5-8-gpu", labels=["glm5"]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            match_all_labels=True,
        )
        assert _names(enabled) == {"tests/e2e/h100/t.py"}

    def test_cross_backend_isolation(self):
        # CPU suite must not pull in CUDA-registered always_on tests.
        tests = [
            _make("tests/fast/t.py", backend=HWBackend.CPU, suite="stage-a-cpu", always_on=True),
            _make("tests/e2e/h100/t.py", backend=HWBackend.CUDA, suite="stage-c-8-gpu-h100", always_on=True),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CPU,
            "stage-a-cpu",
            labels=set(),
        )
        assert _names(enabled) == {"tests/fast/t.py"}

    def test_nightly_dimension_respected(self):
        tests = [
            _make("tests/e2e/per_commit.py", labels=["megatron"], nightly=False),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=False,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/per_commit.py"}
