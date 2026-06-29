import ast
import glob
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto

from tests.ci.labels import KNOWN_LABELS

__all__ = [
    "HWBackend",
    "CIRegistry",
    "CiGateSpec",
    "collect_tests",
    "discover_ci_files",
    "parse_ci_gate_specs",
    "register_cpu_ci",
    "register_cuda_ci",
    "register_rocm_ci",
    "register_ci_gate",
    "ut_parse_one_file",
]

# Only these two parameters may be passed positionally; everything else
# (labels, always_on, nightly, disabled) is keyword-only.
_POSITIONAL_PARAMS = ("est_time", "suite")

# All accepted keyword arguments (in addition to the positional pair above).
_VALID_KWARGS = frozenset({"est_time", "suite", "labels", "nightly", "disabled"})

_REGISTER_NAMES = frozenset({"register_cpu_ci", "register_cuda_ci", "register_rocm_ci"})

_UNSET = object()


class HWBackend(Enum):
    CPU = auto()
    CUDA = auto()
    ROCM = auto()


@dataclass
class CIRegistry:
    backend: HWBackend
    filename: str
    est_time: float
    suite: str
    labels: list[str] = field(default_factory=list)
    nightly: bool = False
    disabled: str | None = None  # None = enabled, string = disabled reason
    # True only when collect_tests synthesized this entry by directory
    # convention (currently for tests/fast/ files that declare no register
    # call); False for every entry parsed from a register_*_ci() call.
    implicit: bool = False


def register_cpu_ci(
    est_time: float,
    suite: str,
    *,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
):
    """Marker for CPU CI registration (parsed via AST; runtime no-op).

    `labels=None` and `labels=[]` are equivalent: the test runs on every PR
    regardless of `run-ci-*` labels. A non-empty `labels` list gates the test
    on PR labels — the test runs when the PR carries `run-ci-<x>` for any
    `<x>` in `labels`.
    """
    return None


def register_cuda_ci(
    est_time: float,
    suite: str,
    *,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
):
    """Marker for CUDA CI registration (parsed via AST; runtime no-op).

    See `register_cpu_ci` for label semantics.
    """
    return None


def register_rocm_ci(
    est_time: float,
    suite: str,
    *,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
):
    """Marker for ROCm CI registration (parsed via AST; runtime no-op).

    See `register_cpu_ci` for label semantics.

    """
    return None


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
        value = _extract_constant(kw.value)
        if value is _UNSET:
            raise ValueError(f"{filename}: {kw.arg} in {_CI_GATE_REGISTER_NAME}() must be a literal constant")
        parsed[kw.arg] = value

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


_REGISTER_BACKEND_MAP = {
    "register_cpu_ci": HWBackend.CPU,
    "register_cuda_ci": HWBackend.CUDA,
    "register_rocm_ci": HWBackend.ROCM,
}


def _extract_constant(node: ast.AST) -> object:
    """Return the literal value of an ast.Constant; otherwise return _UNSET.

    Sentinel return (instead of raising) lets callers compose richer error
    messages with parameter names and file paths.
    """
    if isinstance(node, ast.Constant):
        return node.value
    return _UNSET


def _extract_list_constant(node: ast.AST, *, context: str = "value") -> list:
    """Return a list of literal string constants from `ast.List`.

    Accepts `None` (as `ast.Constant(None)`) and treats it as an empty list,
    so callers may write `labels=None` interchangeably with `labels=[]`.

    Raises ValueError when the node is neither a list literal of string
    constants nor a literal `None`.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return []
    if not isinstance(node, ast.List):
        raise ValueError(f"{context} must be a list of string literals or None (got {type(node).__name__})")
    out: list = []
    for elt in node.elts:
        v = _extract_constant(elt)
        if v is _UNSET:
            raise ValueError(f"{context} must be a list of string literals (non-literal element)")
        if not isinstance(v, str):
            raise ValueError(f"{context} must be a list of string literals (got {type(v).__name__} element)")
        out.append(v)
    return out


class RegistryVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.registries: list[CIRegistry] = []

    def _parse_call_args(self, func_call: ast.Call, func_name: str) -> CIRegistry:
        if any(isinstance(arg, ast.Starred) for arg in func_call.args):
            raise ValueError(f"{self.filename}: starred arguments are not supported in {func_name}()")

        if len(func_call.args) > len(_POSITIONAL_PARAMS):
            raise ValueError(
                f"{self.filename}: too many positional arguments in {func_name}(); "
                f"only {list(_POSITIONAL_PARAMS)} may be positional "
                f"(labels and later are keyword-only)"
            )

        parsed: dict[str, object] = {}

        for name, arg in zip(_POSITIONAL_PARAMS, func_call.args, strict=False):
            v = _extract_constant(arg)
            if v is _UNSET:
                raise ValueError(f"{self.filename}: {name} in {func_name}() must be a literal constant")
            parsed[name] = v

        for kw in func_call.keywords:
            if kw.arg is None:
                raise ValueError(f"{self.filename}: **kwargs are not supported in {func_name}()")
            if kw.arg in parsed:
                raise ValueError(f"{self.filename}: duplicated argument '{kw.arg}' in {func_name}()")
            if kw.arg not in _VALID_KWARGS:
                raise ValueError(f"{self.filename}: unknown argument '{kw.arg}' in {func_name}()")
            if kw.arg == "labels":
                parsed["labels"] = _extract_list_constant(
                    kw.value, context=f"{self.filename}: labels in {func_name}()"
                )
            else:
                v = _extract_constant(kw.value)
                if v is _UNSET:
                    raise ValueError(f"{self.filename}: {kw.arg} in {func_name}() must be a literal constant")
                parsed[kw.arg] = v

        if "est_time" not in parsed:
            raise ValueError(f"{self.filename}: est_time is required in {func_name}()")
        if "suite" not in parsed:
            raise ValueError(f"{self.filename}: suite is required in {func_name}()")

        if not isinstance(parsed["est_time"], (int, float)):
            raise ValueError(f"{self.filename}: est_time must be a number in {func_name}()")
        if not isinstance(parsed["suite"], str):
            raise ValueError(f"{self.filename}: suite must be a string in {func_name}()")

        # `labels` is optional. Missing / None / [] all mean "always run on
        # every PR"; only a non-empty list gates the test on PR labels.
        labels = parsed.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"{self.filename}: labels must be a list or None in {func_name}()")

        nightly = parsed.get("nightly", False)
        if not isinstance(nightly, bool):
            raise ValueError(f"{self.filename}: nightly must be a boolean in {func_name}()")

        disabled = parsed.get("disabled", None)
        if disabled is not None and not isinstance(disabled, str):
            raise ValueError(f"{self.filename}: disabled must be a string or None in {func_name}()")

        unknown = [label for label in labels if label not in KNOWN_LABELS]
        if unknown:
            valid_list = ", ".join(sorted(KNOWN_LABELS))
            raise ValueError(
                f"{self.filename}: unknown labels {unknown} in {func_name}(); "
                f"valid labels: [{valid_list}]. "
                f"To add a new label: edit tests/ci/labels.py + create matching "
                f"`run-ci-<label>` in GitHub repo Settings -> Labels."
            )

        return CIRegistry(
            backend=_REGISTER_BACKEND_MAP[func_name],
            filename=self.filename,
            est_time=float(parsed["est_time"]),
            suite=parsed["suite"],
            labels=list(labels),
            nightly=nightly,
            disabled=disabled,
            implicit=False,
        )

    def _collect_ci_registry(self, func_call: ast.Call):
        if not isinstance(func_call.func, ast.Name):
            return None
        if func_call.func.id not in _REGISTER_NAMES:
            return None
        return self._parse_call_args(func_call, func_call.func.id)

    def visit_Module(self, node):
        for stmt in node.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            cr = self._collect_ci_registry(stmt.value)
            if cr is not None:
                self.registries.append(cr)


def ut_parse_one_file(filename: str) -> list[CIRegistry]:
    with open(filename) as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=filename)
    visitor = RegistryVisitor(filename=filename)
    visitor.visit(tree)
    return visitor.registries


def _is_implicit_fast_cpu_path(filename: str) -> bool:
    return filename.startswith("tests/fast/")


# Directories the CI runner scans.
# 1. tests/fast/ is CPU-only and auto-registers
# 2. the rest require an explicit register_*_ci on each discovered file.
# 3. Only test_*.py are collected.
# 4. Patterns are repo-relative, so the runner must run from the repo root (the same cwd ut_parse_one_file's open() assumes).
_DISCOVERY_ROOTS = ("tests/fast", "tests/fast-gpu", "tests/e2e", "tests/ci")


def discover_ci_files() -> list[str]:
    """Return every CI test file (sorted, repo-relative) across the roots."""
    files: list[str] = []
    for root in _DISCOVERY_ROOTS:
        files.extend(glob.glob(f"{root}/**/test_*.py", recursive=True))
    return sorted(files)


def _file_text_mentions_register(filename: str) -> bool:
    """True when the file's text contains `register_cpu_ci` or
    `register_cuda_ci` as a substring anywhere.

    Used as a defense-in-depth check before synthesizing an implicit CPU
    registry for a tests/fast/ file with zero parsed registries: if the
    file mentions either symbol but the AST visitor found no top-level
    Expr(Call), the file probably has an aliased import, a non-toplevel
    call, or an attribute-style call (`ci_register.register_cpu_ci(...)`)
    -- silently treating it as unregistered would mask the intent.
    """
    try:
        with open(filename) as f:
            content = f.read()
    except OSError:
        return False
    return "register_cpu_ci" in content or "register_cuda_ci" in content


def _make_implicit_cpu_registry(filename: str) -> CIRegistry:
    return CIRegistry(
        backend=HWBackend.CPU,
        filename=filename,
        est_time=1.0,
        suite="stage-a-cpu",
        labels=[],
        nightly=False,
        disabled=None,
        implicit=True,
    )


def collect_tests(files: list[str], sanity_check: bool = True) -> list[CIRegistry]:
    ci_tests: list[CIRegistry] = []
    for file in files:
        registries = ut_parse_one_file(file)
        if _is_implicit_fast_cpu_path(file):
            # tests/fast/ is CPU-only by location;
            for r in registries:
                if r.backend != HWBackend.CPU:
                    raise ValueError(
                        f"{file}: register_cuda_ci is forbidden in tests/fast/; "
                        f"move the file to tests/fast-gpu/ instead"
                    )
            if len(registries) == 0:
                if _file_text_mentions_register(file):
                    raise ValueError(
                        f"{file}: file mentions register_cpu_ci or register_cuda_ci "
                        f"textually but no top-level call was parsed; check for "
                        f"aliased import, non-toplevel call, or attribute access"
                    )
                ci_tests.append(_make_implicit_cpu_registry(file))
                continue
        if len(registries) == 0:
            msg = f"No CI registry found in {file}"
            if sanity_check:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
            continue
        ci_tests.extend(registries)
    return ci_tests
