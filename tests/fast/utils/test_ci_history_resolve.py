"""Unit tests for resolving CI metric-history capture from MILES_CI_GATE_RECORD_DIR.

The resolver helper lives in ``miles.utils.arguments`` but importing that module
pulls heavy training dependencies. To keep this test light, the helper's source
is extracted and exec'd in isolation, so the behavior contract is verified
without importing the full module.
"""

import ast
from pathlib import Path

_ARGUMENTS_PATH = Path(__file__).resolve().parents[3] / "miles" / "utils" / "arguments.py"


def _load_resolver():
    """Exec only the resolver helper (and the env-var constant it uses)."""
    source = _ARGUMENTS_PATH.read_text()
    module = ast.parse(source)

    wanted = {"CI_GATE_RECORD_DIR_ENV", "resolve_ci_enable_metrics_capture"}
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)

    namespace = {"os": __import__("os")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(_ARGUMENTS_PATH), "exec"), namespace)
    return namespace["resolve_ci_enable_metrics_capture"], namespace["CI_GATE_RECORD_DIR_ENV"]


def test_env_unset_resolves_false():
    resolve, env_name = _load_resolver()
    assert resolve(env={}) is False
    # an unrelated key present is still off
    assert resolve(env={"SOMETHING_ELSE": "/x"}) is False
    assert env_name == "MILES_CI_GATE_RECORD_DIR"


def test_env_set_resolves_true():
    resolve, env_name = _load_resolver()
    assert resolve(env={env_name: "/some/record/dir"}) is True


def test_empty_env_value_does_not_enable():
    resolve, env_name = _load_resolver()
    assert resolve(env={env_name: ""}) is False
