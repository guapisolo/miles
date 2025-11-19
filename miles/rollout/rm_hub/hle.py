from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_PARENT = _WORKSPACE_ROOT.parent
_LOCAL_HLE_REQUIREMENTS = _WORKSPACE_ROOT / "examples" / "eval_multi_task" / "requirements_hle.txt"

_HLE_REPO_URL = "https://github.com/centerforaisafety/hle.git"
_HLE_REPO_DIRNAME = "HLE"
_HLE_JUDGE_ENV = "HLE_JUDGE_MODEL"
_DEFAULT_JUDGE_MODEL = "o3-mini-2025-01-31"


def _ensure_hle_repo() -> Path:
    """Clone the HLE repo if needed and add it to sys.path."""

    repo_path = _WORKSPACE_PARENT / _HLE_REPO_DIRNAME
    if not repo_path.exists():
        clone_cmd = ["git", "clone", _HLE_REPO_URL, str(repo_path)]
        try:
            subprocess.run(clone_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:  # pragma: no cover - best effort helper
            raise ImportError(
                f"Unable to clone HLE repo from {_HLE_REPO_URL}. Please clone it into {repo_path}."
            ) from exc

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = repo_str
    elif repo_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pythonpath])

    return repo_path


def _ensure_hle_dependencies(repo_path: Path) -> None:
    """Install the minimal dependencies HLE expects."""

    if not _LOCAL_HLE_REQUIREMENTS.exists():
        logger.debug("Local HLE requirements file not found at %s; skipping install.", _LOCAL_HLE_REQUIREMENTS)
        return

    sentinel = repo_path / ".deps_installed"
    if sentinel.exists():
        return

    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(_LOCAL_HLE_REQUIREMENTS)]
    try:
        subprocess.run(install_cmd, check=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to install HLE dependencies automatically: %s", exc)
    else:
        sentinel.write_text("installed\n")


def _configure_judge_module(module) -> None:
    judge_model = os.environ.get(_HLE_JUDGE_ENV, _DEFAULT_JUDGE_MODEL)
    try:
        num_workers = int(os.environ.get("HLE_JUDGE_NUM_WORKERS", "8"))
    except ValueError:
        num_workers = 8
    num_workers = max(2, num_workers)

    args_obj = getattr(module, "args", None)
    if args_obj is None:
        args_obj = SimpleNamespace()
    args_obj.judge = judge_model
    args_obj.num_workers = num_workers
    module.args = args_obj
    setattr(module, "_miles_configured", True)


def _load_hle_judge_module():
    repo_path = _ensure_hle_repo()
    try:
        module = importlib.import_module("hle_eval.run_judge_results")
    except ImportError:
        _ensure_hle_dependencies(repo_path)
        module = importlib.import_module("hle_eval.run_judge_results")
    if not getattr(module, "_miles_configured", False):
        _configure_judge_module(module)
    return module


_HLE_JUDGE_MODULE = _load_hle_judge_module()


def _extract_from_content_pieces(content: Iterable[Any]) -> str:
    pieces: list[str] = []
    for chunk in content:
        text = ""
        if isinstance(chunk, str):
            text = chunk
        elif isinstance(chunk, dict):
            if isinstance(chunk.get("text"), str):
                text = chunk["text"]
            elif isinstance(chunk.get("content"), str):
                text = chunk["content"]
        if text:
            pieces.append(text)
    return "\n".join(pieces).strip()


def _stringify_prompt(prompt: Any) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        role = prompt.get("role")
        content = prompt.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = _extract_from_content_pieces(content)
        elif content is None and isinstance(prompt.get("text"), str):
            text = prompt["text"]
        else:
            text = str(content) if content is not None else ""
        if role and text:
            return f"{role}: {text}"
        return text
    if isinstance(prompt, list):
        parts = [_stringify_prompt(item) for item in prompt]
        return "\n\n".join(part for part in parts if part)
    return str(prompt)


def _extract_question_text(prompt: Any, metadata: Optional[Dict[str, Any]]) -> str:
    metadata = metadata or {}
    for key in ("question", "question_text", "prompt_text", "hle_question"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompt_text = _stringify_prompt(prompt).strip()
    return prompt_text


def _extract_correct_answer(label: Any, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    metadata = metadata or {}
    for key in ("correct_answer", "answer", "label_text", "hle_answer"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if label is None:
        return None
    return str(label)


async def compute_hle_reward(
    response: str,
    label: Any,
    *,
    prompt: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Score a response using HLE's official judge pipeline backed by OpenAI.
    """

    if response is None:
        return 0.0

    question_text = _extract_question_text(prompt, metadata)
    if not question_text:
        logger.debug("HLE scoring skipped because question text is missing.")
        return 0.0

    correct_answer = _extract_correct_answer(label, metadata)
    if not correct_answer:
        logger.debug("HLE scoring skipped because ground-truth answer is missing.")
        return 0.0

    try:
        judge_response = await _HLE_JUDGE_MODULE.extract_answer(question_text, correct_answer, str(response))
    except Exception as exc:  # pragma: no cover - remote dependency
        logger.warning("HLE judge request failed: %s", exc)
        return 0.0

    if not judge_response:
        return 0.0

    is_correct = str(judge_response.get("correct", "")).lower().startswith("y")
    return 1.0 if is_correct else 0.0
