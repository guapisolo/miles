"""Rule-based reward for MathVision (``--custom-rm-path``).

MathVision rows are either multiple-choice (gold answer is a letter) or free-form
(gold answer is a value). The split is recorded in ``sample.metadata`` by
``prepare_data.py`` (``is_choice`` / ``options``).

  - multiple-choice: 1.0 when the model's selected option letter matches the gold
    letter. The model may emit the letter directly or the option's content, so we
    map content back to its letter before comparing.
  - free-form: 1.0 when the boxed answer matches the gold value via the shared
    math grader (handles numeric / sympy equivalence).
"""

import re
import string

from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.types import Sample


def _strip_think(text: str) -> str:
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def _gold_letter(gold: str, options: list[str]) -> str | None:
    letters = string.ascii_uppercase[: len(options)]
    g = gold.strip()
    if g.upper() in letters:
        return g.upper()
    # gold stored as option content -> map to its letter
    for i, opt in enumerate(options):
        if str(opt).strip() == g:
            return letters[i]
    return None


def _pred_letter(response: str, options: list[str]) -> str | None:
    letters = set(string.ascii_uppercase[: len(options)])
    text = _strip_think(response)

    boxed = extract_boxed_answer(text) if "\\boxed" in text else None
    if boxed:
        # exact letter inside the box
        b = boxed.strip().upper()
        if b in letters:
            return b
        # box holds option content -> map to its letter
        for i, opt in enumerate(options):
            if str(opt).strip() == boxed.strip():
                return string.ascii_uppercase[i]
        m = re.fullmatch(r"\(?([A-Z])\)?", b)
        if m and m.group(1) in letters:
            return m.group(1)

    for pat in (r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\)?", r"\b([A-Z])\b"):
        for cand in reversed(re.findall(pat, text, flags=re.IGNORECASE)):
            if cand.upper() in letters:
                return cand.upper()
    return None


async def reward(args, sample: Sample, **kwargs) -> float:
    response = sample.response or ""
    gold = (sample.label or "").strip()
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    options = list(metadata.get("options") or [])

    if metadata.get("is_choice") and options:
        gold_letter = _gold_letter(gold, options)
        pred_letter = _pred_letter(response, options)
        if gold_letter is None or pred_letter is None:
            return 0.0
        return 1.0 if pred_letter == gold_letter else 0.0

    return 1.0 if grade_answer_verl(response, gold) else 0.0
