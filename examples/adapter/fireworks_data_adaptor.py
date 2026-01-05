"""
python examples/adapter/fireworks_data_adaptor.py \
  --input examples/adapter/gsm8k/gsm8k_sample.jsonl \
  --output examples/adapter/gsm8k/gsm8k_miles_sample.jsonl
"""
import argparse
import json
import logging
import os
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def extract_answer_digits(ground_truth: str) -> Optional[str]:
    if not ground_truth:
        return None

    match = re.search(r"<answer>(.*?)</answer>", ground_truth, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    answer_string = match.group(1)
    digits_match = re.search(r"(\d+)", answer_string)
    return digits_match.group(1) if digits_match else None


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc


def _extract_prompt_messages(messages: list[dict]) -> list[dict]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def convert_fireworks_to_miles_rows(
    rows: Iterable[dict],
    prompt_key: str = "prompt",
    label_key: str = "labels",
) -> Iterable[dict]:
    for row in rows:
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError("Expected 'messages' to be a list of dicts.")

        prompt_messages = _extract_prompt_messages(messages)
        label = extract_answer_digits(str(row.get("ground_truth", "")))

        yield {
            prompt_key: prompt_messages,
            label_key: label,
        }


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    default_input = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "gsm8k", "gsm8k_sample.jsonl")
    )
    default_output = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "gsm8k", "gsm8k_sample_miles.jsonl")
    )

    parser = argparse.ArgumentParser(
        description="Convert Fireworks jsonl data into Miles prompt/label format."
    )
    parser.add_argument("--input", default=default_input, help="Path to Fireworks jsonl file.")
    parser.add_argument("--output", default=default_output, help="Path to save Miles jsonl file.")
    parser.add_argument("--prompt-key", default="prompt", help="Key name for prompt in output jsonl.")
    parser.add_argument("--label-key", default="label", help="Key name for labels in output jsonl.")
    args = parser.parse_args()

    rows = convert_fireworks_to_miles_rows(
        _read_jsonl(args.input),
        prompt_key=args.prompt_key,
        label_key=args.label_key,
    )
    _write_jsonl(args.output, rows)
    logger.info("Converted data written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
