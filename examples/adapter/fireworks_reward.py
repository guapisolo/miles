from argparse import Namespace
from functools import cache
from typing import Any

from eval_protocol.models import EvaluationRow, Message

from miles.utils.misc import load_function
from miles.utils.types import Sample


@cache
def _get_reward_function(path: str):
    return load_function(path)


def _build_messages(sample: Sample) -> list[dict]:
    messages = [{"role": "user", "content": sample.prompt}] if isinstance(sample.prompt, str) else list(sample.prompt)
    messages.append({"role": "assistant", "content": sample.response})
    return messages


async def custom_reward(args: Namespace, messages: list[dict[str, Any]], label: str | None, **kwargs) -> float:
    row = EvaluationRow(
        messages=[Message.model_validate(message) for message in messages], ground_truth=label, **kwargs
    )
    reward_function = _get_reward_function(args.reward_function_path)
    row = reward_function(row, **kwargs)
    return row.evaluation_result.score
