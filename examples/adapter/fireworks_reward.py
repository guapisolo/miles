from functools import lru_cache

from eval_protocol.models import EvaluationRow

from miles.utils.misc import load_function
from miles.utils.types import Sample


@lru_cache(maxsize=None)
def _get_reward_function(path: str):
    return load_function(path)


def _build_messages(sample: Sample) -> list[dict]:
    messages = [{"role": "user", "content": sample.prompt}] if isinstance(sample.prompt, str) else list(sample.prompt)
    messages.append({"role": "assistant", "content": sample.response})
    return messages


async def custom_reward(args, sample: Sample, **kwargs) -> float:
    row = EvaluationRow(messages=_build_messages(sample), ground_truth=sample.label)
    reward_function = _get_reward_function(args.reward_function_path)
    row = reward_function(row, **kwargs)
    return row.evaluation_result.score
