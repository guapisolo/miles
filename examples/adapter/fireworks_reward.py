from eval_protocol.models import EvaluationRow

from examples.adapter.gsm8k.evaluator import gsm8k_reward_row
from miles.utils.types import Sample


def _build_messages(sample: Sample) -> list[dict]:
    messages = [{"role": "user", "content": sample.prompt}] if isinstance(sample.prompt, str) else list(sample.prompt)
    messages.append({"role": "assistant", "content": sample.response})
    return messages


async def custom_reward(args, sample: Sample, **kwargs) -> float:
    row = EvaluationRow(messages=_build_messages(sample), ground_truth=sample.label)
    row = gsm8k_reward_row(row, **kwargs)
    return row.evaluation_result.score
