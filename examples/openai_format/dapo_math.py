"""
DAPO math OpenAI format example for token in/out verification.
"""

import argparse
from typing import Any

from openai import AsyncOpenAI

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples

_DAPO_MATH_SYSTEM_PROMPT = (
    "Solve the math problem and return the final answer as \\boxed{integer}. "
    "Keep the reasoning concise and finish with the boxed answer."
)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)
    messages = _normalize_prompt(input.sample.prompt)
    await _run_single_turn_openai(base_url=tracer.base_url, messages=messages)

    records = await tracer.collect_records()
    samples = compute_samples_from_openai_records(input.sample, records, input.state.tokenizer)
    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments


def build_dapo_math_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _DAPO_MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return prompt
    return build_dapo_math_messages(prompt)


async def _run_single_turn_openai(base_url: str, messages: list[dict[str, Any]]) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="empty")
    await client.chat.completions.create(model="default", messages=messages)
