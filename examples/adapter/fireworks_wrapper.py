"""
Fireworks wrapper for miles.
"""

import argparse

from eval_protocol import InitRequest
from eval_protocol.types.message import Message
from eval_protocol.types.remote_rollout_processor import RolloutMetadata

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.http_utils import post


def _blank_rollout_metadata() -> RolloutMetadata:
    return RolloutMetadata(
        invocation_id="",
        experiment_id="",
        rollout_id="",
        run_id="",
        row_id="",
    )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)

    assert isinstance(input.sample.prompt, list), "prompt must be a list of messages"
    messages = []
    for item in input.sample.prompt:
        messages.append(Message(role=item["role"], content=item["content"]))

    request_kwargs = build_chat_request_kwargs(input.sampling_params)
    init_request = InitRequest(
        completion_params=request_kwargs,
        messages=messages,
        model_base_url=tracer.base_url,
        metadata=_blank_rollout_metadata(),
        tools=None,
        api_key=None,
    )

    response = await post(f"{input.args.agent_base_url}/init", init_request.model_dump())
    if response.get("status") != "success":
        raise ValueError(f"Failed to initialize agent: {response.get('error')}")

    records = await tracer.collect_records()
    samples = compute_samples_from_openai_records(input.sample, records, input.state.tokenizer)
    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--agent-base-url", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)


generate.add_arguments = _add_arguments
