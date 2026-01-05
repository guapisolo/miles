
import os
from argparse import Namespace
from typing import Any

from eval_protocol import InitRequest
from eval_protocol.types.remote_rollout_processor import RolloutMetadata

from miles.utils.http_utils import _wrap_ipv6, post
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

TOKENIZER = None
MASK_GENERATOR = None


def _get_agent_base_url(args: Namespace) -> str:
    base_url = getattr(args, "agent_base_url", None) or os.environ.get("MILES_AGENT_BASE_URL")
    if not base_url:
        base_url = "http://127.0.0.1:8000"
    return base_url.rstrip("/")


def _get_sglang_base_url(args: Namespace) -> str:
    host = _wrap_ipv6(args.sglang_router_ip)
    return f"http://{host}:{args.sglang_router_port}/v1"


def _blank_rollout_metadata() -> RolloutMetadata:
    return RolloutMetadata(
        invocation_id="",
        experiment_id="",
        rollout_id="",
        run_id="",
        row_id="",
    )


def _coerce_messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": prompt}]


def _extract_assistant_content(messages: list[dict[str, Any]]) -> str:
    return messages[-1]["content"]


def _get_tokenizer_and_mask_generator(args: Namespace):
    global TOKENIZER, MASK_GENERATOR
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)
    return TOKENIZER, MASK_GENERATOR


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for the Fireworks adapter."
    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    init_url = f"{_get_agent_base_url(args)}/init"
    messages = _coerce_messages(sample.prompt)
    tools = sample.metadata.get("tools") if sample.metadata else None
    completion_params = dict(sampling_params)

    if "model" not in completion_params:
        model_name = getattr(args, "model_name", None) or getattr(args, "hf_checkpoint", None) or "default"
        completion_params["model"] = model_name

    init_request = InitRequest(
        completion_params=completion_params,
        messages=messages,
        tools=tools,
        model_base_url=_get_sglang_base_url(args),
        metadata=_blank_rollout_metadata(),
    )

    response = await post(init_url, init_request.model_dump(exclude_none=True))
    result_messages = response["result"]["messages"]

    assistant_content = _extract_assistant_content(result_messages)
    _, mask_generator = _get_tokenizer_and_mask_generator(args)
    token_ids, loss_mask = mask_generator.get_loss_mask(result_messages)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]

    sample.tokens = token_ids
    sample.response = assistant_content
    sample.response_length = response_length
    sample.loss_mask = loss_mask[-response_length:] if response_length else []
    sample.status = Sample.Status.COMPLETED

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["rollout_messages"] = result_messages
    return sample
    
