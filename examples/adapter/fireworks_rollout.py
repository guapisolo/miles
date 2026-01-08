from argparse import Namespace
from typing import Any

from eval_protocol import InitRequest
from eval_protocol.types.remote_rollout_processor import RolloutMetadata
from examples.adapter.fireworks_reward import custom_reward

from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

TOKENIZER = None
MASK_GENERATOR = None


def _get_sglang_base_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"


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


def _get_tokenizer_and_mask_generator(args: Namespace, state: GenerateState):
    return state.tokenizer, MultiTurnLossMaskGenerator(state.tokenizer, tokenizer_type=args.loss_mask_type)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for the Fireworks adapter."
    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    state = GenerateState(args)
    init_url = f"{args.agent_base_url}/init"
    messages = _coerce_messages(sample.prompt)
    tools = sample.metadata.get("tools") if sample.metadata else None
    completion_params = dict(sampling_params)

    # print(f"prompt: {sample.prompt}")
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

    resp = await post(init_url, init_request.model_dump(exclude_none=True))
    if resp["status"] != "success":
        raise ValueError(f"Failed to initialize agent: {resp['error']}")

    # TODO: better use miles router to handle the responses

    messages = resp["result"]
    tokenizer, mask_gen = _get_tokenizer_and_mask_generator(args, state)
    token_ids, loss_mask = mask_gen.get_loss_mask(messages, tools=tools)
    response_length = mask_gen.get_response_lengths([loss_mask])[0]

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = loss_mask[-response_length:]

    sample.reward = await custom_reward(args, messages, sample.label)
    # print(f"messages: {messages}, reward: {sample.reward}")
    return sample
