import uuid
from collections.abc import Callable
from typing import Any

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    LogProbs,
    ToolCallConstraint,
    UsageInfo,
)


def sampling_params_to_chat_request(
    messages: list[dict[str, Any]],
    model: str,
    sampling_params: dict[str, Any],
    **extra_fields: Any,
) -> ChatCompletionRequest:
    data: dict[str, Any] = {
        "messages": messages,
        "model": model,
        **sampling_params,
        **extra_fields,
    }
    if "max_new_tokens" in data and "max_tokens" not in data:
        data["max_tokens"] = data.pop("max_new_tokens")
    if "sampling_seed" in data and "seed" not in data:
        data["seed"] = data.pop("sampling_seed")
    if "min_new_tokens" in data and "min_tokens" not in data:
        data["min_tokens"] = data.pop("min_new_tokens")
    return ChatCompletionRequest.model_validate(data)


def openai_payload_to_generate(
    prompt_text: str, retrieve_input_ids: Callable[[str], list[int]], payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        chat_request = ChatCompletionRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid ChatCompletionRequest payload: {exc}") from exc
    return chat_request_to_generate_payload(prompt_text, chat_request, retrieve_input_ids)


def chat_request_to_generate_payload(
    prompt_text: str,
    chat_request: ChatCompletionRequest,
    retrieve_input_ids: Callable[[str], list[int]],
    model_generation_config: dict[str, Any] | None = None,
    tool_call_constraint: ToolCallConstraint | None = None,
) -> dict[str, Any]:
    stop_field = chat_request.stop
    if isinstance(stop_field, str):
        stop_list: list[str] = [stop_field]
    else:
        stop_list = stop_field or []

    sampling_params = chat_request.to_sampling_params(
        stop_list, model_generation_config or chat_request._DEFAULT_SAMPLING_PARAMS, tool_call_constraint
    )

    # Note: return_routed_experts. are not supported yet.
    # image_data is to be tested.
    return {
        "input_ids": retrieve_input_ids(prompt_text),
        "sampling_params": sampling_params,
        "return_logprob": True,
        "return_text_in_logprobs": True,  # Should keep this true to map token/logprob indices to text indices.
        "stream": False,
    }


def build_chat_response(
    model: str,
    output: dict[str, Any],
    prompt_tokens: int,
) -> dict[str, Any]:
    meta = output.get("meta_info", {}) or {}
    finish_reason = meta.get("finish_reason", {}).get("type")
    text = output.get("text", "")

    logprobs_field = None
    output_token_logprobs = meta.get("output_token_logprobs") or output.get("output_token_logprobs")
    if output_token_logprobs:
        token_ids = [item[1] for item in output_token_logprobs]
        token_logps = [item[0] for item in output_token_logprobs]
        tokens = [item[2] for item in output_token_logprobs]
        logprobs_field = LogProbs(
            text_offset=[0 for _ in token_ids],
            token_logprobs=token_logps,
            tokens=tokens,
            top_logprobs=[None for _ in token_ids],
        )
        completion_tokens = len(token_ids)
    else:
        completion_tokens = 0

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=text),
        logprobs=logprobs_field,
        finish_reason=finish_reason,
        matched_stop=meta.get("finish_reason", {}).get("matched"),
    )

    response = ChatCompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        model=model,
        choices=[choice],
        usage=usage,
        metadata=meta if meta else None,
    )
    return response.model_dump()
