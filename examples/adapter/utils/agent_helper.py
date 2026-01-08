from typing import Any

from eval_protocol import InitRequest
from openai import OpenAI
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

OPENAI_COMPATIBLE_PARAMS = {
    "messages",
    "model",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "user",
    "tools",
    "tool_choice",
    "return_hidden_states",
    "reasoning_effort",
}


def build_openai_compatible_params(
    completion_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    openai_compatible_params: dict[str, Any] = {}
    extra_body_params: dict[str, Any] = {}
    for key, value in completion_params.items():
        if key in OPENAI_COMPATIBLE_PARAMS:
            openai_compatible_params[key] = value
        else:
            extra_body_params[key] = value
    return openai_compatible_params, extra_body_params


def _normalize_base_url(base_url: str | None) -> str:
    if not base_url:
        raise ValueError("model_base_url is required for agent execution")
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def call_llm(request: InitRequest, messages: list[dict[str, Any]]) -> ChatCompletionResponse:
    base_url = _normalize_base_url(request.model_base_url)
    completion_params = dict(request.completion_params or {})
    model = completion_params.pop("model", None) or "default"
    if "max_new_tokens" in completion_params and "max_tokens" not in completion_params:
        completion_params["max_tokens"] = completion_params.pop("max_new_tokens")
    openai_compatible_params, extra_body_params = build_openai_compatible_params(completion_params)

    payload = {
        "model": model,
        "messages": messages,  # notice: we do not use request.messages here
        "tools": request.tools,
        **openai_compatible_params,
        "extra_body": {**extra_body_params},
    }

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    json_response = client.chat.completions.with_raw_response.create(**payload)
    response = ChatCompletionResponse.model_validate(json_response.parse().model_dump())
    return response
