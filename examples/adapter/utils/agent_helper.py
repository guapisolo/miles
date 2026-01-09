import asyncio
import logging
import time
from typing import Any

import httpx
from eval_protocol import InitRequest
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

logger = logging.getLogger(__name__)
_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_BASE_URL: str | None = None
_HTTP_TIMEOUT_SECONDS = 60.0
_HTTP_MAX_RETRIES = 60
_HTTP_RETRY_INTERVAL_SECONDS = 1.0
_HTTP_CLIENT_LOCK = asyncio.Lock()


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


HTTP_CLIENT_CONCURRENCY = 2048


async def _get_http_client(base_url: str) -> httpx.AsyncClient:
    global _HTTP_CLIENT, _HTTP_BASE_URL
    async with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None or _HTTP_BASE_URL != base_url:
            if _HTTP_CLIENT is not None:
                await _HTTP_CLIENT.aclose()
            _HTTP_BASE_URL = base_url
            _HTTP_CLIENT = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=HTTP_CLIENT_CONCURRENCY),
                timeout=httpx.Timeout(None),
            )
        return _HTTP_CLIENT


async def call_llm(request: InitRequest, messages: list[dict[str, Any]]) -> ChatCompletionResponse:
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

    url = f"{base_url}/chat/completions"
    client = await _get_http_client(base_url)
    response = None
    for attempt in range(1, _HTTP_MAX_RETRIES + 1):
        try:
            http_response = await client.post(url, json=payload)
            http_response.raise_for_status()
            response = http_response.json()
            break
        except httpx.RequestError:
            if attempt == _HTTP_MAX_RETRIES:
                raise
            logger.warning("OpenAI API error, retrying... (attempt %s/%s)", attempt, _HTTP_MAX_RETRIES)
            time.sleep(_HTTP_RETRY_INTERVAL_SECONDS)
    response = ChatCompletionResponse.model_validate(response)
    return response
