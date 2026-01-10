import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from miles.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware

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
    "top_k",
}


class OpenAICompatMiddleware(RadixTreeMiddleware):
    """Expose /v1/chat/completions using radix tree + tokenizer, leveraging RadixTreeMiddleware."""

    def __init__(self, app, *, router):
        super().__init__(app, router=router)
        self.chat_template_kwargs = getattr(self.args, "apply_chat_template_kwargs", None) or {}

    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/chat/completions":
            return await self.chat_completions(request)
        return await super().dispatch(request, call_next)

    def _split_openai_and_extra_params(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        completion_params = {k: v for k, v in payload.items() if k not in {"messages", "model"}}
        openai_params: dict[str, Any] = {}
        extra_body_params: dict[str, Any] = {}
        for key, value in completion_params.items():
            if key in OPENAI_COMPATIBLE_PARAMS:
                openai_params[key] = value
            else:
                extra_body_params[key] = value
        return openai_params, extra_body_params

    def _build_sampling_params(
        self, openai_params: dict[str, Any], extra_body_params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        sampling_params: dict[str, Any] = {}

        for key in ("temperature", "top_p", "top_k"):
            if key in openai_params:
                sampling_params[key] = openai_params.pop(key)

        stop = openai_params.pop("stop", None)
        if stop is not None:
            sampling_params["stop"] = stop

        max_tokens = openai_params.pop("max_tokens", None) or openai_params.pop("max_completion_tokens", None)
        if max_tokens is not None:
            sampling_params["max_new_tokens"] = max_tokens
        if "max_new_tokens" in extra_body_params and "max_new_tokens" not in sampling_params:
            sampling_params["max_new_tokens"] = extra_body_params.pop("max_new_tokens")

        openai_params.pop("stream", None)
        openai_params.pop("n", None)

        return sampling_params, extra_body_params

    def _build_generate_payload(
        self, prompt_text: str, openai_params: dict[str, Any], extra_body_params: dict[str, Any]
    ) -> dict[str, Any]:
        input_ids, _, _ = self.router.radix_tree.retrieve_from_text(prompt_text, return_logprob=True)
        sampling_params, remaining_extra = self._build_sampling_params(openai_params, extra_body_params)
        if openai_params:
            remaining_extra = {**openai_params, **remaining_extra}

        generate_payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "stream": False,
        }
        if remaining_extra:
            generate_payload["extra_body"] = remaining_extra
        return generate_payload

    def _build_chat_response(self, model: str, output: dict[str, Any]) -> dict[str, Any]:
        finish_reason = output.get("meta_info", {}).get("finish_reason", {}).get("type")
        text = output.get("text", "")
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
        }

    async def chat_completions(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):
            return JSONResponse(status_code=400, content={"error": "stream is not supported"})
        if payload.get("n", 1) != 1:
            return JSONResponse(status_code=400, content={"error": "only n=1 is supported"})

        if self.router.radix_tree is None:
            return JSONResponse(status_code=500, content={"error": "radix tree middleware is required"})

        prompt_text = self.tokenizer.apply_chat_template(
            payload["messages"],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

        openai_params, extra_body_params = self._split_openai_and_extra_params(payload)
        generate_payload = self._build_generate_payload(prompt_text, openai_params, extra_body_params)

        worker_url = self.router._use_url()
        url = f"{worker_url}/generate"
        try:
            response = await self.router.client.post(url, json=generate_payload)
            response.raise_for_status()
            output = response.json()
        finally:
            self.router._finish_url(worker_url)

        model = payload.get("model") or getattr(self.router.args, "hf_checkpoint", "default")
        return JSONResponse(content=self._build_chat_response(model, output))
