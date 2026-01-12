from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware
from miles.utils.openai_utils import build_chat_response, chat_request_to_generate_payload


class OpenAICompatMiddleware(RadixTreeMiddleware):
    """Expose /v1/chat/completions using radix tree + tokenizer, leveraging RadixTreeMiddleware."""

    def __init__(self, app, *, router):
        super().__init__(app, router=router)
        self.chat_template_kwargs = getattr(self.args, "apply_chat_template_kwargs", None) or {}

    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/chat/completions":
            return await self.chat_completions(request)
        return await super().dispatch(request, call_next)

    def _build_generate_payload(self, prompt_text: str, chat_request: ChatCompletionRequest) -> dict[str, Any]:
        return chat_request_to_generate_payload(
            prompt_text,
            chat_request,
            self.tokenizer,
        )

    async def chat_completions(self, request: Request):
        payload = await request.json()
        try:
            chat_request = ChatCompletionRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=400, content={"error": f"invalid request: {exc}"})

        if chat_request.stream:
            return JSONResponse(status_code=400, content={"error": "stream is not supported"})
        if chat_request.n != 1:
            return JSONResponse(status_code=400, content={"error": "only n=1 is supported"})

        if self.router.radix_tree is None:
            return JSONResponse(status_code=500, content={"error": "radix tree middleware is required"})

        prompt_text = self.tokenizer.apply_chat_template(
            chat_request.messages,
            tools=chat_request.tools,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

        generate_payload = self._build_generate_payload(prompt_text, chat_request)

        worker_url = self.router._use_url()
        url = f"{worker_url}/generate"
        try:
            response = await self.router.client.post(url, json=generate_payload)
            response.raise_for_status()
            output = response.json()
        finally:
            self.router._finish_url(worker_url)

        model = chat_request.model or getattr(self.router.args, "hf_checkpoint", "default")
        prompt_tokens = len(generate_payload.get("input_ids") or [])
        return JSONResponse(content=build_chat_response(model, output, prompt_tokens))
