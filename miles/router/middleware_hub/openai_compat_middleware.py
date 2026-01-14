# Copied and adapted from https://github.com/sgl-project/sglang

import json
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.reasoning_parser import ReasoningParser

from miles.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware
from miles.utils.openai_utils import chat_request_to_generate_payload

logger = logging.getLogger(__name__)


class OpenAICompatMiddleware(RadixTreeMiddleware):
    """Expose /v1/chat/completions using radix tree + tokenizer, leveraging RadixTreeMiddleware."""

    def __init__(self, app, *, router):
        super().__init__(app, router=router)
        self.chat_template_kwargs = getattr(self.args, "apply_chat_template_kwargs", None) or {}

        assert self.args.sglang_reasoning_parser, "sglang_reasoning_parser is required"
        self.reasoning_parser = ReasoningParser(model_type=self.args.sglang_reasoning_parser, stream_reasoning=False)

    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/chat/completions":
            return await self.chat_completions(request, call_next)
        return await super().dispatch(request, call_next)

    def _build_generate_payload(self, prompt_text: str, chat_request: ChatCompletionRequest) -> dict[str, Any]:
        return chat_request_to_generate_payload(
            prompt_text,
            chat_request,
            self.tokenizer,
        )

    async def chat_completions(self, request: Request, call_next):
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

        # Route chat request through RadixTreeMiddleware while proxying to worker /generate.
        request._json = generate_payload

        async def _call_generate(inner_request: Request):
            worker_url = self.router._use_url()
            url = f"{worker_url}/generate"
            try:
                response = await self.router.client.post(url, json=await inner_request.json())
                response.raise_for_status()
                return JSONResponse(content=response.json())
            finally:
                self.router._finish_url(worker_url)

        response = await super()._generate(request, _call_generate)
        output = json.loads(response.body.decode("utf-8"))

        # model = chat_request.model or getattr(self.router.args, "hf_checkpoint", "default")
        prompt_tokens = len(generate_payload.get("input_ids") or [])
        # return JSONResponse(content=build_chat_response_deprecated(chat_request.model, output, prompt_tokens))
        return JSONResponse(content=self._build_chat_response(chat_request, output, prompt_tokens))

    # # Limitations compared to sglang serving chat
    # # 1 choice, non-streaming, only token logprobs (no top_logprobs)
    # def _build_chat_response(
    #     self,
    #     chat_request: ChatCompletionRequest,
    #     output: dict[str, Any],
    #     prompt_tokens: int,
    # ) -> dict[str, Any]:
    #     choice_logprobs = self.process_response_logprobs(output, prompt_tokens)

    #     reasoning_content, content = self.handle_reasoning_content(output["text"])

    #     history_tool_calls_cnt = OpenAIServingChat.get_history_tool_calls_cnt(chat_request)
    #     tool_calls, content, finish_reason = OpenAIServingChat.process_tool_calls(
    #         self.args.sglang_tool_call_parser,
    #         content or "",
    #         chat_request.tools or [],
    #         output["meta_info"].get("finish_reason"),
    #         chat_request.tool_choice,
    #         history_tool_calls_cnt,
    #     )

    #     choice = ChatCompletionResponseChoice(
    #         index=0,
    #         message=ChatMessage(
    #             role="assistant",
    #             content=content if content else None,
    #             tool_calls=tool_calls,
    #             reasoning_content=reasoning_content if reasoning_content else None,
    #         ),
    #         logprobs=choice_logprobs,
    #         finish_reason=finish_reason["type"] if finish_reason else None,
    #         matched_stop=(finish_reason["matched"] if finish_reason and "matched" in finish_reason else None),
    #     )

    #     usage = UsageProcessor.calculate_response_usage(
    #         [output],
    #         n_choices=1,
    #         enable_cache_report=self.args.sglang_enable_cache_report,
    #     )

    #     resp = ChatCompletionResponse(
    #         id=output["meta_info"]["id"],
    #         created=int(time.time()),
    #         model=model,
    #         choices=choices,
    #         usage=usage,
    #         metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
    #     )

    #     return resp.model_dump()

    # def process_response_logprobs(
    #     self,
    #     output: dict[str, Any],
    #     prompt_tokens: int,
    # ) -> LogProbs | None:
    #     """
    #     This log probs is just info to users, we don't use it for training.
    #     """
    #     output_token_logprobs = output["meta_info"].get("output_token_logprobs")
    #     if output_token_logprobs:
    #         token_ids = [item[1] for item in output_token_logprobs]
    #         token_logps = [item[0] for item in output_token_logprobs]
    #         tokens = [item[2] for item in output_token_logprobs]
    #         logprobs_field = LogProbs(
    #             text_offset=[0 for _ in token_ids],
    #             token_logprobs=token_logps,
    #             tokens=tokens,
    #             top_logprobs=[None for _ in token_ids],
    #         )
    #         return logprobs_field
    #     else:
    #         return None

    # def handle_reasoning_content(self, text: str) -> tuple[str | None, str | None]:
    #     """
    #     Handle reasoning content in the output.
    #     """
    #     reasoning_content, content = self.reasoning_parser.parse_non_stream(text)
    #     return reasoning_content, content
