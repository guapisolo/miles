import asyncio
import json
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.r3_codec import encode_r3, make_logp_tokens, make_r3, parse_signature, seed_from_signature
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@dataclass(frozen=True)
class ProcessResultMetaInfo:
    weight_version: str | None = None
    routed_experts: str | None = None
    spec_accept_token_num: int | None = None
    spec_draft_token_num: int | None = None
    spec_verify_ct: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ProcessResult:
    text: str
    finish_reason: str = "stop"
    cached_tokens: int = 0
    meta_info: ProcessResultMetaInfo = ProcessResultMetaInfo()


ProcessFn = Callable[[str], ProcessResult]


@dataclass(frozen=True)
class StressMockConfig:
    """Deterministic stress-test mode for ``MockSGLangServer``.

    When enabled, the mock bypasses ``process_fn`` and the tokenizer for chat
    completions, and instead emits a fixed-size synthetic response whose
    contents are uniquely determined by the ``[STRESS sid=<sid> turn=<n>]``
    signature embedded in the LAST user message's content. Mock and verifier
    derive the same ``(sid, turn)`` -> seed -> payload, so byte-identity
    round-trip becomes provable.

    Knobs (all required when ``enabled`` is True):

    - ``output_tokens``: number of generated tokens per response. Drives the
      length of ``output_token_logprobs`` and the leading axis of R3.
    - ``inject_routed_experts``: when True, attaches a base64 R3 blob to
      ``choice.meta_info.routed_experts`` shaped
      ``(output_tokens, r3_num_layers, r3_topk)``.
    - ``r3_num_layers`` / ``r3_topk``: per-token R3 shape. The product is the
      ``num_layers * moe_router_topk`` value the verifier passes back to
      ``decode_r3``. Real models: Qwen3-30B-A3B ~ 48 x 8 = 384 int32/token.
    - ``echo_signature``: when True, the same ``(sid, turn)`` is also surfaced
      back into ``choice.meta_info.stress_signature`` so the verifier can do
      a redundant request/response signature consistency check.
    - ``canonical_json``: when True, the response is rendered with a
      canonical JSON serializer (``separators=(",", ":")``, ``allow_nan=False``,
      ``ensure_ascii=True``); NaN logp values are rejected at the wire edge.
    """

    enabled: bool = False
    output_tokens: int = 0
    inject_routed_experts: bool = False
    r3_num_layers: int = 0
    r3_topk: int = 0
    echo_signature: bool = True
    canonical_json: bool = True


class CanonicalJSONResponse(JSONResponse):
    """JSON response rendered with deterministic options:

    - ``separators=(",", ":")``: no whitespace, stable byte length
    - ``allow_nan=False``: NaN / Inf logp at the wire edge is a hard error
    - ``ensure_ascii=True``: defensive against locale-driven encoding drift
    """

    media_type = "application/json"

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")


class MockSGLangServer:
    def __init__(
        self,
        model_name: str,
        process_fn: ProcessFn,
        host: str,
        port: int,
        latency: float = 0.0,
        chat_template_path: str | None = None,
        stress_config: StressMockConfig | None = None,
    ):
        self.tokenizer = load_tokenizer(model_name, chat_template_path=chat_template_path, trust_remote_code=True)
        self.process_fn = process_fn
        self.host = host
        self.port = port or find_available_port(30000)
        self.latency = latency
        self.stress_config = stress_config or StressMockConfig()

        self.app = FastAPI()
        self._server: UvicornThreadServer | None = None

        self.request_log: list[dict] = []
        self._concurrency = Counter()

        self._setup_routes()

    @property
    def max_concurrent(self) -> int:
        return self._concurrency.max_value

    def reset_stats(self):
        self.request_log.clear()
        self._concurrency.reset()

    def start(self):
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._server.start()

    def stop(self):
        if self._server is not None:
            self._server.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: Request):
            return await self._handle_generate_like_request(request, self._compute_generate_response)

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            # Dispatch to stress-mode response generator when configured; both share
            # the same _handle_generate_like_request wrapper for concurrency tracking,
            # latency injection, and request logging.
            compute_fn = (
                self._compute_chat_completions_response_stress
                if self.stress_config.enabled
                else self._compute_chat_completions_response
            )
            return await self._handle_generate_like_request(request, compute_fn)

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/abort_request")
        async def abort_request(_request: Request):
            return JSONResponse(content={"status": "ok"})

    async def _handle_generate_like_request(self, request: Request, compute_fn: Callable[[dict], dict]):
        payload = await request.json()
        self.request_log.append(payload)
        with self._concurrency.track():
            if self.latency > 0:
                await asyncio.sleep(self.latency)
            response = compute_fn(payload)
        # Canonical JSON wire format only when stress mode is on; legacy callers
        # keep getting the default FastAPI JSONResponse behavior.
        response_cls = (
            CanonicalJSONResponse if self.stress_config.enabled and self.stress_config.canonical_json else JSONResponse
        )
        return response_cls(content=response)

    def _compute_generate_response(self, payload: dict) -> dict:
        assert payload.get("return_logprob", True) is True, "MockSGLangServer requires return_logprob=True"
        input_ids = payload.get("input_ids", [])

        prompt_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        prompt_tokens = len(input_ids)
        completion_tokens = len(output_ids)

        finish_reason_dict = {"type": process_result.finish_reason}
        if process_result.finish_reason == "length":
            finish_reason_dict["length"] = completion_tokens

        output_token_logprobs = [(-1 / 128 * i, token_id) for i, token_id in enumerate(output_ids)]

        meta_info = {
            "finish_reason": finish_reason_dict,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": process_result.cached_tokens,
            "completion_tokens": completion_tokens,
            "output_token_logprobs": output_token_logprobs,
            **process_result.meta_info.to_dict(),
        }

        return {"text": process_result.text, "meta_info": meta_info}

    def _compute_chat_completions_response(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        tools = payload.get("tools")

        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        prompt_ids = None
        if payload.get("return_prompt_token_ids"):
            input_ids = payload.get("input_ids")
            if input_ids is not None:
                prompt_ids = list(input_ids)
            else:
                prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        logprobs_content = [
            {
                "token": self.tokenizer.convert_ids_to_tokens(tid),
                "token_id": tid,
                "logprob": -1 / 128 * i,
            }
            for i, tid in enumerate(output_ids)
        ]

        finish_reason = process_result.finish_reason
        tool_calls = None
        if tools and finish_reason == "stop":
            parser = FunctionCallParser(
                tools=TypeAdapter(list[Tool]).validate_python(tools),
                tool_call_parser="qwen25",
            )
            message_content, parsed_calls = parser.parse_non_stream(process_result.text)
            if parsed_calls:
                finish_reason = "tool_calls"
                tool_calls = [
                    {
                        "id": f"call{i:05d}",
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.parameters or "{}"},
                    }
                    for i, call in enumerate(parsed_calls)
                ]
        else:
            message_content = process_result.text

        output_token_logprobs = [(-1 / 128 * i, tid) for i, tid in enumerate(output_ids)]

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls,
            },
            "logprobs": {"content": logprobs_content},
            "finish_reason": finish_reason,
            "meta_info": {
                "output_token_logprobs": output_token_logprobs,
                "completion_tokens": len(output_ids),
                **process_result.meta_info.to_dict(),
            },
        }
        if prompt_ids is not None:
            choice["prompt_token_ids"] = prompt_ids

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [choice],
        }

    def _compute_chat_completions_response_stress(self, payload: dict) -> dict:
        """Stress-mode response generator. Bypasses ``process_fn`` and the
        tokenizer; derives the entire payload deterministically from the
        ``[STRESS sid=<sid> turn=<n>]`` signature in the last user message.

        Required wire-shape guarantees (every consumer downstream depends on
        these):

        - ``choice.meta_info.output_token_logprobs`` is exactly
          ``self.stress_config.output_tokens`` entries long.
        - Each entry is a JSON list ``[logp_float, token_id_int]``.
        - ``logp`` values are drawn from a ``np.float32`` Uniform(-20, 0)
          distribution so the verifier's ``struct.pack("<f", ...)`` byte
          comparison is bit-exact across the JSON round trip.
        - ``choice.meta_info.completion_tokens`` equals
          ``len(output_token_logprobs)``; the session server's split-lock
          handler raises ``UpstreamResponseError`` if these disagree.
        - ``choice.meta_info.routed_experts`` (when injected) is base64 ASCII
          of an ``np.int32`` buffer of shape
          ``(output_tokens, r3_num_layers, r3_topk)`` as produced by
          :mod:`miles.utils.test_utils.r3_codec`.
        """
        cfg = self.stress_config
        messages = payload.get("messages", [])
        if not messages:
            raise ValueError("stress mock requires at least one message in the request payload")
        last_content = messages[-1].get("content", "") or ""
        sig = parse_signature(last_content)
        if sig is None:
            raise ValueError(
                "stress mock could not find [STRESS sid=... turn=...] in messages[-1].content; "
                f"observed prefix: {last_content[:120]!r}"
            )
        sid, turn = sig
        seed = seed_from_signature(sid, turn)

        logps, tokens = make_logp_tokens(seed, cfg.output_tokens)
        # Python floats / ints, not numpy scalars, so the canonical JSON encoder
        # emits portable wire forms.
        output_token_logprobs = [[float(lp), int(tid)] for lp, tid in zip(logps, tokens, strict=True)]

        meta_info: dict = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": cfg.output_tokens,
        }
        if cfg.echo_signature:
            meta_info["stress_signature"] = {"sid": sid, "turn": turn}
        if cfg.inject_routed_experts:
            r3_arr = make_r3(seed, cfg.output_tokens, cfg.r3_num_layers, cfg.r3_topk)
            meta_info["routed_experts"] = encode_r3(r3_arr)

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"[STRESS_ECHO sid={sid} turn={turn}]" if cfg.echo_signature else "",
                "tool_calls": None,
            },
            "logprobs": {"content": []},
            "finish_reason": "stop",
            "meta_info": meta_info,
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-stress",
            "choices": [choice],
        }


class Counter:
    def __init__(self):
        self._current = 0
        self._max = 0

    @property
    def max_value(self) -> int:
        return self._max

    def reset(self):
        self._current = 0
        self._max = 0

    @contextmanager
    def track(self):
        self._current += 1
        self._max = max(self._max, self._current)
        try:
            yield
        finally:
            self._current -= 1


def default_process_fn(prompt: str) -> ProcessResult:
    match = re.search(r"What is 1\+(\d+)\?", prompt)
    if match:
        num = int(match.group(1))
        ans = 1 + num
        return ProcessResult(text=f"\\boxed{{{ans}}}", finish_reason="stop")
    return ProcessResult(text="I don't understand.", finish_reason="stop")


@contextmanager
def with_mock_server(
    model_name: str = "Qwen/Qwen3-0.6B",
    process_fn: ProcessFn = default_process_fn,
    host: str = "127.0.0.1",
    port: int | None = None,
    latency: float = 0.0,
):
    server = MockSGLangServer(
        model_name=model_name,
        process_fn=process_fn,
        host=host,
        port=port,
        latency=latency,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
