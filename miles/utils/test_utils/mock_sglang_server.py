import asyncio
import base64
import random
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.utils.http_utils import find_available_port


class MockSGLangServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        response_text: str = "Hello, world!",
        finish_reason: str = "stop",
        prompt_tokens: int = 5,
        cached_tokens: int = 0,
        completion_tokens: int | None = None,
        weight_version: str | None = None,
        spec_accept_token_num: int = 0,
        spec_draft_token_num: int = 0,
        spec_verify_ct: int = 0,
    ):
        self.host = host
        self.port = port or find_available_port(30000)
        self.response_text = response_text
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.completion_tokens = completion_tokens or len(response_text.split())
        self.weight_version = weight_version
        self.spec_accept_token_num = spec_accept_token_num
        self.spec_draft_token_num = spec_draft_token_num
        self.spec_verify_ct = spec_verify_ct

        self.requests: list[dict[str, Any]] = []
        self.app = FastAPI()
        self.server: uvicorn.Server | None = None
        self.server_thread: threading.Thread | None = None

        @self.app.post("/generate")
        async def generate(request: Request):
            payload = await request.json()
            self.requests.append(payload)

            return_logprob = payload.get("return_logprob", False)
            return_routed_experts = payload.get("return_routed_experts", False)
            input_ids = payload.get("input_ids", [])

            response = {
                "text": self.response_text,
                "meta_info": {
                    "finish_reason": {"type": self.finish_reason},
                    "prompt_tokens": self.prompt_tokens,
                    "cached_tokens": self.cached_tokens,
                    "completion_tokens": self.completion_tokens,
                },
            }

            if self.finish_reason == "length":
                response["meta_info"]["finish_reason"]["length"] = self.completion_tokens

            if return_logprob:
                num_tokens = self.completion_tokens
                output_token_logprobs = [
                    (random.uniform(-10.0, -0.1), random.randint(1, 50000)) for _ in range(num_tokens)
                ]
                response["meta_info"]["output_token_logprobs"] = output_token_logprobs

            if return_routed_experts:
                num_layers = 32
                moe_router_topk = 2
                total_tokens = len(input_ids) + self.completion_tokens if input_ids else self.prompt_tokens + self.completion_tokens
                num_tokens_for_routing = total_tokens - 1
                routed_experts_array = np.random.randint(0, 8, size=(num_tokens_for_routing, num_layers, moe_router_topk), dtype=np.int32)
                routed_experts_b64 = base64.b64encode(routed_experts_array.tobytes()).decode("ascii")
                response["meta_info"]["routed_experts"] = routed_experts_b64

            if self.weight_version is not None:
                response["meta_info"]["weight_version"] = self.weight_version

            if self.spec_accept_token_num > 0 or self.spec_draft_token_num > 0 or self.spec_verify_ct > 0:
                response["meta_info"]["spec_accept_token_num"] = self.spec_accept_token_num
                response["meta_info"]["spec_draft_token_num"] = self.spec_draft_token_num
                response["meta_info"]["spec_verify_ct"] = self.spec_verify_ct

            return JSONResponse(content=response)

    def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)

        def run_server():
            asyncio.run(self.server.serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        import time

        for _ in range(50):
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Failed to start server on {self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def clear_requests(self):
        self.requests.clear()


@contextmanager
def start_mock_server(
    host: str = "127.0.0.1",
    port: int | None = None,
    response_text: str = "Hello, world!",
    finish_reason: str = "stop",
    **kwargs,
):
    server = MockSGLangServer(
        host=host, port=port, response_text=response_text, finish_reason=finish_reason, **kwargs
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()


@asynccontextmanager
async def start_mock_server_async(
    host: str = "127.0.0.1",
    port: int | None = None,
    response_text: str = "Hello, world!",
    finish_reason: str = "stop",
    **kwargs,
):
    server = MockSGLangServer(
        host=host, port=port, response_text=response_text, finish_reason=finish_reason, **kwargs
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
