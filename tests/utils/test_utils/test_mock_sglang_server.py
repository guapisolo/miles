import asyncio

import httpx
import pytest

from miles.utils.http_utils import post
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, start_mock_server, start_mock_server_async


def test_basic_server_start_stop():
    server = MockSGLangServer(response_text="Test response", finish_reason="stop")
    try:
        server.start()
        assert server.port > 0
        assert f"http://{server.host}:{server.port}" == server.url
    finally:
        server.stop()


def test_generate_endpoint_basic():
    server = MockSGLangServer(response_text="Hello, world!", finish_reason="stop", prompt_tokens=5, cached_tokens=2)
    try:
        server.start()

        response = httpx.post(
            f"{server.url}/generate",
            json={
                "input_ids": [1, 2, 3, 4, 5],
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 10},
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "text" in data
        assert data["text"] == "Hello, world!"
        assert "meta_info" in data
        assert data["meta_info"]["finish_reason"]["type"] == "stop"
        assert data["meta_info"]["prompt_tokens"] == 5
        assert data["meta_info"]["cached_tokens"] == 2
        assert "completion_tokens" in data["meta_info"]

        assert len(server.requests) == 1
        assert server.requests[0]["input_ids"] == [1, 2, 3, 4, 5]
    finally:
        server.stop()


def test_finish_reason_stop():
    server = MockSGLangServer(response_text="Complete response", finish_reason="stop")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "stop"
        assert "length" not in data["meta_info"]["finish_reason"]
    finally:
        server.stop()


def test_finish_reason_length():
    server = MockSGLangServer(response_text="Truncated", finish_reason="length", completion_tokens=32)
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "length"
        assert data["meta_info"]["finish_reason"]["length"] == 32
    finally:
        server.stop()


def test_finish_reason_abort():
    server = MockSGLangServer(response_text="Aborted", finish_reason="abort")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "abort"
    finally:
        server.stop()


def test_return_logprob():
    server = MockSGLangServer(response_text="Test", finish_reason="stop")
    try:
        server.start()

        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "output_token_logprobs" in data["meta_info"]
        logprobs = data["meta_info"]["output_token_logprobs"]
        assert isinstance(logprobs, list)
        assert len(logprobs) > 0
        assert isinstance(logprobs[0], list)
        assert len(logprobs[0]) == 2
        assert isinstance(logprobs[0][0], float)
        assert isinstance(logprobs[0][1], int)
    finally:
        server.stop()


def test_return_routed_experts():
    server = MockSGLangServer(response_text="Test", finish_reason="stop")
    try:
        server.start()

        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3, 4, 5], "sampling_params": {}, "return_routed_experts": True},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "routed_experts" in data["meta_info"]
        routed_experts_b64 = data["meta_info"]["routed_experts"]
        assert isinstance(routed_experts_b64, str)
    finally:
        server.stop()


def test_request_recording():
    server = MockSGLangServer(response_text="Test", finish_reason="stop")
    try:
        server.start()

        request1 = {"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0.7}}
        request2 = {"input_ids": [4, 5, 6], "sampling_params": {"temperature": 0.9}, "return_logprob": True}

        httpx.post(f"{server.url}/generate", json=request1, timeout=5.0)
        httpx.post(f"{server.url}/generate", json=request2, timeout=5.0)

        assert len(server.requests) == 2
        assert server.requests[0] == request1
        assert server.requests[1] == request2

        server.clear_requests()
        assert len(server.requests) == 0
    finally:
        server.stop()


def test_weight_version():
    server = MockSGLangServer(response_text="Test", finish_reason="stop", weight_version="v1.0")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["weight_version"] == "v1.0"
    finally:
        server.stop()


def test_speculative_decoding_fields():
    server = MockSGLangServer(
        response_text="Test",
        finish_reason="stop",
        spec_accept_token_num=10,
        spec_draft_token_num=15,
        spec_verify_ct=5,
    )
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["spec_accept_token_num"] == 10
        assert data["meta_info"]["spec_draft_token_num"] == 15
        assert data["meta_info"]["spec_verify_ct"] == 5
    finally:
        server.stop()


def test_context_manager():
    with start_mock_server(response_text="Context test", finish_reason="stop") as server:
        assert server is not None
        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Context test"


@pytest.mark.asyncio
async def test_async_post():
    async with start_mock_server_async(response_text="Async test", finish_reason="stop") as server:
        url = f"{server.url}/generate"
        payload = {"input_ids": [1, 2, 3], "sampling_params": {}}

        response = await post(url, payload)
        assert response["text"] == "Async test"
        assert response["meta_info"]["finish_reason"]["type"] == "stop"
        assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_async_with_logprob():
    async with start_mock_server_async(response_text="Test response", finish_reason="stop", completion_tokens=2) as server:
        url = f"{server.url}/generate"
        payload = {"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True}

        response = await post(url, payload)
        assert "output_token_logprobs" in response["meta_info"]
        logprobs = response["meta_info"]["output_token_logprobs"]
        assert len(logprobs) == 2


@pytest.mark.asyncio
async def test_async_with_routed_experts():
    async with start_mock_server_async(response_text="Test", finish_reason="stop") as server:
        url = f"{server.url}/generate"
        payload = {"input_ids": [1, 2, 3, 4, 5], "sampling_params": {}, "return_routed_experts": True}

        response = await post(url, payload)
        assert "routed_experts" in response["meta_info"]
        routed_experts_b64 = response["meta_info"]["routed_experts"]
        assert isinstance(routed_experts_b64, str)
