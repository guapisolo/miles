from types import SimpleNamespace

import pytest
import requests
from transformers import AutoTokenizer

from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def router_env():
    def process_fn(_prompt: str) -> ProcessResult:
        return ProcessResult(text="ok", finish_reason="stop")

    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as backend:
        args = SimpleNamespace(
            miles_router_max_connections=10,
            miles_router_timeout=30,
            miles_router_middleware_paths=[],
            rollout_health_check_interval=60,
            miles_router_health_check_failure_threshold=3,
            hf_checkpoint=MODEL_NAME,
            cross_turn_token_out=True,
            inherit_last_assistant=False,
        )
        router = MilesRouter(args)

        port = find_available_port(31000)
        server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
        server.start()

        url = f"http://127.0.0.1:{port}"
        requests.post(f"{url}/add_worker", json={"url": backend.url})

        try:
            yield {"url": url, "backend": backend}
        finally:
            server.stop()


def _create_session(url: str) -> str:
    response = requests.post(f"{url}/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


def test_create_session_and_get_empty_records(router_env):
    url = router_env["url"]
    session_id = _create_session(url)

    response = requests.get(f"{url}/sessions/{session_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == session_id
    assert data["records"] == {
        "tokens": [],
        "token_ids": [],
        "log_probs": [],
        "loss_mask": [],
    }


def test_get_session_not_found(router_env):
    url = router_env["url"]
    response = requests.get(f"{url}/sessions/nonexistent")
    assert response.status_code == 404
    assert response.json()["error"] == "session not found"


def test_delete_session(router_env):
    url = router_env["url"]
    session_id = _create_session(url)

    delete_resp = requests.delete(f"{url}/sessions/{session_id}")
    assert delete_resp.status_code == 204
    assert delete_resp.text == ""

    missing_resp = requests.delete(f"{url}/sessions/{session_id}")
    assert missing_resp.status_code == 404
    assert missing_resp.json()["error"] == "session not found"


def test_proxy_session_not_found(router_env):
    url = router_env["url"]
    response = requests.post(
        f"{url}/sessions/nonexistent/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "session not found"


def test_proxy_inserts_input_ids_and_records_tokens(router_env):
    url = router_env["url"]
    backend = router_env["backend"]
    backend.reset_stats()

    session_id = _create_session(url)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]

    response = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": messages},
    )
    assert response.status_code == 200

    response_body = response.json()
    logprobs_content = response_body["choices"][0]["logprobs"]["content"]

    expected_prompt_ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    backend_payload = backend.request_log[-1]
    assert backend_payload["input_ids"] == expected_prompt_ids

    response_token_ids = [TOKENIZER.convert_tokens_to_ids(item["token"]) for item in logprobs_content]
    response_logprobs = [item["logprob"] for item in logprobs_content]

    get_resp = requests.get(f"{url}/sessions/{session_id}")
    assert get_resp.status_code == 200

    records = get_resp.json()["records"]
    expected_token_ids = expected_prompt_ids + response_token_ids
    assert records["token_ids"] == expected_token_ids
    assert records["tokens"] == TOKENIZER.convert_ids_to_tokens(expected_token_ids)
    assert records["log_probs"] == [0.0] * len(expected_prompt_ids) + response_logprobs
    assert records["loss_mask"] == [0] * len(expected_prompt_ids) + [1] * len(response_token_ids)
