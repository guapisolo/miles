from types import SimpleNamespace

import pytest
import requests
from transformers import AutoTokenizer

from miles.rollout.generate_utils.tokenize_utils import tokenize_messages
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
            cross_turn_token_out=False,
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


def _extract_response_tokens(response_body: dict) -> tuple[list[int], list[float], list[str]]:
    logprobs_content = response_body["choices"][0]["logprobs"]["content"]
    token_ids = [item.get("token_id", TOKENIZER.convert_tokens_to_ids(item["token"])) for item in logprobs_content]
    logprobs = [item["logprob"] for item in logprobs_content]
    tokens = [item["token"] for item in logprobs_content]
    return token_ids, logprobs, tokens


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

    expected_prompt_ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    backend_payload = backend.request_log[-1]
    assert backend_payload["input_ids"] == expected_prompt_ids

    response_token_ids, response_logprobs, response_tokens = _extract_response_tokens(response_body)

    get_resp = requests.get(f"{url}/sessions/{session_id}")
    assert get_resp.status_code == 200

    records = get_resp.json()["records"]
    expected_token_ids = expected_prompt_ids + response_token_ids
    assert records["token_ids"] == expected_token_ids
    assert records["tokens"] == TOKENIZER.convert_ids_to_tokens(expected_prompt_ids) + response_tokens
    assert records["log_probs"] == [0.0] * len(expected_prompt_ids) + response_logprobs
    assert records["loss_mask"] == [0] * len(expected_prompt_ids) + [1] * len(response_token_ids)


def test_proxy_preserves_input_ids_when_provided(router_env):
    url = router_env["url"]
    backend = router_env["backend"]
    backend.reset_stats()

    session_id = _create_session(url)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    base_prompt_ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    custom_input_ids = base_prompt_ids + [base_prompt_ids[-1]]

    response = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": messages, "input_ids": custom_input_ids},
    )
    assert response.status_code == 200

    backend_payload = backend.request_log[-1]
    assert backend_payload["input_ids"] == custom_input_ids

    response_body = response.json()
    response_token_ids, response_logprobs, response_tokens = _extract_response_tokens(response_body)

    get_resp = requests.get(f"{url}/sessions/{session_id}")
    records = get_resp.json()["records"]
    assert records["token_ids"] == response_token_ids
    assert records["tokens"] == response_tokens
    assert records["log_probs"] == response_logprobs
    assert records["loss_mask"] == [1] * len(response_token_ids)


def test_proxy_multi_turn_second_call_uses_only_new_messages(router_env):
    url = router_env["url"]
    backend = router_env["backend"]
    backend.reset_stats()

    session_id = _create_session(url)
    messages_turn1 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    response1 = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": messages_turn1},
    )
    assert response1.status_code == 200

    messages_turn2 = [{"role": "user", "content": "next"}]
    response2 = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": messages_turn2},
    )
    assert response2.status_code == 200

    expected_prompt_ids = tokenize_messages(messages_turn2, TOKENIZER, add_generation_prompt=True)
    backend_payload = backend.request_log[-1]
    assert backend_payload["input_ids"] == expected_prompt_ids

    response2_body = response2.json()
    response2_token_ids, response2_logprobs, response2_tokens = _extract_response_tokens(response2_body)

    get_resp = requests.get(f"{url}/sessions/{session_id}")
    records = get_resp.json()["records"]
    expected_token_ids = expected_prompt_ids + response2_token_ids
    assert records["token_ids"] == expected_token_ids
    assert records["tokens"] == TOKENIZER.convert_ids_to_tokens(expected_prompt_ids) + response2_tokens
    assert records["log_probs"] == [0.0] * len(expected_prompt_ids) + response2_logprobs
    assert records["loss_mask"] == [0] * len(expected_prompt_ids) + [1] * len(response2_token_ids)


def test_proxy_third_call_reuses_first_turn_prefix(router_env):
    url = router_env["url"]
    backend = router_env["backend"]
    backend.reset_stats()

    session_id = _create_session(url)
    messages_turn1 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    response1 = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": messages_turn1},
    )
    assert response1.status_code == 200

    response1_body = response1.json()
    response1_token_ids, _, _ = _extract_response_tokens(response1_body)
    prompt1_ids = TOKENIZER.apply_chat_template(messages_turn1, tokenize=True, add_generation_prompt=True)

    response2 = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "next"}]},
    )
    assert response2.status_code == 200

    assistant_message = response1_body["choices"][0]["message"]
    messages_turn3 = [{"role": "system", "content": "sys"}]
    response3 = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": messages_turn3},
    )
    assert response3.status_code == 200

    remain_messages = [messages_turn1[1], assistant_message]
    expected_prompt_ids = (
        prompt1_ids
        + response1_token_ids
        + tokenize_messages(
            remain_messages,
            TOKENIZER,
            add_generation_prompt=True,
        )
    )
    backend_payload = backend.request_log[-1]
    assert backend_payload["input_ids"] == expected_prompt_ids

    response3_body = response3.json()
    response3_token_ids, response3_logprobs, response3_tokens = _extract_response_tokens(response3_body)

    get_resp = requests.get(f"{url}/sessions/{session_id}")
    records = get_resp.json()["records"]
    expected_token_ids = expected_prompt_ids + response3_token_ids
    assert records["token_ids"] == expected_token_ids
    assert records["tokens"] == TOKENIZER.convert_ids_to_tokens(expected_prompt_ids) + response3_tokens
    assert records["log_probs"] == [0.0] * len(expected_prompt_ids) + response3_logprobs
    assert records["loss_mask"] == [0] * len(expected_prompt_ids) + [1] * len(response3_token_ids)


def test_proxy_respects_token_id_in_logprobs(router_env):
    url = router_env["url"]
    backend = router_env["backend"]
    backend.reset_stats()

    original_compute = backend._compute_chat_completions_response

    def _custom_compute(payload: dict) -> dict:
        response = original_compute(payload)
        for idx, item in enumerate(response["choices"][0]["logprobs"]["content"]):
            item["token_id"] = 900000 + idx
        return response

    backend._compute_chat_completions_response = _custom_compute
    try:
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
        custom_ids, response_logprobs, response_tokens = _extract_response_tokens(response_body)
        prompt_ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

        get_resp = requests.get(f"{url}/sessions/{session_id}")
        records = get_resp.json()["records"]
        expected_token_ids = prompt_ids + custom_ids
        assert records["token_ids"] == expected_token_ids
        assert records["tokens"] == TOKENIZER.convert_ids_to_tokens(prompt_ids) + response_tokens
        assert records["log_probs"] == [0.0] * len(prompt_ids) + response_logprobs
        assert records["loss_mask"] == [0] * len(prompt_ids) + [1] * len(custom_ids)
    finally:
        backend._compute_chat_completions_response = original_compute
