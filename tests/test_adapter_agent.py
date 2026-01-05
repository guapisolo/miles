import types

import pytest
from eval_protocol import InitRequest
from eval_protocol.types.remote_rollout_processor import RolloutMetadata

from examples.adapter.gsm8k import agent


@pytest.mark.unit
def test_execute_agent_builds_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json

        response = types.SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        return response

    monkeypatch.setattr(agent.httpx, "post", fake_post)

    request = InitRequest(
        completion_params={"model": "qwen3-4b", "temperature": 0.1, "max_new_tokens": 16},
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "noop"}}],
        model_base_url="http://localhost:8001",
        metadata=RolloutMetadata(
            invocation_id="",
            experiment_id="",
            rollout_id="",
            run_id="",
            row_id="",
        ),
    )

    result = agent.execute_agent(request)

    assert captured["url"] == "http://localhost:8001/v1/chat/completions"
    assert captured["json"]["model"] == "qwen3-4b"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["json"]["max_tokens"] == 16
    assert "max_new_tokens" not in captured["json"]
    assert captured["json"]["tools"] == [{"type": "function", "function": {"name": "noop"}}]
    assert result["messages"][-1]["content"] == "ok"
