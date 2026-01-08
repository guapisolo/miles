import types

import pytest
from eval_protocol import InitRequest
from eval_protocol.types.remote_rollout_processor import RolloutMetadata

from examples.adapter.math import agent


@pytest.mark.unit
def test_execute_agent_builds_payload(monkeypatch):
    captured = {}

    def fake_call_llm(request, messages):
        captured["request"] = request
        captured["messages"] = list(messages)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(model_dump=lambda: {"role": "assistant", "content": "ok"})
                )
            ]
        )

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)

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

    expected_messages = [msg.dump_mdoel_for_chat_completion_request() for msg in (request.messages or [])]
    expected_messages.append({"role": "assistant", "content": "ok"})

    assert captured["request"] is request
    assert captured["messages"] == expected_messages[:-1]
    assert result == expected_messages
