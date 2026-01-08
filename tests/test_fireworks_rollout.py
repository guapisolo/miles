import asyncio
from argparse import Namespace

import pytest

from examples.adapter import fireworks_rollout
from miles.utils.types import Sample


class DummyMaskGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_loss_mask(self, messages: list[dict[str, str]], tools: list[dict[str, object]] | None = None):
        self.calls.append({"messages": messages, "tools": tools})
        return [11, 12, 13, 14], [0, 1, 1, 1]

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return [2]


class DummyState:
    def __init__(self, _args: Namespace) -> None:
        self.tokenizer = "dummy-tokenizer"


@pytest.mark.unit
def test_fireworks_rollout_generate_populates_sample(monkeypatch):
    captured: dict[str, object] = {}
    dummy_mask = DummyMaskGenerator()

    async def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "status": "success",
            "result": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
        }

    async def fake_custom_reward(_args: Namespace, _sample: Sample) -> float:
        return 0.5

    def fake_get_tokenizer_and_mask_generator(_args: Namespace, _state: DummyState):
        return None, dummy_mask

    monkeypatch.setattr(fireworks_rollout, "GenerateState", DummyState)
    monkeypatch.setattr(fireworks_rollout, "post", fake_post)
    monkeypatch.setattr(fireworks_rollout, "_get_tokenizer_and_mask_generator", fake_get_tokenizer_and_mask_generator)
    monkeypatch.setattr(fireworks_rollout, "custom_reward", fake_custom_reward)

    args = Namespace(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=3000,
        model_name="qwen3-4b",
        hf_checkpoint="qwen3-4b",
        loss_mask_type="qwen",
        agent_base_url="http://127.0.0.1:8000",
    )
    tools = [{"type": "function", "function": {"name": "noop"}}]
    sample = Sample(prompt="hi", metadata={"tools": tools}, status=Sample.Status.PENDING)

    result = asyncio.run(fireworks_rollout.generate(args, sample, {"temperature": 0.2}))

    assert captured["url"] == "http://127.0.0.1:8000/init"
    assert captured["payload"]["completion_params"]["model"] == "qwen3-4b"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert dummy_mask.calls[0]["tools"] == tools
    assert result.response == "ok"
    assert result.tokens == [11, 12, 13, 14]
    assert result.response_length == 2
    assert result.loss_mask == [1, 1]
    assert result.reward == 0.5
    assert result.status == Sample.Status.PENDING


@pytest.mark.unit
def test_fireworks_rollout_helpers():
    args = Namespace(sglang_router_ip="127.0.0.1", sglang_router_port=3000)
    metadata = fireworks_rollout._blank_rollout_metadata()

    assert fireworks_rollout._get_sglang_base_url(args) == "http://127.0.0.1:3000/v1"
    assert fireworks_rollout._coerce_messages("hi") == [{"role": "user", "content": "hi"}]
    assert fireworks_rollout._coerce_messages([{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": "hi"}
    ]
    assert metadata.invocation_id == ""
    assert metadata.rollout_id == ""


@pytest.mark.unit
def test_fireworks_rollout_generate_disallows_partial_rollout():
    args = Namespace(
        partial_rollout=True,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=3000,
        model_name="qwen3-4b",
        hf_checkpoint="qwen3-4b",
        loss_mask_type="qwen",
        agent_base_url="http://127.0.0.1:8000",
    )
    sample = Sample(prompt=[{"role": "user", "content": "hi"}], metadata={}, status=Sample.Status.PENDING)

    with pytest.raises(AssertionError):
        asyncio.run(fireworks_rollout.generate(args, sample, {"max_new_tokens": 8}))


@pytest.mark.unit
def test_fireworks_rollout_generate_raises_on_failure(monkeypatch):
    async def fake_post(_url, _payload):
        return {"status": "error", "error": "bad request"}

    monkeypatch.setattr(fireworks_rollout, "GenerateState", DummyState)
    monkeypatch.setattr(fireworks_rollout, "post", fake_post)

    args = Namespace(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=3000,
        model_name="qwen3-4b",
        hf_checkpoint="qwen3-4b",
        loss_mask_type="qwen",
        agent_base_url="http://127.0.0.1:8000",
    )
    sample = Sample(prompt="hi", metadata={}, status=Sample.Status.PENDING)

    with pytest.raises(ValueError, match="Failed to initialize agent"):
        asyncio.run(fireworks_rollout.generate(args, sample, {"temperature": 0.2}))
