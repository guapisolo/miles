import asyncio
from argparse import Namespace

import pytest

from examples.adapter import fireworks_rollout
from miles.utils.types import Sample


class _DummyMaskGenerator:
    def get_loss_mask(self, messages):
        return [101, 102, 103], [0, 1, 1]

    def get_response_lengths(self, loss_masks):
        return [2]


@pytest.mark.unit
def test_fireworks_rollout_generate_builds_sample(monkeypatch):
    async def fake_post(url, payload):
        return {
            "result": {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        }

    def fake_get_tokenizer_and_mask_generator(args):
        return None, _DummyMaskGenerator()

    monkeypatch.setattr(fireworks_rollout, "post", fake_post)
    monkeypatch.setattr(
        fireworks_rollout, "_get_tokenizer_and_mask_generator", fake_get_tokenizer_and_mask_generator
    )

    args = Namespace(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=3000,
        model_name="qwen3-4b",
        hf_checkpoint="qwen3-4b",
        loss_mask_type="qwen",
        agent_base_url="http://127.0.0.1:8000",
    )
    sample = Sample(prompt=[{"role": "user", "content": "hi"}], metadata={}, status=Sample.Status.PENDING)

    result = asyncio.run(fireworks_rollout.generate(args, sample, {"max_new_tokens": 8}))

    assert result.response == "ok"
    assert result.tokens == [101, 102, 103]
    assert result.response_length == 2
    assert result.loss_mask == [1, 1]
    assert result.status == Sample.Status.COMPLETED


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
