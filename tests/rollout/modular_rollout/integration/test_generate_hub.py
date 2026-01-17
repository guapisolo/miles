from typing import Any

import pytest
from tests.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fixtures.rollout_integration import IntegrationEnvConfig
from tests.rollout.modular_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample


async def _simple_reward_function(args, samples: Sample | list[Sample]) -> float | list[float]:
    """Simple reward function that checks if response contains the label."""
    if isinstance(samples, list):
        # For multi_samples variants, check if the last sample contains the label
        # If so, all samples get reward=1 (as requested by user)
        if len(samples) > 0 and samples[-1].response and samples[-1].label:
            if str(samples[-1].label) in samples[-1].response:
                return [1.0] * len(samples)
        # Otherwise, check each sample individually
        rewards = []
        for sample in samples:
            if sample.response and sample.label:
                reward = 1.0 if str(sample.label) in sample.response else 0.0
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards
    else:
        sample = samples
        if sample.response and sample.label:
            return 1.0 if str(sample.label) in sample.response else 0.0
        return 0.0


TWO_TURN_DATA_ROWS = [{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}]

_VARIANT_NAMES = [
    "multi_turn_single_sample",
    "multi_turn_multi_samples",
    "agentic_tool_call_single_sample",
    "agentic_tool_call_multi_samples",
]


def _config_for_variant(variant: str) -> IntegrationEnvConfig:
    return IntegrationEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + extra_argv_for_variant(variant)
        + [
            "--rollout-batch-size",
            "2",
            "--n-samples-per-prompt",
            "2",
            "--n-samples-per-eval-prompt",
            "2",
            "--custom-rm-path",
            "tests.rollout.modular_rollout.integration.test_generate_hub._simple_reward_function",
        ],
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize(
    "rollout_integration_env",
    [pytest.param(_config_for_variant(variant), id=variant) for variant in _VARIANT_NAMES],
    indirect=True,
)
@pytest.mark.parametrize("test_type", ["train", "eval"])
def test_rollout(rollout_integration_env, request, test_type):
    env = rollout_integration_env
    # Extract variant name from callspec.id (format: "test_type-variant" or "variant")
    callspec_id = request.node.callspec.id
    if "-" in callspec_id:
        variant = callspec_id.split("-", 1)[1]
    else:
        variant = callspec_id

    env.mock_server.process_fn = TwoTurnStub.process_fn

    out = load_and_call_rollout(env.args, env.data_source, mode=test_type)

    if test_type == "train":
        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        _verify_samples(variant, group)
    else:
        assert "toy" in out.data
        samples = out.data["toy"]["samples"]
        _verify_samples(variant, samples)


def _verify_samples(variant: str, samples: list[Any]):
    if variant in ("multi_turn_multi_samples", "agentic_tool_call_multi_samples"):
        # For multi_samples variants, samples can be either:
        # 1. list[list[Sample]] (train mode: grouped by prompt)
        # 2. list[Sample] (eval mode: flattened)
        if len(samples) > 0 and isinstance(samples[0], list):
            # Train mode: list[list[Sample]]
            assert len(samples) == 2, f"n_samples_per_prompt=2, so group should have 2 samples, got {len(samples)}"
            for group_sample in samples:
                assert isinstance(group_sample, list), "multi_samples variant should return list[Sample] per generate"
                assert len(group_sample) == 2, "multi_samples variant should return 2 samples per generate (one per turn)"
                for i, sample in enumerate(group_sample):
                    assert sample.status == Sample.Status.COMPLETED
                    assert sample.reward == 1, f"Sample {i} should have reward=1"
                assert "2008" in group_sample[-1].response, "Last sample should contain final answer '2008'"
        else:
            # Eval mode: list[Sample] (flattened)
            # n_samples_per_eval_prompt=2, and each generate returns 2 turns, so 2*2=4 samples
            assert len(samples) == 4, f"n_samples_per_eval_prompt=2, each generate returns 2 turns, so should have 4 samples, got {len(samples)}"
            # Group samples by prompt (every 2 samples form a group)
            for group_idx in range(2):
                group_samples = samples[group_idx * 2 : (group_idx + 1) * 2]
                assert len(group_samples) == 2, f"Each group should have 2 samples (one per turn)"
                for i, sample in enumerate(group_samples):
                    assert sample.status == Sample.Status.COMPLETED
                    assert sample.reward == 1, f"Sample {i} in group {group_idx} should have reward=1"
                assert "2008" in group_samples[-1].response, f"Last sample in group {group_idx} should contain final answer '2008'"
    else:
        assert len(samples) == 2, f"n_samples_per_prompt=2, so group should have 2 samples, got {len(samples)}"
        for sample in samples:
            assert isinstance(sample, Sample), "single_sample variant should return Sample, not list"
            assert sample.status == Sample.Status.COMPLETED
            assert sample.reward == 1, "multi_turn_single_sample merges all turns, reward should be 1"
            assert "2008" in sample.response, "Response should contain final answer '2008'"
