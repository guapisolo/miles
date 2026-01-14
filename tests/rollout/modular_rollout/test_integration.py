import pytest

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.modular_rollout.orchestration_eval import SimpleEvalRolloutFn
from miles.rollout.modular_rollout.orchestration_train import SimpleTrainRolloutFn
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_simple_train_rollout_fn_integration(rollout_integration_env):
    args, data_source = rollout_integration_env
    fn = SimpleTrainRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    out = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(out.samples) == args.rollout_batch_size
    group = out.samples[0]
    assert len(group) == args.n_samples_per_prompt
    sample = group[0]
    assert "\\boxed" in sample.response
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1


@pytest.mark.asyncio
async def test_simple_eval_rollout_fn_integration(rollout_integration_env):
    args, data_source = rollout_integration_env
    fn = SimpleEvalRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    out = await fn(RolloutFnEvalInput(rollout_id=0))

    assert "toy" in out.data
    rewards = out.data["toy"]["rewards"]
    samples = out.data["toy"]["samples"]
    assert len(rewards) == len(samples) == args.n_samples_per_eval_prompt
    assert rewards[0] == 1
    assert "\\boxed" in samples[0].response
    assert samples[0].status == Sample.Status.COMPLETED
