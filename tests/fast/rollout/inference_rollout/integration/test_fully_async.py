import pytest
from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.types import Sample


def _make_buffer_group(start_rollout_id: int, n_samples: int = 1) -> list[Sample]:
    return [
        Sample(
            group_index=0,
            index=i,
            prompt="buffered prompt",
            response="partial response",
            response_length=3,
            status=Sample.Status.PENDING,
            metadata={"start_rollout_id": start_rollout_id},
        )
        for i in range(n_samples)
    ]


def _load_and_call_train(args, data_source, rollout_id: int = 0):
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        args.rollout_function_path,
    )
    return call_rollout_function(fn, RolloutFnTrainInput(rollout_id=rollout_id))


_FULLY_ASYNC_BASIC = integration_env_config(
    extra_argv=[
        "--fully-async-rollout",
        "--rollout-batch-size",
        "2",
        "--over-sampling-batch-size",
        "4",
    ],
)


@pytest.mark.parametrize("rollout_env", [_FULLY_ASYNC_BASIC], indirect=True)
def test_fully_async_basic(rollout_env):
    env = rollout_env
    out = _load_and_call_train(env.args, env.data_source)
    assert len(out.samples) == env.args.rollout_batch_size
    assert len(env.mock_server.request_log) > 0


_FULLY_ASYNC_CONCURRENCY = integration_env_config(
    extra_argv=[
        "--fully-async-rollout",
        "--rollout-batch-size",
        "2",
        "--over-sampling-batch-size",
        "4",
        "--sglang-server-concurrency",
        "4",
    ],
    latency=0.1,
)


@pytest.mark.parametrize("rollout_env", [_FULLY_ASYNC_CONCURRENCY], indirect=True)
def test_fully_async_maintains_inflight_count(rollout_env):
    env = rollout_env
    out = _load_and_call_train(env.args, env.data_source)
    assert len(out.samples) == env.args.rollout_batch_size
    # continuous mode should push multiple requests concurrently
    assert env.mock_server.max_concurrent > 1


_FULLY_ASYNC_PARTIAL = integration_env_config(
    extra_argv=[
        "--fully-async-rollout",
        "--partial-rollout",
        "--rollout-batch-size",
        "2",
        "--over-sampling-batch-size",
        "4",
    ],
)


@pytest.mark.parametrize("rollout_env", [_FULLY_ASYNC_PARTIAL], indirect=True)
def test_fully_async_with_partial_rollout(rollout_env):
    env = rollout_env
    out = _load_and_call_train(env.args, env.data_source)
    assert len(out.samples) == env.args.rollout_batch_size
    # over-sampling > target means more requests were submitted than needed
    assert len(env.mock_server.request_log) > env.args.rollout_batch_size


_FULLY_ASYNC_STALENESS = integration_env_config(
    extra_argv=[
        "--fully-async-rollout",
        "--partial-rollout",
        "--rollout-batch-size",
        "2",
        "--over-sampling-batch-size",
        "4",
        "--max-buffer-staleness",
        "1",
    ],
)


@pytest.mark.parametrize("rollout_env", [_FULLY_ASYNC_STALENESS], indirect=True)
def test_fully_async_with_staleness_filter(rollout_env):
    env = rollout_env
    # pre-populate buffer with stale samples
    for _ in range(3):
        group = _make_buffer_group(start_rollout_id=0, n_samples=env.args.n_samples_per_prompt)
        env.data_source.buffer.append(group)

    out = _load_and_call_train(env.args, env.data_source, rollout_id=10)
    assert len(out.samples) == env.args.rollout_batch_size
    assert "rollout/buffer/stale_samples_discarded" in out.metrics
    assert out.metrics["rollout/buffer/stale_samples_discarded"] == 3
