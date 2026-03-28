from argparse import Namespace

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.data_source import RolloutDataSourceWithBuffer, pop_first
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


def _make_data_source(max_buffer_staleness=None) -> RolloutDataSourceWithBuffer:
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        max_buffer_staleness=max_buffer_staleness,
        n_samples_per_prompt=1,
    )
    return RolloutDataSourceWithBuffer(args)


# --- Direct data source tests (no mock server needed) ---


def test_staleness_filter_discards_old_samples():
    ds = _make_data_source(max_buffer_staleness=2)
    # groups started at rollout 0 — age = 10 - 0 = 10 > 2
    for _ in range(5):
        ds.buffer.append(_make_buffer_group(start_rollout_id=0))

    ds.get_samples(num_samples=5, rollout_id=10)
    # all 5 groups are stale and discarded; fallback to parent returns empty Samples
    assert ds.stale_samples_discarded == 5
    assert len(ds.buffer) == 0


def test_staleness_filter_keeps_fresh_samples():
    ds = _make_data_source(max_buffer_staleness=2)
    # groups started at rollout 9 — age = 10 - 9 = 1 <= 2
    for _ in range(3):
        ds.buffer.append(_make_buffer_group(start_rollout_id=9))

    samples = ds.get_samples(num_samples=3, rollout_id=10)
    assert ds.stale_samples_discarded == 0
    # all 3 groups returned from buffer
    assert len(samples) == 3
    assert all(s[0].metadata["start_rollout_id"] == 9 for s in samples)
    assert len(ds.buffer) == 0


def test_staleness_disabled_by_default():
    ds = _make_data_source(max_buffer_staleness=None)
    for _ in range(3):
        ds.buffer.append(_make_buffer_group(start_rollout_id=0))

    samples = ds.get_samples(num_samples=3, rollout_id=100)
    # no filtering — all returned
    assert len(samples) == 3
    assert ds.stale_samples_discarded == 0


def test_staleness_mixed_fresh_and_stale():
    ds = _make_data_source(max_buffer_staleness=2)
    ds.buffer.append(_make_buffer_group(start_rollout_id=0))  # stale (age=10)
    ds.buffer.append(_make_buffer_group(start_rollout_id=9))  # fresh (age=1)
    ds.buffer.append(_make_buffer_group(start_rollout_id=5))  # stale (age=5)
    ds.buffer.append(_make_buffer_group(start_rollout_id=8))  # fresh (age=2)

    samples = ds.get_samples(num_samples=4, rollout_id=10)
    assert ds.stale_samples_discarded == 2
    # 2 fresh groups from buffer + 2 from parent (empty Samples since no dataset)
    assert len(samples) == 4
    assert samples[0][0].metadata["start_rollout_id"] == 9
    assert samples[1][0].metadata["start_rollout_id"] == 8


def test_pop_first_without_start_rollout_id_keeps_sample():
    """Groups without start_rollout_id metadata are always kept (not stale)."""
    args = Namespace(max_buffer_staleness=1)
    buffer = [[Sample(prompt="no metadata", metadata={})]]
    result = pop_first(args, rollout_id=100, buffer=buffer, num_samples=1)
    assert len(result) == 1
    assert len(buffer) == 0


# --- Integration tests (with mock server) ---


def _load_and_call_train_with_rollout_id(args, data_source, rollout_id: int):
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        args.rollout_function_path,
    )
    return call_rollout_function(fn, RolloutFnTrainInput(rollout_id=rollout_id))


_STALENESS_INTEGRATION_CONFIG = integration_env_config(
    extra_argv=[
        "--rollout-batch-size",
        "2",
        "--over-sampling-batch-size",
        "2",
        "--partial-rollout",
        "--max-buffer-staleness",
        "1",
    ],
)


@pytest.mark.parametrize("rollout_env", [_STALENESS_INTEGRATION_CONFIG], indirect=True)
def test_fixed_sample_count_after_staleness_discard(rollout_env):
    env = rollout_env
    # pre-populate buffer with stale samples (start_rollout_id=0, will call with rollout_id=10)
    for _ in range(3):
        group = _make_buffer_group(start_rollout_id=0, n_samples=env.args.n_samples_per_prompt)
        env.data_source.buffer.append(group)

    out = _load_and_call_train_with_rollout_id(env.args, env.data_source, rollout_id=10)
    # must still produce exact rollout_batch_size despite stale discards
    assert len(out.samples) == env.args.rollout_batch_size


@pytest.mark.parametrize("rollout_env", [_STALENESS_INTEGRATION_CONFIG], indirect=True)
def test_staleness_metrics_reported(rollout_env):
    env = rollout_env
    for _ in range(3):
        group = _make_buffer_group(start_rollout_id=0, n_samples=env.args.n_samples_per_prompt)
        env.data_source.buffer.append(group)

    out = _load_and_call_train_with_rollout_id(env.args, env.data_source, rollout_id=10)
    assert "rollout/buffer/stale_samples_discarded" in out.metrics
    assert out.metrics["rollout/buffer/stale_samples_discarded"] == 3
    assert "rollout/buffer/remaining_size" in out.metrics
