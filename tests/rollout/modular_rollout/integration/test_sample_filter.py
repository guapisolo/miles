import pytest
from tests.rollout.modular_rollout.integration.utils import (
    MIXED_DATA_ROWS,
    config,
    filter_by_reward,
    load_and_call_train,
)

from miles.utils.misc import function_registry


@pytest.mark.parametrize(
    "rollout_integration_env",
    [
        pytest.param(
            config(
                [
                    "--rollout-batch-size",
                    "2",
                    "--dynamic-sampling-filter-path",
                    "test:filter_by_reward",
                    "--rollout-sample-filter-path",
                    "test:sample_filter",
                    "--rollout-all-samples-process-path",
                    "test:all_samples_process",
                ],
                data_rows=MIXED_DATA_ROWS,
            ),
            id="sample_filter_vs_all_samples",
        ),
    ],
    indirect=True,
)
def test_sample_filter_only_sees_unfiltered(rollout_integration_env):
    env = rollout_integration_env
    sample_filter_log = {"called": False, "data_len": None, "rewards": None}
    all_samples_log = {"called": False, "all_samples_len": None, "has_data_source": False}

    def sample_filter(args, data):
        sample_filter_log["called"] = True
        sample_filter_log["data_len"] = len(data)
        sample_filter_log["rewards"] = [g[0][0].reward if isinstance(g[0], list) else g[0].reward for g in data]

    def all_samples_process(args, all_samples, data_source):
        all_samples_log["called"] = True
        all_samples_log["all_samples_len"] = len(all_samples)
        all_samples_log["has_data_source"] = data_source is not None

    with (
        function_registry.temporary("test:filter_by_reward", filter_by_reward),
        function_registry.temporary("test:sample_filter", sample_filter),
        function_registry.temporary("test:all_samples_process", all_samples_process),
    ):
        load_and_call_train(env.args, env.data_source)

        assert sample_filter_log["called"]
        assert sample_filter_log["data_len"] == env.args.rollout_batch_size
        assert all(r == 1 for r in sample_filter_log["rewards"])

        assert all_samples_log["called"]
        assert all_samples_log["all_samples_len"] >= env.args.rollout_batch_size
        assert all_samples_log["has_data_source"]
