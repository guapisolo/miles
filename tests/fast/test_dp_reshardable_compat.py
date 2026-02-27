import types

import pytest
import torch

from miles.backends.megatron_utils import checkpoint as ckpt


def test_merge_optimizer_param_state_truncates_padding():
    x1 = {
        "optimizer": {
            "param_state": [
                {"a": 1},
                {"a": 2},
                {"a": 3},
                {"a": 4},
            ]
        }
    }
    x2 = {
        "optimizer": {
            "param_state": [
                {"b": 10},
                {"b": 20},
                {"b": 30},
            ]
        }
    }

    ckpt._merge_optimizer_param_state_lists(x1, x2)

    merged = x1["optimizer"]["param_state"]
    assert len(merged) == 3
    assert merged[0]["a"] == 1
    assert merged[0]["b"] == 10
    assert merged[2]["a"] == 3
    assert merged[2]["b"] == 30


def test_merge_list_mismatch_non_optimizer_raises():
    x1 = {"a": [1, 2, 3]}
    x2 = {"a": [1, 2]}
    with pytest.raises(ValueError, match="Cannot merge two lists"):
        ckpt._merge_optimizer_param_state_lists(x1, x2)


def test_filter_dp_reshardable_bucket_state_truncates_padding():
    bucket_state = [
        {"step": 1},
        {"padding": True},
        {"step": 2},
        {"step": 3},
    ]
    filtered = ckpt._filter_dp_reshardable_bucket_state(bucket_state, 2)
    assert len(filtered) == 2
    assert filtered[0]["step"] == 1
    assert filtered[1]["step"] == 2


def test_filter_dp_reshardable_bucket_state_short_raises():
    bucket_state = [{"padding": True}]
    with pytest.raises(AssertionError, match="bucket_state shorter"):
        ckpt._filter_dp_reshardable_bucket_state(bucket_state, 2)


def test_dp_reshardable_compat_patches_and_restores(tmp_path):
    iter_dir = tmp_path / "iter_0000001"
    iter_dir.mkdir()
    torch.save({"optimizer": {"param_state_sharding_type": "dp_reshardable"}}, iter_dir / "common.pt")

    args = types.SimpleNamespace(ckpt_step=1)

    from megatron.core.dist_checkpointing import dict_utils

    original_merge = dict_utils.merge
    with ckpt._dp_reshardable_compat(tmp_path, args):
        assert dict_utils.merge is ckpt._merge_optimizer_param_state_lists
    assert dict_utils.merge is original_merge
