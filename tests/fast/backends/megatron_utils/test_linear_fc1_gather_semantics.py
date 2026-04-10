from argparse import Namespace

import torch

from miles.backends.megatron_utils.update_weight.common import _check_and_fix_partition, _gather_with_stride


def test_linear_fc1_swiglu_gather_restores_gate_then_up_layout():
    # Simulate full gate/up tensor that conversion expects after all-gather.
    gate = torch.arange(0, 24, dtype=torch.float32).reshape(6, 4)
    up = torch.arange(100, 124, dtype=torch.float32).reshape(6, 4)
    expected = torch.cat([gate, up], dim=0)

    # Simulate TP=2 local shards in interleaved order: [gate_shard, up_shard].
    gate_chunks = gate.chunk(2, dim=0)
    up_chunks = up.chunk(2, dim=0)
    rank0 = torch.cat([gate_chunks[0], up_chunks[0]], dim=0)
    rank1 = torch.cat([gate_chunks[1], up_chunks[1]], dim=0)

    gathered = _gather_with_stride([rank0, rank1], partition_dim=0, partition_stride=2)
    torch.testing.assert_close(gathered, expected)


def test_linear_fc1_uses_partition_stride_two_when_swiglu_enabled():
    args = Namespace(swiglu=True)
    stride, dim = _check_and_fix_partition(
        args,
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
        partition_stride=1,
        partition_dim=0,
    )
    assert stride == 2
    assert dim == 0
