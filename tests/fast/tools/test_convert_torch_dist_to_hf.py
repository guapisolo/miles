from argparse import Namespace

import torch

from tools.convert_torch_dist_to_hf import get_expert_param


def test_get_expert_param_keeps_decoder_fused_moe_kernel():
    args = Namespace(num_experts=4)
    name = "module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
    param = torch.randn(4, 12, 16)

    out = list(get_expert_param(args, name, param))
    assert len(out) == 1
    assert out[0][0] == "module.module.decoder.layers.0.mlp.experts.linear_fc1"
    torch.testing.assert_close(out[0][1], param)


def test_get_expert_param_still_splits_non_fused_layout():
    args = Namespace(num_experts=4)
    name = "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight"
    param = torch.randn(4, 6, 16)

    out = list(get_expert_param(args, name, param))
    assert len(out) == 4
    assert out[0][0] == "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight0"
    assert out[3][0] == "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight3"
    torch.testing.assert_close(out[2][1], param[2])
