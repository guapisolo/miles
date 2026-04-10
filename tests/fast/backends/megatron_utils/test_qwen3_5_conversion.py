from argparse import Namespace

import torch

import miles.backends.megatron_utils.megatron_to_hf.qwen3_5 as qwen3_5_conv


def _args(num_experts: int = 4):
    return Namespace(
        kv_channels=None,
        hidden_size=16,
        num_attention_heads=8,
        num_query_groups=4,
        num_experts=num_experts,
    )


def _clear_caches():
    qwen3_5_conv._DECODER_EXPERT_LINEAR_FC1_CACHE.clear()
    qwen3_5_conv._DECODER_EXPERT_LINEAR_FC2_CACHE.clear()


def test_decoder_moe_experts_are_merged_to_fused_hf_kernels():
    _clear_caches()
    args = _args(num_experts=4)

    fc1_params = [torch.randn(12, 16) for _ in range(args.num_experts)]
    for expert_id, param in enumerate(fc1_params):
        out = qwen3_5_conv.convert_qwen3_5_to_hf(
            args,
            f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}",
            param,
        )
        if expert_id < args.num_experts - 1:
            assert out == []
        else:
            assert len(out) == 1
            name, merged = out[0]
            assert name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
            assert merged.shape == (args.num_experts, 12, 16)
            for i in range(args.num_experts):
                torch.testing.assert_close(merged[i], fc1_params[i])

    fc2_params = [torch.randn(6, 16) for _ in range(args.num_experts)]
    for expert_id, param in enumerate(fc2_params):
        out = qwen3_5_conv.convert_qwen3_5_to_hf(
            args,
            f"module.module.decoder.layers.0.mlp.experts.linear_fc2.weight{expert_id}",
            param,
        )
        if expert_id < args.num_experts - 1:
            assert out == []
        else:
            assert len(out) == 1
            name, merged = out[0]
            assert name == "model.language_model.layers.0.mlp.experts.down_proj"
            assert merged.shape == (args.num_experts, 6, 16)
            for i in range(args.num_experts):
                torch.testing.assert_close(merged[i], fc2_params[i])


def test_mtp_experts_keep_per_expert_hf_format():
    _clear_caches()
    args = _args(num_experts=4)
    gate = torch.randn(6, 16)
    up = torch.randn(6, 16)
    param = torch.cat([gate, up], dim=0)
    out = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight3",
        param,
    )
    assert len(out) == 2
    assert out[0][0] == "mtp.layers.0.mlp.experts.3.gate_proj.weight"
    assert out[1][0] == "mtp.layers.0.mlp.experts.3.up_proj.weight"
    torch.testing.assert_close(out[0][1], gate)
    torch.testing.assert_close(out[1][1], up)


def test_linear_attn_a_log_is_exported_as_fp32():
    _clear_caches()
    args = _args(num_experts=4)
    param = torch.randn(8, dtype=torch.bfloat16)
    out = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.linear_attn.A_log",
        param,
    )
    assert len(out) == 1
    name, tensor = out[0]
    assert name == "model.language_model.layers.0.linear_attn.A_log"
    assert tensor.dtype == torch.float32


def test_linear_attn_norm_weight_is_exported_as_fp32():
    _clear_caches()
    args = _args(num_experts=4)
    param = torch.randn(8, dtype=torch.bfloat16)
    out = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.linear_attn.norm.weight",
        param,
    )
    assert len(out) == 1
    name, tensor = out[0]
    assert name == "model.language_model.layers.0.linear_attn.norm.weight"
    assert tensor.dtype == torch.float32


def test_linear_attn_dt_bias_keeps_original_dtype():
    _clear_caches()
    args = _args(num_experts=4)
    param = torch.randn(8, dtype=torch.bfloat16)
    out = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.linear_attn.dt_bias",
        param,
    )
    assert len(out) == 1
    name, tensor = out[0]
    assert name == "model.language_model.layers.0.linear_attn.dt_bias"
    assert tensor.dtype == torch.bfloat16


def test_fused_decoder_expert_weight_name_is_supported():
    _clear_caches()
    args = _args(num_experts=4)
    param = torch.randn(4, 12, 16)
    out = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight",
        param,
    )
    assert len(out) == 1
    assert out[0][0] == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    torch.testing.assert_close(out[0][1], param)


def test_decoder_ungrouped_fc1_keeps_gate_up_fused_tensor_without_chunk():
    _clear_caches()
    args = _args(num_experts=2)

    gate0 = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
    up0 = torch.arange(100, 112, dtype=torch.float32).reshape(3, 4)
    param0 = torch.cat([gate0, up0], dim=0)
    out0 = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        param0,
    )
    assert out0 == []

    gate1 = torch.arange(200, 212, dtype=torch.float32).reshape(3, 4)
    up1 = torch.arange(300, 312, dtype=torch.float32).reshape(3, 4)
    param1 = torch.cat([gate1, up1], dim=0)
    out1 = qwen3_5_conv.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight1",
        param1,
    )

    assert len(out1) == 1
    name, merged = out1[0]
    assert name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    assert merged.shape == (2, 6, 4)
    torch.testing.assert_close(merged[0], param0)
    torch.testing.assert_close(merged[1], param1)
