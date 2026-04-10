import importlib
import sys

import pytest
import torch
import torch.nn as nn


def _load_qwen3_5_gdn_cls():
    for module_name in [
        "miles_plugins.models.qwen3_5",
        "megatron",
        "megatron.core",
        "megatron.core.models",
        "megatron.core.models.gpt",
        "megatron.core.models.gpt.gpt_layer_specs",
        "mbridge",
        "mbridge.core",
        "mbridge.models",
    ]:
        sys.modules.pop(module_name, None)

    try:
        module = importlib.import_module("miles_plugins.models.qwen3_5")
    except Exception as exc:  # pragma: no cover - environment dependent import
        pytest.skip(f"qwen3_5 model module is unavailable in test env: {exc}")
    return module.Qwen3_5GatedDeltaNet


def _build_stub_qwen3_5_gdn(model_cls):
    module = model_cls.__new__(model_cls)
    nn.Module.__init__(module)
    module.A_log = nn.Parameter(torch.randn(8, dtype=torch.float32))
    module.dt_bias = nn.Parameter(torch.randn(8, dtype=torch.float32))
    module.norm = nn.LayerNorm(8)
    module.extra = nn.Parameter(torch.randn(8, dtype=torch.float32))
    return module


def test_qwen3_5_fp32_whitelist_survives_mixed_precision_casts():
    model_cls = _load_qwen3_5_gdn_cls()
    module = _build_stub_qwen3_5_gdn(model_cls)

    module.bfloat16()

    assert module.A_log.dtype == torch.float32
    assert module.norm.weight.dtype == torch.float32
    assert module.dt_bias.dtype == torch.bfloat16
    assert module.extra.dtype == torch.bfloat16


def test_qwen3_5_fp32_whitelist_survives_state_dict_roundtrip():
    model_cls = _load_qwen3_5_gdn_cls()
    source = _build_stub_qwen3_5_gdn(model_cls)
    source.bfloat16()

    state_dict = source.state_dict()
    assert state_dict["A_log"].dtype == torch.float32
    assert state_dict["norm.weight"].dtype == torch.float32
    assert state_dict["dt_bias"].dtype == torch.bfloat16

    target = _build_stub_qwen3_5_gdn(model_cls)
    target.half()
    target.load_state_dict(state_dict, strict=True)

    assert target.A_log.dtype == torch.float32
    assert target.norm.weight.dtype == torch.float32
    assert target.dt_bias.dtype == torch.float16
