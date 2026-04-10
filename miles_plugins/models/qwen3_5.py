import copy

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    pass

from .hf_attention import HuggingfaceAttention, _load_hf_config

DEBUG_LOG_PATH = "/root/miles/.cursor/debug-b8af16.log"
DEBUG_SESSION_ID = "b8af16"


def _append_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    _FP32_PARAM_NAMES = frozenset({"A_log", "norm.weight"})

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # Separate projections for QKV and Z (unlike Qwen3Next which combines QKVZ)
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        projection_size_z = self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, projection_size_qkv, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, projection_size_z, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # time step projection
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        self._debug_logged_forward = False

        # region agent log
        _append_debug_log(
            run_id="baseline",
            hypothesis_id="H1",
            location="miles_plugins/models/qwen3_5.py:90",
            message="Qwen3_5GatedDeltaNet init dtype snapshot",
            data={
                "layer_idx": int(layer_idx),
                "a_log_dtype": str(self.A_log.dtype),
                "dt_bias_dtype": str(self.dt_bias.dtype),
                "config_dtype": str(config.dtype),
            },
        )
        # endregion

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=torch.cuda.current_device(),
            dtype=config.dtype if config.dtype is not None else torch.get_current_dtype(),
        )
        self._maintain_fp32_params()

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _maintain_fp32_params(self):
        with torch.no_grad():
            for name, param in self.named_parameters(recurse=True):
                if name in self._FP32_PARAM_NAMES and param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._maintain_fp32_params()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._maintain_fp32_params()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._maintain_fp32_params()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def half(self):
        ret = super().half()
        self._maintain_fp32_params()
        return ret

    def bfloat16(self):
        ret = super().bfloat16()
        self._maintain_fp32_params()
        return ret

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        self._maintain_fp32_params()
        batch_size, seq_len, _ = hidden_states.shape

        # Projections (flat layout: [Q_all, K_all, V_all])
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Convolution on the flat QKV
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )

        # Split into Q, K, V (flat split, matching HF layout)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        if not self._debug_logged_forward:
            # region agent log
            _append_debug_log(
                run_id="baseline",
                hypothesis_id="H2",
                location="miles_plugins/models/qwen3_5.py:153",
                message="Pre-g dtype snapshot",
                data={
                    "layer_idx": int(self.layer_idx),
                    "hidden_states_dtype": str(hidden_states.dtype),
                    "a_log_dtype": str(self.A_log.dtype),
                    "a_dtype": str(a.dtype),
                    "dt_bias_dtype": str(self.dt_bias.dtype),
                    "query_dtype": str(query.dtype),
                    "key_dtype": str(key.dtype),
                    "value_dtype": str(value.dtype),
                },
            )
            # endregion
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if not self._debug_logged_forward:
            # region agent log
            _append_debug_log(
                run_id="baseline",
                hypothesis_id="H3",
                location="miles_plugins/models/qwen3_5.py:173",
                message="Post-g dtype and range snapshot",
                data={
                    "layer_idx": int(self.layer_idx),
                    "g_dtype": str(g.dtype),
                    "g_min": float(g.min().item()),
                    "g_max": float(g.max().item()),
                    "g_mean": float(g.mean().item()),
                    "beta_dtype": str(beta.dtype),
                },
            )
            # endregion
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        if not self._debug_logged_forward:
            # region agent log
            _append_debug_log(
                run_id="baseline",
                hypothesis_id="H4",
                location="miles_plugins/models/qwen3_5.py:198",
                message="Kernel output dtype snapshot",
                data={
                    "layer_idx": int(self.layer_idx),
                    "core_attn_out_dtype": str(core_attn_out.dtype),
                    "last_recurrent_state_is_none": bool(last_recurrent_state is None),
                },
            )
            # endregion
            self._debug_logged_forward = True

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # Qwen3.5 is a VLM model with nested text_config
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"

        self.linear_attn = Qwen3_5GatedDeltaNet(self.hf_config, self.hf_layer_idx)

        # Use a simple RMSNorm
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
