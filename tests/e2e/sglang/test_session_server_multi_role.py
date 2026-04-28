"""E2E test: multi-role session-server TITO verification under real model inference.

Thin wrapper around
``miles.utils.test_utils.session_verify_runner.run_session_verify`` (driver
and coverage assertions live in ``session_verify_agent``).  Requires 8 GPUs.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-b-short-8-gpu", num_gpus=8)


import os
from dataclasses import dataclass

from miles.utils.test_utils.session_verify_runner import run_session_verify

# ---------------------------------------------------------------------------
# Model registry — one entry per (model, allowed_role) surface to verify.
# Selected by env var SESSION_TEST_MODEL_FAMILY (mirrors the legacy tool-only
# e2e knob); defaults to glm47-multi-role since GLM-4.7 is the model family
# whose ``clear_thinking=False`` auto-merge in TITOTokenizer.SUPPORTED_TEMPLATES
# only kicks in when ``user`` is in the role surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None
    tito_model: str
    allowed_append_roles: tuple[str, ...]
    # Engine tensor-parallel slice (``--rollout-num-gpus-per-engine``).
    # Picked per-model so that ``num_attention_heads % tp_size == 0`` and the
    # weights fit in ``tp_size × 143GB H200``.  Default 1 keeps small-model
    # paths working; larger models override explicitly.
    tp_size: int = 1
    cycles: int = 3
    # Soft-threshold override for assistant_text mismatch ratio.  Default
    # 0.05 matches session_verify_runner; raise per-family when an upstream
    # sglang reasoning parser is known to roundtrip imperfectly (e.g.
    # nemotron_3 keeps trailing newline in reasoning_content) so the gate
    # does not block on a documented out-of-scope issue.
    assistant_text_threshold: float = 0.05


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "glm47-multi-role": ModelConfig(
        # GLM-4.7-Flash: ~100B MoE, num_attention_heads=20.  TP must divide 20,
        # so tp_size=4 (5 heads/rank); tp_size=8 fails the sglang
        # ``num_heads % attn_tp_size == 0`` assertion.
        model_name="zai-org/GLM-4.7-Flash",
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        tito_model="glm47",
        allowed_append_roles=("tool", "user", "system"),
        tp_size=4,
    ),
    "qwen3-tool-user": ModelConfig(
        # Qwen3-30B-A3B (MoE, ~3B activated, ~60GB bf16) over Qwen3-4B for
        # higher tool-call success rate — small Qwen3 emits tool_calls
        # inconsistently, which makes the cross-sample append_tool coverage
        # assertion flaky.  Fits in a single H200 so tp_size=1.
        model_name="Qwen/Qwen3-30B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        allowed_append_roles=("tool", "user"),
        tp_size=1,
        # cycles=2 keeps schedule depth × 4K response budget within Qwen3's 32K
        # context window with headroom for prompt + chat template overhead.
        # GLM-4.7-Flash has a larger context so it stays at the default 3.
        cycles=2,
    ),
    "qwen35-tool-user": ModelConfig(
        # Qwen3.5-35B-A3B (MoE, ~3B activated, ~70GB bf16). Same boundary
        # handling as Qwen3 via Qwen35TITOTokenizer; the {tool, user} row uses
        # clear_thinking=False to keep <think> history across multi-user turns.
        # Qwen3.5 emits tool calls as <tool_call><function=...><parameter=...>;
        # the qwen3_coder parser handles that XML-style wrapping (qwen25 parser
        # only understands <tool_call>{json}</tool_call>).  Fits in a single
        # H200 so tp_size=1.
        model_name="Qwen/Qwen3.5-35B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        tito_model="qwen35",
        allowed_append_roles=("tool", "user"),
        tp_size=1,
        cycles=2,
    ),
    "qwennext-tool-user": ModelConfig(
        # Qwen3-Next-80B-A3B-Thinking (MoE, ~3B activated, ~160GB bf16).
        # Doesn't fit one 143GB H200 — tp_size=2 minimum.  num_key_value_heads
        # is 2, so TP > 2 would either replicate KV or hit divisibility
        # asserts; tp_size=2 is the safe ceiling.  Thinking-only model, so
        # reasoning_parser stays qwen3 and tool_call_parser stays qwen25;
        # the {tool, user} row uses clear_thinking=False to preserve reasoning
        # across user turns.
        model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwennext",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
    ),
    "nemotron3-tool-user": ModelConfig(
        # Nemotron-3-Super-120B-A12B-BF16 (~240GB bf16, A12B activated).
        # num_attention_heads=32, num_key_value_heads=2 — same KV-bottleneck
        # as Qwen3-Next, so tp_size=2 is the safe ceiling.  Tool calls use
        # the same <tool_call><function=...><parameter=...> XML wrapping as
        # Qwen3.5, so qwen3_coder is the right tool_call_parser.  The
        # nemotron_3 reasoning parser is documented (in Nemotron3TITOTokenizer)
        # to leave a trailing newline in reasoning_content — assistant_text
        # roundtrip mismatches on every plain-text turn until upstream sglang
        # is patched, so the soft threshold is relaxed to 1.0 for this row;
        # hard mismatches (special tokens / non-assistant text) still gate.
        model_name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        reasoning_parser="nemotron_3",
        tool_call_parser="qwen3_coder",
        tito_model="nemotron3",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
        assistant_text_threshold=1.0,
    ),
    "kimi25-tool-user": ModelConfig(
        # Kimi-K2.5 (~1.058T total params, MoE).  Weights ship pre-quantized
        # (~555GB on disk, FP8 packed in I32 + BF16 norms/embeds), so tp_size=8
        # gives ~70GB/GPU and fits comfortably on 8×H200.  Reasoning + tool
        # call both use the kimi_k2 sglang parser.  The {tool, user} row in
        # Kimi25TITOTokenizer ships the patched kimi_k25_fixed.jinja with
        # auto-merged preserve_thinking=True to keep history
        # append-only across multi-user turns.
        model_name="moonshotai/Kimi-K2.5",
        reasoning_parser="kimi_k2",
        tool_call_parser="kimi_k2",
        tito_model="kimi25",
        allowed_append_roles=("tool", "user"),
        tp_size=8,
        cycles=2,
    ),
    "kimi26-tool-user": ModelConfig(
        # Kimi-K2.6 (~1.058T total params, MoE — same shape as K2.5).
        # K2.6's HF-native chat template already exposes the preserve_thinking
        # gate that K2.5 needs patched in, so the {tool, user} row in
        # Kimi26TITOTokenizer registers template=None with
        # auto-merged preserve_thinking=True.
        model_name="moonshotai/Kimi-K2.6",
        reasoning_parser="kimi_k2",
        tool_call_parser="kimi_k2",
        tito_model="kimi26",
        allowed_append_roles=("tool", "user"),
        tp_size=8,
        cycles=2,
    ),
}

DEFAULT_MODEL_FAMILY = "glm47-multi-role"


def _get_config() -> ModelConfig:
    family = os.environ.get("SESSION_TEST_MODEL_FAMILY", DEFAULT_MODEL_FAMILY)
    if family not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown SESSION_TEST_MODEL_FAMILY={family!r}. " f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[family]


def test_session_server_multi_role():
    cfg = _get_config()
    run_session_verify(
        hf_checkpoint=cfg.model_name,
        tito_model=cfg.tito_model,
        allowed_append_roles=list(cfg.allowed_append_roles),
        reasoning_parser=cfg.reasoning_parser,
        tool_call_parser=cfg.tool_call_parser,
        tp_size=cfg.tp_size,
        cycles=cfg.cycles,
        assistant_text_threshold=cfg.assistant_text_threshold,
    )


if __name__ == "__main__":
    test_session_server_multi_role()
