"""E2E test: verify TITO session logprobs match a fresh full re-prefill.

After a multi-turn agent conversation through the session server, the
full ``accumulated_token_ids`` (the token sequence built incrementally
by TITO) is sent to ``/generate`` with ``max_new_tokens=0`` for a single
prefill pass.  The resulting ``input_token_logprobs`` are compared
per-turn against the ``output_token_logprobs`` from the session's decode
phase.  Token IDs must match exactly; logprob values are compared with a
tight tolerance for prefill-vs-decode numerical differences.

When an MoE model is used with ``ENABLE_R3=1``, routed_experts arrays
are also compared.

Uses the same infrastructure as ``test_session_server_tool_call``:
``execute_train --debug-rollout-only`` with a custom generate function
that wraps the normal agentic flow with re-prefill verification.

Requires 1 GPU.
"""

import os

import pytest
from tests.e2e.sglang.utils.session_test_config import cleanup_proxy_env, execute, prepare, restore_transformers

PROMPT_DATA_PATH = "/root/datasets/session_logprob_verify.jsonl"
ENABLE_R3 = os.environ.get("ENABLE_R3", "0") == "1"


@pytest.mark.system
def test_tito_logprob_equivalence():
    prepare(PROMPT_DATA_PATH)
    cleanup_proxy_env()
    execute(
        PROMPT_DATA_PATH,
        "tests.e2e.sglang.utils.logprob_verify_generate.generate",
        n_samples_per_prompt=1,
        temperature=0.0,
        extra_sglang_args="--sglang-enable-deterministic-inference ",
        extra_router_args="--use-rollout-routing-replay " if ENABLE_R3 else "",
    )


if __name__ == "__main__":
    prepare(PROMPT_DATA_PATH)
    cleanup_proxy_env()
    execute(
        PROMPT_DATA_PATH,
        "tests.e2e.sglang.utils.logprob_verify_generate.generate",
        n_samples_per_prompt=1,
        temperature=0.0,
        extra_sglang_args="--sglang-enable-deterministic-inference ",
        extra_router_args="--use-rollout-routing-replay " if ENABLE_R3 else "",
    )
    restore_transformers()
