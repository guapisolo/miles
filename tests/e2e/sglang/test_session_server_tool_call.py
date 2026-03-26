"""E2E test: session-server pretokenized TITO with real model inference.

Starts the full miles pipeline (sglang + miles-router with session support)
via ``execute_train --debug-rollout-only``, then runs the agentic_tool_call
generate function with a custom agent that performs multi-turn tool calls and
asserts the pretokenized prefix invariant on every turn.

Requires 1 GPU.
"""

import pytest
from tests.e2e.sglang.utils.session_test_config import cleanup_proxy_env, execute, prepare, restore_transformers

PROMPT_DATA_PATH = "/root/datasets/session_tool_call.jsonl"


@pytest.mark.system
def test_session_server_tool_call():
    prepare(PROMPT_DATA_PATH)
    cleanup_proxy_env()
    execute(
        PROMPT_DATA_PATH,
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        n_samples_per_prompt=4,
        temperature=0.7,
    )


if __name__ == "__main__":
    prepare(PROMPT_DATA_PATH)
    cleanup_proxy_env()
    execute(
        PROMPT_DATA_PATH,
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        n_samples_per_prompt=4,
        temperature=0.7,
    )
    restore_transformers()
