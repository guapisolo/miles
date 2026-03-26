"""Shared infrastructure for TITO session e2e tests.

Both ``test_session_server_tool_call`` and ``test_tito_logprob_equivalence``
use the same model registry, download logic, and ``execute_train`` harness.
This module centralizes that boilerplate so the test files only need to
specify what differs (generate function, rollout params, extra flags).
"""

import json
import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils as U

MODEL_FAMILY = os.environ.get("SESSION_TEST_MODEL_FAMILY", "qwen3")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None = None
    tito_model: str = "default"
    num_gpus: int = 1


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3": ModelConfig(
        model_name="Qwen/Qwen3-4B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    ),
    "glm47": ModelConfig(
        model_name="zai-org/GLM-4.7-Flash",
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        tito_model="glm47",
        num_gpus=1,
    ),
}

WEATHER_PROMPT = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to weather tools. "
                "Use the get_weather tool to look up weather information. "
                "When you have gathered all the information, "
                "wrap your final summary in <final_answer>...</final_answer> tags."
            ),
        },
        {
            "role": "user",
            "content": "What's the weather like in Beijing, Shanghai, Tokyo, and New York?",
        },
    ],
}


def get_config() -> ModelConfig:
    if MODEL_FAMILY not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family {MODEL_FAMILY!r}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[MODEL_FAMILY]


def prepare(prompt_data_path: str) -> None:
    """Download model and write prompt data."""
    cfg = get_config()
    U.exec_command("mkdir -p /root/models /root/datasets")
    if MODEL_FAMILY == "glm47":
        U.exec_command(
            "pip install git+https://github.com/huggingface/transformers.git@"
            "76732b4e7120808ff989edbd16401f61fa6a0afa --break-system-packages"
        )
    U.exec_command(f"hf download {cfg.model_name} --local-dir /root/models/{cfg.model_name.split('/')[-1]}")

    with open(prompt_data_path, "w") as f:
        f.write(json.dumps(WEATHER_PROMPT) + "\n")


def execute(
    prompt_data_path: str,
    generate_function_path: str,
    *,
    n_samples_per_prompt: int = 1,
    temperature: float = 0.7,
    extra_sglang_args: str = "",
    extra_router_args: str = "",
) -> None:
    """Run ``execute_train --debug-rollout-only`` with session support."""
    cfg = get_config()
    local_model_dir = f"/root/models/{cfg.model_name.split('/')[-1]}"
    batch_size = n_samples_per_prompt * 16  # rollout-batch-size is fixed at 16

    ckpt_args = f"--hf-checkpoint {local_model_dir} "

    rollout_args = (
        f"--prompt-data {prompt_data_path} "
        "--input-key messages "
        "--num-rollout 1 "
        "--rollout-batch-size 16 "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        "--rollout-max-response-len 1024 "
        f"--rollout-temperature {temperature} "
        f"--global-batch-size {batch_size} "
    )

    generate_args = (
        f"--custom-generate-function-path {generate_function_path} "
        "--custom-agent-function-path "
        "tests.e2e.sglang.utils.session_tool_agent.run_agent "
    )

    router_args = (
        "--use-miles-router " "--use-session-server " "--chat-template-path autofix " f"--tito-model {cfg.tito_model} " f"{extra_router_args}"
    )

    sglang_args = f"--rollout-num-gpus-per-engine {cfg.num_gpus} --sglang-reasoning-parser {cfg.reasoning_parser} "
    if cfg.tool_call_parser:
        sglang_args += f"--sglang-tool-call-parser {cfg.tool_call_parser} "
    sglang_args += f"--rm-type random {extra_sglang_args}"

    infra_args = (
        "--debug-rollout-only "
        "--ci-test "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {cfg.num_gpus} "
        "--colocate "
        "--train-backend fsdp "
    )

    train_args = f"{ckpt_args}{rollout_args}{generate_args}{router_args}{sglang_args}{infra_args}"

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=cfg.num_gpus,
        megatron_model_type=None,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "SGLANG_E2E_MODEL_PATH": local_model_dir,
            "MILES_TITO_MODEL": cfg.tito_model,
        },
    )


def cleanup_proxy_env() -> None:
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)


def restore_transformers() -> None:
    if MODEL_FAMILY == "glm47":
        U.exec_command("pip install transformers==4.57.1 --break-system-packages")
