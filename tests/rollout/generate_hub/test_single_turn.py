from argparse import Namespace
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.types import Sample


GENERATE_VARIANTS = [
    pytest.param("old", id="old"),
    pytest.param("new", id="new"),
]


def make_process_fn(
    response_text: str = "\\boxed{8}",
    finish_reason: str = "stop",
    cached_tokens: int = 0,
    weight_version: str | None = None,
    routed_experts: bytes | None = None,
):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text=response_text,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            weight_version=weight_version,
            routed_experts=routed_experts,
        )

    return process_fn


def make_args(
    *,
    router_port: int,
    use_rollout_routing_replay: bool = False,
    use_miles_router: bool = False,
    miles_router_middleware_paths: list[str] | None = None,
) -> Namespace:
    argv = [
        "pytest",
        "--train-backend", "fsdp",
        "--rollout-batch-size", "1",
        "--n-samples-per-prompt", "1",
        "--num-rollout", "1",
        "--rollout-num-gpus", "1",
        "--rollout-num-gpus-per-engine", "1",
        "--hf-checkpoint", "Qwen/Qwen3-0.6B",
        "--prompt-data", "/dev/null",
        "--input-key", "input",
        "--label-key", "label",
        "--rm-type", "math",
        "--sglang-router-ip", "127.0.0.1",
        "--sglang-router-port", str(router_port),
        "--rollout-max-response-len", "16",
    ]
    if use_rollout_routing_replay:
        argv.append("--use-rollout-routing-replay")

    from miles.utils.arguments import parse_args
    with patch("sys.argv", argv):
        args = parse_args()

    args.use_miles_router = use_miles_router
    args.miles_router_middleware_paths = miles_router_middleware_paths or []
    args.ci_test = False
    init_http_client(args)
    return args


def make_sample(
    prompt: str = "What is 1+7?",
    tokens: list[int] | None = None,
    response: str = "",
    response_length: int = 0,
    status: Sample.Status = Sample.Status.PENDING,
    multimodal_inputs: dict | None = None,
) -> Sample:
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
    )


def cleanup_singleton():
    SingletonMeta._instances.pop(
        type("GenerateState", (), {"__module__": "miles.rollout.sglang_rollout"}).__class__, None
    )
    for key in list(SingletonMeta._instances.keys()):
        if "GenerateState" in str(key):
            SingletonMeta._instances.pop(key, None)


async def call_generate(variant: str, args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    if variant == "old":
        from miles.rollout.sglang_rollout import generate as old_generate
        return await old_generate(args, sample, sampling_params.copy())
    else:
        from miles.rollout.generate_hub.single_turn import generate as new_generate
        state = GenerateState(args)
        input_obj = GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params=sampling_params.copy(),
            evaluation=False,
        )
        output = await new_generate(input_obj)
        return output.samples


@contextmanager
def generate_env(args_kwargs: dict | None = None, process_fn_kwargs: dict | None = None):
    cleanup_singleton()
    try:
        port = find_available_port(30000)
        process_fn = make_process_fn(**(process_fn_kwargs or {}))

        with with_mock_server(
            model_name="Qwen/Qwen3-0.6B",
            process_fn=process_fn,
            port=port,
        ) as mock_server:
            args = make_args(router_port=port, **(args_kwargs or {}))
            yield args, mock_server
    finally:
        cleanup_singleton()


class TestBasicGeneration:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_basic_generation(self, variant):
        with generate_env() as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.response == "\\boxed{8}"
            assert result.response_length == 5
            assert len(result.tokens) == 7 + 5  # prompt + response
            assert result.rollout_log_probs is not None
            assert len(result.rollout_log_probs) == 5
            assert result.status == Sample.Status.COMPLETED

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_empty_response(self, variant):
        with generate_env(process_fn_kwargs={"response_text": ""}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.response == ""
            assert result.response_length == 0
            assert result.rollout_log_probs == []


class TestPromptProcessingPath:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_tokenizer_path(self, variant):
        with generate_env() as (args, mock_server):
            sample = make_sample(prompt="What is 1+7?")
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert len(mock_server.request_log) == 1
            payload = mock_server.request_log[0]
            assert "input_ids" in payload
            assert len(payload["input_ids"]) == 7


class TestMultiTurn:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_first_turn_initializes_tokens(self, variant):
        with generate_env() as (args, mock_server):
            sample = make_sample(tokens=[])
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert len(result.tokens) == 12  # 7 prompt + 5 response
            assert result.tokens[:7] != []  # prompt tokens initialized

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_subsequent_turn_appends_tokens(self, variant):
        with generate_env() as (args, mock_server):
            existing_tokens = [1, 2, 3, 4, 5, 6, 7, 100, 101, 102]  # prompt + previous response
            sample = make_sample(
                tokens=existing_tokens,
                response="previous",
                response_length=3,
            )
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.response == "previous\\boxed{8}"
            assert result.response_length == 3 + 5
            assert len(result.tokens) == len(existing_tokens) + 5

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_multi_turn_max_tokens_adjusted(self, variant):
        with generate_env() as (args, mock_server):
            existing_tokens = [1, 2, 3, 4, 5, 6, 7, 100, 101, 102]
            sample = make_sample(
                tokens=existing_tokens,
                response="prev",
                response_length=3,
            )
            sampling_params = {"max_new_tokens": 10, "temperature": 0.7}

            run(call_generate(variant, args, sample, sampling_params))

            payload = mock_server.request_log[0]
            assert payload["sampling_params"]["max_new_tokens"] == 10 - 3  # adjusted


class TestBoundaryConditions:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_max_new_tokens_zero_returns_truncated(self, variant):
        with generate_env() as (args, mock_server):
            existing_tokens = [1, 2, 3, 4, 5, 6, 7] + list(range(100, 110))
            sample = make_sample(
                tokens=existing_tokens,
                response="x" * 10,
                response_length=10,
            )
            sampling_params = {"max_new_tokens": 10, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.status == Sample.Status.TRUNCATED
            assert len(mock_server.request_log) == 0  # no request sent


class TestFinishReason:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_finish_stop_sets_completed(self, variant):
        with generate_env(process_fn_kwargs={"finish_reason": "stop"}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.status == Sample.Status.COMPLETED

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_finish_length_sets_truncated(self, variant):
        with generate_env(process_fn_kwargs={"finish_reason": "length"}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_finish_abort_sets_aborted(self, variant):
        with generate_env(process_fn_kwargs={"finish_reason": "abort"}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.status == Sample.Status.ABORTED


class TestRoutedExperts:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_routed_experts_disabled(self, variant):
        with generate_env(args_kwargs={"use_rollout_routing_replay": False}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.rollout_routed_experts is None
            payload = mock_server.request_log[0]
            assert payload.get("return_routed_experts", False) is False

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_routed_experts_enabled_and_parsed(self, variant):
        import numpy as np
        num_layers = 2
        moe_router_topk = 4
        num_tokens = 7 + 5  # prompt + response
        routed_experts_array = np.arange(
            (num_tokens - 1) * num_layers * moe_router_topk, dtype=np.int32
        ).reshape(num_tokens - 1, num_layers, moe_router_topk)
        routed_experts_bytes = routed_experts_array.tobytes()

        with generate_env(
            args_kwargs={"use_rollout_routing_replay": True},
            process_fn_kwargs={"routed_experts": routed_experts_bytes}
        ) as (args, mock_server):
            args.num_layers = num_layers
            args.moe_router_topk = moe_router_topk
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.rollout_routed_experts is not None
            assert result.rollout_routed_experts.shape == (num_tokens - 1, num_layers, moe_router_topk)
            np.testing.assert_array_equal(result.rollout_routed_experts, routed_experts_array)


class TestMetaInfo:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_prefix_cache_info_updated(self, variant):
        with generate_env(process_fn_kwargs={"cached_tokens": 3}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.prefix_cache_info.cached_tokens == 3
            assert result.prefix_cache_info.total_prompt_tokens == 7

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_weight_version_collected(self, variant):
        with generate_env(process_fn_kwargs={"weight_version": "v1.0"}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert "v1.0" in result.weight_versions


class TestPayloadStructure:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_payload_has_required_fields(self, variant):
        with generate_env() as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9}

            run(call_generate(variant, args, sample, sampling_params))

            assert len(mock_server.request_log) == 1
            payload = mock_server.request_log[0]
            assert "input_ids" in payload
            assert "sampling_params" in payload
            assert payload.get("return_logprob") is True

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_payload_routed_experts_flag_when_enabled(self, variant):
        with generate_env(args_kwargs={"use_rollout_routing_replay": True}) as (args, mock_server):
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            run(call_generate(variant, args, sample, sampling_params))

            payload = mock_server.request_log[0]
            assert payload.get("return_routed_experts") is True
