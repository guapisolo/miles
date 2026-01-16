from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_PROMPT,
    MULTI_TURN_FIRST_RESPONSE,
    MULTI_TURN_SECOND_RESPONSE,
    SAMPLE_TOOLS,
    mock_execute_tool_function,
    multi_turn_tool_call_process_fn,
)
from miles.utils.types import Sample
from tests.fixtures.generation_fixtures import GenerateEnv, generation_env

_ = generation_env, SAMPLE_TOOLS, mock_execute_tool_function, multi_turn_tool_call_process_fn

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

MULTI_TURN_EXTRA_ARGV = [
    "--generate-max-turns", "4",
    "--generate-max-tool-calls", "4",
    "--generate-tool-specs-path", "miles.utils.test_utils.mock_tools:SAMPLE_TOOLS",
    "--generate-tool-call-parser", "qwen25",
    "--generate-execute-tool-function-path", "miles.utils.test_utils.mock_tools:mock_execute_tool_function",
]


@dataclass
class GenerateResult:
    sample: Sample
    requests: list[dict]


def make_sample(prompt=None):
    return Sample(
        prompt=prompt or [{"role": "user", "content": "What is 1+1?"}],
        tokens=[],
        response="",
        response_length=0,
        status=Sample.Status.PENDING,
    )


async def call_multi_turn_generate(args, sample: Sample, sampling_params: dict) -> Sample:
    from miles.rollout.generate_hub.multi_turn_single_sample import generate

    state = GenerateState(args)
    output = await generate(
        GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params.copy(), evaluation=False)
    )
    return output.samples


def run_generate(env: GenerateEnv, sample: Sample | None = None, sampling_params: dict | None = None):
    env.mock_server.request_log.clear()
    result_sample = run(
        call_multi_turn_generate(env.args, sample or make_sample(), sampling_params or DEFAULT_SAMPLING_PARAMS)
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


class TestBasicMultiTurn:
    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_single_turn_no_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text="The answer is 2.", finish_reason="stop")

        result = run_generate(generation_env)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED
        assert "The answer is 2." in result.sample.response

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_two_turns_with_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        sample = make_sample(prompt=[{"role": "user", "content": MULTI_TURN_FIRST_PROMPT}])
        result = run_generate(generation_env, sample)

        assert len(result.requests) == 2
        assert result.sample.status == Sample.Status.COMPLETED
        assert "2008" in result.sample.response
