from dataclasses import dataclass

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample
from tests.fixtures.generation_fixtures import GenerateEnv, generation_env

_ = generation_env

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_answer",
            "description": "Get the answer to a math question",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    }
]


async def mock_execute_tool(parsed_tool_call):
    return {"tool_messages": []}


MULTI_TURN_EXTRA_ARGV = [
    "--generate-max-turns", "4",
    "--generate-max-tool-calls", "4",
    "--generate-tool-specs-path", f"{__name__}:TOOL_SPECS",
    "--generate-tool-call-parser", "qwen25",
    "--execute-tool-function-path", f"{__name__}:mock_execute_tool",
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
