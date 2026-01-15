import pytest
from pydantic import TypeAdapter

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


class TestApplyChatTemplateWithTools:
    EXPECTED_PROMPT_WITHOUT_TOOLS = (
        "<|im_start|>user\n"
        "What's the weather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    EXPECTED_PROMPT_WITH_TOOLS = (
        "<|im_start|>system\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        '{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}\n'
        '{"type": "function", "function": {"name": "search", "description": "Search for information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}\n'
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        "<|im_start|>user\n"
        "What's the weather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    @pytest.mark.parametrize(
        "tools,expected",
        [
            pytest.param(None, EXPECTED_PROMPT_WITHOUT_TOOLS, id="without_tools"),
            pytest.param(SAMPLE_TOOLS, EXPECTED_PROMPT_WITH_TOOLS, id="with_tools"),
        ],
    )
    def test_apply_chat_template(self, tools, expected):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        assert prompt == expected


class TestSGLangFunctionCallParser:
    """Test to demonstrate and ensure SGLang function call parser have features we need without breaking changes."""

    @pytest.mark.parametrize(
        "model_output,expected",
        [
            pytest.param(
                'Let me check the weather for you.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
                (
                    "Let me check the weather for you.",
                    [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')],
                ),
                id="single_tool_call",
            ),
            pytest.param(
                "I will search for weather and restaurants.\n"
                '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Shanghai"}}\n</tool_call>\n'
                '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n</tool_call>',
                (
                    "I will search for weather and restaurants.",
                    [
                        ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Shanghai"}'),
                        ToolCallItem(tool_index=1, name="search", parameters='{"query": "restaurants"}'),
                    ],
                ),
                id="multi_tool_calls",
            ),
            pytest.param(
                "The weather is sunny today.",
                ("The weather is sunny today.", []),
                id="no_tool_call",
            ),
        ],
    )
    def test_parse_non_stream(self, model_output, expected):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
        assert parser.parse_non_stream(model_output) == expected
