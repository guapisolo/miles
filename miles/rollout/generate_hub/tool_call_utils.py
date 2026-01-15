from typing import Any


DUMMY_USER = {"role": "user", "content": "dummy"}


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dummy assistant message with tool_calls matching the tool responses."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": resp.get("tool_call_id", f"call_dummy_{i}"),
                "type": "function",
                "function": {
                    "name": resp.get("name", "dummy_func"),
                    "arguments": "{}",
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }


def tokenize_tool_responses(
    tool_messages: list[dict[str, Any]],
    tokenizer,
) -> list[list[int]]:
    """
    Tokenize multiple tool response tool_messages.

    Returns a list of token ID lists, one for each tool response.
    """
    dummy_assistant = _build_dummy_assistant(tool_messages)
    base_messages = [DUMMY_USER, dummy_assistant]

    result = []
    for i, tool_response in enumerate(tool_messages):
        messages_without = base_messages + tool_messages[:i]
        messages_with = base_messages + tool_messages[: i + 1]

        tokens_with = tokenizer.apply_chat_template(
            messages_with, tokenize=True, add_generation_prompt=False
        )
        tokens_without = tokenizer.apply_chat_template(
            messages_without, tokenize=True, add_generation_prompt=False
        )

        assert tokens_with[: len(tokens_without)] == tokens_without, (
            "Token prefix mismatch: the tokens without tool should be a prefix of tokens with tool"
        )

        result.append(tokens_with[len(tokens_without) :])

    return result
