import pytest

from miles.utils.chat_message_utils import get_think_token_end, get_think_token_start, trim_think_tokens


def test_get_think_token_start_end():
    assert get_think_token_start("qwen3") == ("<think>", 151667)
    assert get_think_token_end("qwen3") == ("</think>", 151668)


def test_trim_think_tokens_no_think():
    tokens = [1, 2, 3]
    assert trim_think_tokens(tokens, "qwen3") == tokens


def test_trim_think_tokens_start_only():
    tokens = [1, 151667, 2, 3]
    assert trim_think_tokens(tokens, "qwen3") == [1]


def test_trim_think_tokens_start_and_end():
    tokens = [1, 151667, 2, 151668, 3]
    assert trim_think_tokens(tokens, "qwen3") == [1]


def test_trim_think_tokens_end_without_start():
    tokens = [1, 151668, 2]
    with pytest.raises(ValueError, match="No think token start found"):
        trim_think_tokens(tokens, "qwen3")


def test_trim_think_tokens_multiple_starts():
    tokens = [151667, 1, 151667]
    with pytest.raises(ValueError, match="Multiple think token start found"):
        trim_think_tokens(tokens, "qwen3")


def test_trim_think_tokens_multiple_ends():
    tokens = [151667, 1, 151668, 2, 151668]
    with pytest.raises(ValueError, match="Multiple think token end found"):
        trim_think_tokens(tokens, "qwen3")
