"""Tests for TokenSeqComparator with real HuggingFace tokenizers.

Test matrix: {Qwen3-4B, GLM-4.7-Flash} × {segmentation, comparison scenarios}.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.token_seq_comparator import MismatchType, Segment, TokenSeqComparator

# ---------------------------------------------------------------------------
# Model configs — one per tokenizer in the test matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    trust_remote_code: bool
    assistant_start_str: str
    # A few known special tokens we assert are recognised.
    known_special_tokens: tuple[str, ...]


_CONFIGS: dict[str, ModelConfig] = {
    "qwen3_4b": ModelConfig(
        model_id="Qwen/Qwen3-4B",
        trust_remote_code=True,
        assistant_start_str="<|im_start|>assistant",
        known_special_tokens=("<|im_start|>", "<|im_end|>", "<|endoftext|>"),
    ),
    "glm47_flash": ModelConfig(
        model_id="zai-org/GLM-4.7-Flash",
        trust_remote_code=True,
        assistant_start_str="<|assistant|>",
        known_special_tokens=("<|assistant|>", "<|user|>", "<|system|>", "<|observation|>", "<|endoftext|>"),
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class TokenizerEnv:
    """Everything a test needs: tokenizer + pre-built comparator + token IDs."""

    tokenizer: AutoTokenizer
    config: ModelConfig
    comparator: TokenSeqComparator

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_id(self, token_text: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token_text)


_ENV_CACHE: dict[str, TokenizerEnv] = {}


def _build_env(cfg: ModelConfig) -> TokenizerEnv:
    if cfg.model_id in _ENV_CACHE:
        return _ENV_CACHE[cfg.model_id]

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    comp = TokenSeqComparator(
        tok,
        assistant_start_str=cfg.assistant_start_str,
    )

    env = TokenizerEnv(
        tokenizer=tok,
        config=cfg,
        comparator=comp,
    )
    _ENV_CACHE[cfg.model_id] = env
    return env


@pytest.fixture(params=list(_CONFIGS.keys()))
def env(request) -> TokenizerEnv:
    return _build_env(_CONFIGS[request.param])


# ===========================================================================
# _collect_special_ids
# ===========================================================================


class TestCollectSpecialIds:
    def test_known_specials_are_collected(self, env: TokenizerEnv):
        """All tokens the model documents as special are in _collect_special_ids."""
        collected = TokenSeqComparator._collect_special_ids(env.tokenizer)
        for tok_text in env.config.known_special_tokens:
            tid = env.token_id(tok_text)
            assert tid in collected, f"{tok_text} (id={tid}) not in collected special ids"

    def test_regular_tokens_excluded(self, env: TokenizerEnv):
        """Ordinary text tokens are NOT in the special set."""
        collected = TokenSeqComparator._collect_special_ids(env.tokenizer)
        for tid in env.encode("Hello world"):
            assert tid not in collected, f"regular token id={tid} should not be special"

    def test_non_special_added_tokens_excluded(self, env: TokenizerEnv):
        """Added tokens with special=False are NOT collected."""
        collected = TokenSeqComparator._collect_special_ids(env.tokenizer)
        decoder = getattr(env.tokenizer, "added_tokens_decoder", None)
        if decoder:
            for tid, token_obj in decoder.items():
                if not token_obj.special:
                    assert tid not in collected, (
                        f"non-special added token {token_obj.content!r} (id={tid}) "
                        "should not be in collected special ids"
                    )


# ===========================================================================
# segment_by_special_tokens
# ===========================================================================


class TestSegmentation:
    def test_empty(self, env: TokenizerEnv):
        assert env.comparator.segment_by_special_tokens([]) == []

    def test_plain_text_single_segment(self, env: TokenizerEnv):
        ids = env.encode("The quick brown fox jumps over the lazy dog.")
        segs = env.comparator.segment_by_special_tokens(ids)
        assert len(segs) == 1
        assert segs[0].is_special is False
        assert segs[0].token_ids == ids

    def test_special_tokens_create_boundaries(self, env: TokenizerEnv):
        """<special> text <special> → 3 segments (special, content, special)."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        text_ids = env.encode("some content")
        seq = [sp1] + text_ids + [sp2]
        segs = env.comparator.segment_by_special_tokens(seq)
        assert len(segs) == 3
        assert segs[0] == Segment(token_ids=[sp1], is_special=True)
        assert segs[1] == Segment(token_ids=text_ids, is_special=False)
        assert segs[2] == Segment(token_ids=[sp2], is_special=True)

    def test_consecutive_specials(self, env: TokenizerEnv):
        """Multiple adjacent specials each get their own segment."""
        sp_ids = [env.token_id(t) for t in env.config.known_special_tokens[:3]]
        segs = env.comparator.segment_by_special_tokens(sp_ids)
        assert len(segs) == len(sp_ids)
        assert all(s.is_special for s in segs)
        for seg, expected_id in zip(segs, sp_ids, strict=False):
            assert seg.token_ids == [expected_id]


# ===========================================================================
# compare_sequences — identical
# ===========================================================================


class TestCompareIdentical:
    def test_identical_plain_text(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        ids = [sp] + env.encode("Hello world") + [sp]
        assert env.comparator.compare_sequences(ids, ids) == []

    def test_both_empty(self, env: TokenizerEnv):
        assert env.comparator.compare_sequences([], []) == []


# ===========================================================================
# compare_sequences — SPECIAL_TOKEN mismatches
# ===========================================================================


class TestCompareSpecialToken:
    def test_different_segment_count(self, env: TokenizerEnv):
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        text_ids = env.encode("hi")
        expected = [sp1] + text_ids + [sp2]
        actual = [sp1] + text_ids  # missing trailing special
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_COUNT
        assert result[0].segment_index == -1
        assert "segment count differs" in result[0].detail

    def test_structure_pattern_differs(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        text_ids = env.encode("hello")
        # expected: [special, content, special]
        # actual:   [content, special, content] (same length but different pattern)
        expected = [sp] + text_ids + [sp]
        actual = text_ids + [sp] + text_ids
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) >= 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_COUNT

    def test_special_id_differs(self, env: TokenizerEnv):
        """Same structure, but one special token is swapped for another."""
        sp_tokens = env.config.known_special_tokens
        if len(sp_tokens) < 2:
            pytest.skip("Need at least 2 known special tokens")
        sp_a = env.token_id(sp_tokens[0])
        sp_b = env.token_id(sp_tokens[1])
        text_ids = env.encode("content")
        expected = [sp_a] + text_ids + [sp_a]
        actual = [sp_b] + text_ids + [sp_a]
        result = env.comparator.compare_sequences(expected, actual)
        assert any(m.type == MismatchType.SPECIAL_TOKEN_TYPE and m.segment_index == 0 for m in result)


# ===========================================================================
# compare_sequences — content text mismatches (NON_ASSISTANT_TEXT)
# ===========================================================================


class TestCompareNonAssistantText:
    def test_content_text_differs(self, env: TokenizerEnv):
        """Non-assistant content mismatch → NON_ASSISTANT_TEXT."""
        # Use a non-assistant special token (index 1: <|im_end|> for Qwen3, <|user|> for GLM)
        sp = env.token_id(env.config.known_special_tokens[1])
        expected = [sp] + env.encode("Hello world") + [sp]
        actual = [sp] + env.encode("Goodbye world") + [sp]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.NON_ASSISTANT_TEXT
        assert result[0].segment_index == 1


# ===========================================================================
# _is_assistant_content
# ===========================================================================


@pytest.fixture
def qwen3_env() -> TokenizerEnv:
    return _build_env(_CONFIGS["qwen3_4b"])


@pytest.fixture
def glm47_env() -> TokenizerEnv:
    return _build_env(_CONFIGS["glm47_flash"])


class TestIsAssistantContent:
    def test_qwen3_assistant_content(self, qwen3_env: TokenizerEnv):
        """Qwen3: content after <|im_start|> starting with 'assistant' → True."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        content_ids = env.encode("assistant\nHello!")
        segs = env.comparator.segment_by_special_tokens([im_start] + content_ids + [im_end])
        # seg 0 = <|im_start|>, seg 1 = content, seg 2 = <|im_end|>
        assert env.comparator._is_assistant_content(segs, 1) is True

    def test_qwen3_user_content_not_assistant(self, qwen3_env: TokenizerEnv):
        """Qwen3: content after <|im_start|> starting with 'user' → False."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        content_ids = env.encode("user\nHello!")
        segs = env.comparator.segment_by_special_tokens([im_start] + content_ids + [im_end])
        assert env.comparator._is_assistant_content(segs, 1) is False

    def test_glm_assistant_content(self, glm47_env: TokenizerEnv):
        """GLM: content after <|assistant|> → True."""
        env = glm47_env
        assistant_id = env.token_id("<|assistant|>")
        user_id = env.token_id("<|user|>")
        content_ids = env.encode("Hello!")
        segs = env.comparator.segment_by_special_tokens([assistant_id] + content_ids + [user_id])
        assert env.comparator._is_assistant_content(segs, 1) is True

    def test_glm_user_content_not_assistant(self, glm47_env: TokenizerEnv):
        """GLM: content after <|user|> → False."""
        env = glm47_env
        user_id = env.token_id("<|user|>")
        assistant_id = env.token_id("<|assistant|>")
        content_ids = env.encode("Hello!")
        segs = env.comparator.segment_by_special_tokens([user_id] + content_ids + [assistant_id])
        assert env.comparator._is_assistant_content(segs, 1) is False

    def test_special_segment_returns_false(self, env: TokenizerEnv):
        """Calling on a special segment → False (is_special guard)."""
        sp = env.token_id(env.config.known_special_tokens[0])
        content_ids = env.encode("text")
        segs = env.comparator.segment_by_special_tokens([sp] + content_ids + [sp])
        assert env.comparator._is_assistant_content(segs, 0) is False
        assert env.comparator._is_assistant_content(segs, 2) is False

    def test_content_at_index_zero(self, env: TokenizerEnv):
        """Content at index 0 (no preceding segment) → False."""
        content_ids = env.encode("some text")
        sp = env.token_id(env.config.known_special_tokens[0])
        segs = env.comparator.segment_by_special_tokens(content_ids + [sp])
        assert env.comparator._is_assistant_content(segs, 0) is False


# ===========================================================================
# compare_sequences — assistant text classification
# ===========================================================================


class TestAssistantTextClassification:
    def test_qwen3_assistant_mismatch(self, qwen3_env: TokenizerEnv):
        """Qwen3: mismatch in assistant content → ASSISTANT_TEXT."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        expected = [im_start] + env.encode("assistant\nHello") + [im_end]
        actual = [im_start] + env.encode("assistant\nGoodbye") + [im_end]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.ASSISTANT_TEXT

    def test_qwen3_user_mismatch(self, qwen3_env: TokenizerEnv):
        """Qwen3: mismatch in user content → NON_ASSISTANT_TEXT."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        expected = [im_start] + env.encode("user\nHello") + [im_end]
        actual = [im_start] + env.encode("user\nGoodbye") + [im_end]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.NON_ASSISTANT_TEXT

    def test_glm_assistant_mismatch(self, glm47_env: TokenizerEnv):
        """GLM: mismatch in assistant content → ASSISTANT_TEXT."""
        env = glm47_env
        assistant_id = env.token_id("<|assistant|>")
        user_id = env.token_id("<|user|>")
        expected = [assistant_id] + env.encode("Hello") + [user_id]
        actual = [assistant_id] + env.encode("Goodbye") + [user_id]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.ASSISTANT_TEXT

    def test_glm_user_mismatch(self, glm47_env: TokenizerEnv):
        """GLM: mismatch in user content → NON_ASSISTANT_TEXT."""
        env = glm47_env
        user_id = env.token_id("<|user|>")
        assistant_id = env.token_id("<|assistant|>")
        expected = [user_id] + env.encode("Hello") + [assistant_id]
        actual = [user_id] + env.encode("Goodbye") + [assistant_id]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.NON_ASSISTANT_TEXT


# ===========================================================================
# trim_trailing_ids
# ===========================================================================


class TestTrimTrailingIds:
    def test_trim_removes_trailing_specials(self, env: TokenizerEnv):
        """Trailing eos-like tokens are stripped before comparison."""
        sp = env.token_id(env.config.known_special_tokens[0])
        eos = env.token_id(env.config.known_special_tokens[-1])
        text_ids = env.encode("same content")
        expected = [sp] + text_ids + [sp]
        actual = [sp] + text_ids + [sp, eos, eos]  # extra trailing eos
        # Without trim → segment count differs
        result_no_trim = env.comparator.compare_sequences(expected, actual)
        assert len(result_no_trim) > 0
        # With trim → matches
        result_trim = env.comparator.compare_sequences(expected, actual, trim_trailing_ids={eos})
        assert result_trim == []

    def test_trim_does_not_affect_middle(self, env: TokenizerEnv):
        """trim_trailing_ids only strips from the end, not the middle."""
        sp = env.token_id(env.config.known_special_tokens[0])
        eos = env.token_id(env.config.known_special_tokens[-1])
        text_ids = env.encode("content")
        # eos in the middle should not be stripped
        seq = [sp] + text_ids + [eos] + text_ids + [sp]
        result = env.comparator.compare_sequences(seq, seq, trim_trailing_ids={eos})
        assert result == []


# ===========================================================================
# Mixed mismatches — realistic sequences
# ===========================================================================


class TestCompareMixed:
    def test_assistant_and_non_assistant_mismatch(self, qwen3_env: TokenizerEnv):
        """A sequence with both assistant and non-assistant content mismatches."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")

        # user turn (non-assistant) + assistant turn
        expected = (
            [im_start]
            + env.encode("user\nHello")
            + [im_end]
            + [im_start]
            + env.encode("assistant\nHi there")
            + [im_end]
        )
        actual = (
            [im_start]
            + env.encode("user\nGoodbye")
            + [im_end]
            + [im_start]
            + env.encode("assistant\nBye there")
            + [im_end]
        )

        result = env.comparator.compare_sequences(expected, actual)
        types = {m.type for m in result}
        assert MismatchType.NON_ASSISTANT_TEXT in types
        assert MismatchType.ASSISTANT_TEXT in types


# ===========================================================================
# GLM 4.7 specific: <|user|> vs <|observation|> type mismatch
# ===========================================================================


class TestGlm47SpecialTokenType:
    """After stripping stop tokens from pretokenized (in sessions.py), all
    special-token type differences are fatal for every model — no exceptions.
    """

    def test_user_vs_observation_is_type_mismatch(self, glm47_env: TokenizerEnv):
        """Swapping <|user|> for <|observation|> at the same position → TYPE mismatch."""
        env = glm47_env
        user_id = env.token_id("<|user|>")
        obs_id = env.token_id("<|observation|>")
        assistant_id = env.token_id("<|assistant|>")
        text_ids = env.encode("some content")

        expected = [assistant_id] + text_ids + [user_id] + text_ids + [assistant_id]
        actual = [assistant_id] + text_ids + [obs_id] + text_ids + [assistant_id]

        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_TYPE
        assert result[0].segment_index == 2
