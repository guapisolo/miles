"""Tests for TITOTokenizer.create_comparator() and compare_sequences().

Verifies that model-specific subclasses produce correctly-configured
comparators with the right assistant detection and trailing-ID trimming.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.tito_tokenizer import (
    GLM47TITOTokenizer,
    Qwen3TITOTokenizer,
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str, trust_remote_code: bool = True) -> AutoTokenizer:
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return _TOK_CACHE[model_id]


@pytest.fixture
def qwen3_tito() -> Qwen3TITOTokenizer:
    tok = _get_tokenizer("Qwen/Qwen3-4B")
    return Qwen3TITOTokenizer(tok)


@pytest.fixture
def glm47_tito() -> GLM47TITOTokenizer:
    tok = _get_tokenizer("zai-org/GLM-4.7-Flash")
    return GLM47TITOTokenizer(tok)


@pytest.fixture
def default_tito() -> TITOTokenizer:
    tok = _get_tokenizer("Qwen/Qwen3-4B")
    return TITOTokenizer(tok)


# ---------------------------------------------------------------------------
# create_comparator — assistant detection config
# ---------------------------------------------------------------------------


class TestCreateComparator:
    def test_qwen3_has_assistant_config(self, qwen3_tito: Qwen3TITOTokenizer):
        comp = qwen3_tito.create_comparator()
        assert comp._assistant_start_str == "<|im_start|>assistant"

    def test_glm47_has_assistant_config(self, glm47_tito: GLM47TITOTokenizer):
        comp = glm47_tito.create_comparator()
        assert comp._assistant_start_str == "<|assistant|>"

    def test_default_has_no_assistant_config(self, default_tito: TITOTokenizer):
        comp = default_tito.create_comparator()
        assert comp._assistant_start_str is None

    def test_returns_new_instance(self, qwen3_tito: Qwen3TITOTokenizer):
        """Each call creates a fresh comparator."""
        comp1 = qwen3_tito.create_comparator()
        comp2 = qwen3_tito.create_comparator()
        assert comp1 is not comp2


# ---------------------------------------------------------------------------
# Subclass init — boundary token IDs
# ---------------------------------------------------------------------------


class TestSubclassInit:
    def test_qwen3_boundary_tokens(self, qwen3_tito: Qwen3TITOTokenizer):
        tok = qwen3_tito.tokenizer
        assert qwen3_tito._im_end_id == tok.convert_tokens_to_ids("<|im_end|>")
        assert qwen3_tito._newline_id == tok.encode("\n", add_special_tokens=False)[0]

    def test_glm47_boundary_tokens(self, glm47_tito: GLM47TITOTokenizer):
        tok = glm47_tito.tokenizer
        assert glm47_tito._user_id == tok.convert_tokens_to_ids("<|user|>")
        assert glm47_tito._observation_id == tok.convert_tokens_to_ids("<|observation|>")
        assert glm47_tito._ambiguous_boundary_ids == {glm47_tito._user_id, glm47_tito._observation_id}


# ---------------------------------------------------------------------------
# get_comparator_ignore_trailing_ids / max_trim_tokens
# ---------------------------------------------------------------------------


class TestTrailingConfig:
    def test_default_no_trailing(self, default_tito: TITOTokenizer):
        assert default_tito.get_comparator_ignore_trailing_ids() is None
        assert default_tito.max_trim_tokens == 0

    def test_qwen3_trailing_newline(self, qwen3_tito: Qwen3TITOTokenizer):
        trailing = qwen3_tito.get_comparator_ignore_trailing_ids()
        assert trailing == {qwen3_tito._newline_id}

    def test_glm47_trailing_boundary(self, glm47_tito: GLM47TITOTokenizer):
        trailing = glm47_tito.get_comparator_ignore_trailing_ids()
        assert trailing == {glm47_tito._user_id, glm47_tito._observation_id}
        assert glm47_tito.max_trim_tokens == 1


# ---------------------------------------------------------------------------
# compare_sequences — convenience method
# ---------------------------------------------------------------------------


class TestCompareSequences:
    def test_identical_sequences(self, qwen3_tito: Qwen3TITOTokenizer):
        ids = list(range(10))
        assert qwen3_tito.compare_sequences(ids, ids) == []

    def test_caches_comparator(self, qwen3_tito: Qwen3TITOTokenizer):
        """compare_sequences lazily creates and reuses the comparator."""
        ids = list(range(5))
        qwen3_tito.compare_sequences(ids, ids)
        comp = qwen3_tito._comparator
        qwen3_tito.compare_sequences(ids, ids)
        assert qwen3_tito._comparator is comp

    def test_qwen3_trailing_newline_trimmed(self, qwen3_tito: Qwen3TITOTokenizer):
        """Qwen3's trailing newline should be trimmed before comparison."""
        tok = qwen3_tito.tokenizer
        nl_id = tok.encode("\n", add_special_tokens=False)[0]
        base = [tok.convert_tokens_to_ids("<|im_start|>"), 100, tok.convert_tokens_to_ids("<|im_end|>")]
        with_nl = base + [nl_id]
        # With trailing newline trimming, these should be equivalent.
        assert qwen3_tito.compare_sequences(base, with_nl) == []

    def test_glm47_trailing_boundary_trimmed(self, glm47_tito: GLM47TITOTokenizer):
        """GLM47's ambiguous boundary tokens should be trimmed before comparison."""
        tok = glm47_tito.tokenizer
        user_id = tok.convert_tokens_to_ids("<|user|>")
        obs_id = tok.convert_tokens_to_ids("<|observation|>")
        base = [tok.convert_tokens_to_ids("<|assistant|>"), 100]
        # Both trailing <|user|> and <|observation|> should be trimmed.
        assert glm47_tito.compare_sequences(base, base + [user_id]) == []
        assert glm47_tito.compare_sequences(base, base + [obs_id]) == []


# ---------------------------------------------------------------------------
# get_tito_tokenizer factory — assistant config propagated
# ---------------------------------------------------------------------------


class TestFactoryAssistantConfig:
    def test_factory_qwen3(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type="qwen3")
        assert isinstance(tito, Qwen3TITOTokenizer)
        comp = tito.create_comparator()
        assert comp._assistant_start_str == "<|im_start|>assistant"

    def test_factory_glm47(self):
        tok = _get_tokenizer("zai-org/GLM-4.7-Flash")
        tito = get_tito_tokenizer(tok, tokenizer_type="glm47")
        assert isinstance(tito, GLM47TITOTokenizer)
        comp = tito.create_comparator()
        assert comp._assistant_start_str == "<|assistant|>"

    def test_factory_default(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type="default")
        assert isinstance(tito, TITOTokenizer)
        comp = tito.create_comparator()
        assert comp._assistant_start_str is None

    def test_factory_enum_input(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type=TITOTokenizerType.QWEN3)
        assert isinstance(tito, Qwen3TITOTokenizer)

    def test_factory_invalid_type(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        with pytest.raises(ValueError):
            get_tito_tokenizer(tok, tokenizer_type="nonexistent")
