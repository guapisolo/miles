"""Incremental message tokenization for pretokenized prefix reuse.

``AdditionalMessageTokenizer`` computes the token IDs for messages appended
after an already-tokenized prefix.  The default implementation uses a
dummy-message diff (mirrors sglang's ``calc_additional_message_tokenization_by_dummy``).
Model-specific subclasses handle quirks such as GLM 4.7's ambiguous boundary
tokens and lack of mid-conversation system message support.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from miles.utils.chat_template_utils.template import apply_chat_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy message helpers (mirrors sglang utils.py)
# ---------------------------------------------------------------------------

_DUMMY_USER: dict[str, Any] = {"role": "user", "content": "dummy"}


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dummy assistant message whose tool_calls match *tool_responses*."""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": resp.get("tool_call_id") or f"call0000{i}",
                "type": "function",
                "function": {
                    "name": resp.get("name") or "dummy_func",
                    "arguments": {},
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }


def _build_dummy_tool_response() -> dict[str, Any]:
    """Build a minimal dummy tool response."""
    return {
        "role": "tool",
        "content": "",
        "tool_call_id": "call0000",
        "type": "function",
        "function": {
            "name": "dummy_func",
            "arguments": {},
        },
    }


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class AdditionalMessageTokenizer(ABC):
    """Base class for computing incremental token IDs for new messages."""

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    @abstractmethod
    def tokenize_additional(
        self,
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for *new_messages*.

        Args:
            new_messages: Messages to tokenize (tool responses, system
                injections, etc.) appended after the pretokenized prefix.
            pretokenized_token_ids: Token IDs covering everything before
                *new_messages*, including the last assistant turn's
                completion tokens.
            tools: Tool definitions in OpenAI format (may vary per call).

        Returns:
            Incremental token IDs that, when concatenated to
            *pretokenized_token_ids*, form the full prompt token IDs.
        """
        ...

    def preprocess_messages(
        self,
        all_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Optional hook to transform messages before template application.

        Default: identity (no-op).  GLM47 overrides to merge mid-conversation
        system messages into user messages.
        """
        return all_messages


# ---------------------------------------------------------------------------
# Default implementation (dummy-prefix diff)
# ---------------------------------------------------------------------------


class DefaultAdditionalMessageTokenizer(AdditionalMessageTokenizer):
    """Dummy-prefix diff approach.

    1. Build dummy base: ``[dummy_user, dummy_assistant]``
    2. ``tokens_without`` = tokenize base (no generation prompt)
    3. ``tokens_with``    = tokenize base + *new_messages* (with generation prompt)
    4. ``incremental_ids  = tokens_with[len(tokens_without):]``
    5. Call ``postprocess`` hook (identity by default, subclasses may override)
    """

    def tokenize_additional(
        self,
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        dummy_assistant = _build_dummy_assistant(new_messages)
        base_messages = [_DUMMY_USER, dummy_assistant]

        tokens_without = apply_chat_template(
            base_messages,
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            **self.chat_template_kwargs,
        )
        tokens_with = apply_chat_template(
            base_messages + list(new_messages),
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            **self.chat_template_kwargs,
        )

        incremental_ids = list(tokens_with[len(tokens_without) :])

        return self.postprocess(incremental_ids, tokens_without, pretokenized_token_ids)

    def postprocess(
        self,
        incremental_ids: list[int],
        tokens_without: list[int],
        pretokenized_token_ids: list[int],
    ) -> list[int]:
        """Post-process incremental IDs.  Default: return as-is."""
        return incremental_ids


# ---------------------------------------------------------------------------
# Qwen3 implementation (trailing whitespace alignment)
# ---------------------------------------------------------------------------


class Qwen3AdditionalMessageTokenizer(DefaultAdditionalMessageTokenizer):
    """Qwen3 variant: prepends trailing whitespace token for alignment.

    Qwen3 templates insert a trailing ``\\n`` after the last assistant
    message (loss-mask 0).  When this token differs from the real
    pretokenized suffix, we prepend it to ``incremental_ids`` so that
    concatenation with the stored prefix produces the correct sequence.
    """

    def postprocess(
        self,
        incremental_ids: list[int],
        tokens_without: list[int],
        pretokenized_token_ids: list[int],
    ) -> list[int]:
        if (
            pretokenized_token_ids
            and tokens_without
            and tokens_without[-1] != pretokenized_token_ids[-1]
            and self.tokenizer.decode([tokens_without[-1]]).strip() == ""
        ):
            incremental_ids = [tokens_without[-1]] + incremental_ids

        return incremental_ids


# ---------------------------------------------------------------------------
# GLM 4.7 implementation
# ---------------------------------------------------------------------------


class GLM47AdditionalMessageTokenizer(AdditionalMessageTokenizer):
    """GLM 4.7 specific incremental tokenizer.

    Handles two GLM 4.7 quirks:

    1. **Stop-token ambiguity**: ``<|user|>`` and ``<|observation|>`` are both
       assistant stop tokens *and* next-message start tokens in the chat
       template.  After computing the dummy-prefix diff, we strip these
       boundary tokens from ``tokens_without`` when they don't match the
       real pretokenized suffix.

    2. **Mid-conversation system messages**: The GLM 4.7 template does not
       support system messages after position 0.  ``preprocess_messages``
       merges them into adjacent user messages.
    """

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(tokenizer, chat_template_kwargs)
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}

    # -- preprocess: merge mid-conversation system messages -----------------

    def preprocess_messages(
        self,
        all_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge mid-conversation system messages into user messages.

        The first system message (position 0) is kept as-is.  Any later
        system messages are prepended to the next user message's content.
        If no subsequent user message exists, the system content is wrapped
        in a standalone user message.
        """
        result: list[dict[str, Any]] = []
        pending_system: list[str] = []
        first_system_seen = False

        for msg in all_messages:
            if msg["role"] == "system":
                if not first_system_seen:
                    result.append(msg)
                    first_system_seen = True
                else:
                    pending_system.append(msg["content"])
            else:
                if not first_system_seen:
                    first_system_seen = True

                if pending_system and msg["role"] == "user":
                    merged = "\n\n".join(pending_system + [msg["content"]])
                    result.append({**msg, "content": merged})
                    pending_system = []
                else:
                    if pending_system:
                        result.append({"role": "user", "content": "\n\n".join(pending_system)})
                        pending_system = []
                    result.append(msg)

        if pending_system:
            result.append({"role": "user", "content": "\n\n".join(pending_system)})

        return result

    # -- tokenize: boundary-aware dummy diff --------------------------------

    def tokenize_additional(
        self,
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        new_messages = self.preprocess_messages(list(new_messages))

        dummy_assistant = _build_dummy_assistant(new_messages)
        # A dummy tool response anchors the boundary so that <|observation|>
        # (used both as assistant stop token and tool-response start token)
        # produces a stable prefix.
        base_messages = [_DUMMY_USER, dummy_assistant, _build_dummy_tool_response()]

        tokens_without = apply_chat_template(
            base_messages,
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            **self.chat_template_kwargs,
        )
        tokens_with = apply_chat_template(
            base_messages + list(new_messages),
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            **self.chat_template_kwargs,
        )

        # Strip ambiguous boundary tokens (<|user|>, <|observation|>) that the
        # template appended to tokens_without but that do not appear at the
        # end of the real pretokenized sequence.
        strip_count = 0
        if (
            tokens_without
            and pretokenized_token_ids
            and tokens_without[-1] in self._ambiguous_boundary_ids
            and tokens_without[-1] != pretokenized_token_ids[-1]
        ):
            strip_count = 1

        adjusted_len = len(tokens_without) - strip_count
        return list(tokens_with[adjusted_len:])


# ---------------------------------------------------------------------------
# Enum + Registry + Factory
# ---------------------------------------------------------------------------


class AdditionalMessageTokenizerType(str, Enum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    GLM47 = "glm47"


_TOKENIZER_REGISTRY: dict[AdditionalMessageTokenizerType, type[AdditionalMessageTokenizer]] = {
    AdditionalMessageTokenizerType.DEFAULT: DefaultAdditionalMessageTokenizer,
    AdditionalMessageTokenizerType.QWEN3: Qwen3AdditionalMessageTokenizer,
    AdditionalMessageTokenizerType.GLM47: GLM47AdditionalMessageTokenizer,
}


def get_additional_message_tokenizer(
    tokenizer: Any,
    tokenizer_type: AdditionalMessageTokenizerType | str = AdditionalMessageTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> AdditionalMessageTokenizer:
    """Create an ``AdditionalMessageTokenizer`` instance.

    Args:
        tokenizer: HuggingFace tokenizer object.
        tokenizer_type: Explicit type (string or enum).  Corresponds to the
            ``--additional-tokenizer`` CLI argument.
        chat_template_kwargs: Extra kwargs forwarded to ``apply_chat_template``.
    """
    if isinstance(tokenizer_type, str):
        resolved = AdditionalMessageTokenizerType(tokenizer_type)
    else:
        resolved = tokenizer_type

    cls = _TOKENIZER_REGISTRY[resolved]
    return cls(tokenizer, chat_template_kwargs=chat_template_kwargs)
