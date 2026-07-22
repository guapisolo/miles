"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── TruncatedSegmentError      → 409  extending a length-truncated segment (fork mode)
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
└── UpstreamResponseError      → 502  SGLang response invalid or unexpected
"""


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class MessageValidationError(SessionError):
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class TruncatedSegmentError(SessionError):
    """Raised when a request extends a segment closed by length truncation.

    Only reachable in fork mode: truncation ends a segment, so its tail can
    never be extended (409 Conflict — the request conflicts with the
    segment's terminal state; 400 stays reserved for structural errors).
    """

    status_code: int = 409


class TokenizationError(SessionError):
    """Raised when TITO tokenization invariants are violated.

    Examples: pretokenized prefix mismatch between stored and new token IDs.
    """

    status_code: int = 500


class UpstreamResponseError(SessionError):
    """Raised when the upstream SGLang response is invalid or unexpected.

    Examples: missing meta_info, assistant content is None,
    output_token_logprobs length mismatch.
    """

    status_code: int = 502
