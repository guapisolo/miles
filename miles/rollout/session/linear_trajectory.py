import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils import assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


# TODO: hardcoded to 1 for now; if multi-step rollback is actually needed,
#  raise this limit or make it configurable and remove the restriction.
MAX_ASSISTANT_ROLLBACK_STEPS = 1


@dataclass(frozen=True)
class RollbackPlan:
    """Mutation recipe for ``LinearTrajectory.apply_rollback``.

    Produced by ``dispatch.classify_extension`` (the judgment half of retry
    handling). ``checkpoint_index`` indexes the lineage's own generated
    assistants (= ``trajectory_token_ids``); ``discard_count`` counts the own
    assistants dropped.
    """

    checkpoint_index: int
    rollback_msg_end: int
    discard_count: int


@dataclass
class LinearTrajectory:
    """State for a linear trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, …],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the dispatch layer rolls the trajectory back at most one
    assistant step (``dispatch.classify_extension`` + ``apply_rollback``).

    Concurrency contract: all mutating methods must be called under the owning
    ``SessionState.lock``.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    num_assistant: int = 0
    # Assistants carried by the FIRST request (few-shot examples, replayed
    # foreign history) are prompt, not checkpoints: they have no entry in
    # trajectory_token_ids and are never rollback anchors.
    prompt_assistant_count: int = 0

    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord) -> None:
        self.records.append(record)

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
    ) -> list[int]:
        """Build the full prompt input_ids for *request_messages*.

        On the first turn (no stored token_ids), renders *request_messages*
        from scratch via the chat template.  On subsequent turns, validates
        that *request_messages* extends the stored history and reuses the
        stored token_ids as the pretokenized prefix.  Retry judgment happens
        BEFORE this call: the dispatch layer classifies the request and
        applies any ``RollbackPlan`` via ``apply_rollback``.

        Must be called under the owning ``SessionState.lock``.
        """
        if not self.token_ids:
            return tito_tokenizer.apply_chat_template(
                request_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )

        # Confirm the (possibly rolled-back) stored messages are a prefix of request,
        # and that each appended message role is in tito_tokenizer.allowed_append_roles.
        try:
            assert_messages_append_only_with_allowed_role(
                self.messages, request_messages, tito_tokenizer.allowed_append_roles
            )
        except ValueError as e:
            raise MessageValidationError(f"{e}; to allow more roles use --tito-allowed-append-roles") from e

        return tito_tokenizer.merge_tokens(
            old_messages=self.messages,
            new_messages=request_messages,
            pretokenized_token_ids=self.token_ids,
            tools=tools,
        )

    def update_pretokenized_state(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        max_trim_tokens: int,
    ) -> None:
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as a new checkpoint.
        Validates that the previously stored token_ids are a prefix of the new
        checkpoint (tolerating up to ``max_trim_tokens`` trailing differences).
        Must be called under the owning ``SessionState.lock``.
        """
        all_token_ids = prompt_token_ids + completion_token_ids

        prev = self.token_ids
        if prev:
            check_len = len(prev) - max_trim_tokens
            if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
                first_mismatch = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True))
                        if a != b
                    ),
                    min(len(all_token_ids), check_len),
                )
                raise TokenizationError(
                    f"pretokenized prefix mismatch: "
                    f"stored {len(prev)} tokens (checking first {check_len}, "
                    f"allowing {max_trim_tokens} trailing) are not a prefix of "
                    f"prompt_token_ids + completion_token_ids "
                    f"({len(all_token_ids)} tokens), "
                    f"first mismatch at index {first_mismatch}, "
                    f"matched {first_mismatch}/{check_len} prefix tokens\n"
                    f"request_messages={request_messages}\n"
                    f"assistant_message={assistant_message}"
                )

        if not self.trajectory_token_ids:
            self.prompt_assistant_count = sum(1 for m in request_messages if m.get("role") == "assistant")
        self.messages = list(request_messages) + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.num_assistant += 1

    def apply_rollback(self, plan: RollbackPlan) -> None:
        """Truncate messages/checkpoints/records back to the plan's checkpoint.

        The mutation half of retry handling; the judgment half is
        ``dispatch.classify_extension``, which produced *plan* against this
        trajectory's current state under the same lock hold.

        Example — agent retries after the first tool call::

            stored:  [sys, user, assistant₁, tool₁, assistant₂]
                      ───────────────────── ▲
                      checkpoint 0 (assistant₁)   checkpoint 1 (assistant₂)

            request: [sys, user, assistant₁, tool₁_different, ...]
                                             ↑ diverges here (index 3)

            plan: checkpoint_index=0, rollback_msg_end=3, discard_count=1

            After rollback:
              messages             = [sys, user, assistant₁]
              trajectory_token_ids = [checkpoint_0_ids]
              records              = [record_0]
              num_assistant        = 1

        Must be called under the owning ``SessionState.lock``.
        """
        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> "
            "checkpoint %d (messages[:%d]), discarding %d assistant(s)",
            len(self.messages),
            self.num_assistant,
            plan.checkpoint_index,
            plan.rollback_msg_end,
            plan.discard_count,
        )
        self.messages = self.messages[: plan.rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: plan.checkpoint_index + 1]
        self.records = self.records[: plan.checkpoint_index + 1]
        self.num_assistant = plan.checkpoint_index + 1


@dataclass
class SessionState:
    """Per-session concurrency container plus its trajectory lineages.

    Owns the lock/closing gate (previously on ``LinearTrajectory``); the lock
    guards the lineage list and every trajectory's state. Today each session
    holds exactly one lineage.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    lineages: list[LinearTrajectory] = field(default_factory=lambda: [LinearTrajectory()])


class SessionRegistry:
    """Session ID -> session state mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch).
    Does NOT mutate session state - all mutations are methods on
    LinearTrajectory; called by the route handler under ``SessionState.lock``.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer):
        self.sessions: dict[str, SessionState] = {}
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SessionState()
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

    def compute_session_mismatch(self, trajectory: LinearTrajectory) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Read-only: does not mutate trajectory state.
        """
        if not trajectory.token_ids:
            return None
        try:
            tools = trajectory.records[-1].request.get("tools") if trajectory.records else None
            expected_ids = self.tito_tokenizer.apply_chat_template(
                trajectory.messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
            mismatches = self.comparator.compare_sequences(expected_ids, trajectory.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e
