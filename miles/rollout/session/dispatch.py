"""Dispatch policies for the session chat endpoint.

The judgment half of retry handling: ``classify_extension`` decides how a
request relates to a lineage's stored history without mutating anything, and a
policy function turns that classification into a ``DispatchDecision``. The
mutation half lives in ``LinearTrajectory.apply_rollback`` and runs only on
the dispatched lineage, under the same ``SessionState.lock`` hold.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.linear_trajectory import (
    MAX_ASSISTANT_ROLLBACK_STEPS,
    LinearTrajectory,
    RollbackPlan,
    SessionState,
)
from miles.utils.chat_template_utils import message_matches

logger = logging.getLogger(__name__)


class Kind(Enum):
    EXTEND = "extend"
    ROLLBACK = "rollback"
    DIVERGED = "diverged"


@dataclass(frozen=True)
class Classification:
    kind: Kind
    match_len: int
    rollback: RollbackPlan | None = None
    # DIVERGED diagnostics: the discard_count that made the rollback illegal,
    # or None when the matched prefix holds no own-assistant anchor at all.
    diverged_discard_count: int | None = None


@dataclass(frozen=True)
class DispatchDecision:
    lineage: LinearTrajectory
    rollback: RollbackPlan | None = None


def classify_extension(lineage: LinearTrajectory, request_messages: list[dict[str, Any]]) -> Classification:
    """Judge how *request_messages* relates to the lineage's stored history.

    Pure — never mutates the lineage, so it can be evaluated against any
    number of lineages before a policy commits to one. Anchor rule: only the
    lineage's own generated assistants (those after ``prompt_assistant_count``)
    are rollback checkpoints; assistants carried by the first request are
    prompt, not checkpoints.
    """
    stored = lineage.messages
    if not stored or not lineage.trajectory_token_ids:
        return Classification(Kind.EXTEND, match_len=0)

    match_len = 0
    for i in range(min(len(request_messages), len(stored))):
        if message_matches(stored[i], request_messages[i]):
            match_len = i + 1
        else:
            break

    if match_len >= len(stored):
        return Classification(Kind.EXTEND, match_len)

    # Find the last OWN assistant within the matched prefix.
    assistant_count = 0
    checkpoint_index = -1
    rollback_msg_end = 0
    for i in range(match_len):
        if stored[i].get("role") == "assistant":
            assistant_count += 1
            if assistant_count > lineage.prompt_assistant_count:
                checkpoint_index = assistant_count - lineage.prompt_assistant_count - 1
                rollback_msg_end = i + 1

    if checkpoint_index < 0:
        return Classification(Kind.DIVERGED, match_len)

    discard_count = lineage.num_assistant - (checkpoint_index + 1)
    if discard_count > MAX_ASSISTANT_ROLLBACK_STEPS:
        return Classification(Kind.DIVERGED, match_len, diverged_discard_count=discard_count)

    return Classification(
        Kind.ROLLBACK,
        match_len,
        rollback=RollbackPlan(
            checkpoint_index=checkpoint_index, rollback_msg_end=rollback_msg_end, discard_count=discard_count
        ),
    )


def dispatch_retry(state: SessionState, request_messages: list[dict[str, Any]]) -> DispatchDecision:
    """Today's retry semantics: at most one own-assistant rollback, everything
    else rejected with the historical 400 texts (byte-exact contract, pinned
    by ``TestRollbackPins``)."""
    lineage = state.lineages[0]
    c = classify_extension(lineage, request_messages)
    if c.kind is Kind.DIVERGED:
        if c.diverged_discard_count is None:
            raise MessageValidationError(
                f"rollback failed: no assistant message found in the first "
                f"{c.match_len} matched messages (stored has {len(lineage.messages)} messages, "
                f"request has {len(request_messages)} messages)"
            )
        raise MessageValidationError(
            f"rollback failed: discard_count={c.diverged_discard_count} exceeds "
            f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
            f"(stored has {len(lineage.messages)} messages, "
            f"request has {len(request_messages)} messages)"
        )
    return DispatchDecision(lineage, rollback=c.rollback)
