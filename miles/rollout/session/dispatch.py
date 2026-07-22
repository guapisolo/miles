"""Dispatch policies for the session chat endpoint.

The judgment half of retry handling: ``classify_extension`` decides how a
request relates to a segment's stored history without mutating anything, and a
policy function turns that classification into a ``DispatchDecision``. The
mutation half lives in ``LinearTrajectory.apply_rollback`` and runs only on
the dispatched segment, under the same ``SessionState.lock`` hold.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from miles.rollout.session.errors import MessageValidationError, TruncatedSegmentError
from miles.rollout.session.linear_trajectory import (
    MAX_ASSISTANT_ROLLBACK_STEPS,
    LinearTrajectory,
    RollbackPlan,
    SessionState,
)
from miles.utils.chat_template_utils import message_matches

logger = logging.getLogger(__name__)

# Fork-mode backstop: a runaway harness (replay drift, scrambled history)
# would otherwise fork a fresh segment on every request, silently and
# forever. Normal sessions (main line + a handful of subagents) stay far
# below this; hitting it is near-certain pathology and fails loud. Hardcoded
# like MAX_ASSISTANT_ROLLBACK_STEPS; promote to a knob if real demand shows.
MAX_SEGMENTS = 64


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
    segment: LinearTrajectory
    rollback: RollbackPlan | None = None


def classify_extension(segment: LinearTrajectory, request_messages: list[dict[str, Any]]) -> Classification:
    """Judge how *request_messages* relates to the segment's stored history.

    Pure — never mutates the segment, so it can be evaluated against any
    number of segments before a policy commits to one. Anchor rule: only the
    segment's own generated assistants (those after ``prompt_assistant_count``)
    are rollback checkpoints; assistants carried by the first request are
    prompt, not checkpoints.
    """
    stored = segment.messages
    if not stored or not segment.trajectory_token_ids:
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
            if assistant_count > segment.prompt_assistant_count:
                checkpoint_index = assistant_count - segment.prompt_assistant_count - 1
                rollback_msg_end = i + 1

    if checkpoint_index < 0:
        return Classification(Kind.DIVERGED, match_len)

    discard_count = segment.num_assistant - (checkpoint_index + 1)
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
    segment = state.segments[0]
    c = classify_extension(segment, request_messages)
    if c.kind is Kind.DIVERGED:
        if c.diverged_discard_count is None:
            raise MessageValidationError(
                f"rollback failed: no assistant message found in the first "
                f"{c.match_len} matched messages (stored has {len(segment.messages)} messages, "
                f"request has {len(request_messages)} messages)"
            )
        raise MessageValidationError(
            f"rollback failed: discard_count={c.diverged_discard_count} exceeds "
            f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
            f"(stored has {len(segment.messages)} messages, "
            f"request has {len(request_messages)} messages)"
        )
    return DispatchDecision(segment, rollback=c.rollback)


def dispatch_disabled(state: SessionState, request_messages: list[dict[str, Any]]) -> DispatchDecision:
    """Strict white-box mode: any request that is not a strict extension of
    the stored history is a harness bug and fails loud."""
    segment = state.segments[0]
    c = classify_extension(segment, request_messages)
    if c.kind is not Kind.EXTEND:
        raise MessageValidationError(
            f"session rollback is disabled (--session-rollback-mode=disabled): "
            f"request must strictly extend the stored history "
            f"(matched {c.match_len} of {len(segment.messages)} stored messages)"
        )
    return DispatchDecision(segment)


def _extends_seed(seed: list[dict[str, Any]], request_messages: list[dict[str, Any]]) -> bool:
    if len(request_messages) < len(seed):
        return False
    return all(message_matches(seed[i], request_messages[i]) for i in range(len(seed)))


def _pick_most_recent(state: SessionState, candidates: list[LinearTrajectory]) -> LinearTrajectory:
    """Tie-break among extension candidates: latest committed turn wins
    (``records[-1].timestamp``); never-committed segments rank by creation
    order. Twin segments with identical stored text stay ambiguous — see the
    design's twin risk note."""
    return max(
        candidates,
        key=lambda segment: (
            segment.records[-1].timestamp if segment.records else float("-inf"),
            state.segments.index(segment),
        ),
    )


def dispatch_fork(state: SessionState, request_messages: list[dict[str, Any]]) -> DispatchDecision:
    """Fork mode: only strict extensions route to an existing segment; every
    other shape (pure-drop, divergence, zero overlap) starts a new segment.
    No destructive rollback ever happens — abandoned turns stay on their
    segment and still produce training samples.

    Placeholder (seed) rule: an uncommitted segment matches only requests
    extending its seed, and any uncommitted segment this function returns has
    a seed. Both halves are load-bearing for concurrency: without them, two
    sibling first-requests in flight would both classify as "first turn" of
    the same empty segment and the loser's sampled tokens would be silently
    dropped by the num_assistant guard in Phase 3.
    """
    extends: list[LinearTrajectory] = []
    best_match_len = 0
    for segment in state.segments:
        if not segment.trajectory_token_ids:
            # Uncommitted: the seed (if any) governs matching; an unseeded
            # segment (the session's root) accepts anything, like a first turn.
            if segment.seed_messages is None or _extends_seed(segment.seed_messages, request_messages):
                extends.append(segment)
            continue
        c = classify_extension(segment, request_messages)
        best_match_len = max(best_match_len, c.match_len)
        if c.kind is Kind.EXTEND:
            extends.append(segment)

    live = [segment for segment in extends if not segment.truncated]
    if live:
        segment = _pick_most_recent(state, live)
        if not segment.trajectory_token_ids and segment.seed_messages is None:
            segment.seed_messages = list(request_messages)
        return DispatchDecision(segment)

    if extends:
        raise TruncatedSegmentError(
            "truncated segment cannot be extended: the matching segment ended with "
            "finish_reason='length' and truncation closes a segment for good"
        )

    if len(state.segments) >= MAX_SEGMENTS:
        raise MessageValidationError(
            f"segment cap reached ({MAX_SEGMENTS}): request does not extend any segment "
            f"and the session cannot fork further — this almost always means the harness "
            f"is not replaying history verbatim"
        )

    segment = LinearTrajectory(seed_messages=list(request_messages))
    state.segments.append(segment)
    logger.info(
        "Forking new segment: request(%d msgs) extends no segment "
        "(best overlap %d msgs), session now has %d segments",
        len(request_messages),
        best_match_len,
        len(state.segments),
    )
    return DispatchDecision(segment)
