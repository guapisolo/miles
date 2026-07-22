"""Semantic matrix for the three dispatch policies (disabled / retry / fork).

Pure SessionState-level tests: no HTTP server, no tokenizer — segment state is
constructed directly, dispatch functions are called the way ``SessionCore``
calls them (sequentially, as under the session lock). The retry column's HTTP
fidelity is separately pinned by ``TestRollbackPins`` in ``test_sessions.py``.
"""

import pytest

from miles.rollout.session.dispatch import MAX_SEGMENTS, dispatch_disabled, dispatch_fork, dispatch_retry
from miles.rollout.session.errors import MessageValidationError, TruncatedSegmentError
from miles.rollout.session.linear_trajectory import LinearTrajectory, SessionState
from miles.rollout.session.types import SessionRecord

SYS = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What's the weather in Beijing?"}
ASST_1 = {"role": "assistant", "content": "Let me check."}
TOOL_1 = {"role": "tool", "content": '{"temperature": 25}', "tool_call_id": "call_1"}
ASST_2 = {"role": "assistant", "content": "It's 25°C."}
TOOL_1_DIFF = {"role": "tool", "content": '{"temperature": 99}', "tool_call_id": "call_1"}
TOOL_2 = {"role": "tool", "content": "more", "tool_call_id": "call_2"}
SUBAGENT_SYS = {"role": "system", "content": "You are a search subagent."}
SUBAGENT_USER = {"role": "user", "content": "Find the report."}


def _record(finish_reason: str, timestamp: float) -> SessionRecord:
    return SessionRecord(
        timestamp=timestamp,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={},
        response={"choices": [{"finish_reason": finish_reason, "message": {}}]},
    )


def _committed_segment(messages, num_assistant, *, finish_reason="stop", prompt_assistant_count=0):
    """A segment with committed turns; one checkpoint and record per own assistant."""
    return LinearTrajectory(
        messages=list(messages),
        records=[_record("stop", float(i)) for i in range(num_assistant - 1)]
        + [_record(finish_reason, float(num_assistant - 1))],
        trajectory_token_ids=[[i] for i in range(num_assistant)],
        num_assistant=num_assistant,
        prompt_assistant_count=prompt_assistant_count,
    )


def _two_turn_state(finish_reason: str = "stop") -> SessionState:
    """Stored: [sys, user, asst1, tool1, asst2] with 2 committed turns."""
    segment = _committed_segment([SYS, USER, ASST_1, TOOL_1, ASST_2], 2, finish_reason=finish_reason)
    return SessionState(segments=[segment])


STRICT_EXTENSION = [SYS, USER, ASST_1, TOOL_1, ASST_2, TOOL_2]
DEGENERATE_EQUAL = [SYS, USER, ASST_1, TOOL_1, ASST_2]
PURE_DROP_ONE = [SYS, USER, ASST_1, TOOL_1]
DIVERGENT_ONE = [SYS, USER, ASST_1, TOOL_1_DIFF]
NO_ANCHOR = [SYS, {"role": "user", "content": "a different question"}]
SUBAGENT_FIRST = [SUBAGENT_SYS, SUBAGENT_USER]


class TestSemanticsMatrix:
    """Rows of the design's semantics table, one policy per column."""

    @pytest.mark.parametrize("request_messages", [STRICT_EXTENSION, DEGENERATE_EQUAL])
    def test_extension_accepted_by_all_modes(self, request_messages):
        for policy in (dispatch_disabled, dispatch_retry, dispatch_fork):
            state = _two_turn_state()
            decision = policy(state, request_messages)
            assert decision.segment is state.segments[0]
            assert decision.rollback is None
            assert len(state.segments) == 1

    @pytest.mark.parametrize("request_messages", [PURE_DROP_ONE, DIVERGENT_ONE])
    def test_one_step_shapes(self, request_messages):
        # disabled: 400
        with pytest.raises(MessageValidationError, match="session rollback is disabled"):
            dispatch_disabled(_two_turn_state(), request_messages)

        # retry: destructive rollback plan against the single segment
        state = _two_turn_state()
        decision = dispatch_retry(state, request_messages)
        assert decision.rollback is not None
        assert decision.rollback.discard_count == 1

        # fork: new seeded segment, old segment untouched
        state = _two_turn_state()
        decision = dispatch_fork(state, request_messages)
        assert len(state.segments) == 2
        assert decision.segment is state.segments[1]
        assert decision.segment.seed_messages == request_messages
        assert state.segments[0].num_assistant == 2

    def test_deep_divergence(self):
        segment = _committed_segment(
            [SYS, USER, ASST_1, TOOL_1, ASST_2, TOOL_2, {"role": "assistant", "content": "f"}], 3
        )
        request = [SYS, USER, ASST_1, TOOL_1_DIFF]

        with pytest.raises(MessageValidationError, match="session rollback is disabled"):
            dispatch_disabled(SessionState(segments=[segment]), request)
        with pytest.raises(MessageValidationError, match="exceeds max_assistant_rollback_steps"):
            dispatch_retry(SessionState(segments=[segment]), request)

        state = SessionState(segments=[_committed_segment(segment.messages, 3)])
        decision = dispatch_fork(state, request)
        assert len(state.segments) == 2
        assert decision.segment.seed_messages == request

    def test_zero_overlap_subagent_first_request(self):
        with pytest.raises(MessageValidationError, match="session rollback is disabled"):
            dispatch_disabled(_two_turn_state(), SUBAGENT_FIRST)
        with pytest.raises(MessageValidationError, match="no assistant message found"):
            dispatch_retry(_two_turn_state(), SUBAGENT_FIRST)

        state = _two_turn_state()
        decision = dispatch_fork(state, SUBAGENT_FIRST)
        assert len(state.segments) == 2
        assert decision.segment.seed_messages == SUBAGENT_FIRST

    def test_truncated_tail_extension(self):
        # disabled/retry: accepted exactly as before truncation existed
        for policy in (dispatch_disabled, dispatch_retry):
            state = _two_turn_state(finish_reason="length")
            decision = policy(state, STRICT_EXTENSION)
            assert decision.segment is state.segments[0]

        # fork: 409 — truncation closes the segment
        with pytest.raises(TruncatedSegmentError):
            dispatch_fork(_two_turn_state(finish_reason="length"), STRICT_EXTENSION)

    def test_truncated_segment_still_forkable_before_the_cut(self):
        """T2: diverging before the truncated turn is not a tail extension."""
        state = _two_turn_state(finish_reason="length")
        decision = dispatch_fork(state, DIVERGENT_ONE)
        assert len(state.segments) == 2
        assert decision.segment is state.segments[1]

    def test_segment_cap(self):
        state = _two_turn_state()
        state.segments.extend(
            LinearTrajectory(seed_messages=[{"role": "user", "content": f"task {i}"}]) for i in range(MAX_SEGMENTS - 1)
        )
        with pytest.raises(MessageValidationError, match=rf"segment cap reached \({MAX_SEGMENTS}\)"):
            dispatch_fork(state, SUBAGENT_FIRST)


class TestForkSeedPlaceholder:
    """The seed rule closes the concurrent-sibling race: dispatch happens under
    the session lock, commits (Phase 3) happen later, so every uncommitted
    segment a dispatch returns must already be claimed."""

    def test_empty_session_two_different_first_requests(self):
        """Two concurrent first requests on a fresh session get separate segments."""
        state = SessionState()
        first = dispatch_fork(state, [SYS, USER])
        assert first.segment is state.segments[0]
        assert first.segment.seed_messages == [SYS, USER]

        second = dispatch_fork(state, SUBAGENT_FIRST)
        assert second.segment is not first.segment
        assert len(state.segments) == 2
        assert second.segment.seed_messages == SUBAGENT_FIRST

    def test_concurrent_siblings_fork_separately(self):
        state = _two_turn_state()
        sibling_a = dispatch_fork(state, [SUBAGENT_SYS, {"role": "user", "content": "task A"}])
        sibling_b = dispatch_fork(state, [SUBAGENT_SYS, {"role": "user", "content": "task B"}])
        assert sibling_a.segment is not sibling_b.segment
        assert len(state.segments) == 3

    def test_identical_retry_routes_to_seeded_segment(self):
        """Resending the seed (lost first response) rejoins the same segment."""
        state = _two_turn_state()
        first = dispatch_fork(state, SUBAGENT_FIRST)
        retry = dispatch_fork(state, SUBAGENT_FIRST)
        assert retry.segment is first.segment
        assert len(state.segments) == 2

    def test_seed_extension_routes_to_seeded_segment(self):
        state = _two_turn_state()
        first = dispatch_fork(state, SUBAGENT_FIRST)
        extended = dispatch_fork(state, [*SUBAGENT_FIRST, ASST_1, TOOL_1])
        assert extended.segment is first.segment

    def test_failed_first_turn_leftover_does_not_capture_main_line(self):
        """A seeded-but-never-committed segment must not swallow other traffic."""
        state = _two_turn_state()
        dispatch_fork(state, SUBAGENT_FIRST)  # proxy for this one never succeeds

        main = dispatch_fork(state, STRICT_EXTENSION)
        assert main.segment is state.segments[0]
        assert len(state.segments) == 2

    def test_most_recent_segment_wins_ties(self):
        """Two committed segments both extended by the request: latest commit wins."""
        older = _committed_segment([SYS, USER, ASST_1], 1)
        newer = _committed_segment([SYS, USER, ASST_1], 1)
        older.records[-1] = _record("stop", 1.0)
        newer.records[-1] = _record("stop", 2.0)
        state = SessionState(segments=[older, newer])

        decision = dispatch_fork(state, [SYS, USER, ASST_1, TOOL_1])
        assert decision.segment is newer
