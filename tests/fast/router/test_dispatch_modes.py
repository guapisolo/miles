"""Semantic matrix for the three dispatch policies (disabled / retry / fork).

Pure SessionState-level tests: no HTTP server, no tokenizer — lineage state is
constructed directly, dispatch functions are called the way ``SessionCore``
calls them (sequentially, as under the session lock). The retry column's HTTP
fidelity is separately pinned by ``TestRollbackPins`` in ``test_sessions.py``.
"""

import pytest

from miles.rollout.session.dispatch import MAX_LINEAGES, dispatch_disabled, dispatch_fork, dispatch_retry
from miles.rollout.session.errors import MessageValidationError, TruncatedLineageError
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


def _committed_lineage(messages, num_assistant, *, finish_reason="stop", prompt_assistant_count=0):
    """A lineage with committed turns; one checkpoint and record per own assistant."""
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
    lineage = _committed_lineage([SYS, USER, ASST_1, TOOL_1, ASST_2], 2, finish_reason=finish_reason)
    return SessionState(lineages=[lineage])


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
            assert decision.lineage is state.lineages[0]
            assert decision.rollback is None
            assert len(state.lineages) == 1

    @pytest.mark.parametrize("request_messages", [PURE_DROP_ONE, DIVERGENT_ONE])
    def test_one_step_shapes(self, request_messages):
        # disabled: 400
        with pytest.raises(MessageValidationError, match="session rollback is disabled"):
            dispatch_disabled(_two_turn_state(), request_messages)

        # retry: destructive rollback plan against the single lineage
        state = _two_turn_state()
        decision = dispatch_retry(state, request_messages)
        assert decision.rollback is not None
        assert decision.rollback.discard_count == 1

        # fork: new seeded lineage, old lineage untouched
        state = _two_turn_state()
        decision = dispatch_fork(state, request_messages)
        assert len(state.lineages) == 2
        assert decision.lineage is state.lineages[1]
        assert decision.lineage.seed_messages == request_messages
        assert state.lineages[0].num_assistant == 2

    def test_deep_divergence(self):
        lineage = _committed_lineage(
            [SYS, USER, ASST_1, TOOL_1, ASST_2, TOOL_2, {"role": "assistant", "content": "f"}], 3
        )
        request = [SYS, USER, ASST_1, TOOL_1_DIFF]

        with pytest.raises(MessageValidationError, match="session rollback is disabled"):
            dispatch_disabled(SessionState(lineages=[lineage]), request)
        with pytest.raises(MessageValidationError, match="exceeds max_assistant_rollback_steps"):
            dispatch_retry(SessionState(lineages=[lineage]), request)

        state = SessionState(lineages=[_committed_lineage(lineage.messages, 3)])
        decision = dispatch_fork(state, request)
        assert len(state.lineages) == 2
        assert decision.lineage.seed_messages == request

    def test_zero_overlap_subagent_first_request(self):
        with pytest.raises(MessageValidationError, match="session rollback is disabled"):
            dispatch_disabled(_two_turn_state(), SUBAGENT_FIRST)
        with pytest.raises(MessageValidationError, match="no assistant message found"):
            dispatch_retry(_two_turn_state(), SUBAGENT_FIRST)

        state = _two_turn_state()
        decision = dispatch_fork(state, SUBAGENT_FIRST)
        assert len(state.lineages) == 2
        assert decision.lineage.seed_messages == SUBAGENT_FIRST

    def test_truncated_tail_extension(self):
        # disabled/retry: accepted exactly as before truncation existed
        for policy in (dispatch_disabled, dispatch_retry):
            state = _two_turn_state(finish_reason="length")
            decision = policy(state, STRICT_EXTENSION)
            assert decision.lineage is state.lineages[0]

        # fork: 409 — truncation closes the lineage
        with pytest.raises(TruncatedLineageError):
            dispatch_fork(_two_turn_state(finish_reason="length"), STRICT_EXTENSION)

    def test_truncated_lineage_still_forkable_before_the_cut(self):
        """T2: diverging before the truncated turn is not a tail extension."""
        state = _two_turn_state(finish_reason="length")
        decision = dispatch_fork(state, DIVERGENT_ONE)
        assert len(state.lineages) == 2
        assert decision.lineage is state.lineages[1]

    def test_lineage_cap(self):
        state = _two_turn_state()
        state.lineages.extend(
            LinearTrajectory(seed_messages=[{"role": "user", "content": f"task {i}"}]) for i in range(MAX_LINEAGES - 1)
        )
        with pytest.raises(MessageValidationError, match=rf"lineage cap reached \({MAX_LINEAGES}\)"):
            dispatch_fork(state, SUBAGENT_FIRST)


class TestForkSeedPlaceholder:
    """The seed rule closes the concurrent-sibling race: dispatch happens under
    the session lock, commits (Phase 3) happen later, so every uncommitted
    lineage a dispatch returns must already be claimed."""

    def test_empty_session_two_different_first_requests(self):
        """Two concurrent first requests on a fresh session get separate lineages."""
        state = SessionState()
        first = dispatch_fork(state, [SYS, USER])
        assert first.lineage is state.lineages[0]
        assert first.lineage.seed_messages == [SYS, USER]

        second = dispatch_fork(state, SUBAGENT_FIRST)
        assert second.lineage is not first.lineage
        assert len(state.lineages) == 2
        assert second.lineage.seed_messages == SUBAGENT_FIRST

    def test_concurrent_siblings_fork_separately(self):
        state = _two_turn_state()
        sibling_a = dispatch_fork(state, [SUBAGENT_SYS, {"role": "user", "content": "task A"}])
        sibling_b = dispatch_fork(state, [SUBAGENT_SYS, {"role": "user", "content": "task B"}])
        assert sibling_a.lineage is not sibling_b.lineage
        assert len(state.lineages) == 3

    def test_identical_retry_routes_to_seeded_lineage(self):
        """Resending the seed (lost first response) rejoins the same lineage."""
        state = _two_turn_state()
        first = dispatch_fork(state, SUBAGENT_FIRST)
        retry = dispatch_fork(state, SUBAGENT_FIRST)
        assert retry.lineage is first.lineage
        assert len(state.lineages) == 2

    def test_seed_extension_routes_to_seeded_lineage(self):
        state = _two_turn_state()
        first = dispatch_fork(state, SUBAGENT_FIRST)
        extended = dispatch_fork(state, [*SUBAGENT_FIRST, ASST_1, TOOL_1])
        assert extended.lineage is first.lineage

    def test_failed_first_turn_leftover_does_not_capture_main_line(self):
        """A seeded-but-never-committed lineage must not swallow other traffic."""
        state = _two_turn_state()
        dispatch_fork(state, SUBAGENT_FIRST)  # proxy for this one never succeeds

        main = dispatch_fork(state, STRICT_EXTENSION)
        assert main.lineage is state.lineages[0]
        assert len(state.lineages) == 2

    def test_most_recent_lineage_wins_ties(self):
        """Two committed lineages both extended by the request: latest commit wins."""
        older = _committed_lineage([SYS, USER, ASST_1], 1)
        newer = _committed_lineage([SYS, USER, ASST_1], 1)
        older.records[-1] = _record("stop", 1.0)
        newer.records[-1] = _record("stop", 2.0)
        state = SessionState(lineages=[older, newer])

        decision = dispatch_fork(state, [SYS, USER, ASST_1, TOOL_1])
        assert decision.lineage is newer
