from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from miles.rollout.generate_utils.tokenize_utils import tokenize_messages
from miles.router.session import seq_trajectory
from miles.utils.chat_message_utils import get_think_token_start

MODEL_NAME = "Qwen/Qwen3-4B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def _messages(items: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in items]


def _token_info_from_ids(token_ids: list[int]) -> seq_trajectory.TokenInfo:
    return seq_trajectory.TokenInfo(
        tokens=TOKENIZER.convert_ids_to_tokens(token_ids),
        token_ids=token_ids,
        log_probs=[0.0] * len(token_ids),
        loss_mask=[1] * len(token_ids),
    )


def _turn(messages: list[dict[str, str]], prompt_ids: list[int], response_ids: list[int]) -> seq_trajectory.Turn:
    payload = {
        "messages": messages,
        "prompt_tokens": _token_info_from_ids(prompt_ids),
        "response_tokens": _token_info_from_ids(response_ids),
    }
    if hasattr(seq_trajectory.Turn, "model_construct"):
        return seq_trajectory.Turn.model_construct(**payload)
    return seq_trajectory.Turn.construct(**payload)


def _turn_from_messages(messages: list[dict[str, str]]) -> seq_trajectory.Turn:
    prompt_token_ids = TOKENIZER.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    response_token_ids = TOKENIZER.encode(messages[-1]["content"], add_special_tokens=False)
    return _turn(messages, prompt_token_ids, response_token_ids)


def _assert_prompt_token_info(token_info: seq_trajectory.TokenInfo, expected_token_ids: list[int]) -> None:
    assert token_info.token_ids == expected_token_ids
    assert token_info.tokens == TOKENIZER.convert_ids_to_tokens(expected_token_ids)
    assert token_info.log_probs == [0.0] * len(expected_token_ids)
    assert token_info.loss_mask == [0] * len(expected_token_ids)


def _make_manager(*, cross_turn_token_out: bool, inherit_last_assistant: bool) -> seq_trajectory.SeqTrajectoryManager:
    args = SimpleNamespace(
        cross_turn_token_out=cross_turn_token_out,
        inherit_last_assistant=inherit_last_assistant,
    )
    return seq_trajectory.SeqTrajectoryManager(args, TOKENIZER)


def test_turn_match_prefix_messages_returns_remaining():
    messages = _messages([("user", "hi"), ("assistant", "ok"), ("user", "next"), ("assistant", "done")])
    turn = _turn(messages, [], [])

    remaining = turn.match_prefix_messages_and_return_remaining(messages[:2])

    assert remaining == messages[2:]


def test_turn_match_prefix_messages_exact_match_returns_empty():
    messages = _messages([("user", "hi"), ("assistant", "ok")])
    turn = _turn(messages, [], [])

    remaining = turn.match_prefix_messages_and_return_remaining(messages)

    assert remaining == []


def test_turn_match_prefix_messages_mismatch_returns_none():
    messages = _messages([("user", "hi"), ("assistant", "ok")])
    turn = _turn(messages, [], [])

    assert turn.match_prefix_messages_and_return_remaining([{"role": "user", "content": "nope"}]) is None
    assert (
        turn.match_prefix_messages_and_return_remaining(messages + [{"role": "assistant", "content": "extra"}]) is None
    )


def test_calc_prompt_tokens_info_multi_turn_cross_turn_disabled_uses_last_turn():
    trajectory = seq_trajectory.SeqTrajectory()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn2_messages = _messages(
        [
            ("system", "sys"),
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ]
    )

    trajectory.insert_new_turn(_turn_from_messages(turn1_messages))
    trajectory.insert_new_turn(_turn_from_messages(turn2_messages))

    token_info = trajectory.calc_prompt_tokens_info(
        turn2_messages,
        TOKENIZER,
        cross_turn_token_out=False,
        inherit_last_assistant=True,
    )
    expected_token_ids = TOKENIZER.apply_chat_template(turn2_messages, tokenize=True, add_generation_prompt=True)
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_calc_prompt_tokens_info_multi_turn_cross_turn_uses_prefix_suffix():
    trajectory = seq_trajectory.SeqTrajectory()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn2_messages = _messages(
        [
            ("system", "sys"),
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ]
    )
    turn1 = _turn_from_messages(turn1_messages)
    trajectory.insert_new_turn(turn1)
    trajectory.insert_new_turn(_turn_from_messages(turn2_messages))

    input_messages = _messages([("system", "sys")])
    remain_messages = _messages([("user", "u1"), ("assistant", "a1")])

    token_info = trajectory.calc_prompt_tokens_info(
        input_messages,
        TOKENIZER,
        cross_turn_token_out=True,
        inherit_last_assistant=False,
    )
    expected_new_token_ids = tokenize_messages(remain_messages, TOKENIZER, add_generation_prompt=True)
    expected_token_ids = turn1.prompt_tokens.token_ids + turn1.response_tokens.token_ids + expected_new_token_ids
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_calc_prompt_tokens_info_multi_turn_cross_turn_matches_two_turns():
    trajectory = seq_trajectory.SeqTrajectory()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn2_messages = _messages([("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")])
    turn3_messages = _messages([("user", "u3"), ("assistant", "a3")])
    turn2 = _turn_from_messages(turn2_messages)

    trajectory.insert_new_turn(_turn_from_messages(turn1_messages))
    trajectory.insert_new_turn(turn2)
    trajectory.insert_new_turn(_turn_from_messages(turn3_messages))

    input_messages = _messages([("system", "sys")])
    remain_messages = _messages([("user", "u2"), ("assistant", "a2")])

    token_info = trajectory.calc_prompt_tokens_info(
        input_messages,
        TOKENIZER,
        cross_turn_token_out=True,
        inherit_last_assistant=False,
    )
    expected_new_token_ids = tokenize_messages(remain_messages, TOKENIZER, add_generation_prompt=True)
    expected_token_ids = turn2.prompt_tokens.token_ids + turn2.response_tokens.token_ids + expected_new_token_ids
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_calc_prompt_tokens_info_multi_turn_cross_turn_empty_remaining_messages():
    trajectory = seq_trajectory.SeqTrajectory()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn2_messages = _messages(
        [
            ("system", "sys"),
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ]
    )
    turn1 = _turn_from_messages(turn1_messages)

    trajectory.insert_new_turn(turn1)
    trajectory.insert_new_turn(_turn_from_messages(turn2_messages))

    token_info = trajectory.calc_prompt_tokens_info(
        turn1_messages,
        TOKENIZER,
        cross_turn_token_out=True,
        inherit_last_assistant=False,
    )
    expected_token_ids = turn1.prompt_tokens.token_ids + turn1.response_tokens.token_ids
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_tokenize_messages_trims_complete_think_content():
    messages_with_think = _messages([("assistant", "<think>thought</think>answer")])
    messages_plain = _messages([("assistant", "answer")])

    tokens_with_think = tokenize_messages(messages_with_think, TOKENIZER, add_generation_prompt=True)
    tokens_plain = tokenize_messages(messages_plain, TOKENIZER, add_generation_prompt=True)

    think_start_id = get_think_token_start("qwen3")[1]

    assert tokens_with_think == tokens_plain
    assert think_start_id not in tokens_with_think


def test_tokenize_messages_does_not_trim_incomplete_think_content():
    messages_incomplete_think = _messages([("assistant", "<think>thought answer")])
    messages_plain = _messages([("assistant", "answer")])

    tokens_incomplete = tokenize_messages(messages_incomplete_think, TOKENIZER, add_generation_prompt=True)
    tokens_plain = tokenize_messages(messages_plain, TOKENIZER, add_generation_prompt=True)

    think_start_id = get_think_token_start("qwen3")[1]

    assert tokens_incomplete != tokens_plain
    assert think_start_id in tokens_incomplete


def test_manager_calc_prompt_tokens_missing_session_returns_none():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    messages = _messages([("system", "sys"), ("user", "hi")])

    assert manager.calc_prompt_tokens("missing", messages) is None


def test_manager_get_session_by_id_empty_returns_empty_token_info():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()

    token_info = manager.get_session_by_id(session_id)
    assert token_info is not None
    assert token_info.tokens == []
    assert token_info.token_ids == []
    assert token_info.log_probs == []
    assert token_info.loss_mask == []


def test_manager_calc_prompt_tokens_no_turns_retokens_messages():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()
    messages = _messages([("system", "sys"), ("user", "u1")])

    token_info = manager.calc_prompt_tokens(session_id, messages)

    expected_token_ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_manager_calc_prompt_tokens_inherit_last_assistant_raises():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=True)
    session_id = manager.create_session()
    turn_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    manager.add_record(session_id, _turn_from_messages(turn_messages))

    with pytest.raises(NotImplementedError):
        manager.calc_prompt_tokens(session_id, turn_messages)


def test_manager_calc_prompt_tokens_cross_turn_single_turn_uses_tokenize_messages():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()
    turn_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    manager.add_record(session_id, _turn_from_messages(turn_messages))

    messages = _messages([("system", "sys"), ("user", "next")])
    token_info = manager.calc_prompt_tokens(session_id, messages)

    expected_token_ids = tokenize_messages(messages, TOKENIZER, add_generation_prompt=True)
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_manager_calc_prompt_tokens_cross_turn_multi_turn_prefix_success():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn2_messages = _messages([("user", "u2"), ("assistant", "a2")])
    turn1 = _turn_from_messages(turn1_messages)
    manager.add_record(session_id, turn1)
    manager.add_record(session_id, _turn_from_messages(turn2_messages))

    input_messages = _messages([("system", "sys")])
    token_info = manager.calc_prompt_tokens(session_id, input_messages)

    remain_messages = _messages([("user", "u1"), ("assistant", "a1")])
    expected_new_token_ids = tokenize_messages(remain_messages, TOKENIZER, add_generation_prompt=True)
    expected_token_ids = turn1.prompt_tokens.token_ids + turn1.response_tokens.token_ids + expected_new_token_ids
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_manager_calc_prompt_tokens_cross_turn_multi_turn_prefix_mismatch_raises():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    manager.add_record(session_id, _turn_from_messages(turn1_messages))
    manager.add_record(session_id, _turn_from_messages(_messages([("user", "u2"), ("assistant", "a2")])))

    with pytest.raises(ValueError):
        manager.calc_prompt_tokens(session_id, _messages([("system", "nope")]))


def test_manager_calc_prompt_tokens_cross_turn_disabled_retokens_messages():
    manager = _make_manager(cross_turn_token_out=False, inherit_last_assistant=True)
    session_id = manager.create_session()
    manager.add_record(
        session_id, _turn_from_messages(_messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")]))
    )

    messages = _messages([("system", "sys"), ("user", "new")])
    token_info = manager.calc_prompt_tokens(session_id, messages)

    expected_token_ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    _assert_prompt_token_info(token_info, expected_token_ids)


def test_manager_get_session_by_id_after_add_record_returns_combined_tokens():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()
    messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn = _turn_from_messages(messages)
    manager.add_record(session_id, turn)

    token_info = manager.get_session_by_id(session_id)

    expected_token_ids = turn.prompt_tokens.token_ids + turn.response_tokens.token_ids
    assert token_info.token_ids == expected_token_ids
    assert token_info.tokens == TOKENIZER.convert_ids_to_tokens(expected_token_ids)
    assert token_info.log_probs == turn.prompt_tokens.log_probs + turn.response_tokens.log_probs
    assert token_info.loss_mask == turn.prompt_tokens.loss_mask + turn.response_tokens.loss_mask


def test_manager_delete_session_by_id():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()

    assert manager.delete_session_by_id(session_id) is True
    assert manager.delete_session_by_id(session_id) is False


def test_manager_add_record_missing_session_raises():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn = _turn_from_messages(messages)

    with pytest.raises(ValueError):
        manager.add_record("missing", turn)


def test_manager_calc_prompt_tokens_cross_turn_multi_turn_empty_remaining_messages():
    manager = _make_manager(cross_turn_token_out=True, inherit_last_assistant=False)
    session_id = manager.create_session()
    turn1_messages = _messages([("system", "sys"), ("user", "u1"), ("assistant", "a1")])
    turn2_messages = _messages([("user", "u2"), ("assistant", "a2")])
    turn1 = _turn_from_messages(turn1_messages)
    manager.add_record(session_id, turn1)
    manager.add_record(session_id, _turn_from_messages(turn2_messages))

    token_info = manager.calc_prompt_tokens(session_id, turn1_messages)

    expected_token_ids = turn1.prompt_tokens.token_ids + turn1.response_tokens.token_ids
    _assert_prompt_token_info(token_info, expected_token_ids)
