import pytest
from unittest.mock import MagicMock

from miles.rollout.generate_hub.sample_utils import merge_samples
from miles.utils.types import Sample


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode = lambda tokens: f"<decoded:{tokens}>"
    return tokenizer


def make_sample(
    prompt="test_prompt",
    tokens=None,
    response="",
    response_length=0,
    loss_mask=None,
    rollout_log_probs=None,
    status=Sample.Status.COMPLETED,
    label="test_label",
    reward=1.0,
    index=0,
    group_index=0,
):
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        status=status,
        label=label,
        reward=reward,
        index=index,
        group_index=group_index,
    )


class TestMergeSamplesBasic:
    def test_basic_merge(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 3, 10, 11, 12],
            response="response1",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        b = make_sample(
            tokens=[1, 2, 3, 10, 11, 12, 20, 21, 30, 31, 32],
            response="response2",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.4, -0.5, -0.6],
        )

        merged = merge_samples(a, b, mock_tokenizer)

        assert merged.tokens == b.tokens
        assert merged.response_length == 3 + 2 + 3
        assert merged.loss_mask == [1, 1, 1, 0, 0, 1, 1, 1]
        assert merged.rollout_log_probs == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5, -0.6]
        assert merged.prompt == a.prompt
        assert merged.status == b.status
        assert merged.label == a.label
        assert merged.index == a.index
        assert merged.group_index == a.group_index

    def test_response_concatenation(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response="hello",
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 21, 30],
            response="world",
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.2],
        )

        merged = merge_samples(a, b, mock_tokenizer)

        assert "hello" in merged.response
        assert "world" in merged.response
        assert "<decoded:[20, 21]>" in merged.response


class TestMergeSamplesValidation:
    def test_prompt_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            prompt="prompt_a",
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            prompt="prompt_b",
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="prompt mismatch"):
            merge_samples(a, b, mock_tokenizer)

    def test_tokens_prefix_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 3],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 99, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="must start with"):
            merge_samples(a, b, mock_tokenizer)

    def test_loss_mask_none_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="loss_mask is None"):
            merge_samples(a, b, mock_tokenizer)

    def test_loss_mask_none_sample2_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="loss_mask is None"):
            merge_samples(a, b, mock_tokenizer)

    def test_loss_mask_length_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10, 11],
            response_length=2,
            loss_mask=[1],
            rollout_log_probs=[-0.1, -0.2],
        )
        b = make_sample(
            tokens=[1, 2, 10, 11, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="loss_mask length"):
            merge_samples(a, b, mock_tokenizer)

    def test_rollout_log_probs_none_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=None,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="rollout_log_probs is None"):
            merge_samples(a, b, mock_tokenizer)

    def test_rollout_log_probs_length_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10, 11],
            response_length=2,
            loss_mask=[1, 1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 11, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="rollout_log_probs length"):
            merge_samples(a, b, mock_tokenizer)

    def test_obs_len_zero_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        with pytest.raises(AssertionError, match="obs_len.*must be > 0"):
            merge_samples(a, b, mock_tokenizer)

    def test_obs_len_negative_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10, 11, 12],
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        b = make_sample(
            tokens=[1, 2, 10, 11, 12, 30],
            response_length=2,
            loss_mask=[1, 1],
            rollout_log_probs=[-0.1, -0.2],
        )

        with pytest.raises(AssertionError, match="obs_len.*must be > 0"):
            merge_samples(a, b, mock_tokenizer)

    def test_index_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            index=0,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            index=1,
        )

        with pytest.raises(AssertionError, match="index mismatch"):
            merge_samples(a, b, mock_tokenizer)

    def test_group_index_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            group_index=0,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            group_index=1,
        )

        with pytest.raises(AssertionError, match="group_index mismatch"):
            merge_samples(a, b, mock_tokenizer)

    def test_label_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            label="label_a",
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            label="label_b",
        )

        with pytest.raises(AssertionError, match="label mismatch"):
            merge_samples(a, b, mock_tokenizer)


class TestMergeSamplesEdgeCases:
    def test_response_length_zero_sample1(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2],
            response="",
            response_length=0,
            loss_mask=[],
            rollout_log_probs=[],
        )
        b = make_sample(
            tokens=[1, 2, 20, 30],
            response="response2",
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )

        merged = merge_samples(a, b, mock_tokenizer)

        assert merged.response_length == 0 + 1 + 1
        assert merged.loss_mask == [0, 1]
        assert merged.rollout_log_probs == [0.0, -0.1]

    def test_single_token_observation(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.2],
        )

        merged = merge_samples(a, b, mock_tokenizer)

        assert merged.response_length == 1 + 1 + 1
        assert merged.loss_mask == [1, 0, 1]

    def test_reward_from_b(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            reward=0.5,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            reward=0.8,
        )

        merged = merge_samples(a, b, mock_tokenizer)

        assert merged.reward == 0.8

    def test_status_from_b(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            status=Sample.Status.COMPLETED,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            status=Sample.Status.TRUNCATED,
        )

        merged = merge_samples(a, b, mock_tokenizer)

        assert merged.status == Sample.Status.TRUNCATED
