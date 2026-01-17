from miles.utils.types import Sample


def merge_samples(a: Sample, b: Sample, tokenizer) -> Sample:
    _validate_samples(a, b)

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    assert obs_len > 0, (
        f"obs_len (observation/intermediate tokens) must be > 0, got {obs_len}. "
        f"b.tokens length: {len(b.tokens)}, "
        f"a.tokens length: {len(a.tokens)}, "
        f"b.response_length: {b.response_length}"
    )

    obs_tokens = b.tokens[len(a.tokens): len(a.tokens) + obs_len]
    obs_text = tokenizer.decode(obs_tokens)

    return Sample(
        prompt=a.prompt,
        tokens=b.tokens,
        response=a.response + obs_text + b.response,
        response_length=a.response_length + obs_len + b.response_length,
        loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
        rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
        status=b.status,
        label=_merge_equal_value(a.label, b.label, "label"),
        reward=b.reward,
        index=_merge_equal_value(a.index, b.index, "index"),
        group_index=_merge_equal_value(a.group_index, b.group_index, "group_index"),
    )


def _validate_samples(sample1: Sample, sample2: Sample):
    assert sample1.prompt == sample2.prompt, (
        f"prompt mismatch: sample1.prompt={sample1.prompt}, sample2.prompt={sample2.prompt}"
    )

    assert sample2.tokens[: len(sample1.tokens)] == sample1.tokens, (
        f"sample2.tokens must start with sample1.tokens. "
        f"sample1.tokens: {sample1.tokens}, "
        f"sample2.tokens prefix: {sample2.tokens[:len(sample1.tokens)]}"
    )

    assert sample1.loss_mask is not None, "sample1.loss_mask is None"
    assert sample2.loss_mask is not None, "sample2.loss_mask is None"
    assert len(sample1.loss_mask) == sample1.response_length, (
        f"sample1.loss_mask length ({len(sample1.loss_mask)}) != "
        f"sample1.response_length ({sample1.response_length})"
    )
    assert len(sample2.loss_mask) == sample2.response_length, (
        f"sample2.loss_mask length ({len(sample2.loss_mask)}) != "
        f"sample2.response_length ({sample2.response_length})"
    )

    assert sample1.rollout_log_probs is not None, "sample1.rollout_log_probs is None"
    assert sample2.rollout_log_probs is not None, "sample2.rollout_log_probs is None"
    assert len(sample1.rollout_log_probs) == sample1.response_length, (
        f"sample1.rollout_log_probs length ({len(sample1.rollout_log_probs)}) != "
        f"sample1.response_length ({sample1.response_length})"
    )
    assert len(sample2.rollout_log_probs) == sample2.response_length, (
        f"sample2.rollout_log_probs length ({len(sample2.rollout_log_probs)}) != "
        f"sample2.response_length ({sample2.response_length})"
    )


def _merge_equal_value(x, y, name):
    assert x == y, f"{name} mismatch: a.{name}={x}, b.{name}={y}"
    return x
