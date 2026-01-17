from miles.utils.types import Sample


def merge_samples(a: Sample, b: Sample, tokenizer) -> Sample:
    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    a.validate()
    b.validate()
    assert b.tokens[: len(a.tokens)] == a.tokens, (
        f"b.tokens must start with a.tokens. "
        f"a.tokens: {a.tokens}, "
        f"b.tokens prefix: {b.tokens[:len(a.tokens)]}"
    )

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    assert obs_len > 0, (
        f"obs_len (observation/intermediate tokens) must be > 0, got {obs_len}. "
        f"b.tokens length: {len(b.tokens)}, "
        f"a.tokens length: {len(a.tokens)}, "
        f"b.response_length: {b.response_length}"
    )

    a_loss_mask = a.loss_mask if a.loss_mask is not None else [1] * a.response_length
    b_loss_mask = b.loss_mask if b.loss_mask is not None else [1] * b.response_length
    a_log_probs = a.rollout_log_probs if a.rollout_log_probs is not None else [0.0] * a.response_length
    b_log_probs = b.rollout_log_probs if b.rollout_log_probs is not None else [0.0] * b.response_length

    obs_tokens = b.tokens[len(a.tokens): len(a.tokens) + obs_len]
    obs_text = tokenizer.decode(obs_tokens)

    return Sample(
        group_index=_merge_equal_value("group_index"),
        index=_merge_equal_value("index"),
        prompt=_merge_equal_value("prompt"),
        tokens=b.tokens,
        response=a.response + obs_text + b.response,
        response_length=a.response_length + obs_len + b.response_length,
        label=_merge_equal_value("label"),
        reward=_merge_equal_value("reward"),
        loss_mask=a_loss_mask + [0] * obs_len + b_loss_mask,
        rollout_log_probs=a_log_probs + [0.0] * obs_len + b_log_probs,
        status=b.status,
    )
