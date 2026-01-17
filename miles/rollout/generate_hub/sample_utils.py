from copy import deepcopy

from miles.utils.types import Sample


def merge_samples(a: Sample, b: Sample, tokenizer) -> Sample:
    a, b = deepcopy(a), deepcopy(b)

    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    def _fill_default_loss_mask(sample: Sample):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

    _fill_default_loss_mask(a)
    _fill_default_loss_mask(b)
    obs_len = len(b.tokens) - len(a.tokens) - b.response_length

    try:
        a.validate()
        b.validate()
        assert b.tokens[: len(a.tokens)] == a.tokens
        assert b.response.startswith(a.response)
        assert b.response_length >= a.response_length
        assert obs_len > 0
    except AssertionError as e:
        e.add_note(f"{a=} {b=}")
        raise

    return Sample(
        group_index=_merge_equal_value("group_index"),
        index=_merge_equal_value("index"),
        prompt=_merge_equal_value("prompt"),
        tokens=b.tokens,
        response=b.response,
        response_length=b.response_length,
        label=_merge_equal_value("label"),
        reward=_merge_equal_value("reward"),
        loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
        rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
        status=b.status,
    )
