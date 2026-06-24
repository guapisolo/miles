"""Single-turn MathVision rollout for the refactored (experimental) rollout.

Enabled by ``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``, the custom generate hook
takes a ``GenerateFnInput`` and returns a ``GenerateFnOutput`` (the new signature),
instead of the legacy ``(args, sample, sampling_params)`` form. Here we just wrap
the built-in single-turn generate and keep ``multimodal_train_inputs`` tensor-only
for the Megatron VLM data path.
"""

import torch

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as _single_turn_generate


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    output = await _single_turn_generate(input)
    sample = output.samples
    # The Qwen VL processor returns some text-modality fields (e.g.
    # mm_token_type_ids) as Python lists. The Megatron data path calls ``.to(...)``
    # on every multimodal_train_inputs value, which a list-valued field cannot
    # satisfy, so drop non-tensor fields (matching the geo3k VLM example).
    mm = sample.multimodal_train_inputs
    if mm:
        sample.multimodal_train_inputs = {k: v for k, v in mm.items() if isinstance(v, torch.Tensor)} or None
    return GenerateFnOutput(samples=sample)
