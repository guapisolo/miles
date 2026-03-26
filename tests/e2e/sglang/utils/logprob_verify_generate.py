"""Custom generate function that wraps the agentic tool-call flow with
re-prefill logprob verification.

After the agent finishes its multi-turn conversation through the session
server, this function sends the full ``accumulated_token_ids`` (the token
sequence built incrementally by TITO across all turns) to the SGLang
``/generate`` endpoint with ``max_new_tokens=0`` and ``return_logprob=True``.
This produces ``input_token_logprobs`` via a single prefill pass over the
entire sequence.  The resulting logprobs are then compared per-turn against
the ``output_token_logprobs`` collected during the session's decode phase.

Any mismatch in **token IDs** is fatal (indicates a TITO tokenization bug).
Logprob values are compared with a tight tolerance to account for minor
numerical differences between the prefill and decode GPU kernels.

When ``use_rollout_routing_replay`` is enabled (MoE models), routed
expert arrays are also compared.
"""

import logging
import statistics
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pybase64

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
    truncate_samples_by_total_tokens,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

LOGPROB_ABS_TOL = 1e-8  # with deterministic inference, prefill and decode must be bit-identical
LOGPROB_WARN_TOL = 0.0  # any nonzero diff is worth logging


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run the production agentic flow, then verify logprobs via re-prefill.

    Inlines the core flow from ``agentic_tool_call.generate`` so that we
    can insert verification between record collection and finalization,
    without modifying the production code.
    """

    # === Step 1: core agentic flow (same as production generate) ===
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}

    agent_metadata = await custom_agent_function(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        request_kwargs=build_chat_request_kwargs(input.sampling_params),
        metadata=metadata,
    )

    records, session_metadata = await tracer.collect_records()

    if not records:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    # === Step 2: session-level checks ===
    mismatch = session_metadata.get("tito_session_mismatch")
    assert mismatch == [], f"tito_session_mismatch is not empty: {mismatch}"

    accumulated = session_metadata.get("accumulated_token_ids")
    assert accumulated and len(accumulated) > 0, "accumulated_token_ids is empty"

    max_trim_tokens = session_metadata.get("max_trim_tokens", 0)

    # === Step 3: re-prefill verification ===
    assert len(records) >= 2, f"Expected at least 2 turns for TITO verification, got {len(records)}"

    sglang_url = f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}"
    use_r3 = getattr(input.args, "use_rollout_routing_replay", False)

    await _verify_logprobs_via_reprefill(
        sglang_url,
        records,
        accumulated,
        max_trim_tokens=max_trim_tokens,
        use_r3=use_r3,
    )

    logger.info(
        "Logprob equivalence verified: %d turns, %d accumulated tokens",
        len(records),
        len(accumulated),
    )

    # === Step 4: finalize (same as production generate) ===
    samples = compute_samples_from_openai_records(
        input.args,
        input.sample,
        records,
        input.state.tokenizer,
        accumulated_token_ids=accumulated,
        max_trim_tokens=max_trim_tokens,
    )

    for s in samples:
        s.metadata.update(agent_metadata or {})

    if max_seq_len is not None:
        samples = truncate_samples_by_total_tokens(samples, max_seq_len, input.state.tokenizer)

    if not samples:
        logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
        samples.metadata.update(session_metadata)
    else:
        samples[-1].metadata.update(session_metadata)
    return GenerateFnOutput(samples=samples)


# reuse the same CLI arguments as agentic_tool_call
generate.add_arguments = None  # set below after import


def _init_add_arguments():
    from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate

    generate.add_arguments = _base_generate.add_arguments


_init_add_arguments()


# ---------------------------------------------------------------------------
# Verification logic (unique to this test generate function)
# ---------------------------------------------------------------------------


async def _verify_logprobs_via_reprefill(
    sglang_url: str,
    records: list,
    accumulated_token_ids: list[int],
    max_trim_tokens: int,
    use_r3: bool,
) -> None:
    """Re-prefill the full accumulated token sequence and compare logprobs.

    Sends ``accumulated_token_ids`` to ``/generate`` with
    ``max_new_tokens=0`` and ``return_logprob=True``.  Compares the
    resulting ``input_token_logprobs`` (prefill) against per-turn
    ``output_token_logprobs`` (decode) from the session records.
    """
    first_prompt_len = len(records[0].response["choices"][0]["prompt_token_ids"])

    # --- Step A: send re-prefill request ---
    payload = {
        "input_ids": accumulated_token_ids,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": first_prompt_len,
    }
    if use_r3:
        payload["return_routed_experts"] = True

    reprefill_resp = await post(f"{sglang_url}/generate", payload)
    reprefill_logprobs = reprefill_resp["meta_info"]["input_token_logprobs"]

    expected_len = len(accumulated_token_ids) - first_prompt_len
    assert len(reprefill_logprobs) == expected_len, (
        f"Re-prefill returned {len(reprefill_logprobs)} input_token_logprobs, "
        f"expected {expected_len} (accumulated={len(accumulated_token_ids)}, "
        f"first_prompt={first_prompt_len})"
    )

    # --- Step B: walk records and compare per-turn ---
    all_diffs: list[float] = []

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        choice = record.response["choices"][0]
        prompt_ids = choice["prompt_token_ids"]
        session_output_logprobs = choice["meta_info"]["output_token_logprobs"]
        output_ids = [t[1] for t in session_output_logprobs]

        if not output_ids:
            logger.warning("Turn %d: no output tokens, skipping", i)
            continue

        # Position cursor after this turn's prompt
        cursor = len(prompt_ids)

        # Greedily match output_ids against accumulated[cursor:]
        matched = 0
        for j in range(len(output_ids)):
            idx = cursor + j
            if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                matched += 1
            else:
                break

        trim_count = len(output_ids) - matched
        allowed = 0 if is_last else max_trim_tokens
        assert trim_count <= allowed, f"Turn {i}: trim_count {trim_count} exceeds allowed={allowed}"

        # Compare matched tokens
        turn_reprefill_start = cursor - first_prompt_len
        mismatches = []
        warnings = []

        for j in range(matched):
            rp_entry = reprefill_logprobs[turn_reprefill_start + j]  # (logprob, token_id, text)
            sp_entry = session_output_logprobs[j]  # (logprob, token_id)

            rp_tid = rp_entry[1]
            sp_tid = sp_entry[1]
            assert rp_tid == sp_tid, (
                f"Turn {i}, token {j}: token_id mismatch — " f"reprefill={rp_tid} vs session={sp_tid}"
            )

            rp_lp = rp_entry[0]
            sp_lp = sp_entry[0]
            if rp_lp is None or sp_lp is None:
                continue

            diff = abs(rp_lp - sp_lp)
            all_diffs.append(diff)

            if diff > LOGPROB_ABS_TOL:
                mismatches.append(f"  token {j}: prefill={rp_lp:.8f} decode={sp_lp:.8f} " f"diff={diff:.4f}")
            elif diff > LOGPROB_WARN_TOL:
                warnings.append(f"  token {j}: prefill={rp_lp:.8f} decode={sp_lp:.8f} " f"diff={diff:.4f}")

        if warnings:
            logger.warning(
                "Turn %d: %d tokens with diff > %.4f (but within tolerance):\n%s",
                i,
                len(warnings),
                LOGPROB_WARN_TOL,
                "\n".join(warnings[:10]),
            )

        assert (
            not mismatches
        ), f"Turn {i}: {len(mismatches)} logprob differences exceed " f"tolerance {LOGPROB_ABS_TOL}:\n" + "\n".join(
            mismatches
        )

        logger.info("Turn %d: verified %d output tokens (trimmed %d)", i, matched, trim_count)

    # --- Step C: R3 comparison ---
    if use_r3:
        _verify_routed_experts(records, reprefill_resp, accumulated_token_ids, first_prompt_len, max_trim_tokens)

    # --- Step D: summary statistics ---
    if all_diffs:
        sorted_diffs = sorted(all_diffs)
        p99_idx = min(int(len(sorted_diffs) * 0.99), len(sorted_diffs) - 1)
        logger.info(
            "Logprob diff stats: mean=%.6f, max=%.6f, p99=%.6f, count=%d",
            statistics.mean(all_diffs),
            max(all_diffs),
            sorted_diffs[p99_idx],
            len(all_diffs),
        )


def _verify_routed_experts(
    records: list,
    reprefill_resp: dict,
    accumulated_token_ids: list[int],
    first_prompt_len: int,
    max_trim_tokens: int,
) -> None:
    """Compare per-turn routed_experts from session decode against re-prefill."""
    reprefill_re_b64 = reprefill_resp["meta_info"].get("routed_experts")
    if reprefill_re_b64 is None:
        logger.warning("Re-prefill response missing routed_experts, skipping R3 check")
        return

    reprefill_re_flat = np.frombuffer(pybase64.b64decode(reprefill_re_b64.encode("ascii")), dtype=np.int32)

    for i, record in enumerate(records):
        choice = record.response["choices"][0]
        prompt_ids = choice["prompt_token_ids"]
        session_output_logprobs = choice["meta_info"]["output_token_logprobs"]
        output_ids = [t[1] for t in session_output_logprobs]

        session_re_b64 = choice["meta_info"].get("routed_experts")
        if session_re_b64 is None:
            logger.warning("Turn %d: session missing routed_experts, skipping", i)
            continue

        # Match count (same logic as logprob comparison)
        cursor = len(prompt_ids)
        matched = 0
        for j in range(len(output_ids)):
            idx = cursor + j
            if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                matched += 1
            else:
                break

        session_re = np.frombuffer(pybase64.b64decode(session_re_b64.encode("ascii")), dtype=np.int32)

        # routed_experts shape: [total_tokens - 1, num_layers, top_k]
        # where total_tokens = len(prompt_ids) + len(output_ids)
        # Entry k corresponds to routing for token at position k+1.
        total_tokens = len(prompt_ids) + len(output_ids)
        if total_tokens <= 1 or len(session_re) == 0 or matched == 0:
            continue

        per_token_size = len(session_re) // (total_tokens - 1)

        # Session: output tokens start at position P (=len(prompt_ids)),
        # so their experts start at index P-1 in the experts array.
        session_start = (len(prompt_ids) - 1) * per_token_size
        session_slice = session_re[session_start : session_start + matched * per_token_size]

        # Re-prefill: experts array covers all of accumulated_token_ids.
        # Output at accumulated[cursor] -> expert at index cursor-1.
        rp_offset = (cursor - 1) * per_token_size
        rp_slice = reprefill_re_flat[rp_offset : rp_offset + matched * per_token_size]

        np.testing.assert_array_equal(
            session_slice,
            rp_slice,
            err_msg=f"Turn {i}: routed_experts mismatch",
        )
        logger.info("Turn %d: routed_experts match (%d entries)", i, len(session_slice))
