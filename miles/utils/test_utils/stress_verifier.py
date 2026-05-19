"""Stress-harness correctness verifier (C1 through C7).

Given a session's ``records`` list (as returned by ``GET /sessions/{sid}``)
plus a few drivers-side knowns (expected sid, expected turn count, the R3
shape the mock used), this module produces a :class:`VerifyReport` whose
``overall_pass`` is True iff every invariant in the contract finalised by
``docs/cpu-stress/invariants.md`` holds.

Each invariant is implemented as a small standalone function; this batch
version is used by the in-process harness self-test in Round 1, and the
Round 3 streaming verifier will key off the same invariant names.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from miles.utils.test_utils.r3_codec import decode_r3, make_logp_tokens, make_r3, parse_signature, seed_from_signature


@dataclass
class InvariantResult:
    """Outcome of one named invariant check."""

    name: str
    passed: bool
    mismatches: list[dict] = field(default_factory=list)
    note: str | None = None


@dataclass
class VerifyReport:
    overall_pass: bool
    by_invariant: dict[str, InvariantResult]

    def first_failure(self) -> InvariantResult | None:
        for r in self.by_invariant.values():
            if not r.passed:
                return r
        return None

    def __bool__(self) -> bool:
        return self.overall_pass

    def summary(self) -> str:
        lines = ["VerifyReport: " + ("PASS" if self.overall_pass else "FAIL")]
        for name, r in self.by_invariant.items():
            status = "ok" if r.passed else "FAIL"
            lines.append(f"  [{status}] {name}")
            for m in r.mismatches[:3]:
                lines.append(f"      - {m}")
            if len(r.mismatches) > 3:
                lines.append(f"      ... ({len(r.mismatches) - 3} more)")
        return "\n".join(lines)


def _check_c1(records: list[dict], expected_turn_count: int) -> InvariantResult:
    name = "C1_record_count"
    actual = len(records)
    passed = actual == expected_turn_count
    mismatches: list[dict] = []
    if not passed:
        mismatches.append({"expected_turn_count": expected_turn_count, "actual_record_count": actual})
    return InvariantResult(name=name, passed=passed, mismatches=mismatches)


def _check_c2_and_c7(
    records: list[dict],
    expected_sid: str,
    max_mismatches: int,
) -> tuple[InvariantResult, InvariantResult]:
    """C2 (signature ordering inside a single session) and C7 (no foreign
    sid leakage) are both extracted from the per-record signature embedded
    in ``request.messages[-1].content``."""
    c2 = InvariantResult(name="C2_signature_order", passed=True, mismatches=[])
    c7 = InvariantResult(name="C7_cross_session_isolation", passed=True, mismatches=[])

    for i, r in enumerate(records):
        messages = r.get("request", {}).get("messages", [])
        if not messages:
            c2.passed = False
            if len(c2.mismatches) < max_mismatches:
                c2.mismatches.append({"index": i, "issue": "no messages in request"})
            continue
        last_content = messages[-1].get("content", "") or ""
        sig = parse_signature(last_content)
        if sig is None:
            c2.passed = False
            if len(c2.mismatches) < max_mismatches:
                c2.mismatches.append(
                    {
                        "index": i,
                        "issue": "no [STRESS sid=... turn=...] signature",
                        "content_prefix": last_content[:80],
                    }
                )
            continue
        obs_sid, obs_turn = sig
        if obs_turn != i:
            c2.passed = False
            if len(c2.mismatches) < max_mismatches:
                c2.mismatches.append(
                    {
                        "index": i,
                        "expected_turn": i,
                        "observed_turn": obs_turn,
                        "observed_sid": obs_sid,
                    }
                )
        if obs_sid != expected_sid:
            c7.passed = False
            if len(c7.mismatches) < max_mismatches:
                c7.mismatches.append(
                    {
                        "index": i,
                        "expected_sid": expected_sid,
                        "observed_sid": obs_sid,
                        "observed_turn": obs_turn,
                    }
                )
    return c2, c7


def _check_c3_and_c4(
    records: list[dict],
    r3_num_layers: int,
    r3_topk: int,
    max_mismatches: int,
) -> tuple[InvariantResult, InvariantResult]:
    """C3 (byte-identity of R3 routing meta) and C4 (IEEE 754 32-bit
    bit-identity of per-token logprobs). Both per-record."""
    c3 = InvariantResult(name="C3_r3_byte_identity", passed=True, mismatches=[])
    c4 = InvariantResult(name="C4_logp_bit_identity", passed=True, mismatches=[])

    for i, r in enumerate(records):
        try:
            choice = r["response"]["choices"][0]
            meta_info = choice["meta_info"]
        except (KeyError, IndexError, TypeError) as e:
            c3.passed = False
            c4.passed = False
            if len(c3.mismatches) < max_mismatches:
                c3.mismatches.append({"index": i, "issue": f"meta_info path missing: {e}"})
            if len(c4.mismatches) < max_mismatches:
                c4.mismatches.append({"index": i, "issue": f"meta_info path missing: {e}"})
            continue

        messages = r.get("request", {}).get("messages", [])
        if not messages:
            continue  # C2 will have flagged this
        sig = parse_signature(messages[-1].get("content", "") or "")
        if sig is None:
            continue  # C2 will have flagged this
        sid, turn = sig
        seed = seed_from_signature(sid, turn)

        otl = meta_info.get("output_token_logprobs", [])
        num_tokens = len(otl)
        expected_logps, expected_tokens = make_logp_tokens(seed, num_tokens)

        # C4: per-token (logp_float32_bytes, token_id_int) bit equality
        for j, entry in enumerate(otl):
            try:
                lp_obs, tid_obs = entry[0], entry[1]
            except (TypeError, IndexError):
                c4.passed = False
                if len(c4.mismatches) < max_mismatches:
                    c4.mismatches.append({"index": i, "token_pos": j, "issue": "malformed logp tuple"})
                break
            lp_obs_bytes = struct.pack("<f", float(lp_obs))
            lp_exp_bytes = struct.pack("<f", float(expected_logps[j]))
            if lp_obs_bytes != lp_exp_bytes or int(tid_obs) != int(expected_tokens[j]):
                c4.passed = False
                if len(c4.mismatches) < max_mismatches:
                    c4.mismatches.append(
                        {
                            "index": i,
                            "token_pos": j,
                            "expected_lp_bytes": lp_exp_bytes.hex(),
                            "observed_lp_bytes": lp_obs_bytes.hex(),
                            "expected_token_id": int(expected_tokens[j]),
                            "observed_token_id": int(tid_obs),
                        }
                    )
                break  # one mismatch per record is enough to flag C4

        # C3: routed_experts byte-identity, if R3 shape was requested
        r3_str = meta_info.get("routed_experts")
        if r3_num_layers > 0 and r3_topk > 0:
            if r3_str is None:
                c3.passed = False
                if len(c3.mismatches) < max_mismatches:
                    c3.mismatches.append({"index": i, "issue": "routed_experts missing from meta_info"})
                continue
            try:
                r3_obs_arr = decode_r3(r3_str, num_tokens, r3_num_layers, r3_topk)
            except Exception as e:
                c3.passed = False
                if len(c3.mismatches) < max_mismatches:
                    c3.mismatches.append({"index": i, "issue": f"R3 decode failed: {e!s}"})
                continue
            r3_exp_arr = make_r3(seed, num_tokens, r3_num_layers, r3_topk)
            if r3_obs_arr.tobytes() != r3_exp_arr.tobytes():
                c3.passed = False
                if len(c3.mismatches) < max_mismatches:
                    flat_obs = r3_obs_arr.flatten()
                    flat_exp = r3_exp_arr.flatten()
                    diff_idx = int(np.argmax(flat_obs != flat_exp))
                    c3.mismatches.append(
                        {
                            "index": i,
                            "first_diff_index": diff_idx,
                            "expected_value": int(flat_exp[diff_idx]),
                            "observed_value": int(flat_obs[diff_idx]),
                        }
                    )
        elif r3_str is not None:
            # mock did not promise R3; tolerated. Not a failure either way.
            pass

    return c3, c4


def _check_c5(records: list[dict], accumulated_token_ids: list[int]) -> InvariantResult:
    """C5: ``accumulated_token_ids`` (the latest assistant checkpoint exposed
    by ``GET /sessions/{sid}.metadata``) equals the LAST turn's prompt
    ``input_ids`` plus the LAST turn's completion token IDs.

    This is the externally-observable form of the per-turn token concat
    invariant — see docs/cpu-stress/invariants.md for the rationale."""
    name = "C5_token_concat"
    mismatches: list[dict] = []
    if not records:
        return InvariantResult(name=name, passed=False, mismatches=[{"issue": "no records to check"}])
    last = records[-1]
    last_prompt = list(last.get("request", {}).get("input_ids", []))
    last_meta = last.get("response", {}).get("choices", [{}])[0].get("meta_info", {})
    last_completion = [int(t[1]) for t in last_meta.get("output_token_logprobs", [])]
    expected = last_prompt + last_completion
    accumulated = list(accumulated_token_ids)
    passed = accumulated == expected
    if not passed:
        first_diff = 0
        common_len = min(len(expected), len(accumulated))
        while first_diff < common_len and expected[first_diff] == accumulated[first_diff]:
            first_diff += 1
        mismatches.append(
            {
                "expected_len": len(expected),
                "observed_len": len(accumulated),
                "first_diff_index": first_diff,
                "expected_window": expected[max(0, first_diff - 2) : first_diff + 3],
                "observed_window": accumulated[max(0, first_diff - 2) : first_diff + 3],
            }
        )
    return InvariantResult(name=name, passed=passed, mismatches=mismatches)


def _check_c6(records: list[dict], expected_turn_count: int) -> InvariantResult:
    """C6 derived form: count of 200-status chat-completion records equals
    the expected number of assistant turns. The internal
    ``LinearTrajectory.num_assistant`` is not exposed via the public GET
    endpoint; if and when an admin route is added, this check can be
    strengthened to also compare the in-memory counter."""
    name = "C6_assistant_count_derived"
    actual = sum(1 for r in records if r.get("status_code") == 200 and r.get("path") == "/v1/chat/completions")
    passed = actual == expected_turn_count
    mismatches: list[dict] = []
    if not passed:
        mismatches.append({"expected_assistant_count": expected_turn_count, "observed_200_chat_records": actual})
    return InvariantResult(
        name=name,
        passed=passed,
        mismatches=mismatches,
        note="Derived from external GET; internal num_assistant not exposed.",
    )


def verify_session(
    records: list[dict],
    expected_sid: str,
    expected_turn_count: int,
    accumulated_token_ids: list[int],
    r3_num_layers: int = 0,
    r3_topk: int = 0,
    max_mismatches: int = 20,
) -> VerifyReport:
    """Run C1-C7 against a single session's records + accumulated_token_ids.

    Args:
        records: list of SessionRecord-shaped dicts from GET /sessions/{sid}.
        expected_sid: the session_id this verifier is checking.
        expected_turn_count: how many turns the driver expected to be
            appended successfully.
        accumulated_token_ids: GET response's metadata.accumulated_token_ids.
        r3_num_layers, r3_topk: R3 shape; both 0 disables C3 (no R3 expected).
        max_mismatches: cap on per-invariant mismatch samples for bounded
            memory; the verifier short-circuits when a check has accumulated
            this many failures.

    Returns:
        VerifyReport; ``overall_pass`` is True iff every invariant passed.
    """
    results: dict[str, InvariantResult] = {}
    results["C1_record_count"] = _check_c1(records, expected_turn_count)
    c2, c7 = _check_c2_and_c7(records, expected_sid, max_mismatches)
    results["C2_signature_order"] = c2
    results["C7_cross_session_isolation"] = c7
    c3, c4 = _check_c3_and_c4(records, r3_num_layers, r3_topk, max_mismatches)
    results["C3_r3_byte_identity"] = c3
    results["C4_logp_bit_identity"] = c4
    results["C5_token_concat"] = _check_c5(records, accumulated_token_ids)
    results["C6_assistant_count_derived"] = _check_c6(records, expected_turn_count)

    overall = all(r.passed for r in results.values())
    return VerifyReport(overall_pass=overall, by_invariant=results)
