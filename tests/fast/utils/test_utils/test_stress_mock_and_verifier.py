"""Harness self-test for the CPU stress mock + verifier pair.

Brings up the extended ``MockSGLangServer`` (stress mode on) + a real
``SessionServer`` in-process via ``UvicornThreadServer``, pushes deterministic
turns through ``/v1/chat/completions``, and runs
``stress_verifier.verify_session`` against the resulting record dumps. The
test then mutates the records to inject the four AC-13 negative cases
(R3 single-bit flip, logp value change, record order swap, sid mismatch)
and asserts each mutation is caught by the correct invariant.

Single-process deployment is intentional: Round 1's scope is harness
self-test only. Round 2 will move stress runs to a subprocess deployment
per AC-2.
"""

from __future__ import annotations

import copy
import os

# Force CPU-only before any miles import chain runs; the test must remain
# runnable on a GPU-less CI host.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import struct  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import requests  # noqa: E402
from tests.ci.ci_register import register_cpu_ci  # noqa: E402

from miles.rollout.session.session_server import SessionServer  # noqa: E402
from miles.utils.http_utils import find_available_port  # noqa: E402
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, StressMockConfig  # noqa: E402
from miles.utils.test_utils.r3_codec import (  # noqa: E402
    decode_r3,
    encode_r3,
    format_signature,
    make_logp_tokens,
    make_r3,
    seed_from_signature,
)
from miles.utils.test_utils.stress_verifier import VerifyReport, verify_session  # noqa: E402
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer  # noqa: E402

register_cpu_ci(est_time=120, suite="stage-a-fast")


HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
OUTPUT_TOKENS = 100
R3_NUM_LAYERS = 28
R3_TOPK = 8


@pytest.fixture(scope="module")
def stress_env():
    """One mock + session server pair for the whole module; each test resets
    the mock's request log via ``backend.reset_stats()``."""
    cfg = StressMockConfig(
        enabled=True,
        output_tokens=OUTPUT_TOKENS,
        inject_routed_experts=True,
        r3_num_layers=R3_NUM_LAYERS,
        r3_topk=R3_TOPK,
        echo_signature=True,
        canonical_json=True,
    )
    mock_port = find_available_port(32000)
    backend = MockSGLangServer(
        model_name=HF_CHECKPOINT,
        process_fn=lambda _: None,  # never called in stress mode
        host="127.0.0.1",
        port=mock_port,
        latency=0.0,
        stress_config=cfg,
    )
    backend.start()

    args = SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint=HF_CHECKPOINT,
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        tito_allowed_append_roles=["tool"],
        trajectory_manager="linear_trajectory",
        session_server_instance_id=uuid.uuid4().hex,
        use_rollout_routing_replay=True,
    )
    server_obj = SessionServer(args, backend_url=backend.url)
    session_port = find_available_port(33000)
    ut_server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=session_port)
    ut_server.start()
    base_url = f"http://127.0.0.1:{session_port}"

    try:
        yield SimpleNamespace(url=base_url, backend=backend, config=cfg)
    finally:
        ut_server.stop()
        backend.stop()


def _create_session(url: str) -> str:
    r = requests.post(f"{url}/sessions", timeout=10)
    r.raise_for_status()
    return r.json()["session_id"]


def _post_one_turn(url: str, sid: str, turn: int) -> requests.Response:
    sig = format_signature(sid, turn)
    payload = {
        "messages": [{"role": "user", "content": f"{sig} please respond"}],
        "model": "mock-stress",
    }
    return requests.post(
        f"{url}/sessions/{sid}/v1/chat/completions",
        json=payload,
        timeout=30,
    )


def _fetch_session(url: str, sid: str) -> dict:
    r = requests.get(f"{url}/sessions/{sid}", timeout=10)
    r.raise_for_status()
    return r.json()


def test_happy_path_all_invariants_pass(stress_env):
    sid = _create_session(stress_env.url)
    resp = _post_one_turn(stress_env.url, sid, 0)
    assert resp.status_code == 200, resp.text

    dump = _fetch_session(stress_env.url, sid)
    assert dump["session_id"] == sid
    assert len(dump["records"]) == 1

    report = verify_session(
        records=dump["records"],
        expected_sid=sid,
        expected_turn_count=1,
        accumulated_token_ids=dump["metadata"]["accumulated_token_ids"],
        r3_num_layers=R3_NUM_LAYERS,
        r3_topk=R3_TOPK,
    )
    assert report.overall_pass, report.summary()


def test_cross_session_isolation_holds_for_two_sessions(stress_env):
    sid_a = _create_session(stress_env.url)
    sid_b = _create_session(stress_env.url)
    assert sid_a != sid_b

    assert _post_one_turn(stress_env.url, sid_a, 0).status_code == 200
    assert _post_one_turn(stress_env.url, sid_b, 0).status_code == 200

    dump_a = _fetch_session(stress_env.url, sid_a)
    dump_b = _fetch_session(stress_env.url, sid_b)

    report_a = verify_session(
        records=dump_a["records"],
        expected_sid=sid_a,
        expected_turn_count=1,
        accumulated_token_ids=dump_a["metadata"]["accumulated_token_ids"],
        r3_num_layers=R3_NUM_LAYERS,
        r3_topk=R3_TOPK,
    )
    report_b = verify_session(
        records=dump_b["records"],
        expected_sid=sid_b,
        expected_turn_count=1,
        accumulated_token_ids=dump_b["metadata"]["accumulated_token_ids"],
        r3_num_layers=R3_NUM_LAYERS,
        r3_topk=R3_TOPK,
    )
    assert report_a.overall_pass, report_a.summary()
    assert report_b.overall_pass, report_b.summary()


# ---------------------------------------------------------------------------
# AC-13 negative-case self-tests
#
# Each negative test takes the happy-path session dump, mutates the records
# in driver-side memory in a specific way, then re-runs the verifier and
# asserts the corresponding invariant flips to FAIL while every other
# invariant remains PASS.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def happy_path_dump(stress_env):
    sid = _create_session(stress_env.url)
    resp = _post_one_turn(stress_env.url, sid, 0)
    assert resp.status_code == 200, resp.text
    dump = _fetch_session(stress_env.url, sid)
    return {"sid": sid, "dump": dump}


def _run_verifier(dump_pair: dict, *, records_override=None, accum_override=None, sid_override=None) -> VerifyReport:
    return verify_session(
        records=records_override if records_override is not None else dump_pair["dump"]["records"],
        expected_sid=sid_override if sid_override is not None else dump_pair["sid"],
        expected_turn_count=len(records_override if records_override is not None else dump_pair["dump"]["records"]),
        accumulated_token_ids=(
            accum_override if accum_override is not None else dump_pair["dump"]["metadata"]["accumulated_token_ids"]
        ),
        r3_num_layers=R3_NUM_LAYERS,
        r3_topk=R3_TOPK,
    )


def test_negative_r3_bit_flip_fails_only_c3(happy_path_dump):
    records = copy.deepcopy(happy_path_dump["dump"]["records"])
    r3_str = records[0]["response"]["choices"][0]["meta_info"]["routed_experts"]
    arr = decode_r3(r3_str, OUTPUT_TOKENS, R3_NUM_LAYERS, R3_TOPK).copy()
    # Flip a single int32 entry (deterministic location).
    arr[0, 0, 0] = arr[0, 0, 0] ^ 1
    records[0]["response"]["choices"][0]["meta_info"]["routed_experts"] = encode_r3(arr)

    report = _run_verifier(happy_path_dump, records_override=records)
    assert not report.overall_pass
    assert not report.by_invariant["C3_r3_byte_identity"].passed, report.summary()
    # No other invariant should flip.
    for name, res in report.by_invariant.items():
        if name == "C3_r3_byte_identity":
            continue
        assert res.passed, f"unexpected collateral fail in {name}: {res.mismatches}"


def test_negative_logp_value_change_fails_only_c4(happy_path_dump):
    records = copy.deepcopy(happy_path_dump["dump"]["records"])
    otl = records[0]["response"]["choices"][0]["meta_info"]["output_token_logprobs"]
    # Mutate first token's logp into a value that survives JSON but flips the
    # float32 bit pattern (nudge by ~2 ULPs at -10.0).
    old_lp = float(otl[0][0])
    new_lp = struct.unpack("<f", (int.from_bytes(struct.pack("<f", old_lp), "little") + 2).to_bytes(4, "little"))[0]
    otl[0][0] = new_lp

    report = _run_verifier(happy_path_dump, records_override=records)
    assert not report.overall_pass
    assert not report.by_invariant["C4_logp_bit_identity"].passed, report.summary()
    for name, res in report.by_invariant.items():
        if name == "C4_logp_bit_identity":
            continue
        assert res.passed, f"unexpected collateral fail in {name}: {res.mismatches}"


def test_negative_record_signature_reorder_fails_c2_when_multi_turn():
    """Build two synthetic records with swapped per-turn signatures and feed
    them directly to the verifier. C2 (ordering) should catch the swap.

    This case does NOT go through the session server; mutating per-turn
    ordering on the wire would require multi-turn driver wiring which lands
    in Round 2. Verifying the invariant function itself is in scope for
    Round 1 AC-13 coverage.
    """
    sid = "synthetic-multi-turn"

    def synth_record(turn: int) -> dict:
        seed = seed_from_signature(sid, turn)
        logps, tokens = make_logp_tokens(seed, 8)
        r3_arr = make_r3(seed, 8, R3_NUM_LAYERS, R3_TOPK)
        return {
            "timestamp": 0.0,
            "method": "POST",
            "path": "/v1/chat/completions",
            "status_code": 200,
            "request": {
                "messages": [{"role": "user", "content": f"{format_signature(sid, turn)} hi"}],
                "input_ids": [1, 2, 3],
            },
            "response": {
                "choices": [
                    {
                        "meta_info": {
                            "output_token_logprobs": [
                                [float(lp), int(tid)] for lp, tid in zip(logps, tokens, strict=True)
                            ],
                            "completion_tokens": 8,
                            "routed_experts": encode_r3(r3_arr),
                        }
                    }
                ]
            },
        }

    # Build records in the wrong order: turn 1 comes before turn 0.
    records = [synth_record(1), synth_record(0)]
    accum = records[-1]["request"]["input_ids"] + [
        int(t[1]) for t in records[-1]["response"]["choices"][0]["meta_info"]["output_token_logprobs"]
    ]

    report = verify_session(
        records=records,
        expected_sid=sid,
        expected_turn_count=2,
        accumulated_token_ids=accum,
        r3_num_layers=R3_NUM_LAYERS,
        r3_topk=R3_TOPK,
    )
    assert not report.overall_pass
    assert not report.by_invariant["C2_signature_order"].passed, report.summary()


def test_negative_sid_mismatch_fails_only_c7(happy_path_dump):
    bogus_sid = "definitely-not-the-session-id"
    report = _run_verifier(happy_path_dump, sid_override=bogus_sid)
    assert not report.overall_pass
    assert not report.by_invariant["C7_cross_session_isolation"].passed, report.summary()
    for name, res in report.by_invariant.items():
        if name == "C7_cross_session_isolation":
            continue
        assert res.passed, f"unexpected collateral fail in {name}: {res.mismatches}"


# ---------------------------------------------------------------------------
# Sanity tests on the codec itself (cheap belt-and-suspenders).
# ---------------------------------------------------------------------------


def test_r3_codec_roundtrip_is_deterministic():
    seed = seed_from_signature("alpha", 7)
    arr = make_r3(seed, 16, 12, 4)
    encoded = encode_r3(arr)
    decoded = decode_r3(encoded, 16, 12, 4)
    assert decoded.tobytes() == arr.tobytes()
    # Same seed -> same array
    arr2 = make_r3(seed, 16, 12, 4)
    assert arr.tobytes() == arr2.tobytes()


def test_r3_codec_rejects_non_int32():
    import numpy as np

    bad = np.zeros(8, dtype=np.float32)
    with pytest.raises(TypeError):
        encode_r3(bad)
