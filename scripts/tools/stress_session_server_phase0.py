#!/usr/bin/env python3
"""Phase 0 probe for the CPU-only session-server stress harness.

Brings up MockSGLangServer + SessionServer in the same process (via
UvicornThreadServer), pushes 1 session x 1 turn with a deterministic
synthetic R3 payload, then dumps the real meta_info schema and verifies
byte-identity of R3 through the round trip.

The probe answers two questions the larger stress plan depends on:

1. Is meta_info.routed_experts inline-encoded as base64-ascii-of-int32?
   (Equivalent to the decoder in
    miles/rollout/generate_utils/generate_endpoint_utils.py:105-111.)

2. Does the session server pass meta_info through byte-identical, so the
   downstream byte-level round-trip verifier (AC-3) is actually feasible?

The probe writes meta_info_schema.json + full_session_dump.json into the
output directory and exits non-zero on any byte-identity failure.

CPU-only: CUDA_VISIBLE_DEVICES is forced empty before any miles imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Force CPU-only BEFORE miles / transformers / torch import chain runs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pybase64  # noqa: E402
import requests  # noqa: E402

from miles.rollout.session.session_server import SessionServer  # noqa: E402
from miles.utils.http_utils import find_available_port  # noqa: E402
from miles.utils.processing_utils import load_tokenizer  # noqa: E402
from miles.utils.test_utils.mock_sglang_server import (  # noqa: E402
    MockSGLangServer,
    ProcessResult,
    ProcessResultMetaInfo,
    with_mock_server,
)
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
# R3 shape knobs for the synthetic payload. Realistic-ish values for a small
# MoE model (e.g. Qwen3-30B-A3B has 48 layers x 8 topk = 384 int32/token, which
# is the same order of magnitude as the user-stated ~250 int32/token estimate).
# Phase 0 uses smaller numbers to keep the dump compact while still exercising
# the (output_tokens, num_layers, topk) reshape.
PHASE0_NUM_LAYERS = 28
PHASE0_TOPK = 8


def make_synthetic_r3(
    seed_material: str,
    num_output_tokens: int,
    num_layers: int,
    topk: int,
) -> tuple[np.ndarray, str]:
    """Generate deterministic R3 routing IDs and their base64 ascii string.

    Uses numpy default_rng (PCG64) seeded from blake2s of seed_material for
    cross-platform reproducibility; np.random.RandomState is intentionally
    avoided since its algorithm is not guaranteed stable across versions.
    """
    seed_int = int.from_bytes(hashlib.blake2s(seed_material.encode("ascii")).digest()[:8], "big")
    rng = np.random.default_rng(seed_int)
    arr = rng.integers(0, 128, size=(num_output_tokens, num_layers, topk), dtype=np.int32)
    encoded = pybase64.b64encode(arr.tobytes()).decode("ascii")
    return arr, encoded


def build_phase0_chat_patch():
    """Patch MockSGLangServer._compute_chat_completions_response so that the
    response carries (a) output_token_logprobs in (logprob, token_id) tuple
    form the session server validates and (b) any routed_experts the process_fn
    set on ProcessResultMetaInfo. The stock mock spreads meta_info via
    to_dict() so routed_experts already round-trips; we keep that behavior
    and only adjust the shape so session_server's length check passes.
    """

    def patched(self, payload):
        messages = payload.get("messages", [])
        tools = payload.get("tools")

        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )
        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        output_token_logprobs = [(-1.0 / 128.0 * i, tid) for i, tid in enumerate(output_ids)]

        meta_info = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_ids),
            **process_result.meta_info.to_dict(),
        }

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": process_result.text,
                "tool_calls": None,
            },
            "logprobs": {"content": []},
            "finish_reason": process_result.finish_reason,
            "meta_info": meta_info,
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [choice],
        }

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)


def run_phase0(output_dir: Path, hf_checkpoint: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", hf_checkpoint)
    tokenizer = load_tokenizer(hf_checkpoint, chat_template_path=None, trust_remote_code=True)
    output_text = "Phase 0 probe response."
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    num_output_tokens = len(output_ids)
    logger.info("Output text tokenizes to %d tokens", num_output_tokens)

    seed_material = "phase0-probe"
    r3_arr, r3_encoded = make_synthetic_r3(seed_material, num_output_tokens, PHASE0_NUM_LAYERS, PHASE0_TOPK)
    logger.info(
        "Synthetic R3: shape=%s dtype=%s nbytes=%d base64_len=%d",
        r3_arr.shape,
        r3_arr.dtype,
        r3_arr.nbytes,
        len(r3_encoded),
    )

    def process_fn(_prompt: str) -> ProcessResult:
        return ProcessResult(
            text=output_text,
            finish_reason="stop",
            meta_info=ProcessResultMetaInfo(routed_experts=r3_encoded),
        )

    with build_phase0_chat_patch(), with_mock_server(model_name=hf_checkpoint, process_fn=process_fn) as backend:
        args = SimpleNamespace(
            miles_router_timeout=30,
            hf_checkpoint=hf_checkpoint,
            chat_template_path=None,
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model="default",
            tito_allowed_append_roles=["tool"],
            trajectory_manager="linear_trajectory",
            session_server_instance_id=uuid.uuid4().hex,
            use_rollout_routing_replay=True,
        )
        server_obj = SessionServer(args, backend_url=backend.url)
        port = find_available_port(31000)
        ut_server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        ut_server.start()
        url = f"http://127.0.0.1:{port}"
        try:
            r = requests.post(f"{url}/sessions", timeout=10)
            r.raise_for_status()
            sid = r.json()["session_id"]
            logger.info("Created session: %s", sid)

            payload = {
                "messages": [{"role": "user", "content": "ping"}],
                "model": "mock-model",
            }
            r = requests.post(
                f"{url}/sessions/{sid}/v1/chat/completions",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            chat_response = r.json()
            logger.info("Chat completion returned %d top-level keys", len(chat_response))

            r = requests.get(f"{url}/sessions/{sid}", timeout=10)
            r.raise_for_status()
            session_dump = r.json()
        finally:
            ut_server.stop()

    if len(session_dump["records"]) != 1:
        raise AssertionError(f"expected 1 record, got {len(session_dump['records'])}")

    record = session_dump["records"][0]
    meta_info = record["response"]["choices"][0]["meta_info"]
    if "routed_experts" not in meta_info:
        raise AssertionError(
            "Phase 0 STOP GATE: meta_info.routed_experts not present in record after round trip. "
            "Session server may be stripping the field. Keys observed: "
            f"{sorted(meta_info.keys())}"
        )

    r3_returned_b64 = meta_info["routed_experts"]
    r3_returned_bytes = pybase64.b64decode(r3_returned_b64.encode("ascii"))
    r3_returned_arr = np.frombuffer(r3_returned_bytes, dtype=np.int32).reshape(
        num_output_tokens, PHASE0_NUM_LAYERS, PHASE0_TOPK
    )
    byte_identity = bool(r3_returned_arr.tobytes() == r3_arr.tobytes())
    encoded_identity = bool(r3_returned_b64 == r3_encoded)

    schema = {
        "phase0_inputs": {
            "hf_checkpoint": hf_checkpoint,
            "num_output_tokens": num_output_tokens,
            "num_layers": PHASE0_NUM_LAYERS,
            "topk": PHASE0_TOPK,
            "seed_material": seed_material,
        },
        "r3_synthetic": {
            "shape": list(r3_arr.shape),
            "dtype": str(r3_arr.dtype),
            "nbytes": int(r3_arr.nbytes),
            "base64_len": len(r3_encoded),
            "first_8_bytes_hex": r3_arr.tobytes()[:8].hex(),
        },
        "r3_round_trip": {
            "decoded_byte_identity": byte_identity,
            "encoded_string_identity": encoded_identity,
            "returned_shape": list(r3_returned_arr.shape),
            "returned_dtype": str(r3_returned_arr.dtype),
            "returned_nbytes": int(r3_returned_arr.nbytes),
        },
        "session_get_keys": sorted(session_dump.keys()),
        "session_metadata_keys": sorted(session_dump.get("metadata", {}).keys()),
        "record_keys": sorted(record.keys()),
        "record_request_keys": sorted(record["request"].keys()),
        "record_response_keys": sorted(record["response"].keys()),
        "choice_keys": sorted(record["response"]["choices"][0].keys()),
        "meta_info_keys": sorted(meta_info.keys()),
        "meta_info_field_types": {k: type(v).__name__ for k, v in meta_info.items()},
        "output_token_logprobs_sample": meta_info["output_token_logprobs"][:3],
        "completion_tokens": meta_info["completion_tokens"],
        "accumulated_token_ids_len": len(session_dump.get("metadata", {}).get("accumulated_token_ids", [])),
        "max_trim_tokens": session_dump.get("metadata", {}).get("max_trim_tokens"),
    }

    schema_path = output_dir / "meta_info_schema.json"
    full_dump_path = output_dir / "full_session_dump.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    full_dump_path.write_text(json.dumps(session_dump, indent=2, ensure_ascii=False))

    logger.info("Wrote %s", schema_path)
    logger.info("Wrote %s", full_dump_path)

    if not byte_identity:
        logger.error(
            "R3 byte-identity FAILED: session server did NOT preserve meta_info.routed_experts byte-for-byte. "
            "The downstream stress harness verifier (AC-3) cannot proceed with byte-level comparison; "
            "either the session server is rewriting meta_info, or our encoder/decoder pair is wrong."
        )
        raise SystemExit(2)

    if not encoded_identity:
        logger.warning(
            "R3 encoded-string identity FAILED but decoded byte-identity PASSED. "
            "Session server may be re-encoding base64 with different padding; "
            "downstream verifier must compare on decoded int32 buffer, NOT the base64 string."
        )

    logger.info(
        "R3 byte-identity PASSED (shape=%s, nbytes=%d, encoded_str_identity=%s)",
        r3_arr.shape,
        r3_arr.nbytes,
        encoded_identity,
    )
    return schema


def main():
    parser = argparse.ArgumentParser(description="Phase 0 probe for CPU-only session-server stress harness")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cpu-stress/phase0"),
        help="Directory to write meta_info_schema.json and full_session_dump.json",
    )
    parser.add_argument(
        "--hf-checkpoint",
        default=DEFAULT_HF_CHECKPOINT,
        help="HF model name for tokenizer (must be cached locally for fully offline run)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    schema = run_phase0(args.output_dir, args.hf_checkpoint)
    print(json.dumps(schema, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
