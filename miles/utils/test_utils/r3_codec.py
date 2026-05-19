"""Shared R3 (Rollout Routing Replay) codec for the CPU stress harness.

This module is the single source of truth for how synthetic R3 payloads are
generated, encoded into base64 ASCII, and decoded back into ``np.int32``
arrays. Both the mock SGLang backend (encoder side) and the stress verifier
(decoder side) MUST go through these functions — re-implementing either side
inline risks subtle drift that defeats the byte-identity invariant.

Wire format is taken from the existing decoder in
``miles/rollout/generate_utils/generate_endpoint_utils.py:105-111``:

    np.frombuffer(pybase64.b64decode(s.encode("ascii")), dtype=np.int32).reshape(
        num_output_tokens, num_layers, moe_router_topk
    )

The encoder is the inverse: ``arr.astype(np.int32).tobytes()`` then
``pybase64.b64encode(...).decode("ascii")``.

Seed derivation uses ``hashlib.blake2s`` over a ``f"{sid}-{turn}"`` string for
cross-platform reproducibility; ``numpy.random.default_rng`` (PCG64) is used
rather than the legacy ``RandomState`` because its byte output is guaranteed
stable across numpy versions.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pybase64

# Recognised in mock prompts and verifier inputs; embedded into the LAST user
# message of every stress-mode request so both sides can re-derive the same
# (sid, turn_idx) tuple.
STRESS_SIG_RE = re.compile(r"\[STRESS sid=(?P<sid>[^\s\]]+) turn=(?P<turn>\d+)\]")


def format_signature(sid: str, turn: int) -> str:
    return f"[STRESS sid={sid} turn={turn}]"


def parse_signature(content: str) -> tuple[str, int] | None:
    """Extract (sid, turn) from a message content string. Returns None on miss."""
    m = STRESS_SIG_RE.search(content)
    if not m:
        return None
    return m.group("sid"), int(m.group("turn"))


def seed_from_signature(sid: str, turn: int) -> int:
    """Derive a 64-bit RNG seed from (sid, turn). Stable across platforms."""
    digest = hashlib.blake2s(f"{sid}-{turn}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def make_r3(
    seed: int,
    num_output_tokens: int,
    num_layers: int,
    topk: int,
    num_experts: int = 128,
) -> np.ndarray:
    """Generate a deterministic R3 int32 array of shape (T, L, K).

    ``num_experts`` only controls the integer range; the verifier does not
    rely on a specific value, only on byte-identity of the produced buffer.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_experts, size=(num_output_tokens, num_layers, topk), dtype=np.int32)


def encode_r3(arr: np.ndarray) -> str:
    """Encode an R3 int32 array to base64 ASCII, matching the existing wire
    format. Asserts dtype to avoid accidental float / int64 silent passes."""
    if arr.dtype != np.int32:
        raise TypeError(f"R3 array must be np.int32, got {arr.dtype}")
    return pybase64.b64encode(arr.tobytes()).decode("ascii")


def decode_r3(s: str, num_output_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    """Decode a base64 ASCII R3 string back into an int32 array of shape
    ``(num_output_tokens, num_layers, topk)``. Mirrors the decoder in
    ``miles/rollout/generate_utils/generate_endpoint_utils.py:105-111``."""
    buf = pybase64.b64decode(s.encode("ascii"))
    expected_nbytes = num_output_tokens * num_layers * topk * 4
    if len(buf) != expected_nbytes:
        raise ValueError(
            f"R3 byte length mismatch: got {len(buf)} bytes, expected "
            f"{expected_nbytes} (T={num_output_tokens}, L={num_layers}, K={topk})"
        )
    return np.frombuffer(buf, dtype=np.int32).reshape(num_output_tokens, num_layers, topk)


def make_logp_tokens(
    seed: int,
    num_output_tokens: int,
    vocab_size: int = 32000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic per-token logp (float32) and token_id (int32)
    arrays. Returns ``(logps_float32, tokens_int32)`` of length
    ``num_output_tokens`` each.

    Logps are drawn from a Uniform(-20, 0) and explicitly stored as float32
    so the byte-identity verifier can compare ``struct.pack("<f", g)`` to
    ``struct.pack("<f", e)`` without precision-loss surprises across the
    JSON serialization boundary.
    """
    rng = np.random.default_rng(seed ^ 0xA5A5_A5A5_5A5A_5A5A)
    logps = rng.uniform(-20.0, 0.0, size=num_output_tokens).astype(np.float32)
    tokens = rng.integers(0, vocab_size, size=num_output_tokens, dtype=np.int32)
    return logps, tokens
