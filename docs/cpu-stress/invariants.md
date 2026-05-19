# Session-Server Stress Verifier: Invariants Contract (Phase 0 Findings)

Derived from `scripts/tools/stress_session_server_phase0.py` end-to-end probe.
See `outputs/cpu-stress/phase0/meta_info_schema.json` for the raw dump this
contract is grounded in.

## Phase 0 Probe Result — STOP GATE PASSED

- R3 payload survives the session-server round trip **byte-identical** on
  both the decoded `int32` buffer and the encoded base64 string (the encoded
  string identity was a bonus — the harness verifier must not rely on it,
  see C3 below).
- `meta_info.routed_experts` is preserved as a top-level field on
  `choice.meta_info`; no lossy rewrite by the session server.
- `output_token_logprobs` survives as a `list[list[logp_float, token_id_int]]`
  shape (JSON 2-tuples), matching what the planned verifier expects.
- Field types, JSON paths, and overall shape of the GET response are stable
  enough to support a deterministic verifier.

The user’s pre-stated assumption that the session server does not do lossy
rewriting of `meta_info` is empirically confirmed at the smallest end of the
range (1 session × 6 output tokens). The C3 / C4 byte-identity invariants
are achievable; verifier may remain byte-level with no field-level fallback.

## Wire Shape Observations

GET `/sessions/{session_id}` returns:

```
{
  "session_id": str,
  "records": list[SessionRecord],
  "metadata": {
    "accumulated_token_ids": list[int],   # latest assistant checkpoint, not per-turn
    "max_trim_tokens": int,
    "tito_session_mismatch": <may be present>
  }
}
```

Each `SessionRecord`:

```
{
  "timestamp": float,
  "method": str,                # e.g. "POST"
  "path": str,                  # e.g. "/v1/chat/completions"
  "status_code": int,           # 200 for happy path
  "request": {                  # full request body the session server sent upstream
    "messages": list[dict],
    "model": str,
    "input_ids": list[int],     # injected by session_server (Phase 1 prep)
    "logprobs": True,
    "return_meta_info": True,
    "return_routed_experts": True,   # only if use_rollout_routing_replay=True
    "no_stop_trim": False
  },
  "response": {                 # full JSON response from the mock backend
    "id": str,
    "object": str,
    "created": int,
    "model": str,
    "choices": [
      {
        "index": int,
        "message": {"role": "assistant", "content": str, "tool_calls": null},
        "logprobs": {"content": list},
        "finish_reason": str,
        "meta_info": {
          "output_token_logprobs": list[[float, int]],
          "completion_tokens": int,
          "routed_experts": str        # base64 ASCII of int32 buffer, INLINE
        }
      }
    ]
  }
}
```

Note: `response` body is whatever the backend returned. With the real SGLang
backend, additional `meta_info` keys may appear (e.g. `finish_reason` dict,
`prompt_tokens`, `cached_tokens`); the stress harness must tolerate this and
key into specific fields rather than expecting an exhaustive schema match.

## Signature Channel

The driver embeds the per-turn signature `f"{sid}-{turn_idx}"` into the
**last user message** of `request.messages`. Justification:

- `request.messages` is round-tripped byte-identically into `record.request`.
- `messages` is also the input the mock’s `process_fn` sees (decoded prompt),
  so the mock can derive the same `(sid, turn_idx)` seed from the prompt
  without any new request field.
- The other request fields (`input_ids`, `logprobs`, etc.) are written by
  the session server itself and would not preserve driver-side state.

Convention (will be hard-coded in mock + verifier):

```
messages[-1]["content"] starts with:  "[STRESS sid=<sid> turn=<turn_idx>] ..."
```

The mock parses this regex on every chat-completions call to derive its
seed; the verifier reads the same field from `record.request.messages[-1]`
to re-derive the expected seed at compare time.

## Decoded R3 Codec (Authoritative)

This codec is the single source of truth shared between mock and verifier.
A future `r3_codec.py` shared module will host these two functions; for
Round 0 the probe inlines them and Round 1+ extracts them.

```python
def encode_r3(arr: np.ndarray) -> str:
    """arr.dtype == np.int32, arr.shape == (T, L, K). Returns base64 ASCII."""
    assert arr.dtype == np.int32
    return pybase64.b64encode(arr.tobytes()).decode("ascii")

def decode_r3(s: str, num_output_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    """Mirrors miles/rollout/generate_utils/generate_endpoint_utils.py:105-111."""
    buf = pybase64.b64decode(s.encode("ascii"))
    return np.frombuffer(buf, dtype=np.int32).reshape(num_output_tokens, num_layers, topk)
```

R3 byte size per token: `num_layers × moe_router_topk × 4` bytes.
For Qwen3-30B-A3B (48 layers × 8 topk): 1536 bytes/token, 49,152 bytes/100 tok.
For 50k trajectory: ~75 MB/session × 1024 sessions = ~75 GB (worst case),
which is the order of magnitude the stress plan targets.

## C1–C7 Invariants — Finalized

Each invariant lists (a) what it asserts, (b) how the verifier checks it
against the GET response, (c) what failure says about the system.

### C1 — Record count

**Assertion**: After K successful `POST /sessions/{sid}/v1/chat/completions`
calls, `GET /sessions/{sid}` returns `len(records) == K`.

**Check**: `len(session_dump["records"]) == client_side_K`.

**Failure means**: session server dropped one or more appends silently;
worst case is silent data loss under concurrency.

### C2 — Per-record signature + ordering

**Assertion**: `records[i].request.messages[-1].content` starts with the
signature `f"[STRESS sid={sid} turn={i}]"`, in order, for every i ∈ [0, K).

**Check**: parse the signature regex from each record’s last user message;
assert the (sid, turn_idx) tuple equals `(expected_sid, i)`.

**Failure means**: either the session server reordered records, or a record
from another session leaked in (cross-session bleed — also caught by C7,
but C2 catches the per-session ordering layer first).

### C3 — R3 byte-identity

**Assertion**: For every record i, decode
`records[i].response.choices[0].meta_info.routed_experts` per the codec above
and assert the resulting `int32` buffer `.tobytes()` equals the buffer the
mock generated from the same `(sid, i)` seed.

**Check**:
```python
expected_arr, _ = make_synthetic_r3(f"{sid}-{i}", T_i, NUM_LAYERS, TOPK)
got_arr = decode_r3(records[i].response.choices[0].meta_info.routed_experts,
                   T_i, NUM_LAYERS, TOPK)
assert got_arr.tobytes() == expected_arr.tobytes()
```

**Failure means**: bytes corrupted between mock and session-record storage;
this would block all R3 RL downstream usage and is the highest-severity
correctness signal.

**Important**: do NOT compare the base64 string. The probe observed encoded-
string identity in this run, but the verifier must tolerate alternative
base64 encoders that re-pad equivalently (the encoded form is not the
canonical layer — the int32 buffer is).

### C4 — Logprob bit-identity

**Assertion**: For every record i, `meta_info.output_token_logprobs` is a
list of `[logp_float, token_id_int]` pairs whose entries are bit-identical
to what the mock generated.

**Check**:
```python
expected_logps_int = expected_logps_for_seed(f"{sid}-{i}", T_i)  # list[(np.float32, int)]
got = records[i].response.choices[0].meta_info.output_token_logprobs  # list[[float, int]]
assert len(got) == len(expected_logps_int)
for (g_lp, g_tok), (e_lp_f32, e_tok) in zip(got, expected_logps_int):
    assert g_tok == e_tok
    assert struct.pack("<f", float(g_lp)) == struct.pack("<f", float(e_lp_f32))
```

**Failure means**: server is mutating logprob values, dropping precision,
or reordering token alignment. Bit-pattern comparison sidesteps the
non-canonical JSON float representation.

**Constraint**: mock generates logprobs as `np.float32` then casts to
Python `float` for JSON serialization. The verifier casts back to
`np.float32` via `struct.pack("<f", ...)` to compare. This is the same
canonical layer both sides agree on.

### C5 — `accumulated_token_ids` cumulative correctness

**Assertion**: After K successful turns, `metadata.accumulated_token_ids`
equals the **last turn’s prompt_token_ids** concatenated with **each
turn’s completion_token_ids in order**.

**Observation from probe**: `metadata.accumulated_token_ids` exposes only
the LATEST assistant checkpoint (per `LinearTrajectory.token_ids`), not
per-turn `trajectory_token_ids`. So C5 verifies the final cumulative state,
not per-turn deltas.

**Check**:
```python
last_prompt = records[-1].request.input_ids   # what session-server sent upstream for the last turn
all_completions = []
for r in records:
    completion = [pair[1] for pair in r.response.choices[0].meta_info.output_token_logprobs]
    all_completions.extend(completion)
# accumulated_token_ids should equal the last assistant checkpoint:
#   = last_prompt + completion_of_last_turn
# but verifier wants global cumulative, so we compute:
expected_cumulative = last_prompt + [tok for tok in completion_of_last_turn]
assert session_dump.metadata.accumulated_token_ids == expected_cumulative
```

**Rolling-hash optimization** (per AC-6, O(1) memory): instead of storing
all completion tokens, the verifier computes a rolling FNV-1a hash of the
expected token id sequence as it iterates and compares to the same hash
computed over `accumulated_token_ids`. Length comparison is independent
and cheap.

**Failure means**: token-id list and record list diverge — possible
double-checkpoint, lost completion, or wrong prompt prefix.

### C6 — `num_assistant` count

**Observation**: `metadata` does NOT expose `num_assistant` directly. It is
internal to `LinearTrajectory`. Externally, the count can be derived as
`len([r for r in records if r.path == "/v1/chat/completions" and r.status_code == 200])`.

**Assertion (revised)**: For a session where the driver made K successful
chat-completions calls, `len([r for r in records if r.status_code == 200])`
equals K. (Per C1 this is a tautology unless C1 already failed.)

**Future strengthening**: If a test endpoint or admin route exposing
`num_assistant` is added later, C6 should also verify the internal counter
equals K. For Round 0 we accept the externally-derivable form.

**Failure means**: same failure space as C1 + ordering — internal counter
out of sync with observed records. This is also captured indirectly by
C5 (cumulative tokens would diverge).

### C7 — Cross-session isolation

**Assertion**: For every record across all sessions, the signature parsed
from `record.request.messages[-1].content` has `sid` equal to the session
the record belongs to. No exceptions, full coverage.

**Check**:
```python
for sid, session_records in all_dumps.items():
    for r in session_records.records:
        m = STRESS_SIG_REGEX.search(r.request.messages[-1].content)
        assert m, f"missing signature on record in session {sid}"
        observed_sid = m.group("sid")
        assert observed_sid == sid, (
            f"record stamped sid={observed_sid} found in session {sid}"
        )
```

**Failure means**: server has a concurrency bug that writes one session’s
record to another session’s record list. Highest-severity bug class.

**Coverage**: every record, every session, no sampling. This is cheap
per-record (string regex + comparison) so full coverage is affordable.

## Updates to the Plan

The plan’s draft C1–C7 contract referenced `trajectory_token_ids` (the
internal per-turn list). The probe shows the GET response only exposes
`accumulated_token_ids` (the latest checkpoint, flat list). C5 is therefore
re-formulated above to operate on `accumulated_token_ids` plus the in-record
`completion_token_ids` derived from `output_token_logprobs[i][1]`.

C6 is re-formulated to compare derived assistant-record count rather than
reading `num_assistant` directly, since the field is not exposed via
`GET /sessions/{sid}`.

These re-formulations preserve the user-facing intent of every invariant.
The plan acceptance criteria (AC-5 through AC-8) remain unchanged; only
the internal "how we check" mapping is tightened to observed wire shape.

## Stop Gate Evaluation

| Question | Phase 0 Answer | Action |
|----------|----------------|--------|
| Is `routed_experts` inline in `meta_info`? | Yes, base64 ASCII string | proceed |
| Decoder = `np.frombuffer(b64decode(s), int32).reshape(...)`? | Yes, matches `generate_endpoint_utils.py:105-111` | proceed |
| Does session server preserve byte-identity? | Yes (both decoded buffer and encoded string at probe scale) | proceed |
| Is signature channel survivable? | Yes, via `request.messages[-1].content` regex | proceed |
| Does logprob shape match the verifier’s expectation? | Yes, `list[[float, int]]` | proceed |
| Are there fields the verifier must NOT depend on? | `tito_session_mismatch` may or may not be present; `record.response.id` is non-deterministic | document |
| Does `accumulated_token_ids` expose per-turn deltas? | No, only the latest assistant checkpoint | re-formulate C5 |
| Is `num_assistant` exposed via GET? | No | re-formulate C6 |

**Decision**: PROCEED to Round 1 (mock extension + driver + smoke).

No need to re-scope the plan’s memory targets or replace the byte-identity
verifier strategy. The two re-formulations above are minor and live in
this doc; the plan AC text remains correct.
