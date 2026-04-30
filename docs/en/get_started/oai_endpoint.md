# OAI Endpoint Usage

This document explains how to use the OpenAI-format chat endpoint through Miles
Router sessions. For the `/generate` endpoint, see
`docs/en/get_started/gen_endpoint.md`.

## 1. Minimal `run_agent` loop

Your `run_agent` receives a session-scoped `base_url`. Send OpenAI-format chat
requests to `base_url/v1/chat/completions` and pass the `messages` list as the
prompt.

Minimal custom agent example:

```python
from miles.utils.http_utils import post

async def run_agent(base_url: str, prompt, request_kwargs: dict | None = None) -> None:
    payload = {"model": "default", "messages": prompt, **(request_kwargs or {})}
    await post(f"{base_url}/v1/chat/completions", payload)
```

Notes for `run_agent`:

- `base_url` already includes the session path (e.g. `/sessions/<id>`), so you
  should not manually add the session id. Just append the OpenAI route.
- `request_kwargs` already contains the default sampling settings from
  `agentic_tool_call.build_chat_request_kwargs`, so you can directly expand it
  into the chat request payload.
- If you pass rollout sampling params, `max_new_tokens` will be mapped to the
  OpenAI `max_tokens` field before the request is sent.
- If you need structured parsing payloads, use SGLang's
  `ChatCompletionRequest`-compatible format. It is compatible with native OpenAI
  fields, plus extra SGLang parameters.

## 2. OpenAI chat messages and the basic request

The OpenAI-format chat API uses a list of `messages`, each with a `role` and
`content`.

Minimal request shape:

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Answer with one word: 2+2?"}
  ],
  "logprobs": true,
  "return_prompt_token_ids": true
}
```

You can pass any OpenAI-compatible parameters in the payload, or any
SGLang-compatible `ChatCompletionRequest` parameters. Note:
`logprobs=True` and `return_prompt_token_ids=True` are set in
`request_kwargs` to extract token ids and logprobs for TITO (see below).
Do **not** set `logprob_start_len=0` — it would disable SGLang's prefix
cache.

## 3. Quickstart index

If you just want something runnable, start here:

Generator entry point:

- `miles/rollout/generate_hub/agentic_tool_call.py`
  - OpenAI-format agent loop via router sessions.

OpenAI-format examples that use `agentic_tool_call.generate`:

- `examples/openai_format/dapo_math.py`
  - Single-turn OpenAI format agent (DAPO math).
- Launcher scripts:
  - `examples/openai_format/run-qwen3-4B.sh`


You can customize generate function like:
```
CUSTOM_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path examples.openai_format.dapo_math.run_agent
)
```

For OpenAI format, do not add `--apply-chat-template`; the
prompt must remain a `messages` list.

More agentic multi-turn examples will come in the future.

## 4. Further customization (OpenAI wrapper generate function)

For OpenAI-format rollout, the key generate function is
`miles/rollout/generate_hub/agentic_tool_call.generate`. It is a thin wrapper
around your custom agent:

1. Create a session on Miles Router and build a session-scoped `base_url`.
2. Call the custom agent (from `--custom-agent-function-path`) to send one or
   more chat requests to `base_url/v1/chat/completions`, typically using
   `prompt` and `request_kwargs`.
3. Collect session records via `OpenAIEndpointTracer`.
4. Convert records into `Sample` objects with
   `compute_samples_from_openai_records`.

If you want general generate-function customization beyond the OpenAI wrapper,
see `docs/en/get_started/gen_endpoint.md`.

## 5. TITO (token-in token-out)

TITO needs two things from the SGLang response:

1. **Prompt token ids** — extracted from `response.choices[0].prompt_token_ids`.
   This field is returned by SGLang when the request sets
   `return_prompt_token_ids=True`.
2. **Output token ids and logprobs** — extracted from
   `response.choices[0].meta_info.output_token_logprobs[*]`.
   These fields are returned when the request sets `logprobs=True` and
   `return_meta_info=True`.

When you route chat requests through the session server, the middleware
forces the SGLang flags required by TITO:

- `logprobs=True`
- `return_prompt_token_ids=True`
- `return_meta_info=True`
- `no_stop_trim=False`

The first turn is sent normally. After that, the session server may inject
`input_ids` built from:

- the exact token prefix returned by previous turns, plus
- incremental tokens computed for newly appended non-assistant messages

This is the core TITO optimization: multi-turn sessions reuse tokens
across turns instead of re-tokenizing the full conversation on every request.

At the end of the session, `OpenAIEndpointTracer.collect_records` fetches the
per-turn records and metadata, and `compute_samples_from_openai_records` turns
them into `Sample` objects while trimming model-specific trailing boundary
tokens when needed.

### Common pitfalls

- Ensure `logprobs=True` and `return_prompt_token_ids=True` in OpenAI chat
  requests (both are already set in `request_kwargs`).
- Do **not** set `logprob_start_len=0` — it forces SGLang to compute
  logprobs for every prompt token, which destroys the prefix cache and hurts
  performance. Use `return_prompt_token_ids=True` instead, which returns
  prompt token ids at zero cost without affecting caching.

## 6. Session server message constraints

The session server enforces two invariants on every request's `messages`.
Violations raise `MessageValidationError`.

- **Append-only.** Each new request's `messages` must contain the previous
  turn's full `messages` as an exact prefix (same roles, same
  template-relevant content). You can extend at the tail; you cannot edit,
  reorder, or drop earlier messages. This is how TITO can reuse tokens
  across turns: the stored token prefix is only valid if the client
  confirms the same message prefix on every request.

- **Allowed append roles.** After the first assistant message, every newly
  appended message must have a role in `--tito-allowed-append-roles`
  (default `{tool}`; valid values: `tool`, `user`, `system`). Typical usage:

  - Pure tool-calling agent — `tool` (default).
  - Multi-turn chat with mid-conversation user follow-ups — add `user`.
  - Mid-conversation system reminders — also add `system`.

  See [chat-template verification](../agentic/chat_template_verification.md)
  for how this flag drives template resolution at training time.

Both invariants depend on the chat template's rendering behavior under the
chosen role surface. Adding miles support for a new model family therefore
involves both updating the chat template and adapting the TITO tokenizer
subclass for that family.
