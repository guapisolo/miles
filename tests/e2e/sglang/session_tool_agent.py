"""Custom agent function for the session-server tool-call e2e test.

Performs a multi-turn tool-calling conversation through the session proxy and
verifies that the session's prompt_token_ids, when decoded, exactly match
the text produced by locally applying the chat template to the same messages.

The agent is loaded at runtime by ``agentic_tool_call.generate`` via
``--custom-agent-function-path tests.e2e.sglang.session_tool_agent.run_agent``.
"""

import logging
import os

import httpx
from sglang.srt.entrypoints.openai.protocol import Tool
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import try_get_fixed_chat_template

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("SGLANG_E2E_MODEL_PATH", "Qwen/Qwen3-4B")

MAX_TOOL_TURNS = 8
MAX_RETRIES = 2

RETRY_SYSTEM_MESSAGE = (
    "Your previous response was not a valid tool call and did not contain a "
    "final answer. Please either call a tool or provide your final answer "
    "inside <final_answer>...</final_answer> tags."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. Beijing",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

MOCK_TOOL_RESULTS = [
    '{"temperature_celsius": 22, "condition": "sunny", "humidity": 45}',
    '{"temperature_celsius": 15, "condition": "cloudy", "humidity": 70}',
    '{"temperature_celsius": 30, "condition": "rainy", "humidity": 90}',
    '{"temperature_celsius": 8, "condition": "snowy", "humidity": 85}',
]


def _load_chat_template(model_path: str) -> str:
    """Load the fixed chat template for the model."""
    template_path = try_get_fixed_chat_template(model_path)
    assert template_path is not None, f"No fixed chat template found for {model_path}"
    with open(template_path) as f:
        return f.read()


def _is_task_complete(assistant_msg: dict) -> bool:
    """Check if the assistant has produced a final answer."""
    content = assistant_msg.get("content") or ""
    return "<final_answer>" in content and "</final_answer>" in content


def _extract_tool_calls(assistant_msg: dict) -> list[dict] | None:
    """Return structured tool_calls from the assistant message, or None.

    Only trusts the structured ``tool_calls`` field populated by sglang's
    tool-call parser.  No fallback parsing — if the parser didn't produce
    structured output, the caller should treat it as a failed tool call
    and retry via a system message.
    """
    if assistant_msg.get("tool_calls"):
        return assistant_msg["tool_calls"]
    return None


async def _chat(
    client: httpx.AsyncClient,
    url: str,
    messages,
    rk,
    label="",
    tool_choice=None,
):
    """Send a chat completions request and return (response_json, prompt_token_ids)."""
    payload = {
        "messages": messages,
        "tools": TOOLS,
        **rk,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    resp = await client.post(f"{url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200, f"{label} failed ({resp.status_code}): {resp.text}"
    data = resp.json()
    prompt_ids = data["choices"][0].get("prompt_token_ids")
    assert prompt_ids is not None, f"{label}: prompt_token_ids missing"
    return data, prompt_ids


def _verify_turn(
    turn: int,
    session_prompt_ids: list[int],
    messages: list[dict],
    prev_session_text: str | None,
    tokenizer,
    chat_template: str,
):
    """Assert that decoded session prompt text matches locally rendered template."""
    session_text = tokenizer.decode(session_prompt_ids, skip_special_tokens=False)
    tool_defs = [Tool(**t).function.model_dump() for t in TOOLS]
    expected_text = tokenizer.apply_chat_template(
        messages, tools=tool_defs, chat_template=chat_template, tokenize=False, add_generation_prompt=True
    )

    if session_text != expected_text:
        first_char_diff = next(
            (i for i, (a, b) in enumerate(zip(session_text, expected_text, strict=False)) if a != b),
            min(len(session_text), len(expected_text)),
        )
        ctx_lo = max(0, first_char_diff - 80)
        ctx_hi = min(max(len(session_text), len(expected_text)), first_char_diff + 80)
        raise AssertionError(
            f"TITO TEXT MISMATCH on turn {turn}!\n"
            f"Session decoded length: {len(session_text)} chars "
            f"({len(session_prompt_ids)} tokens)\n"
            f"Template rendered length: {len(expected_text)} chars\n"
            f"First char diff at index {first_char_diff}\n"
            f"\n--- Session decoded text around diff ---\n"
            f"{session_text[ctx_lo:ctx_hi]!r}\n"
            f"\n--- Template rendered text around diff ---\n"
            f"{expected_text[ctx_lo:ctx_hi]!r}\n"
        )

    if prev_session_text is not None:
        assert session_text.startswith(prev_session_text), (
            f"Prefix mismatch on turn {turn}: previous turn's decoded text "
            f"({len(prev_session_text)} chars) is not a prefix of this turn's "
            f"({len(session_text)} chars).\n"
            f"Previous tail: {prev_session_text[-100:]!r}\n"
            f"Current head:  {session_text[:len(prev_session_text)][-100:]!r}"
        )

    return session_text


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Multi-turn tool-call agent that verifies session prompt correctness.

    On every turn, decodes the session's prompt_token_ids and compares with
    locally rendered chat template text — they must be identical.
    """
    messages = list(prompt)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    rk.setdefault("return_prompt_token_ids", True)
    rk.setdefault("logprobs", True)

    model_path = metadata.get("model_path", MODEL_NAME) if metadata else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chat_template = _load_chat_template(model_path)

    turns_completed = 0
    total_tool_calls = 0
    prev_session_text = None
    consecutive_retries = 0

    async with httpx.AsyncClient(
        timeout=180, limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    ) as client:
        for turn in range(1, MAX_TOOL_TURNS + 1):
            label = f"Session Turn {turn}"
            resp_data, session_ids = await _chat(
                client,
                base_url,
                messages,
                rk,
                label=label,
                tool_choice="auto",
            )
            logger.info("%s: %d prompt tokens", label, len(session_ids))

            session_text = _verify_turn(
                turn,
                session_ids,
                messages,
                prev_session_text,
                tokenizer,
                chat_template,
            )
            prev_session_text = session_text
            logger.info(
                "Turn %d VERIFIED: session decoded text matches template (%d tokens, %d chars)",
                turn,
                len(session_ids),
                len(session_text),
            )

            turns_completed = turn

            assistant_msg = resp_data["choices"][0]["message"]
            messages.append(assistant_msg)

            tool_calls = _extract_tool_calls(assistant_msg)
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    mock_idx = (total_tool_calls + i) % len(MOCK_TOOL_RESULTS)
                    messages.append(
                        {
                            "role": "tool",
                            "content": MOCK_TOOL_RESULTS[mock_idx],
                            "tool_call_id": tc["id"],
                        }
                    )
                total_tool_calls += len(tool_calls)
                logger.info(
                    "Turn %d: appended %d tool result(s), total tool calls so far: %d",
                    turn,
                    len(tool_calls),
                    total_tool_calls,
                )
                consecutive_retries = 0
            elif _is_task_complete(assistant_msg):
                logger.info("Turn %d: task complete, ending loop", turn)
                break
            else:
                consecutive_retries += 1
                if consecutive_retries > MAX_RETRIES:
                    logger.warning("Turn %d: exceeded %d retries, ending loop", turn, MAX_RETRIES)
                    break
                messages.append({"role": "system", "content": RETRY_SYSTEM_MESSAGE})
                logger.info(
                    "Turn %d: no tool calls and not complete, appended retry system message (%d/%d)",
                    turn,
                    consecutive_retries,
                    MAX_RETRIES,
                )

    MIN_TOOL_CALLS = 3
    if total_tool_calls < MIN_TOOL_CALLS:
        raise AssertionError(
            f"Only made {total_tool_calls} successful tool call(s) in {turns_completed} turn(s), "
            f"need >= {MIN_TOOL_CALLS} for meaningful multi-turn prefix invariant verification"
        )
    return {
        "turns_completed": turns_completed,
        "total_tool_calls": total_tool_calls,
        "prefix_invariant_verified": True,
    }
