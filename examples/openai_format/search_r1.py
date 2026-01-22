"""
Search-R1 agent via OpenAI-compatible chat/completions.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.utils.http_utils import post
from miles.utils.types import Sample

SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    "format_score": 0.2,
}

SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])

SEARCH_R1_DIR = Path(__file__).resolve().parents[1] / "search-r1"
if str(SEARCH_R1_DIR) not in sys.path:
    sys.path.insert(0, str(SEARCH_R1_DIR))

from google_search_server import google_search
from local_search_server import local_search
from qa_em_format import compute_score_em


def _passages2string(retrieval_result: list[dict[str, Any]]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. Must be either 'local' or 'google'.")

    return _passages2string(result)


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
    else:
        content = ""
        action = None

    return action, content


async def execute_predictions(prediction: str) -> tuple[list[dict[str, Any]], bool]:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        async with SEMAPHORE:
            search_results = await search(content)
        next_obs = f"<information>{search_results.strip()}</information>"
        return [{"role": "user", "content": next_obs}], False
    if action == "answer":
        return [], True

    next_obs = (
        "My previous action is invalid. "
        "If I want to search, I should put the query between <search> and </search>. "
        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
        "Let me try again."
    )
    return [{"role": "user", "content": next_obs}], False


def _normalize_messages(prompt: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(message) for message in prompt]


def _build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}


async def run_agent(base_url: str, prompt: list[dict[str, Any]] | str, sampling_params: dict[str, Any]) -> None:
    request_kwargs = _build_chat_request_kwargs(sampling_params)
    messages = _normalize_messages(prompt)

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {
            "model": "default",
            "messages": messages,
            "logprobs": True,
            **request_kwargs,
        }
        response = await post(base_url + "/v1/chat/completions", payload)
        choice = response["choices"][0]
        content = choice["message"].get("content") or ""
        messages.append({"role": "assistant", "content": content})

        if choice.get("finish_reason") == "length":
            break

        next_messages, done = await execute_predictions(content)
        if done:
            break
        messages.extend(next_messages)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    # Best-effort formatting for reward parsing.
    parts = []
    for message in messages:
        role = message["role"]
        content = message.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    prompt = sample.prompt
    if isinstance(prompt, list):
        prompt = _messages_to_prompt(prompt)

    score = compute_score_em(
        solution_str=prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
