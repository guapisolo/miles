"""
Tau2-bench agent via OpenAI-compatible chat/completions.
"""

from __future__ import annotations

import json
from typing import Any
from tau2.gym.gym_agent import AgentGymEnv

from miles.utils.http_utils import post
from miles.utils.types import Sample

TAU2_CONFIGS = {
    "domain": "retail",
    "max_steps": 40,
    "max_turns": 40,
    "solo_mode": False,
    "user_llm": None,
    "user_llm_args": None,
}

_REWARD_CACHE: dict[str, list[float]] = {}


def _cache_reward(task_id: str, reward: float) -> None:
    _REWARD_CACHE.setdefault(task_id, []).append(reward)


def _pop_cached_reward(task_id: str) -> float:
    rewards = _REWARD_CACHE.get(task_id)
    if not rewards:
        return 0.0
    return rewards.pop(0)


async def run_agent(base_url: str, prompt: list[dict[str, Any]] | str, request_kwargs: dict[str, Any]) -> None:
    task_id = str(prompt)

    env = AgentGymEnv(
        domain=TAU2_CONFIGS["domain"],
        task_id=task_id,
        max_steps=TAU2_CONFIGS["max_steps"],
        solo_mode=TAU2_CONFIGS["solo_mode"],
        user_llm=TAU2_CONFIGS["user_llm"],
        user_llm_args=TAU2_CONFIGS["user_llm_args"],
    )

    observation, info = env.reset()
    tools = [tool.openai_schema for tool in info["tools"]]
    messages: list[dict[str, Any]] = [{"role": "system", "content": info["policy"]}]
    if observation:
        messages.append({"role": "user", "content": observation})

    reward = 0.0
    for _ in range(TAU2_CONFIGS["max_turns"]):
        payload = {
            "model": "default",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "logprobs": True,
            **request_kwargs,
        }
        response = await post(base_url + "/v1/chat/completions", payload)
        choice = response["choices"][0]
        message = choice["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            arguments = json.loads(call["function"]["arguments"]) if call["function"].get("arguments") else {}
            action = json.dumps({"name": call["function"]["name"], "arguments": arguments})
        else:
            action = message.get("content") or ""

        observation, reward, terminated, truncated, _info = env.step(action)
        if observation:
            messages.append({"role": "user", "content": observation})
        if terminated or truncated:
            break

    _cache_reward(task_id, reward)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    return _pop_cached_reward(str(sample.prompt))
