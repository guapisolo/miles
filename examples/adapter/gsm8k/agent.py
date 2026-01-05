import logging
from typing import Any

import httpx
from fastapi import FastAPI
import uvicorn
from eval_protocol import InitRequest, FireworksTracingHttpHandler, RolloutIdFilter, Status

app = FastAPI()

# Configure Fireworks tracing handler globally
fireworks_handler = FireworksTracingHttpHandler()
logging.getLogger().addHandler(fireworks_handler)


def _normalize_base_url(base_url: str | None) -> str:
    if not base_url:
        raise ValueError("model_base_url is required for agent execution")
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _build_completion_payload(request: InitRequest) -> dict[str, Any]:
    completion_params = dict(request.completion_params or {})
    model = completion_params.pop("model", None) or "default"
    if "max_new_tokens" in completion_params and "max_tokens" not in completion_params:
        completion_params["max_tokens"] = completion_params.pop("max_new_tokens")

    payload = {
        "model": model,
        "messages": request.messages,
        **{k: v for k, v in completion_params.items() if v is not None},
    }
    if request.tools:
        payload["tools"] = request.tools
    return payload


def execute_agent(request: InitRequest) -> dict:
    """Minimal GSM8K agent flow: call the SGLang OpenAI-compatible endpoint."""
    base_url = _normalize_base_url(request.model_base_url)
    url = f"{base_url}/chat/completions"
    payload = _build_completion_payload(request)

    response = httpx.post(url, json=payload, timeout=300.0)
    response.raise_for_status()
    data = response.json()

    assistant_message = data["choices"][0]["message"]

    full_messages = list(request.messages or [])
    full_messages.append(assistant_message)
    return {
        "messages": full_messages,
        "assistant_message": assistant_message,
        "raw_response": data,
    }


@app.post("/init")
def init(request: InitRequest):
    # Create rollout-specific logger with filter
    rollout_logger = logging.getLogger(f"eval_server.{request.metadata.rollout_id}")
    rollout_logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))

    try:
        result = execute_agent(request)

        rollout_logger.info(
            f"Rollout {request.metadata.rollout_id} completed",
            extra={"status": Status.rollout_finished()}
        )

        return {"status": "success", "result": result}
    except Exception as exc:
        rollout_logger.error(
            f"Rollout {request.metadata.rollout_id} failed: {exc}",
            extra={"status": Status.rollout_error(str(exc))}
        )
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
