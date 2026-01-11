from typing import Any

import uvicorn
from eval_protocol import InitRequest
from examples.adapter.utils.agent_helper import post
from fastapi import FastAPI

app = FastAPI()


async def execute_agent(request: InitRequest) -> list[dict[str, Any]]:
    """Minimal GSM8K agent flow: call the SGLang OpenAI-compatible endpoint."""

    chat_completion_url = f"{request.model_base_url}/chat/completions"
    completion_params = (
        request.completion_params
    )  # These are default params for the request. You can modify them as you like.
    messages = [
        msg.dump_mdoel_for_chat_completion_request() for msg in (request.messages or [])
    ]  # fetch messages from the request
    payload = {
        "model": completion_params.model,
        "messages": messages,
        "tools": request.tools,
        **completion_params,
    }
    response = await post(chat_completion_url, payload)
    messages.append(response.choices[0].message.model_dump())
    return messages


@app.post("/init")
async def init(request: InitRequest):
    # Create rollout-specific logger with filter
    # rollout_logger = logging.getLogger(f"eval_server.{request.metadata.rollout_id}")
    # rollout_logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))

    try:
        result = await execute_agent(request)

        # rollout_logger.info(
        #     f"Rollout {request.metadata.rollout_id} completed", extra={"status": Status.rollout_finished()}
        # )

        return {"status": "success", "result": result}
    except Exception:
        # rollout_logger.error(
        #     f"Rollout {request.metadata.rollout_id} failed: {exc}", extra={"status": Status.rollout_error(str(exc))}
        # )
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
