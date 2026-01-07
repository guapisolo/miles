import logging

import uvicorn
from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status
from fastapi import FastAPI
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

from .utils.agent_helper import call_llm

app = FastAPI()

# Configure Fireworks tracing handler globally
fireworks_handler = FireworksTracingHttpHandler()
logging.getLogger().addHandler(fireworks_handler)


def execute_agent(request: InitRequest) -> ChatCompletionResponse:
    """Minimal GSM8K agent flow: call the SGLang OpenAI-compatible endpoint."""

    messages = [msg.dump_mdoel_for_chat_completion_request() for msg in (request.messages or [])]

    # ... agent logic ...

    response = call_llm(request, messages)
    return response


@app.post("/init")
def init(request: InitRequest):
    # Create rollout-specific logger with filter
    rollout_logger = logging.getLogger(f"eval_server.{request.metadata.rollout_id}")
    rollout_logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))

    try:
        result = execute_agent(request)

        rollout_logger.info(
            f"Rollout {request.metadata.rollout_id} completed", extra={"status": Status.rollout_finished()}
        )

        return {"status": "success", "result": result}
    except Exception as exc:
        rollout_logger.error(
            f"Rollout {request.metadata.rollout_id} failed: {exc}", extra={"status": Status.rollout_error(str(exc))}
        )
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
