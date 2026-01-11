from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree
from miles.utils.http_utils import post
from miles.utils.openai_utils import sampling_params_to_chat_request
from miles.utils.types import Sample


async def openai_generate(args, sample: Sample, sampling_params: dict):
    messages = sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": sample.prompt}]
    model = getattr(args, "hf_checkpoint", None) or "default"
    chat_request = sampling_params_to_chat_request(messages, model, sampling_params)

    data = await openai_rollout(args, chat_request)

    choice = data["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")

    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "stop":
        sample.status = Sample.Status.COMPLETED

    sample = await postprocess_sample_with_radix_tree(args, sample, content)

    return sample


async def openai_rollout(args, chat_request: ChatCompletionRequest) -> list:
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    payload = chat_request.model_dump(exclude_none=True)
    data = await post(base_url, payload, max_retries=3)
    return data
