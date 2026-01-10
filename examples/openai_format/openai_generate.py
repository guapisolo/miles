from miles.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def openai_generate(args, sample: Sample, sampling_params: dict):
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    messages = sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": sample.prompt}]

    payload = {
        "model": getattr(args, "hf_checkpoint", None) or "default",
        "messages": messages,
    }

    for key in ("temperature", "top_p", "top_k", "stop"):
        if key in sampling_params:
            payload[key] = sampling_params[key]
    if "max_new_tokens" in sampling_params:
        payload["max_tokens"] = sampling_params["max_new_tokens"]

    data = await post(base_url, payload, max_retries=3)

    choice = data["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")

    sample.response = content
    sample.response_length = len(content)
    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "stop":
        sample.status = Sample.Status.COMPLETED

    sample = await postprocess_sample_with_radix_tree(args, sample, content)

    return sample
