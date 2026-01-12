import logging
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.sglang_rollout import GenerateState
from miles.router.middleware_hub.radix_tree_middleware import postprocess_sample_from_messages
from miles.utils.http_utils import post
from miles.utils.openai_utils import sampling_params_to_chat_request
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def openai_generate(args, sample: Sample, sampling_params: dict):
    assert (
        args.apply_chat_template is False
    ), "OpenAI format does not support apply_chat_template during data preparation"
    assert isinstance(sample.prompt, list), "OpenAI format only supports list of messages as prompt"
    messages = sample.prompt
    state = GenerateState(args)
    model = getattr(args, "hf_checkpoint", None) or "default"
    chat_request = sampling_params_to_chat_request(messages, model, sampling_params)

    messages = await openai_rollout(args, chat_request)
    sample.status = Sample.Status.COMPLETED
    sample = await postprocess_sample_from_messages(args, sample, messages, state.tokenizer)
    # logger.info(f"sample: {sample}")
    return sample


async def openai_rollout(args, chat_request: ChatCompletionRequest) -> list:
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    payload = chat_request.model_dump(exclude_none=True)
    data = await post(base_url, payload, max_retries=3)

    choice = data["choices"][0]
    assistant_msg = choice["message"]  # {"role": "assistant", "content": "..."}
    messages = payload["messages"] + [assistant_msg]

    return messages
