import sys
import types


def _ensure_package(name: str) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def install_sglang_stub() -> None:
    _ensure_package("sglang")
    _ensure_package("sglang.srt")
    _ensure_package("sglang.srt.endpoints")
    _ensure_package("sglang.srt.endpoints.openai")
    _ensure_package("sglang.srt.entrypoints")
    _ensure_package("sglang.srt.entrypoints.openai")

    # protocol_module = types.ModuleType("sglang.srt.endpoints.openai.protocol")

    class ChatCompletionMessageGenericParam:
        def __init__(self, role: str, content: str | None = None, **kwargs):
            self.role = role
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_copy(self, update: dict):
            data = self.__dict__.copy()
            data.update(update)
            return self.__class__(**data)

    class ChatCompletionMessageUserParam(ChatCompletionMessageGenericParam):
        pass

    # ChatCompletionMessageParam = Union[ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam]

    # protocol_module.ChatCompletionMessageGenericParam = ChatCompletionMessageGenericParam
    # protocol_module.ChatCompletionMessageUserParam = ChatCompletionMessageUserParam
    # protocol_module.ChatCompletionMessageParam = ChatCompletionMessageParam
    # sys.modules["sglang.srt.endpoints.openai.protocol"] = protocol_module
    # sys.modules["sglang.srt.entrypoints.openai.protocol"] = protocol_module
