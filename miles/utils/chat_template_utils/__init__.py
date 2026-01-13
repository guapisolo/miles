from pathlib import Path

_QWEN3_TEMPLATE_PATH = Path(__file__).resolve().parent / "config" / "qwen3.jinja"


def load_qwen3_chat_template() -> str:
    return _QWEN3_TEMPLATE_PATH.read_text(encoding="utf-8")
