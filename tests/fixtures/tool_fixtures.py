import json

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_year",
            "description": "Get current year",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get temperature for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]

TOOL_EXECUTORS = {
    "get_year": lambda params: {"year": 2025},
    "get_temperature": lambda params: {"temperature": 25, "location": params.get("location", "unknown")},
}


def execute_tool_call(tool_call: dict) -> dict:
    name = tool_call["name"]
    params = json.loads(tool_call["parameters"]) if isinstance(tool_call["parameters"], str) else tool_call["parameters"]
    return TOOL_EXECUTORS[name](params)
