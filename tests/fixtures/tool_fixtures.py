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
