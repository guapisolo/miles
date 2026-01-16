from miles.utils.test_utils.mock_tools import execute_tool_call


class TestExecuteToolCall:
    def test_execute_get_year(self):
        result = execute_tool_call("get_year", {})
        assert result == {"year": 2026}

    def test_execute_get_temperature(self):
        result = execute_tool_call("get_temperature", {"location": "Mars"})
        assert result == {"temperature": -60}
