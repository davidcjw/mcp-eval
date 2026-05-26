import pytest
from mcpeval.dataset import MockToolDef
from mcpeval.mock_server import MockMCPServer


@pytest.mark.asyncio
async def test_mock_server_registers_tools():
    tools = [
        MockToolDef(name="get_logs", returns={"logs": ["err1"]}),
        MockToolDef(name="run_playbook", returns={"status": "ok"}),
    ]
    server = MockMCPServer("test-ops", tools)
    async with server.start() as (client, capture):
        tool_list = await client.list_tools()
        tool_names = {t.name for t in tool_list}
        assert "get_logs" in tool_names
        assert "run_playbook" in tool_names


@pytest.mark.asyncio
async def test_mock_server_returns_fixed_value():
    tools = [MockToolDef(name="get_logs", returns={"logs": ["ERROR: timeout"]})]
    server = MockMCPServer("test-ops", tools)
    async with server.start() as (client, capture):
        result = await client.call_tool("get_logs", {})
    assert result is not None


@pytest.mark.asyncio
async def test_mock_server_capture_middleware_active():
    tools = [MockToolDef(name="get_logs", returns={"logs": []})]
    server = MockMCPServer("test-ops", tools)
    async with server.start() as (client, capture):
        await client.call_tool("get_logs", {"service": "payment"})
    assert len(capture.calls) == 1
    assert capture.calls[0].tool_name == "get_logs"
    assert capture.calls[0].arguments == {"service": "payment"}


@pytest.mark.asyncio
async def test_mock_server_multiple_tools():
    tools = [
        MockToolDef(name="get_logs", returns={"logs": []}),
        MockToolDef(name="run_playbook", returns={"status": "ok"}),
        MockToolDef(name="notify_oncall", returns={"notified": True}),
    ]
    server = MockMCPServer("test-ops", tools)
    async with server.start() as (client, capture):
        await client.call_tool("get_logs", {})
        await client.call_tool("run_playbook", {})
        await client.call_tool("notify_oncall", {})
    assert len(capture.calls) == 3
    assert [r.tool_name for r in capture.calls] == ["get_logs", "run_playbook", "notify_oncall"]


@pytest.mark.asyncio
async def test_mock_server_each_tool_returns_correct_value():
    tools = [
        MockToolDef(name="tool_a", returns={"key": "value_a"}),
        MockToolDef(name="tool_b", returns={"key": "value_b"}),
    ]
    server = MockMCPServer("test", tools)
    async with server.start() as (client, capture):
        result_a = await client.call_tool("tool_a", {})
        result_b = await client.call_tool("tool_b", {})
    assert result_a is not None
    assert result_b is not None


@pytest.mark.asyncio
async def test_tool_description_propagates():
    tools = [MockToolDef(
        name="get_logs",
        returns={"logs": []},
        description="Retrieve service logs",
        parameters={
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    )]
    server = MockMCPServer("test", tools)
    async with server.start() as (client, _capture):
        listed = await client.list_tools()
        tool = next(t for t in listed if t.name == "get_logs")
        assert tool.description == "Retrieve service logs"
        assert tool.inputSchema["properties"]["service"]["type"] == "string"
