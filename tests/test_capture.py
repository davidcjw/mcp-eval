import pytest
import time
from fastmcp import FastMCP, Client
from mcpeval.capture import CaptureMiddleware, ToolCallRecord


@pytest.mark.asyncio
async def test_capture_records_tool_call():
    capture = CaptureMiddleware()
    server = FastMCP("test", middleware=[capture])

    @server.tool
    def hello(name: str) -> str:
        return f"hello {name}"

    async with Client(server) as client:
        await client.call_tool("hello", {"name": "world"})

    assert len(capture.calls) == 1
    assert capture.calls[0].tool_name == "hello"
    assert capture.calls[0].arguments == {"name": "world"}


@pytest.mark.asyncio
async def test_capture_records_multiple_calls():
    capture = CaptureMiddleware()
    server = FastMCP("test", middleware=[capture])

    @server.tool
    def tool_a() -> str:
        return "a"

    @server.tool
    def tool_b() -> str:
        return "b"

    async with Client(server) as client:
        await client.call_tool("tool_a", {})
        await client.call_tool("tool_b", {})

    assert len(capture.calls) == 2
    assert capture.calls[0].tool_name == "tool_a"
    assert capture.calls[1].tool_name == "tool_b"


@pytest.mark.asyncio
async def test_capture_reset():
    capture = CaptureMiddleware()
    server = FastMCP("test", middleware=[capture])

    @server.tool
    def ping() -> str:
        return "pong"

    async with Client(server) as client:
        await client.call_tool("ping", {})
        assert len(capture.calls) == 1
        capture.reset()
        assert capture.calls == []


@pytest.mark.asyncio
async def test_capture_records_empty_args():
    capture = CaptureMiddleware()
    server = FastMCP("test", middleware=[capture])

    @server.tool
    def no_args() -> str:
        return "ok"

    async with Client(server) as client:
        await client.call_tool("no_args", {})

    assert capture.calls[0].arguments == {}


@pytest.mark.asyncio
async def test_capture_preserves_tool_result():
    capture = CaptureMiddleware()
    server = FastMCP("test", middleware=[capture])

    @server.tool
    def add(a: int, b: int) -> int:
        return a + b

    async with Client(server) as client:
        result = await client.call_tool("add", {"a": 2, "b": 3})

    assert len(capture.calls) == 1
    assert result is not None


@pytest.mark.asyncio
async def test_capture_timestamp_monotonic():
    capture = CaptureMiddleware()
    server = FastMCP("test", middleware=[capture])

    @server.tool
    def tick() -> str:
        return "tick"

    async with Client(server) as client:
        await client.call_tool("tick", {})
        await client.call_tool("tick", {})

    assert capture.calls[0].timestamp <= capture.calls[1].timestamp
    assert capture.calls[0].timestamp > 0
