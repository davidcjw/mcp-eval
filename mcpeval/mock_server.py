from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastmcp import FastMCP, Client
from fastmcp.tools.function_tool import FunctionTool
from fastmcp.tools.base import ToolResult
from pydantic import Field

from mcpeval.capture import CaptureMiddleware
from mcpeval.dataset import MockToolDef


class _StubTool(FunctionTool):
    """FunctionTool subclass that ignores arguments and returns a fixed value."""

    stub_returns: dict[str, Any] = Field(default_factory=dict)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return self.convert_result(self.stub_returns)


def _register_tools(server: FastMCP, tool_defs: list[MockToolDef]) -> None:
    for tdef in tool_defs:
        async def _placeholder() -> dict:
            return {}

        base = _StubTool.from_function(_placeholder, name=tdef.name)
        update: dict = {"stub_returns": tdef.returns}
        if tdef.description:
            update["description"] = tdef.description
        if tdef.parameters:
            update["parameters"] = tdef.parameters
        stub = base.model_copy(update=update)
        server.add_tool(stub)


class MockMCPServer:
    def __init__(self, name: str, tools: list[MockToolDef]) -> None:
        self._name = name
        self._tool_defs = tools

    @asynccontextmanager
    async def start(self) -> AsyncIterator[tuple[Client, CaptureMiddleware]]:
        capture = CaptureMiddleware()
        server = FastMCP(name=self._name, middleware=[capture])
        _register_tools(server, self._tool_defs)
        async with Client(server) as client:
            yield client, capture
