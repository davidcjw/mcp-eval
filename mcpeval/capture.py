from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    timestamp: float


class CaptureMiddleware(Middleware):
    def __init__(self) -> None:
        self.calls: list[ToolCallRecord] = []

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        record = ToolCallRecord(
            tool_name=context.message.name,
            arguments=dict(context.message.arguments or {}),
            timestamp=time.monotonic(),
        )
        self.calls.append(record)
        return await call_next(context)

    def reset(self) -> None:
        self.calls.clear()
