from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcpeval.dataset import ExpectedGraph


@dataclass
class Step:
    tool: str
    params_contain: dict[str, Any] | None = None
    must_follow: str | None = None
    optional: bool = False


@dataclass
class ToolCallGraph:
    steps: list[Step]
    max_steps: int = 10
    must_terminate: bool = True

    @classmethod
    def from_expected(cls, eg: ExpectedGraph) -> ToolCallGraph:
        return cls(
            steps=[
                Step(
                    tool=s.tool,
                    params_contain=s.params_contain,
                    must_follow=s.must_follow,
                    optional=s.optional,
                )
                for s in eg.steps
            ],
            max_steps=eg.max_steps,
            must_terminate=eg.must_terminate,
        )

    @property
    def required_steps(self) -> list[Step]:
        return [s for s in self.steps if not s.optional]

    @property
    def step_names(self) -> list[str]:
        return [s.tool for s in self.steps]
