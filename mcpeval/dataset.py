from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class MockToolDef:
    name: str
    returns: dict[str, Any]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphStep:
    tool: str
    params_contain: dict[str, Any] | None = None
    must_follow: str | None = None
    optional: bool = False


@dataclass
class ExpectedGraph:
    steps: list[GraphStep]
    max_steps: int = 10
    must_terminate: bool = True


@dataclass
class EvaluatorConfig:
    type: str
    strict: bool = False
    checks: list[dict] | None = None
    criteria: str | None = None
    judge_model: str | None = None
    threshold: float = 0.7


@dataclass
class Case:
    id: str
    input: str
    expected_graph: ExpectedGraph
    evaluators: list[EvaluatorConfig] = field(default_factory=list)


@dataclass
class EvalSuite:
    name: str
    model: str
    mcp_server: str
    mock_tools: list[MockToolDef]
    cases: list[Case]


def _parse_graph_step(raw: dict) -> GraphStep:
    return GraphStep(
        tool=raw["tool"],
        params_contain=raw.get("params_contain"),
        must_follow=raw.get("must_follow"),
        optional=bool(raw.get("optional", False)),
    )


def _parse_expected_graph(raw: dict) -> ExpectedGraph:
    return ExpectedGraph(
        steps=[_parse_graph_step(s) for s in raw.get("steps", [])],
        max_steps=int(raw.get("max_steps", 10)),
        must_terminate=bool(raw.get("must_terminate", True)),
    )


def _parse_evaluator(raw: dict) -> EvaluatorConfig:
    return EvaluatorConfig(
        type=raw["type"],
        strict=bool(raw.get("strict", False)),
        checks=raw.get("checks"),
        criteria=raw.get("criteria"),
        judge_model=raw.get("judge_model"),
        threshold=float(raw.get("threshold", 0.7)),
    )


def _parse_case(raw: dict) -> Case:
    return Case(
        id=raw["id"],
        input=raw["input"],
        expected_graph=_parse_expected_graph(raw["expected_graph"]),
        evaluators=[_parse_evaluator(e) for e in raw.get("evaluators", [])],
    )


def _parse_mock_tools(raw: dict) -> list[MockToolDef]:
    return [
        MockToolDef(
            name=name,
            returns=cfg.get("returns", {}),
            description=cfg.get("description", ""),
            parameters=cfg.get("parameters", {}),
        )
        for name, cfg in raw.items()
    ]


def load_suite(path: str | Path) -> EvalSuite:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval suite not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvalSuite(
        name=data["name"],
        model=data["model"],
        mcp_server=data["mcp_server"],
        mock_tools=_parse_mock_tools(data.get("mock_tools", {})),
        cases=[_parse_case(c) for c in data.get("cases", [])],
    )
