from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from mcpeval.capture import CaptureMiddleware
from mcpeval.dataset import Case, EvalSuite, EvaluatorConfig
from mcpeval.evaluators.graph_match import GraphMatchEvaluator
from mcpeval.graph import ToolCallGraph
from mcpeval.mock_server import MockMCPServer


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    tool_calls_made: list[dict[str, Any]]
    tool_calls_expected: list[dict[str, Any]]
    graph_match_score: float
    llm_judge_score: float | None
    rule_score: float | None
    steps_taken: int
    terminated_cleanly: bool
    raw_output: str
    error: str | None = None


@dataclass
class RunResult:
    run_id: int | None
    eval_suite: str
    model: str
    total_cases: int
    passed: int
    failed: int
    overall_score: float
    case_results: list[CaseResult] = field(default_factory=list)


def _mcp_result_to_str(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, list):
        parts = []
        for item in result:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(result, "text"):
        return result.text
    return json.dumps(result) if isinstance(result, dict) else str(result)


class EvalRunner:
    def __init__(
        self,
        store=None,
        anthropic_api_key: str | None = None,
        max_agent_turns: int = 10,
    ) -> None:
        self._store = store
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_turns = max_agent_turns

    async def run_suite(self, suite: EvalSuite) -> RunResult:
        mock_server = MockMCPServer(suite.mcp_server, suite.mock_tools)
        case_results: list[CaseResult] = []

        async with mock_server.start() as (client, capture):
            for case in suite.cases:
                capture.reset()
                cr = await self.run_case(case, suite, capture, client)
                case_results.append(cr)

        passed = sum(1 for cr in case_results if cr.passed)
        # overall_score uses graph_match_score only; rule/llm_judge scores inform per-case passed but not this aggregate
        overall_score = (
            sum(cr.graph_match_score for cr in case_results) / len(case_results)
            if case_results
            else 0.0
        )

        result = RunResult(
            run_id=None,
            eval_suite=suite.name,
            model=suite.model,
            total_cases=len(case_results),
            passed=passed,
            failed=len(case_results) - passed,
            overall_score=overall_score,
            case_results=case_results,
        )

        if self._store is not None:
            run_id = self._store.save_run(result)
            result.run_id = run_id
            for cr in case_results:
                self._store.save_case_result(run_id, cr)

        return result

    async def run_case(
        self,
        case: Case,
        suite: EvalSuite,
        capture: CaptureMiddleware,
        client: Any,
    ) -> CaseResult:
        try:
            return await self._run_case_inner(case, suite, capture, client)
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                passed=False,
                tool_calls_made=[],
                tool_calls_expected=[
                    {"tool": s.tool, "params_contain": s.params_contain}
                    for s in case.expected_graph.steps
                ],
                graph_match_score=0.0,
                llm_judge_score=None,
                rule_score=None,
                steps_taken=0,
                terminated_cleanly=False,
                raw_output="",
                error=str(exc),
            )

    async def _run_case_inner(
        self,
        case: Case,
        suite: EvalSuite,
        capture: CaptureMiddleware,
        client: Any,
    ) -> CaseResult:
        anthropic = AsyncAnthropic(api_key=self._api_key)

        mcp_tools = await client.list_tools()
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description or f"Tool: {t.name}",
                "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {"type": "object"},
            }
            for t in mcp_tools
        ]

        messages: list[dict] = [{"role": "user", "content": case.input}]
        terminated_cleanly = False
        raw_output = ""

        for _ in range(self._max_turns):
            response = await anthropic.messages.create(
                model=suite.model,
                tools=anthropic_tools,
                messages=messages,
                max_tokens=4096,
            )

            if response.stop_reason == "end_turn":
                terminated_cleanly = True
                text_blocks = [
                    b for b in response.content if hasattr(b, "type") and b.type == "text"
                ]
                raw_output = text_blocks[-1].text if text_blocks else ""
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if not (hasattr(block, "type") and block.type == "tool_use"):
                        continue
                    try:
                        mcp_result = await client.call_tool(block.name, block.input or {})
                        result_text = _mcp_result_to_str(mcp_result)
                    except Exception as e:
                        result_text = f"Error: {e}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

        graph = ToolCallGraph.from_expected(case.expected_graph)
        eval_cfg = next(
            (e for e in case.evaluators if e.type == "graph_match"),
            EvaluatorConfig(type="graph_match", strict=False),
        )
        evaluator = GraphMatchEvaluator(strict=eval_cfg.strict)
        graph_result = evaluator.evaluate(capture.calls, graph, terminated_cleanly)

        llm_judge_score: float | None = None
        rule_score: float | None = None
        rule_passed: bool | None = None
        judge_passed: bool | None = None

        for ecfg in case.evaluators:
            if ecfg.type == "rule" and ecfg.checks is not None:
                from mcpeval.evaluators.rule import RuleEvaluator

                rule_eval = RuleEvaluator(
                    checks=ecfg.checks,
                    strict=ecfg.strict,
                    threshold=ecfg.threshold,
                )
                rule_result = rule_eval.evaluate(raw_output)
                rule_score = rule_result.score
                rule_passed = rule_result.passed
            elif ecfg.type == "llm_judge" and ecfg.criteria is not None:
                from mcpeval.evaluators.llm_judge import LLMJudgeEvaluator

                judge = LLMJudgeEvaluator(
                    criteria=ecfg.criteria,
                    model=ecfg.judge_model or suite.model,
                    api_key=self._api_key,
                    threshold=ecfg.threshold,
                )
                judge_result = await judge.evaluate(
                    raw_output=raw_output,
                    task_input=case.input,
                    tool_calls=capture.calls,
                )
                llm_judge_score = judge_result.score
                judge_passed = judge_result.passed

        passed = graph_result.passed
        if rule_passed is not None:
            passed = passed and rule_passed
        if judge_passed is not None:
            passed = passed and judge_passed

        return CaseResult(
            case_id=case.id,
            passed=passed,
            tool_calls_made=[
                {"tool_name": r.tool_name, "arguments": r.arguments} for r in capture.calls
            ],
            tool_calls_expected=[
                {"tool": s.tool, "params_contain": s.params_contain} for s in graph.steps
            ],
            graph_match_score=graph_result.score,
            llm_judge_score=llm_judge_score,
            rule_score=rule_score,
            steps_taken=graph_result.steps_taken,
            terminated_cleanly=terminated_cleanly,
            raw_output=raw_output,
        )
