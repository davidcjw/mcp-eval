# mcpeval

A CI-embeddable eval harness for MCP-native agentic workflows, focused on **tool-call graph correctness** and regression tracking over time.

## Why

Existing eval tools (promptfoo, deepeval, RAGAS, LangSmith) score LLM output as text. In agentic workflows, the output **is the sequence of tool calls**. `mcpeval` is built around that insight.

| Tool | What it evaluates |
|------|-------------------|
| promptfoo | Generic prompt/text quality |
| RAGAS | RAG pipeline quality |
| LangSmith / Braintrust | Cloud eval platforms |
| **mcpeval** | **Tool-call graph correctness, MCP-native, self-hosted** |

## How it works

1. Define a YAML eval suite with mock tool responses and expected tool-call graphs
2. `mcpeval` spins up an in-process mock MCP server, runs your agent against it, and captures every tool call
3. A `GraphMatchEvaluator` scores the actual call sequence against your expectations
4. Results are stored in SQLite and printed with Rich — with regression detection across runs

## Install

```bash
pip install mcpeval
# or
uv add mcpeval
```

## Quick start

```yaml
# evals/my_suite.yaml
name: "Incident Response Agent"
model: claude-sonnet-4-20250514
mcp_server: "ops-mcp"
mock_tools:
  get_logs:
    returns:
      logs:
        - "ERROR payment-service: timeout at 14:32"
  run_playbook:
    returns:
      status: "ok"
      steps_completed: 3
cases:
  - id: "incident_001"
    input: "Payment service is throwing errors, investigate and resolve"
    expected_graph:
      steps:
        - tool: get_logs
          params_contain:
            service: "payment"
        - tool: run_playbook
          must_follow: get_logs
      max_steps: 6
      must_terminate: true
    evaluators:
      - type: graph_match
        strict: false
```

```python
import asyncio
from mcpeval import load_suite, EvalRunner, Reporter

async def main():
    suite = load_suite("evals/my_suite.yaml")
    result = await EvalRunner().run_suite(suite)
    Reporter().print_run_summary(result)

asyncio.run(main())
```

Output:

```
Eval Suite: Incident Response Agent
Model: claude-sonnet-4-20250514
Results: 1/1 passed  | Overall score: 1.00

 Case ID        Status    Score   Steps   Terminated
 ─────────────────────────────────────────────────────
 incident_001   PASS       1.00       3   ✓
```

## Core concepts

### MockMCPServer

Spin up a fake MCP server with scripted tool responses — no real infrastructure needed:

```python
from mcpeval import MockMCPServer
from mcpeval.dataset import MockToolDef

tools = [
    MockToolDef(name="get_logs", returns={"logs": ["ERROR: timeout"]}),
    MockToolDef(name="run_playbook", returns={"status": "ok"}),
]

async with MockMCPServer("ops-mcp", tools).start() as (client, capture):
    # run your agent here
    # capture.calls contains every tool call made
```

### ToolCallGraph

Define what tool calls you expect and in what order:

```python
from mcpeval import ToolCallGraph, Step

graph = ToolCallGraph([
    Step("get_logs", params_contain={"service": "payment"}),
    Step("run_playbook", must_follow="get_logs"),
    Step("notify_oncall", optional=True),
])
```

### GraphMatchEvaluator

Score actual calls against expected graph:

```python
from mcpeval.evaluators.graph_match import GraphMatchEvaluator

result = GraphMatchEvaluator(strict=False).evaluate(
    actual_calls=capture.calls,
    expected=graph,
    terminated_cleanly=True,
)
print(result.score)           # 0.0 – 1.0
print(result.missing_required)  # steps that weren't called
print(result.ordering_violations)  # must_follow violations
```

### Result storage

```python
from mcpeval import ResultStore, EvalRunner

store = ResultStore("results.db")
store.initialize()

runner = EvalRunner(store=store)
result = await runner.run_suite(suite)
# result.run_id is set — all runs queryable via store.list_runs()
```

## Evaluator types

| Type | What it checks | When to use |
|------|---------------|-------------|
| `graph_match` | Tool call sequence correctness | Always — core differentiator |
| `llm_judge` | Open-ended output quality *(Phase 2)* | When output is prose/reasoning |
| `rule` | Deterministic checks *(Phase 2)* | Redundant calls, step count, termination |

## CI integration

```yaml
# .github/workflows/eval.yml
- name: Run evals
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    uv run python -c "
    import asyncio
    from mcpeval import load_suite, EvalRunner
    async def main():
        suite = load_suite('evals/incident_response.yaml')
        result = await EvalRunner().run_suite(suite)
        if result.overall_score < 0.85:
            raise SystemExit(f'Score {result.overall_score:.2f} below threshold 0.85')
    asyncio.run(main())
    "
```

## Development

```bash
git clone https://github.com/davidcjw/mcpeval
cd mcpeval
uv sync
uv run pytest tests/ -m "not integration"     # unit tests, no API key needed
uv run pytest tests/ -m "integration"          # requires ANTHROPIC_API_KEY
```

## Project structure

```
mcpeval/
├── mcpeval/
│   ├── dataset.py          # YAML loader → typed dataclasses
│   ├── graph.py            # ToolCallGraph + Step
│   ├── capture.py          # FastMCP middleware: records tool calls
│   ├── mock_server.py      # In-process MCP server with stub tools
│   ├── runner.py           # EvalRunner: Anthropic agent loop
│   ├── store.py            # SQLite result persistence
│   ├── reporter.py         # Rich terminal output
│   └── evaluators/
│       └── graph_match.py  # Graph matching scorer
└── evals/
    └── examples/
        └── incident_response.yaml
```

## Roadmap

- [x] Phase 1 — Core engine: MockMCPServer, ToolCallGraph, GraphMatchEvaluator, SQLite, Rich reporter
- [ ] Phase 2 — Intelligence: LLM-as-judge evaluator, rule-based checks, regression detection
- [ ] Phase 3 — Polish: HTML report export, CI exit codes/thresholds, multi-model comparison
