# mcpeval

**Know when your AI agent stops doing the right thing.**

`mcpeval` tests MCP-native agents by checking the *sequence of tool calls they make* — not just what they say. Define the tool-call graph you expect, run your agent against a mock MCP server, and get a pass/fail score with regression tracking across runs.

One command to run, one exit code for CI:

```bash
mcpeval run evals/incident_response.yaml --threshold 0.85
```

## Why tool-call graph evaluation

Existing eval tools score LLM *text output*. In agentic workflows, the output **is the sequence of tool calls** — did the agent check logs before running the playbook? Did it avoid redundant calls? Did it terminate cleanly?

`mcpeval` is built around that insight: no cloud dependency, no prompt scoring, no RAG metrics — just structured verification that your agent takes the right actions in the right order.

## Install

```bash
pip install mcpeval
# or
uv add mcpeval
```

## Quick start

**1. Define your eval suite (YAML):**

```yaml
# evals/incident_response.yaml
name: "Incident Response Agent"
model: claude-sonnet-4-6
mcp_server: "ops-mcp"
mock_tools:
  get_logs:
    description: "Fetch recent logs for a service"
    parameters:
      type: object
      properties:
        service: { type: string }
    returns:
      logs:
        - "ERROR payment-service: timeout at 14:32"
  run_playbook:
    description: "Execute a remediation playbook"
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
      - type: rule
        checks:
          - contains: "resolved"
```

**2. Run it:**

```bash
mcpeval run evals/incident_response.yaml --threshold 0.85
```

```
Eval Suite: Incident Response Agent
Model:       claude-sonnet-4-6
Results:     1/1 passed  | Overall score: 1.00

 Case ID        Status    Score   Steps   Terminated
 ─────────────────────────────────────────────────────
 incident_001   PASS       1.00       3   ✓
```

Exit code `0` if score ≥ threshold, `1` if below — ready for CI.

## CLI reference

```
mcpeval run SUITE_FILE [OPTIONS]

Arguments:
  SUITE_FILE          Path to YAML eval suite

Options:
  --threshold FLOAT   Exit 1 if overall_score < threshold
  --models TEXT       Comma-separated model IDs or aliases (haiku/sonnet/opus)
                      Runs suite once per model and prints comparison table
  --output PATH       Write HTML report to this path
  --db PATH           SQLite DB path for storing results [default: mcpeval.db]
```

### CI gate

```yaml
# .github/workflows/eval.yml
- name: Run evals
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: mcpeval run evals/incident_response.yaml --threshold 0.85
```

### Compare models side by side

```bash
mcpeval run evals/incident_response.yaml --models haiku,sonnet --threshold 0.7
```

```
Multi-Model Comparison
Suite: Incident Response Agent

 Model                        Passed   Failed   Score
 ────────────────────────────────────────────────────
 claude-haiku-4-5-20251001         3        0    1.00
 claude-sonnet-4-6                 3        0    1.00
```

### HTML report

```bash
mcpeval run evals/incident_response.yaml --output report.html
# or with multi-model
mcpeval run evals/incident_response.yaml --models haiku,sonnet --output report.html
```

Generates a self-contained HTML file with per-case scores, rule/LLM judge results, error details, and a model comparison table.

## Evaluator types

| Type | What it checks | When to use |
|------|---------------|-------------|
| `graph_match` | Tool call sequence, ordering, required steps | Always — core differentiator |
| `rule` | Output contains/not_contains/regex patterns | Deterministic output checks |
| `llm_judge` | Open-ended output quality scored by a model | Prose reasoning, summaries |

### Rule evaluator

```yaml
evaluators:
  - type: rule
    checks:
      - contains: "resolved"
      - not_contains: "error"
      - regex: "ticket-\\d+"
    threshold: 0.8   # fraction of checks that must pass
```

### LLM-as-judge evaluator

```yaml
evaluators:
  - type: llm_judge
    criteria: "Did the agent correctly identify the root cause and provide a clear resolution?"
    judge_model: claude-haiku-4-5-20251001   # optional, defaults to suite model
    threshold: 0.7
```

## Regression detection

Results are stored in SQLite across runs. Compare a run against a baseline:

```python
from mcpeval import ResultStore

store = ResultStore("mcpeval.db")
report = store.get_regression_report(run_id=5, baseline_run_id=1)
# {
#   "overall_score_delta": -0.12,
#   "regressed": [{"case_id": "incident_002", "delta": -0.33}],
#   "improved": [],
#   "unchanged": ["incident_001"]
# }
```

## Python API

For programmatic use or custom pipelines:

```python
import asyncio
from mcpeval import load_suite, EvalRunner, ResultStore, Reporter

async def main():
    suite = load_suite("evals/incident_response.yaml")
    store = ResultStore("mcpeval.db")
    store.initialize()

    result = await EvalRunner(store=store).run_suite(suite)
    Reporter().print_run_summary(result)

asyncio.run(main())
```

## Core concepts

### MockMCPServer

Spin up a fake MCP server with scripted tool responses — no real infrastructure needed:

```python
from mcpeval import MockMCPServer, EvalRunner
from mcpeval.dataset import MockToolDef

tools = [
    MockToolDef(
        name="get_logs",
        description="Fetch logs for a service",
        parameters={"type": "object", "properties": {"service": {"type": "string"}}},
        returns={"logs": ["ERROR: timeout"]},
    ),
]

async with MockMCPServer("ops-mcp", tools).start() as (client, capture):
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
print(result.score)                # 0.0 – 1.0
print(result.missing_required)     # steps that weren't called
print(result.ordering_violations)  # must_follow violations
```

## Development

```bash
git clone https://github.com/davidcjw/mcp-eval
cd mcp-eval
uv sync
uv run pytest tests/ -m "not integration"    # unit tests, no API key needed
uv run pytest tests/ -m "integration"         # requires ANTHROPIC_API_KEY
```

## Project structure

```
mcpeval/
├── mcpeval/
│   ├── cli.py              # mcpeval run CLI entrypoint
│   ├── dataset.py          # YAML loader → typed dataclasses
│   ├── graph.py            # ToolCallGraph + Step
│   ├── capture.py          # FastMCP middleware: records tool calls
│   ├── mock_server.py      # In-process MCP server with stub tools
│   ├── runner.py           # EvalRunner: Anthropic agent loop
│   ├── store.py            # SQLite result persistence + regression detection
│   ├── reporter.py         # Rich terminal output + multi-model table
│   ├── html_reporter.py    # Self-contained HTML report generator
│   └── evaluators/
│       ├── graph_match.py  # Graph matching scorer
│       ├── rule.py         # Contains / not_contains / regex checks
│       └── llm_judge.py    # LLM-as-judge scorer
└── evals/
    └── examples/
        └── incident_response.yaml
```

## Roadmap

- [x] Phase 1 — Core engine: MockMCPServer, ToolCallGraph, GraphMatchEvaluator, SQLite, Rich reporter
- [x] Phase 2 — Intelligence: LLM-as-judge evaluator, rule-based checks, regression detection, richer tool schemas
- [x] Phase 3 — CLI: `mcpeval run` entrypoint, threshold exit codes, multi-model comparison, HTML reports
