# mcpeval Phase 1 — Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core eval harness for MCP-native agentic workflows — tool-call graph evaluation with mock MCP servers, SQLite storage, and Rich terminal reporting.

**Architecture:** A FastMCP-based MockMCPServer captures tool calls via a server-side middleware (`CaptureMiddleware`). The `EvalRunner` drives an Anthropic agent loop, collects `ToolCallRecord`s, then scores them against an `ExpectedGraph` using `GraphMatchEvaluator`. Results are persisted to SQLite and displayed via Rich.

**Tech Stack:** Python ≥3.10, FastMCP ≥2.14.7, Anthropic SDK ≥0.40.0, PyYAML, Rich, SQLite (stdlib), pytest + pytest-asyncio

---

## Module Dependency Graph (build order)

```
dataset.py → graph.py → capture.py → mock_server.py → graph_match.py → store.py → runner.py → reporter.py
```

No circular imports.

---

## Interface Contracts (critical — all modules must match)

### `dataset.py` exports
```python
@dataclass class MockToolDef:
    name: str
    returns: dict[str, Any]

@dataclass class GraphStep:
    tool: str
    params_contain: dict[str, Any] | None = None
    must_follow: str | None = None
    optional: bool = False

@dataclass class ExpectedGraph:
    steps: list[GraphStep]
    max_steps: int = 10
    must_terminate: bool = True

@dataclass class EvaluatorConfig:
    type: str
    strict: bool = False

@dataclass class Case:
    id: str
    input: str
    expected_graph: ExpectedGraph
    evaluators: list[EvaluatorConfig]

@dataclass class EvalSuite:
    name: str
    model: str
    mcp_server: str
    mock_tools: list[MockToolDef]
    cases: list[Case]

def load_suite(path: str | Path) -> EvalSuite: ...
```

### `graph.py` exports
```python
@dataclass class Step:
    tool: str
    params_contain: dict[str, Any] | None = None
    must_follow: str | None = None
    optional: bool = False

@dataclass class ToolCallGraph:
    steps: list[Step]
    max_steps: int = 10
    must_terminate: bool = True

    @classmethod
    def from_expected(cls, eg: ExpectedGraph) -> "ToolCallGraph": ...
    @property
    def required_steps(self) -> list[Step]: ...
    @property
    def step_names(self) -> list[str]: ...
```

### `capture.py` exports
```python
@dataclass class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    timestamp: float  # time.monotonic()

class CaptureMiddleware(Middleware):
    calls: list[ToolCallRecord]
    async def on_call_tool(self, context, call_next) -> ToolResult: ...
    def reset(self) -> None: ...
```

### `mock_server.py` exports
```python
class MockMCPServer:
    def __init__(self, name: str, tools: list[MockToolDef]) -> None: ...

    @asynccontextmanager
    async def start(self) -> AsyncIterator[tuple[Client, CaptureMiddleware]]: ...
```

### `evaluators/graph_match.py` exports
```python
@dataclass class GraphMatchResult:
    score: float  # 0.0 – 1.0
    matched_steps: list[str]
    missing_required: list[str]
    ordering_violations: list[str]
    params_mismatches: list[str]
    steps_taken: int
    terminated_cleanly: bool
    passed: bool

class GraphMatchEvaluator:
    def __init__(self, strict: bool = False) -> None: ...
    def evaluate(
        self,
        actual_calls: list[ToolCallRecord],
        expected: ToolCallGraph,
        terminated_cleanly: bool,
    ) -> GraphMatchResult: ...
```

Scoring algorithm:
1. `required_steps` = non-optional steps
2. For each required step: find first matching call (tool name + params_contain subset check)
3. Check `must_follow` ordering via index in actual_calls
4. `base_score = matched_count / max(len(required_steps), 1)`
5. Penalties: ordering_violation = -0.1 each; `must_terminate + not terminated_cleanly` = -0.2; strict + steps > max_steps = -(excess/max_steps)*0.1
6. `score = max(0.0, min(1.0, base_score - penalties))`
7. `passed = score >= 1.0 if strict else score >= 0.7`

### `runner.py` exports
```python
@dataclass class CaseResult:
    case_id: str
    passed: bool
    tool_calls_made: list[dict]    # serialized ToolCallRecord
    tool_calls_expected: list[dict]  # serialized Step
    graph_match_score: float
    llm_judge_score: float | None
    steps_taken: int
    terminated_cleanly: bool
    raw_output: str

@dataclass class RunResult:
    run_id: int | None
    eval_suite: str
    model: str
    total_cases: int
    passed: int
    failed: int
    overall_score: float
    case_results: list[CaseResult]

class EvalRunner:
    def __init__(self, store=None, anthropic_api_key=None, max_agent_turns=10): ...
    async def run_suite(self, suite: EvalSuite) -> RunResult: ...
    async def run_case(self, case, suite, capture, client) -> CaseResult: ...
```

Agent loop in `run_case`:
- Convert FastMCP `list_tools()` results to Anthropic tool specs (name, description, input_schema)
- Loop up to `max_agent_turns` calling `anthropic.messages.create()`
- On `tool_use` blocks: call `client.call_tool(name, input)` → append tool_result to messages
- On `end_turn`: `terminated_cleanly = True`, break
- Capture happens server-side via `CaptureMiddleware` — no wrapping needed
- Convert `mcp.types.TextContent` from tool result to string for Anthropic's `tool_result` content

### `store.py` exports
```python
class ResultStore:
    def __init__(self, db_path: str | Path = "mcpeval.db") -> None: ...
    def initialize(self) -> None: ...  # CREATE TABLE IF NOT EXISTS, idempotent
    def save_run(self, result: RunResult) -> int: ...  # returns run_id
    def save_case_result(self, run_id: int, cr: CaseResult) -> int: ...
    def get_run(self, run_id: int) -> dict | None: ...
    def get_case_results(self, run_id: int) -> list[dict]: ...
    def list_runs(self) -> list[dict]: ...  # ordered by run_at DESC
```

SQLite schema:
```sql
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    eval_suite TEXT,
    model TEXT,
    total_cases INTEGER,
    passed INTEGER,
    failed INTEGER,
    overall_score REAL
);
CREATE TABLE IF NOT EXISTS case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    case_id TEXT,
    passed BOOLEAN,
    tool_calls_made TEXT,
    tool_calls_expected TEXT,
    graph_match_score REAL,
    llm_judge_score REAL,
    steps_taken INTEGER,
    terminated_cleanly BOOLEAN,
    raw_output TEXT
);
```

### `reporter.py` exports
```python
class Reporter:
    def __init__(self, console: Console | None = None) -> None: ...
    def print_run_summary(self, result: RunResult) -> None: ...
    def print_case_detail(self, cr: CaseResult) -> None: ...
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `mcpeval/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "mcpeval"
version = "0.1.0"
description = "CI-embeddable MCP tool-call eval harness"
requires-python = ">=3.10"
dependencies = [
    "fastmcp>=2.14.7",
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "rich>=13.0",
    "anyio>=4.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: marks tests requiring ANTHROPIC_API_KEY (deselect with '-m not integration')",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["mcpeval"]
```

- [ ] **Step 2: Create `mcpeval/__init__.py`** (empty, marks package)

- [ ] **Step 3: Create `tests/__init__.py`** (empty)

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import pytest
from pathlib import Path
from mcpeval.dataset import (
    EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig
)
from mcpeval.runner import RunResult, CaseResult


@pytest.fixture
def sample_mock_tools() -> list[MockToolDef]:
    return [
        MockToolDef(name="get_logs", returns={"logs": ["ERROR: timeout"]}),
        MockToolDef(name="run_playbook", returns={"status": "ok", "steps_completed": 3}),
        MockToolDef(name="notify_oncall", returns={"notified": True}),
    ]


@pytest.fixture
def sample_suite(sample_mock_tools) -> EvalSuite:
    return EvalSuite(
        name="Test Suite",
        model="claude-sonnet-4-20250514",
        mcp_server="test-mcp",
        mock_tools=sample_mock_tools,
        cases=[
            Case(
                id="test_001",
                input="Check logs and run playbook",
                expected_graph=ExpectedGraph(
                    steps=[
                        GraphStep(tool="get_logs", params_contain={"service": "payment"}),
                        GraphStep(tool="run_playbook", must_follow="get_logs"),
                        GraphStep(tool="notify_oncall", optional=True),
                    ],
                    max_steps=6,
                    must_terminate=True,
                ),
                evaluators=[EvaluatorConfig(type="graph_match", strict=False)],
            )
        ],
    )


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def passing_case_result() -> CaseResult:
    return CaseResult(
        case_id="test_001",
        passed=True,
        tool_calls_made=[{"tool_name": "get_logs", "arguments": {"service": "payment"}}],
        tool_calls_expected=[{"tool": "get_logs", "params_contain": {"service": "payment"}}],
        graph_match_score=1.0,
        llm_judge_score=None,
        steps_taken=2,
        terminated_cleanly=True,
        raw_output="Playbook executed successfully.",
    )


@pytest.fixture
def failing_case_result() -> CaseResult:
    return CaseResult(
        case_id="test_002",
        passed=False,
        tool_calls_made=[],
        tool_calls_expected=[{"tool": "get_logs"}],
        graph_match_score=0.0,
        llm_judge_score=None,
        steps_taken=0,
        terminated_cleanly=False,
        raw_output="",
    )


@pytest.fixture
def sample_run_result(passing_case_result, failing_case_result) -> RunResult:
    return RunResult(
        run_id=None,
        eval_suite="Test Suite",
        model="claude-sonnet-4-20250514",
        total_cases=2,
        passed=1,
        failed=1,
        overall_score=0.5,
        case_results=[passing_case_result, failing_case_result],
    )
```

- [ ] **Step 5: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
*.db
.env
.DS_Store
eval_report.html
```

- [ ] **Step 6: Install dependencies**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv sync
```

Expected: dependencies install without errors.

- [ ] **Step 7: Verify pytest is importable**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest --collect-only
```

Expected: "no tests ran" or empty collection — no errors.

- [ ] **Step 8: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add .
git commit -m "feat: scaffold mcpeval project with pyproject.toml and test fixtures"
```

---

## Task 2: `dataset.py` — YAML Loader

**Files:**
- Create: `mcpeval/dataset.py`
- Create: `tests/test_dataset.py`
- Create: `tests/fixtures/minimal_suite.yaml` (test fixture)

- [ ] **Step 1: Write failing tests in `tests/test_dataset.py`**

```python
import pytest
from pathlib import Path
import yaml
from mcpeval.dataset import load_suite, EvalSuite, MockToolDef, GraphStep, EvaluatorConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_suite_minimal():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert isinstance(suite, EvalSuite)
    assert suite.name == "Minimal Test Suite"
    assert suite.model == "claude-sonnet-4-20250514"
    assert suite.mcp_server == "test-mcp"
    assert len(suite.cases) == 1


def test_load_suite_mock_tools():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert len(suite.mock_tools) == 2
    get_logs = next(t for t in suite.mock_tools if t.name == "get_logs")
    assert "logs" in get_logs.returns
    assert isinstance(get_logs.returns["logs"], list)


def test_load_suite_graph_steps():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    steps = suite.cases[0].expected_graph.steps
    assert len(steps) >= 1
    assert steps[0].tool == "get_logs"


def test_load_suite_step_params_contain():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    step = suite.cases[0].expected_graph.steps[0]
    assert step.params_contain == {"service": "payment"}


def test_load_suite_optional_step():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    steps = suite.cases[0].expected_graph.steps
    optional_step = next((s for s in steps if s.optional), None)
    assert optional_step is not None
    assert optional_step.tool == "notify_oncall"


def test_load_suite_must_follow():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    steps = suite.cases[0].expected_graph.steps
    run_playbook = next((s for s in steps if s.tool == "run_playbook"), None)
    assert run_playbook is not None
    assert run_playbook.must_follow == "get_logs"


def test_load_suite_max_steps():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert suite.cases[0].expected_graph.max_steps == 6


def test_load_suite_must_terminate():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert suite.cases[0].expected_graph.must_terminate is True


def test_load_suite_evaluators():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    evals = suite.cases[0].evaluators
    assert len(evals) >= 1
    assert evals[0].type == "graph_match"
    assert evals[0].strict is False


def test_load_suite_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_suite("/nonexistent/path/suite.yaml")


def test_load_suite_invalid_yaml(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("name: [unclosed")
    with pytest.raises(Exception):
        load_suite(bad_file)
```

- [ ] **Step 2: Create `tests/fixtures/minimal_suite.yaml`**

```yaml
name: "Minimal Test Suite"
model: claude-sonnet-4-20250514
mcp_server: "test-mcp"
mock_tools:
  get_logs:
    returns:
      logs:
        - "ERROR payment-service: timeout at 14:32"
        - "ERROR payment-service: timeout at 14:33"
  run_playbook:
    returns:
      status: "ok"
      steps_completed: 3
cases:
  - id: "test_001"
    input: "Check payment service logs and run playbook"
    expected_graph:
      steps:
        - tool: get_logs
          params_contain:
            service: "payment"
        - tool: run_playbook
          must_follow: get_logs
        - tool: notify_oncall
          optional: true
      max_steps: 6
      must_terminate: true
    evaluators:
      - type: graph_match
        strict: false
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_dataset.py -v
```

Expected: ImportError or AttributeError — `dataset.py` doesn't exist yet.

- [ ] **Step 4: Implement `mcpeval/dataset.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class MockToolDef:
    name: str
    returns: dict[str, Any]


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
        MockToolDef(name=name, returns=cfg.get("returns", {}))
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
```

- [ ] **Step 5: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_dataset.py -v
```

Expected: 11/11 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/dataset.py tests/test_dataset.py tests/fixtures/minimal_suite.yaml tests/conftest.py
git commit -m "feat: implement YAML dataset loader with EvalSuite dataclasses"
```

---

## Task 3: `graph.py` — ToolCallGraph

**Files:**
- Create: `mcpeval/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing tests in `tests/test_graph.py`**

```python
import pytest
from mcpeval.dataset import ExpectedGraph, GraphStep as DatasetStep
from mcpeval.graph import Step, ToolCallGraph


def test_step_defaults():
    s = Step(tool="get_logs")
    assert s.params_contain is None
    assert s.optional is False
    assert s.must_follow is None


def test_step_custom_fields():
    s = Step(tool="run_playbook", must_follow="get_logs", optional=False)
    assert s.must_follow == "get_logs"
    assert s.optional is False


def test_toolcallgraph_from_expected():
    eg = ExpectedGraph(
        steps=[
            DatasetStep(tool="get_logs", params_contain={"service": "payment"}),
            DatasetStep(tool="run_playbook", must_follow="get_logs"),
            DatasetStep(tool="notify_oncall", optional=True),
        ],
        max_steps=6,
        must_terminate=True,
    )
    graph = ToolCallGraph.from_expected(eg)
    assert len(graph.steps) == 3
    assert graph.max_steps == 6
    assert graph.must_terminate is True
    assert graph.steps[0].tool == "get_logs"
    assert graph.steps[0].params_contain == {"service": "payment"}
    assert graph.steps[1].must_follow == "get_logs"
    assert graph.steps[2].optional is True


def test_toolcallgraph_required_steps():
    graph = ToolCallGraph(steps=[
        Step(tool="get_logs"),
        Step(tool="run_playbook"),
        Step(tool="notify_oncall", optional=True),
    ])
    required = graph.required_steps
    assert len(required) == 2
    assert all(not s.optional for s in required)


def test_toolcallgraph_step_names():
    graph = ToolCallGraph(steps=[
        Step(tool="get_logs"),
        Step(tool="run_playbook"),
        Step(tool="notify_oncall"),
    ])
    assert graph.step_names == ["get_logs", "run_playbook", "notify_oncall"]


def test_toolcallgraph_defaults():
    graph = ToolCallGraph(steps=[])
    assert graph.max_steps == 10
    assert graph.must_terminate is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_graph.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `mcpeval/graph.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

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
    def from_expected(cls, eg: "ExpectedGraph") -> "ToolCallGraph":
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
```

- [ ] **Step 4: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_graph.py -v
```

Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/graph.py tests/test_graph.py
git commit -m "feat: implement ToolCallGraph dataclasses with from_expected factory"
```

---

## Task 4: `capture.py` — CaptureMiddleware

**Files:**
- Create: `mcpeval/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write failing tests in `tests/test_capture.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_capture.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `mcpeval/capture.py`**

```python
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastmcp.server.middleware import Middleware, MiddlewareContext


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    timestamp: float


class CaptureMiddleware(Middleware):
    def __init__(self) -> None:
        self.calls: list[ToolCallRecord] = []

    async def on_call_tool(self, context: MiddlewareContext, call_next: Callable):
        record = ToolCallRecord(
            tool_name=context.message.name,
            arguments=dict(context.message.arguments or {}),
            timestamp=time.monotonic(),
        )
        self.calls.append(record)
        return await call_next(context)

    def reset(self) -> None:
        self.calls.clear()
```

- [ ] **Step 4: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_capture.py -v
```

Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/capture.py tests/test_capture.py
git commit -m "feat: implement CaptureMiddleware for server-side tool call interception"
```

---

## Task 5: `mock_server.py` — MockMCPServer

**Files:**
- Create: `mcpeval/mock_server.py`
- Create: `tests/test_mock_server.py`

- [ ] **Step 1: Write failing tests in `tests/test_mock_server.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_mock_server.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `mcpeval/mock_server.py`**

```python
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastmcp import FastMCP, Client

from mcpeval.capture import CaptureMiddleware
from mcpeval.dataset import MockToolDef


def _register_tools(server: FastMCP, tool_defs: list[MockToolDef]) -> None:
    for tdef in tool_defs:
        rv = tdef.returns

        def _make_stub(returns_value: dict) -> Any:
            async def stub(**kwargs: Any) -> dict:
                return returns_value
            stub.__name__ = tdef.name
            return stub

        server.tool(name=tdef.name)(_make_stub(rv))


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
```

- [ ] **Step 4: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_mock_server.py -v
```

Expected: 5/5 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/mock_server.py tests/test_mock_server.py
git commit -m "feat: implement MockMCPServer with FastMCP in-process transport and capture"
```

---

## Task 6: `evaluators/graph_match.py` — Scoring

**Files:**
- Create: `mcpeval/evaluators/__init__.py`
- Create: `mcpeval/evaluators/graph_match.py`
- Create: `tests/test_evaluators.py`

- [ ] **Step 1: Write failing tests in `tests/test_evaluators.py`**

```python
import pytest
from mcpeval.capture import ToolCallRecord
from mcpeval.graph import Step, ToolCallGraph
from mcpeval.evaluators.graph_match import GraphMatchEvaluator, GraphMatchResult


def _record(tool_name: str, arguments: dict | None = None, ts: float = 0.0) -> ToolCallRecord:
    return ToolCallRecord(tool_name=tool_name, arguments=arguments or {}, timestamp=ts)


def test_perfect_match():
    graph = ToolCallGraph(steps=[Step("get_logs"), Step("run_playbook")])
    calls = [_record("get_logs"), _record("run_playbook")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.passed is True
    assert result.missing_required == []


def test_missing_required_step():
    graph = ToolCallGraph(steps=[Step("get_logs"), Step("run_playbook")])
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score < 1.0
    assert "run_playbook" in result.missing_required


def test_optional_step_absent_still_passes():
    graph = ToolCallGraph(steps=[
        Step("get_logs"),
        Step("notify_oncall", optional=True),
    ])
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.passed is True


def test_params_contain_match():
    graph = ToolCallGraph(steps=[Step("get_logs", params_contain={"service": "payment"})])
    calls = [_record("get_logs", {"service": "payment", "extra": "ignored"})]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.missing_required == []


def test_params_contain_mismatch():
    graph = ToolCallGraph(steps=[Step("get_logs", params_contain={"service": "payment"})])
    calls = [_record("get_logs", {"service": "database"})]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score < 1.0
    assert "get_logs" in result.missing_required


def test_must_follow_respected():
    graph = ToolCallGraph(steps=[
        Step("get_logs"),
        Step("run_playbook", must_follow="get_logs"),
    ])
    calls = [_record("get_logs", ts=0.0), _record("run_playbook", ts=1.0)]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.ordering_violations == []


def test_must_follow_violated():
    graph = ToolCallGraph(steps=[
        Step("get_logs"),
        Step("run_playbook", must_follow="get_logs"),
    ])
    # run_playbook appears before get_logs
    calls = [_record("run_playbook", ts=0.0), _record("get_logs", ts=1.0)]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert len(result.ordering_violations) > 0
    assert result.score < 1.0


def test_max_steps_exceeded_strict():
    graph = ToolCallGraph(steps=[Step("get_logs")], max_steps=1)
    calls = [_record("get_logs"), _record("extra_call"), _record("another_call")]
    evaluator = GraphMatchEvaluator(strict=True)
    result = evaluator.evaluate(calls, graph, terminated_cleanly=True)
    assert result.steps_taken == 3
    assert result.score < 1.0


def test_terminate_penalty():
    graph = ToolCallGraph(steps=[Step("get_logs")], must_terminate=True)
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=False)
    assert result.score < 1.0


def test_score_never_below_zero():
    graph = ToolCallGraph(steps=[
        Step("a"), Step("b"), Step("c"), Step("d"),
    ], must_terminate=True)
    calls = []
    result = GraphMatchEvaluator(strict=True).evaluate(calls, graph, terminated_cleanly=False)
    assert result.score >= 0.0


def test_score_never_above_one():
    graph = ToolCallGraph(steps=[Step("get_logs")])
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score <= 1.0


def test_all_optional_no_calls():
    graph = ToolCallGraph(steps=[
        Step("a", optional=True),
        Step("b", optional=True),
    ])
    calls = []
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0


def test_non_strict_passes_at_0_7():
    # 1 of 2 required steps present → base_score = 0.5, no penalties → 0.5 < 0.7 → failed
    graph = ToolCallGraph(steps=[Step("a"), Step("b")])
    calls = [_record("a")]
    result = GraphMatchEvaluator(strict=False).evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == pytest.approx(0.5)
    assert result.passed is False


def test_strict_requires_score_1():
    graph = ToolCallGraph(steps=[Step("a"), Step("b")])
    calls = [_record("a")]
    result = GraphMatchEvaluator(strict=True).evaluate(calls, graph, terminated_cleanly=True)
    assert result.passed is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_evaluators.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `mcpeval/evaluators/__init__.py`** (empty)

- [ ] **Step 4: Implement `mcpeval/evaluators/graph_match.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from mcpeval.capture import ToolCallRecord
from mcpeval.graph import Step, ToolCallGraph


@dataclass
class GraphMatchResult:
    score: float
    matched_steps: list[str]
    missing_required: list[str]
    ordering_violations: list[str]
    params_mismatches: list[str]
    steps_taken: int
    terminated_cleanly: bool
    passed: bool


def _params_match(actual_args: dict[str, Any], params_contain: dict[str, Any]) -> bool:
    return all(actual_args.get(k) == v for k, v in params_contain.items())


def _find_matching_call_index(
    calls: list[ToolCallRecord],
    step: Step,
) -> int | None:
    for i, call in enumerate(calls):
        if call.tool_name != step.tool:
            continue
        if step.params_contain and not _params_match(call.arguments, step.params_contain):
            continue
        return i
    return None


class GraphMatchEvaluator:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def evaluate(
        self,
        actual_calls: list[ToolCallRecord],
        expected: ToolCallGraph,
        terminated_cleanly: bool,
    ) -> GraphMatchResult:
        matched_steps: list[str] = []
        missing_required: list[str] = []
        ordering_violations: list[str] = []
        params_mismatches: list[str] = []

        required_steps = expected.required_steps
        # Track last matched index for must_follow checking
        last_match_index: dict[str, int] = {}

        for step in required_steps:
            idx = _find_matching_call_index(actual_calls, step)
            if idx is None:
                missing_required.append(step.tool)
                continue

            matched_steps.append(step.tool)

            if step.must_follow:
                predecessor_idx = last_match_index.get(step.must_follow)
                if predecessor_idx is None or predecessor_idx >= idx:
                    ordering_violations.append(step.tool)

            last_match_index[step.tool] = idx

        total_required = max(len(required_steps), 1)
        matched_count = len(matched_steps) - len(ordering_violations)
        base_score = matched_count / total_required

        penalty = 0.0
        penalty += len(ordering_violations) * 0.1

        if expected.must_terminate and not terminated_cleanly:
            penalty += 0.2

        steps_taken = len(actual_calls)
        if self.strict and steps_taken > expected.max_steps:
            excess = steps_taken - expected.max_steps
            penalty += (excess / max(expected.max_steps, 1)) * 0.1

        score = max(0.0, min(1.0, base_score - penalty))
        pass_threshold = 1.0 if self.strict else 0.7
        passed = score >= pass_threshold

        return GraphMatchResult(
            score=score,
            matched_steps=matched_steps,
            missing_required=missing_required,
            ordering_violations=ordering_violations,
            params_mismatches=params_mismatches,
            steps_taken=steps_taken,
            terminated_cleanly=terminated_cleanly,
            passed=passed,
        )
```

- [ ] **Step 5: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_evaluators.py -v
```

Expected: 14/14 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/evaluators/ tests/test_evaluators.py
git commit -m "feat: implement GraphMatchEvaluator with ordering, params_contain, and penalty scoring"
```

---

## Task 7: `store.py` — SQLite Persistence

**Files:**
- Create: `mcpeval/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing tests in `tests/test_store.py`**

```python
import json
import sqlite3
import pytest
from pathlib import Path
from mcpeval.store import ResultStore
from mcpeval.runner import RunResult, CaseResult


def _make_run_result(score: float = 0.9, cases: list[CaseResult] | None = None) -> RunResult:
    if cases is None:
        cases = [CaseResult(
            case_id="c001",
            passed=True,
            tool_calls_made=[{"tool_name": "get_logs", "arguments": {}}],
            tool_calls_expected=[{"tool": "get_logs"}],
            graph_match_score=score,
            llm_judge_score=None,
            steps_taken=2,
            terminated_cleanly=True,
            raw_output="Done.",
        )]
    return RunResult(
        run_id=None,
        eval_suite="My Suite",
        model="claude-sonnet-4-20250514",
        total_cases=len(cases),
        passed=sum(1 for c in cases if c.passed),
        failed=sum(1 for c in cases if not c.passed),
        overall_score=score,
        case_results=cases,
    )


def test_initialize_creates_tables(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    conn = sqlite3.connect(tmp_db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "runs" in tables
    assert "case_results" in tables


def test_initialize_idempotent(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    store.initialize()  # should not raise


def test_save_run_returns_id(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    run_id = store.save_run(_make_run_result())
    assert isinstance(run_id, int)
    assert run_id >= 1


def test_save_run_persists_fields(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    run_id = store.save_run(_make_run_result(score=0.85))
    row = store.get_run(run_id)
    assert row is not None
    assert row["eval_suite"] == "My Suite"
    assert row["model"] == "claude-sonnet-4-20250514"
    assert abs(row["overall_score"] - 0.85) < 0.001
    assert row["total_cases"] == 1
    assert row["passed"] == 1
    assert row["failed"] == 0


def test_save_case_result(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    result = _make_run_result()
    run_id = store.save_run(result)
    row_id = store.save_case_result(run_id, result.case_results[0])
    assert isinstance(row_id, int)
    rows = store.get_case_results(run_id)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c001"
    assert rows[0]["passed"] == 1


def test_tool_calls_made_is_valid_json(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    result = _make_run_result()
    run_id = store.save_run(result)
    store.save_case_result(run_id, result.case_results[0])
    rows = store.get_case_results(run_id)
    calls = json.loads(rows[0]["tool_calls_made"])
    assert isinstance(calls, list)
    assert calls[0]["tool_name"] == "get_logs"


def test_list_runs_ordered_desc(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    store.save_run(_make_run_result(score=0.5))
    store.save_run(_make_run_result(score=0.8))
    runs = store.list_runs()
    assert len(runs) == 2
    assert runs[0]["overall_score"] == pytest.approx(0.8)


def test_get_run_returns_none_for_missing(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    assert store.get_run(999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_store.py -v
```

Expected: ImportError (runner and store don't exist yet — we need to create runner stubs first).

- [ ] **Step 3: Create `mcpeval/runner.py` stub** (just the dataclasses needed by store)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    tool_calls_made: list[dict]
    tool_calls_expected: list[dict]
    graph_match_score: float
    llm_judge_score: float | None
    steps_taken: int
    terminated_cleanly: bool
    raw_output: str


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
```

The full `EvalRunner` class will be added in Task 9.

- [ ] **Step 4: Implement `mcpeval/store.py`**

```python
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

from mcpeval.runner import CaseResult, RunResult

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    eval_suite TEXT,
    model TEXT,
    total_cases INTEGER,
    passed INTEGER,
    failed INTEGER,
    overall_score REAL
)
"""

_CREATE_CASE_RESULTS = """
CREATE TABLE IF NOT EXISTS case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    case_id TEXT,
    passed BOOLEAN,
    tool_calls_made TEXT,
    tool_calls_expected TEXT,
    graph_match_score REAL,
    llm_judge_score REAL,
    steps_taken INTEGER,
    terminated_cleanly BOOLEAN,
    raw_output TEXT
)
"""


class ResultStore:
    def __init__(self, db_path: str | Path = "mcpeval.db") -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS)
            conn.execute(_CREATE_CASE_RESULTS)
            conn.commit()

    def save_run(self, result: RunResult) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (eval_suite, model, total_cases, passed, failed, overall_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (result.eval_suite, result.model, result.total_cases,
                 result.passed, result.failed, result.overall_score),
            )
            conn.commit()
            return cursor.lastrowid

    def save_case_result(self, run_id: int, cr: CaseResult) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO case_results "
                "(run_id, case_id, passed, tool_calls_made, tool_calls_expected, "
                "graph_match_score, llm_judge_score, steps_taken, terminated_cleanly, raw_output) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    cr.case_id,
                    int(cr.passed),
                    json.dumps(cr.tool_calls_made),
                    json.dumps(cr.tool_calls_expected),
                    cr.graph_match_score,
                    cr.llm_judge_score,
                    cr.steps_taken,
                    int(cr.terminated_cleanly),
                    cr.raw_output,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_run(self, run_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def get_case_results(self, run_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM case_results WHERE run_id = ?", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_runs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
```

- [ ] **Step 5: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_store.py -v
```

Expected: 8/8 PASSED.

- [ ] **Step 6: Run full suite to check nothing broke**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/ -v --ignore=tests/test_runner.py
```

Expected: All passing.

- [ ] **Step 7: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/runner.py mcpeval/store.py tests/test_store.py
git commit -m "feat: implement SQLite ResultStore and CaseResult/RunResult dataclasses"
```

---

## Task 8: `reporter.py` — Rich Terminal Output

**Files:**
- Create: `mcpeval/reporter.py`
- Create: `tests/test_reporter.py`

- [ ] **Step 1: Write failing tests in `tests/test_reporter.py`**

```python
import io
import pytest
from rich.console import Console
from mcpeval.reporter import Reporter
from mcpeval.runner import RunResult, CaseResult


def _make_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True)
    return console, buf


def test_print_run_summary_renders(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert len(output) > 0
    assert "test_001" in output or "Test Suite" in output


def test_print_run_summary_shows_pass_fail_counts(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert "1" in output  # at least one "1" for pass or fail count


def test_print_run_summary_shows_score(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert "0.5" in output or "50" in output


def test_print_run_summary_shows_case_ids(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert "test_001" in output
    assert "test_002" in output


def test_print_case_detail_shows_score(passing_case_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_case_detail(passing_case_result)
    output = buf.getvalue()
    assert "test_001" in output
    assert "1.0" in output or "100" in output


def test_default_console_created_if_none():
    reporter = Reporter(console=None)
    assert reporter._console is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_reporter.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `mcpeval/reporter.py`**

```python
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich import box

from mcpeval.runner import CaseResult, RunResult


class Reporter:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def print_run_summary(self, result: RunResult) -> None:
        self._console.print(f"\n[bold]Eval Suite:[/bold] {result.eval_suite}")
        self._console.print(f"[bold]Model:[/bold] {result.model}")
        self._console.print(
            f"[bold]Results:[/bold] {result.passed}/{result.total_cases} passed  "
            f"| Overall score: {result.overall_score:.2f}"
        )

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Case ID", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("Terminated", justify="center")

        for cr in result.case_results:
            status = "[green]PASS[/green]" if cr.passed else "[red]FAIL[/red]"
            table.add_row(
                cr.case_id,
                status,
                f"{cr.graph_match_score:.2f}",
                str(cr.steps_taken),
                "✓" if cr.terminated_cleanly else "✗",
            )

        self._console.print(table)

    def print_case_detail(self, cr: CaseResult) -> None:
        status = "[green]PASS[/green]" if cr.passed else "[red]FAIL[/red]"
        self._console.print(f"\n[bold]Case:[/bold] {cr.case_id}  {status}")
        self._console.print(f"  Score: {cr.graph_match_score:.2f}")
        self._console.print(f"  Steps taken: {cr.steps_taken}")
        self._console.print(f"  Terminated cleanly: {cr.terminated_cleanly}")
        if cr.tool_calls_made:
            self._console.print(f"  Tool calls: {[c['tool_name'] for c in cr.tool_calls_made]}")
        if cr.raw_output:
            self._console.print(f"  Output: {cr.raw_output[:200]}")
```

- [ ] **Step 4: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_reporter.py -v
```

Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/reporter.py tests/test_reporter.py
git commit -m "feat: implement Rich terminal reporter with case table and detail view"
```

---

## Task 9: `runner.py` — EvalRunner (Full Implementation)

**Files:**
- Modify: `mcpeval/runner.py` (add EvalRunner class to existing dataclasses)
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests in `tests/test_runner.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcpeval.runner import EvalRunner, RunResult, CaseResult
from mcpeval.dataset import EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig


def _make_suite(cases: list[Case] | None = None) -> EvalSuite:
    if cases is None:
        cases = [Case(
            id="c001",
            input="Do the thing",
            expected_graph=ExpectedGraph(
                steps=[GraphStep(tool="get_logs")],
                max_steps=5,
                must_terminate=True,
            ),
            evaluators=[EvaluatorConfig(type="graph_match", strict=False)],
        )]
    return EvalSuite(
        name="Test Suite",
        model="claude-sonnet-4-20250514",
        mcp_server="test-mcp",
        mock_tools=[MockToolDef(name="get_logs", returns={"logs": []})],
        cases=cases,
    )


def _mock_end_turn_response():
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [MagicMock(type="text", text="All done.")]
    return response


def _mock_tool_use_then_end(tool_name: str, tool_input: dict):
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_use_123"
    tool_block.name = tool_name
    tool_block.input = tool_input

    response1 = MagicMock()
    response1.stop_reason = "tool_use"
    response1.content = [tool_block]

    response2 = MagicMock()
    response2.stop_reason = "end_turn"
    response2.content = [MagicMock(type="text", text="Done.")]

    return [response1, response2]


@pytest.mark.asyncio
async def test_run_suite_returns_run_result(sample_suite):
    runner = EvalRunner(anthropic_api_key="test-key")

    mock_create = AsyncMock(return_value=_mock_end_turn_response())
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(sample_suite)

    assert isinstance(result, RunResult)
    assert result.eval_suite == "Test Suite"
    assert result.total_cases == 1
    assert result.model == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_run_case_terminated_cleanly_true(sample_suite):
    runner = EvalRunner(anthropic_api_key="test-key")

    mock_create = AsyncMock(return_value=_mock_end_turn_response())
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(sample_suite)

    assert result.case_results[0].terminated_cleanly is True


@pytest.mark.asyncio
async def test_run_case_terminated_cleanly_false(sample_suite):
    runner = EvalRunner(anthropic_api_key="test-key", max_agent_turns=1)

    # Return tool_use repeatedly so it never ends
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu_1"
    tool_block.name = "get_logs"
    tool_block.input = {}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]

    mock_create = AsyncMock(return_value=response)
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(sample_suite)

    assert result.case_results[0].terminated_cleanly is False


@pytest.mark.asyncio
async def test_run_case_captures_tool_calls(sample_suite):
    runner = EvalRunner(anthropic_api_key="test-key")
    responses = _mock_tool_use_then_end("get_logs", {})

    mock_create = AsyncMock(side_effect=responses)
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(sample_suite)

    cr = result.case_results[0]
    assert len(cr.tool_calls_made) >= 1
    assert cr.tool_calls_made[0]["tool_name"] == "get_logs"


@pytest.mark.asyncio
async def test_run_suite_aggregates_scores():
    case_a = Case(
        id="a",
        input="do a",
        expected_graph=ExpectedGraph(steps=[GraphStep(tool="get_logs")], must_terminate=True),
        evaluators=[EvaluatorConfig(type="graph_match")],
    )
    case_b = Case(
        id="b",
        input="do b",
        expected_graph=ExpectedGraph(steps=[GraphStep(tool="run_playbook")], must_terminate=True),
        evaluators=[EvaluatorConfig(type="graph_match")],
    )
    suite = EvalSuite(
        name="Multi",
        model="claude-sonnet-4-20250514",
        mcp_server="test",
        mock_tools=[
            MockToolDef(name="get_logs", returns={}),
            MockToolDef(name="run_playbook", returns={}),
        ],
        cases=[case_a, case_b],
    )
    runner = EvalRunner(anthropic_api_key="test-key")

    # Case a: calls get_logs → matches → score 1.0 (but not terminated, penalty)
    # Case b: no matching calls → score 0.0

    mock_responses_a = _mock_tool_use_then_end("get_logs", {})
    mock_responses_b = [_mock_end_turn_response()]

    call_count = 0
    async def side_effect(*args, **kwargs):
        nonlocal call_count
        # For messages in case a, return tool use then end
        # Detect by checking what's in kwargs['tools'] or messages length
        msgs = kwargs.get("messages", args[0] if args else [])
        # Simple approach: alternate between cases
        nonlocal mock_responses_a, mock_responses_b, call_count
        responses = mock_responses_a if call_count < 2 else mock_responses_b
        result = responses[min(call_count % 2, len(responses)-1)]
        call_count += 1
        return result

    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(side_effect=side_effect)
        result = await runner.run_suite(suite)

    assert result.total_cases == 2
    assert 0.0 <= result.overall_score <= 1.0


@pytest.mark.asyncio
async def test_runner_saves_to_store(sample_suite, tmp_db_path):
    from mcpeval.store import ResultStore
    store = ResultStore(tmp_db_path)
    store.initialize()

    runner = EvalRunner(store=store, anthropic_api_key="test-key")
    mock_create = AsyncMock(return_value=_mock_end_turn_response())
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(sample_suite)

    assert result.run_id is not None
    stored = store.get_run(result.run_id)
    assert stored is not None
    assert stored["eval_suite"] == "Test Suite"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_runner.py -v
```

Expected: ImportError (EvalRunner not in runner.py yet).

- [ ] **Step 3: Implement `EvalRunner` in `mcpeval/runner.py`** (append to existing dataclasses)

```python
# Add these imports at the top of runner.py:
import dataclasses
import json
import os
from typing import Any

from anthropic import AsyncAnthropic

from mcpeval.dataset import EvalSuite, Case, EvaluatorConfig
from mcpeval.mock_server import MockMCPServer
from mcpeval.capture import CaptureMiddleware, ToolCallRecord
from mcpeval.graph import ToolCallGraph
from mcpeval.evaluators.graph_match import GraphMatchEvaluator


# Add EvalRunner class after the dataclasses:

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
        overall_score = (
            sum(cr.graph_match_score for cr in case_results) / len(case_results)
            if case_results else 0.0
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
        anthropic = AsyncAnthropic(api_key=self._api_key)

        # Convert FastMCP tools to Anthropic tool spec format
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
                text_blocks = [b for b in response.content if hasattr(b, "type") and b.type == "text"]
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
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

        # Evaluate
        graph = ToolCallGraph.from_expected(case.expected_graph)
        eval_cfg = next(
            (e for e in case.evaluators if e.type == "graph_match"),
            EvaluatorConfig(type="graph_match", strict=False),
        )
        evaluator = GraphMatchEvaluator(strict=eval_cfg.strict)
        graph_result = evaluator.evaluate(capture.calls, graph, terminated_cleanly)

        return CaseResult(
            case_id=case.id,
            passed=graph_result.passed,
            tool_calls_made=[
                {"tool_name": r.tool_name, "arguments": r.arguments}
                for r in capture.calls
            ],
            tool_calls_expected=[
                {"tool": s.tool, "params_contain": s.params_contain}
                for s in graph.steps
            ],
            graph_match_score=graph_result.score,
            llm_judge_score=None,
            steps_taken=graph_result.steps_taken,
            terminated_cleanly=terminated_cleanly,
            raw_output=raw_output,
        )


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
```

- [ ] **Step 4: Run tests — expect all green**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/test_runner.py -v
```

Expected: 7/7 PASSED.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/ -v -m "not integration"
```

Expected: All tests passing.

- [ ] **Step 6: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add mcpeval/runner.py tests/test_runner.py
git commit -m "feat: implement EvalRunner with Anthropic agent loop and graph evaluation"
```

---

## Task 10: Example Eval Suite + `mcpeval/__init__.py` Exports

**Files:**
- Create: `evals/examples/incident_response.yaml`
- Modify: `mcpeval/__init__.py`

- [ ] **Step 1: Create `evals/examples/incident_response.yaml`**

```yaml
name: "Incident Response Agent"
model: claude-sonnet-4-20250514
mcp_server: "ops-mcp"
mock_tools:
  get_logs:
    returns:
      logs:
        - "ERROR payment-service: timeout at 14:32"
        - "ERROR payment-service: timeout at 14:33"
        - "ERROR payment-service: connection refused at 14:34"
  run_playbook:
    returns:
      status: "ok"
      steps_completed: 3
      playbook: "payment-service-recovery"
  notify_oncall:
    returns:
      notified: true
      channel: "#incidents"
      message: "Payment service incident resolved"
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
        - tool: notify_oncall
          optional: true
      max_steps: 6
      must_terminate: true
    evaluators:
      - type: graph_match
        strict: false

  - id: "incident_002"
    input: "Check if there are any issues with the payment system"
    expected_graph:
      steps:
        - tool: get_logs
      max_steps: 3
      must_terminate: true
    evaluators:
      - type: graph_match
        strict: true
```

- [ ] **Step 2: Update `mcpeval/__init__.py`** to expose public API

```python
from mcpeval.dataset import load_suite, EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep
from mcpeval.graph import ToolCallGraph, Step
from mcpeval.capture import CaptureMiddleware, ToolCallRecord
from mcpeval.mock_server import MockMCPServer
from mcpeval.evaluators.graph_match import GraphMatchEvaluator, GraphMatchResult
from mcpeval.store import ResultStore
from mcpeval.runner import EvalRunner, RunResult, CaseResult
from mcpeval.reporter import Reporter

__all__ = [
    "load_suite",
    "EvalSuite",
    "MockToolDef",
    "Case",
    "ExpectedGraph",
    "GraphStep",
    "ToolCallGraph",
    "Step",
    "CaptureMiddleware",
    "ToolCallRecord",
    "MockMCPServer",
    "GraphMatchEvaluator",
    "GraphMatchResult",
    "ResultStore",
    "EvalRunner",
    "RunResult",
    "CaseResult",
    "Reporter",
]
```

- [ ] **Step 3: Verify the public API is importable**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run python -c "import mcpeval; print(mcpeval.__all__)"
```

Expected: Prints the `__all__` list without errors.

- [ ] **Step 4: Run full test suite one final time**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/ -v -m "not integration"
```

Expected: All green.

- [ ] **Step 5: Commit**

```bash
cd /Users/SP12923/Desktop/code/mcpeval
git add evals/examples/incident_response.yaml mcpeval/__init__.py
git commit -m "feat: add incident response example eval suite and expose public API"
```

---

## Verification

After all tasks complete, run the full test suite:

```bash
cd /Users/SP12923/Desktop/code/mcpeval
uv run pytest tests/ -v -m "not integration" --tb=short
```

All tests should pass. Verify module imports work:

```bash
uv run python -c "
import mcpeval
from mcpeval import MockMCPServer, ToolCallGraph, EvalRunner, load_suite
print('All imports OK')
print('Public API:', mcpeval.__all__)
"
```

For a real end-to-end test (requires `ANTHROPIC_API_KEY`):

```bash
ANTHROPIC_API_KEY=your-key uv run pytest tests/ -v -m "integration"
```
