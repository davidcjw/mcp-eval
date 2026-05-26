import pytest
from mcpeval.store import ResultStore
from mcpeval.runner import RunResult, CaseResult


def _make_run(store: ResultStore, suite: str, score: float, case_scores: list[tuple[str, float]]) -> int:
    case_results = [
        CaseResult(
            case_id=cid,
            passed=s >= 0.7,
            tool_calls_made=[],
            tool_calls_expected=[],
            graph_match_score=s,
            llm_judge_score=None,
            rule_score=None,
            steps_taken=1,
            terminated_cleanly=True,
            raw_output="",
        )
        for cid, s in case_scores
    ]
    run = RunResult(
        run_id=None,
        eval_suite=suite,
        model="test-model",
        total_cases=len(case_results),
        passed=sum(1 for cr in case_results if cr.passed),
        failed=sum(1 for cr in case_results if not cr.passed),
        overall_score=score,
        case_results=case_results,
    )
    run_id = store.save_run(run)
    for cr in case_results:
        store.save_case_result(run_id, cr)
    return run_id


def test_regression_detects_score_drop(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    baseline_id = _make_run(store, "s", 1.0, [("c001", 1.0), ("c002", 1.0)])
    current_id = _make_run(store, "s", 0.5, [("c001", 0.5), ("c002", 0.5)])

    report = store.get_regression_report(current_id, baseline_id)

    assert report["overall_score_delta"] == pytest.approx(-0.5)
    assert len(report["regressed"]) == 2
    assert len(report["improved"]) == 0
    assert all(r["delta"] < 0 for r in report["regressed"])


def test_regression_detects_improvement(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    baseline_id = _make_run(store, "s", 0.5, [("c001", 0.5)])
    current_id = _make_run(store, "s", 1.0, [("c001", 1.0)])

    report = store.get_regression_report(current_id, baseline_id)

    assert report["overall_score_delta"] == pytest.approx(0.5)
    assert len(report["improved"]) == 1
    assert report["improved"][0]["case_id"] == "c001"
    assert len(report["regressed"]) == 0


def test_regression_unchanged_within_tolerance(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    baseline_id = _make_run(store, "s", 0.8, [("c001", 0.8)])
    current_id = _make_run(store, "s", 0.82, [("c001", 0.82)])

    report = store.get_regression_report(current_id, baseline_id)

    assert len(report["unchanged"]) == 1
    assert report["unchanged"][0] == "c001"


def test_regression_skips_cases_not_in_baseline(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    baseline_id = _make_run(store, "s", 1.0, [("c001", 1.0)])
    current_id = _make_run(store, "s", 0.75, [("c001", 1.0), ("c002_new", 0.5)])

    report = store.get_regression_report(current_id, baseline_id)

    all_case_ids = (
        [r["case_id"] for r in report["regressed"]]
        + [r["case_id"] for r in report["improved"]]
        + report["unchanged"]
    )
    assert "c002_new" not in all_case_ids


def test_regression_report_has_expected_keys(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    baseline_id = _make_run(store, "s", 1.0, [("c001", 1.0)])
    current_id = _make_run(store, "s", 1.0, [("c001", 1.0)])

    report = store.get_regression_report(current_id, baseline_id)

    assert "run_id" in report
    assert "baseline_run_id" in report
    assert "overall_score_delta" in report
    assert "regressed" in report
    assert "improved" in report
    assert "unchanged" in report
