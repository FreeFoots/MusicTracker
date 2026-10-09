from pathlib import Path

from musictracker.evaluation import assess, regression_environment, source_fingerprints, summarize


def test_no_data_never_counts_as_available_research():
    query = {"plan": {}, "report": {"answer": {"status": "verified", "citations": []}}, "execution_status": "completed"}
    result = assess(query, [], [], {})
    assert result["research_available"] is False and result["answer_grounded"] is None
    assert summarize([result])["research_available"] == 0


def test_mixed_outcomes_have_separate_denominators():
    cases = [{"status": "partial", "research_available": True, "plan_matches": True,
              "expected_status_matches": None, "usage": {"budget_tokens": 20}},
             {"status": "waiting_input", "expected_status_matches": True},
             {"status": "rejected", "expected_status_matches": True}]
    summary = summarize(cases)
    assert summary["plan_checks"] == {"passed": 1, "evaluated": 1}
    assert summary["behavior_checks"] == {"passed": 2, "evaluated": 2}
    assert summary["research_available"] == 1 and summary["budget_tokens_total"] == 20


def test_forged_reference_is_not_grounded():
    query = {"plan": {}, "execution_status": "partial", "report": {
        "answer": {"status": "verified", "citations": [{"observation_id": "invented"}]}}}
    results = [{"platform": "qq", "metrics": [{"name": "comment_count", "status": "exact", "observation_id": "real"}]}]
    result = assess(query, results, [], {})
    assert result["answer_grounded"] is False and result["research_available"] is False


def test_waiting_input_token_usage_is_not_omitted():
    query = {"plan": {}, "report": {}, "execution_status": "waiting_input",
             "checkpoint": {"agent_state": {"rounds": 4, "tool_count": 6, "budget_tokens": 12345}}}
    result = assess(query, [], [], {})
    assert result["usage"]["budget_tokens"] == 12345
    assert summarize([result])["budget_tokens_total"] == 12345


def test_offline_regression_subprocess_does_not_inherit_provider_key(monkeypatch):
    for key in ["DEEPSEEK_API_KEY", "PLANNER_API_KEY", "MUSICTRACKER_API_TOKEN", "DATABASE_URL"]:
        monkeypatch.setenv(key, "test-placeholder")
        assert key not in regression_environment()


def test_evaluation_fingerprints_only_source_and_corpus():
    root = Path(__file__).resolve().parent.parent
    values = source_fingerprints(root, root / "evals/agent_cases.json")
    assert "evals/agent_cases.json" in values and "musictracker/agent.py" in values
    assert not any(".env" in name for name in values)
    assert all(len(value) == 64 for value in values.values())
