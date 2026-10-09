"""Reproducible protocol regression and opt-in, honest live-model behavioral evaluation."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path

from .agent import MusicAgent
from .config import settings
from .deepseek import DeepSeekClient
from .models import uid
from .schemas import QueryRequest, now
from .store import Store
from .tools import ToolRegistry
from .transport import Transport


def regression_environment() -> dict[str, str]:
    excluded = {"DEEPSEEK_API_KEY", "PLANNER_API_KEY", "MUSICTRACKER_API_TOKEN", "DATABASE_URL", "PLANNER_BASE_URL", "PLANNER_MODEL"}
    return {key: value for key, value in os.environ.items() if key not in excluded}


def assess(query: dict, results: list[dict], calls: list[dict], expected: dict, steps: list[dict] | None = None) -> dict:
    plan = query["plan"]
    actual_ids = {m["observation_id"] for song in results for m in song["metrics"] if m["status"] in {"exact", "approximate"}}
    answer = query["report"].get("answer") or {}
    references = {c["observation_id"] for c in answer.get("citations", [])}
    steps = steps or []
    state = query.get("checkpoint", {}).get("agent_state", {})
    usage = query["report"].get("agent_usage") or {"rounds": state.get("rounds", 0),
        "tool_calls": state.get("tool_count", 0), "budget_tokens": state.get("budget_tokens", 0)}
    fields = [m for song in results for m in song["metrics"] if m["name"] in {"comment_count", "favorite_count"}]
    grounded = bool(references) and references.issubset(actual_ids) if answer.get("status") == "verified" and actual_ids else None
    return {"plan_matches": (plan.get("title") == expected["expected_title"] and plan.get("days") == expected["expected_days"])
            if "expected_title" in expected else None,
            "expected_status_matches": query["execution_status"] == expected["expected_status"] if "expected_status" in expected else None,
            "tools_allowlisted": all(c["tool"] in {p + "." + name for p in ["qq", "netease"] for name in ["search_songs", "song_detail", "song_metrics"]} for c in calls),
            "answer_grounded": grounded,
            "research_available": bool(results) and grounded is True,
            "platform_results": {p: sum(s["platform"] == p for s in results) for p in plan.get("platforms", [])},
            "field_counts": {status: sum(m["status"] == status for m in fields) for status in ["exact", "approximate", "unavailable", "error"]},
            "invalid_id_attempts": sum(s["kind"] == "tool_feedback" and s["payload"].get("error") == "unknown_song_id" for s in steps),
            "answer_repairs": sum(s["kind"] == "answer_rejected" for s in steps),
            "model_duration_ms": sum(s["payload"].get("duration_ms", 0) for s in steps if s["kind"] == "model_usage"),
            "status": query["execution_status"], "model_error": query["report"].get("model_error"),
            "result_count": len(results), "citation_count": len(references), "tool_calls": len(calls),
            "usage": usage, "tool_duration_ms": sum(c["duration_ms"] for c in calls)}


def summarize(cases: list[dict]) -> dict:
    statuses = {s: sum(c["status"] == s for c in cases) for s in ["completed", "partial", "waiting_input", "failed", "rejected"]}
    return {"cases": len(cases), "statuses": statuses,
            "research_available": sum(c.get("research_available") is True for c in cases),
            "plan_checks": {"passed": sum(c.get("plan_matches") is True for c in cases),
                            "evaluated": sum(c.get("plan_matches") is not None for c in cases)},
            "behavior_checks": {"passed": sum(c.get("expected_status_matches") is True for c in cases),
                                "evaluated": sum(c.get("expected_status_matches") is not None for c in cases)},
            "model_errors": sum(bool(c.get("model_error")) for c in cases),
            "invalid_id_attempts": sum(c.get("invalid_id_attempts", 0) for c in cases),
            "budget_tokens_total": sum(c.get("usage", {}).get("budget_tokens", 0) for c in cases)}


def source_fingerprints(root: Path, corpus: Path | None = None) -> dict:
    paths = [root / "musictracker" / f"{name}.py" for name in
             ["agent", "deepseek", "workflow", "tools", "platforms", "schemas", "store", "api", "config", "evaluation"]]
    if corpus is not None:
        paths.append(corpus)
    return {str(p.relative_to(root) if p.is_relative_to(root) else p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def evaluate(live: bool = False, limit: int = 3, cases_path: str | None = None):
    root = Path(__file__).resolve().parent.parent
    output = root / "artifacts"
    output.mkdir(exist_ok=True)
    stamp = now().strftime("%Y%m%dT%H%M%SZ")
    if not live:
        junit = output / f"agent-regression-{stamp}.xml"
        result = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests/test_agent.py"), "-q", f"--junitxml={junit}"], cwd=root,
                                env=regression_environment())
        suite = ET.parse(junit).getroot().find("testsuite")
        report = {"mode": "scripted_protocol_regression", "real_model": False,
                  "tests": int(suite.get("tests")), "failures": int(suite.get("failures")),
                  "errors": int(suite.get("errors")), "skipped": int(suite.get("skipped")),
                  "seconds": float(suite.get("time")), "exit_code": result.returncode,
                  "note": "MockTransport测试，不是DeepSeek真实音乐任务准确率"}
    else:
        if not settings.deepseek_key:
            raise SystemExit("未配置 DEEPSEEK_API_KEY，不运行真实评测、不生成虚构成绩。请填写本机.env后重试。")
        corpus_path = (Path(cases_path) if cases_path else root / "evals/agent_cases.json").resolve()
        corpus = json.loads(corpus_path.read_text())
        cases = corpus["cases"][:max(1, min(limit, 8))]
        async def run():
            store, transport, model = Store(settings.database_url), Transport(settings), DeepSeekClient(settings)
            store.initialize()
            tools = ToolRegistry(transport)
            agent = MusicAgent(store, tools, settings, model)
            rows = []
            try:
                for case in cases:
                    try:
                        request = QueryRequest.model_validate(case["request"] | {"execution_mode": "agent"})
                    except ValueError:
                        rows.append({"case": case["id"], "status": "rejected", "expected_status_matches": case.get("expected_status") == "rejected"})
                        continue
                    qid, _ = store.create(request)
                    store.update(qid, status="resolving", lease_owner=uid(), lease_until=now() + timedelta(seconds=settings.agent_timeout + 60))
                    started = time.monotonic()
                    try:
                        await agent.run(qid)
                    except ValueError:
                        store.finish(qid, {"summary": "本用例执行契约异常，保留检查点供审查", "execution_mode": "agent", "model_error": "evaluation_contract_error"}, "failed")
                    row = {"case": case["id"], "query_id": qid, **assess(store.get(qid), store.results(qid), store.calls(qid), case, store.agent_steps(qid)),
                           "elapsed_ms": round((time.monotonic() - started) * 1000)}
                    rows.append(row)
                    print(json.dumps({"case_finished": row["case"], "query_id": qid, "status": row["status"],
                                      "model_error": row["model_error"], "elapsed_ms": row["elapsed_ms"]}, ensure_ascii=False), flush=True)
                return rows
            finally:
                await transport.close()
                await model.close()
                store.engine.dispose()
        report = {"mode": "live_behavior_evaluation", "real_model": True, "model": settings.deepseek_model,
                  "started_at_utc": stamp, "source_sha256": source_fingerprints(root, corpus_path),
                  "limits": {"rounds": settings.agent_max_rounds, "tool_attempts": settings.agent_max_tools,
                             "budget_tokens": settings.agent_token_budget, "execution_seconds": settings.agent_timeout},
                  "corpus_version": corpus["version"], "corpus_file": str(Path(cases_path).name) if cases_path else "agent_cases.json",
                  "cases": asyncio.run(run()),
                  "note": "缺数据/等待确认/模型失败单独报告；不将工程契约通过率等同原唱准确率或全曲库召回率"}
        report["summary"] = summarize(report["cases"])
        report["finished_at"] = now().isoformat()
    path = output / f"agent-{'live' if live else 'regression'}-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"评测文件：{path}")
    if not live and result.returncode:
        raise SystemExit(result.returncode)
