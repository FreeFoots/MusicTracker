"""Opt-in real answer-only retry; records whether saved platform evidence changed."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musictracker.evaluation import assess  # noqa: E402
from musictracker.schemas import now  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_id", help="本机已保存音乐事实的Agent查询ID；消耗模型额度")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}", args.query_id):
        parser.error("查询ID必须是32位小写hex")
    with httpx.Client(base_url="http://127.0.0.1:8100/api/v1", trust_env=False, timeout=20) as client:
        def request(method, path):
            response = client.request(method, path)
            response.raise_for_status()
            return response.json()

        before = request("GET", f"/queries/{args.query_id}")
        started = time.monotonic()
        request("POST", f"/queries/{args.query_id}/regenerate-answer")
        deadline = started + 210
        while time.monotonic() < deadline:
            after = request("GET", f"/queries/{args.query_id}")
            if after["execution_status"] in {"completed", "partial", "failed", "cancelled"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError("本机回答任务超时，检查点保留")
    report = {"mode": "live_answer_only_retry", "real_model": True, "query_id": args.query_id,
              "elapsed_ms": round((time.monotonic() - started) * 1000),
              "before_status": before["execution_status"], "before_model_error": before["report"].get("model_error"),
              "results_unchanged": before["results"] == after["results"],
              "platform_calls_unchanged": before["tool_calls"] == after["tool_calls"],
              "before_budget_tokens": before["report"].get("agent_usage", {}).get("budget_tokens"),
              "new_steps": after["agent_steps"][len(before["agent_steps"]):],
              **assess(after, after["results"], after["tool_calls"], {}, after["agent_steps"])}
    artifact = ROOT / "artifacts" / f"agent-answer-retry-{now().strftime('%Y%m%dT%H%M%SZ')}.json"
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"评测文件：{artifact}")


if __name__ == "__main__":
    main()
