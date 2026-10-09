"""Opt-in live API smoke: clarification → named artist → follow-up and human selection.

Run from the project root with the local service running; creates records and incurs API use.
No provider key is read here. A saved report records real statuses, not fabricated successes.
"""
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
    parser.add_argument("clarification_query_id", help="本机等待补充歌名的测试查询ID")
    parser.add_argument("selection_query_id", help="本机等待确认光泽空心版本的90天测试查询ID")
    args = parser.parse_args()
    source, selection_source = args.clarification_query_id, args.selection_query_id
    if any(not re.fullmatch(r"[a-f0-9]{32}", value) for value in [source, selection_source]):
        parser.error("查询ID必须是32位小写hex")
    rows = []
    with httpx.Client(base_url="http://127.0.0.1:8100/api/v1", trust_env=False, timeout=20) as client:
        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        def wait(qid):
            deadline = time.monotonic() + 210
            while time.monotonic() < deadline:
                query = request("GET", f"/queries/{qid}")
                if query["execution_status"] in {"completed", "partial", "failed", "waiting_input", "cancelled"}:
                    return query
                time.sleep(1)
            raise TimeoutError("本机API查询超时，后台记录保留")

        for name, text, platforms, days in [
            ("clarification_artist_without_id", "我指光泽唱的《空心》，获取近期评论和收藏数据", ["qq", "netease"], 7),
            ("followup_platform_and_window", "刚才那首歌，只看网易云近30天的评论变化", ["netease"], 30),
        ]:
            result = request("POST", f"/queries/{source}/follow-up", json={"input": text, "platforms": platforms})
            source = result["query_id"]
            query = wait(source)
            row = {"case": name, "query_id": source, **assess(query, query["results"], query["tool_calls"],
                   {"expected_title": "空心", "expected_days": days}, query["agent_steps"]),
                   "conversation_id": query["conversation_id"], "anchor_id": query["plan"].get("anchor_id"),
                   "selected_platforms": query["plan"].get("platforms")}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if query["execution_status"] in {"failed", "waiting_input"}:
                break

        before = request("GET", f"/queries/{selection_source}")
        choice = next((c for c in before["checkpoint"].get("choices", []) if "光泽" in c["artists"]), None)
        if choice and before["execution_status"] == "waiting_input":
            result = request("POST", f"/queries/{selection_source}/select", json={"anchor_id": choice["key"]})
            after = wait(result["query_id"])
            row = {"case": "human_selection_resume", "query_id": selection_source,
                   **assess(after, after["results"], after["tool_calls"], {"expected_title": "空心", "expected_days": 90}, after["agent_steps"]),
                   "no_new_platform_calls": len(before["tool_calls"]) == len(after["tool_calls"])}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    artifact = ROOT / "artifacts" / f"agent-scenarios-{now().strftime('%Y%m%dT%H%M%SZ')}.json"
    artifact.write_text(json.dumps({"mode": "live_api_scenarios", "real_model": True, "cases": rows}, ensure_ascii=False, indent=2))
    print(f"评测文件：{artifact}")


if __name__ == "__main__":
    main()
