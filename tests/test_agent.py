import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from musictracker.agent import MusicAgent, render_answer
from musictracker.api import create_app
from musictracker.deepseek import DeepSeekClient, EvidenceAnswer, ModelError
from musictracker.models import Query, ReportVersion
from musictracker.schemas import QueryRequest
from musictracker.transport import PlatformError
from musictracker.workflow import OverviewWorkflow


async def test_agent_cache_keeps_times_and_generates_new_grounded_answer(store, registry, config):
    rules = OverviewWorkflow(store, registry, config)
    original, _ = store.create(QueryRequest(input="测试歌"))
    await rules.run(original)
    agent, provider, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent", mode="cache_only"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "completed" and q["report"]["origin"] == "cache"
        assert q["report"]["execution_mode"] == "agent" and q["report"]["answer"]["status"] == "verified"
        assert len(store.calls(qid)) == 0 and len(provider.requests) == 2
        assert [s["fetched_at"] for s in store.results(qid)] == [s["fetched_at"] for s in store.results(original)]
        assert {c["observation_id"] for c in q["report"]["answer"]["citations"]}.issubset(
            {m["observation_id"] for s in store.results(original) for m in s["metrics"]})
    finally:
        await model.close()


async def test_agent_cache_miss_is_empty_completed_not_platform_failure(store, registry, config):
    agent, _, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent", mode="cache_only"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "completed" and q["report"]["songs"] == []
        assert q["report"]["summary"] == "未找到历史数据" and store.calls(qid) == []
    finally:
        await model.close()


@pytest.mark.parametrize("url", ["https://api.deepseek.com:invalid", "https://[invalid"])
async def test_malformed_model_url_rejected_before_sending(config, url):
    calls = []
    client = DeepSeekClient(replace(config, deepseek_key="not-real", deepseek_base_url=url),
                            httpx.AsyncClient(transport=httpx.MockTransport(lambda r: calls.append(r))))
    try:
        with pytest.raises(ModelError, match="未发送密钥"):
            await client.complete([])
        assert not calls
    finally:
        await client.close()


async def test_duplicate_tool_call_ids_rejected(config):
    call = {"id": "same", "function": {"name": "finish_research", "arguments": "{}"}}
    client = DeepSeekClient(replace(config, deepseek_key="not-real"), httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [call, call]}}]}))))
    try:
        with pytest.raises(ModelError, match="工具调用结构无效"):
            await client.complete([])
    finally:
        await client.close()


def test_legacy_idempotent_request_preserved(store):
    request = QueryRequest(input="测试歌")
    qid, _ = store.create(request, key="before-v06")
    with Session(store.engine) as session, session.begin():
        query = session.get(Query, qid)
        payload = request.model_dump(exclude={"execution_mode"})
        query.request = payload
        query.request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert store.create(request, key="before-v06") == (qid, False)
    with pytest.raises(ValueError):
        store.create(QueryRequest(input="其他歌"), key="before-v06")


def test_credentials_in_title_rejected():
    with pytest.raises(ValueError, match="凭据"):
        QueryRequest(input="测试歌", title="authorization: private")


async def test_token_budget_stops_before_model_or_platform_request(store, registry, config):
    agent, provider, model, _ = setup_agent(store, registry, config, agent_token_budget=2048)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert provider.requests == [] and store.calls(qid) == []
        assert store.get(qid)["report"]["model_error"] == "model_budget"
    finally:
        await model.close()


async def test_agent_deadline_stops_pending_model_request(store, registry, config):
    import asyncio
    config = replace(config, deepseek_key="not-real", agent_timeout=0.02)
    async def delayed(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={})
    model = DeepSeekClient(config, httpx.AsyncClient(transport=httpx.MockTransport(delayed)))
    agent = MusicAgent(store, registry, config, model)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["report"]["model_error"] == "agent_deadline"
        assert store.calls(qid) == []
    finally:
        await model.close()


def action(name, **args):
    return (name, args)


NORMAL = [action("netease__search_songs", keyword="测试歌"), action("netease__song_detail", ids=["1", "2"]),
          action("netease__song_metrics", ids=["1", "2"]), action("qq__search_songs", keyword="测试歌"),
          action("qq__song_detail", ids=["mid1", "mid2"]), action("qq__song_metrics", ids=["1", "2"]),
          action("finish_research")]


class ScriptedProvider:
    """Protocol simulation, not a real-model quality benchmark."""
    def __init__(self, actions=None, plan=None, bad_answer=False):
        self.actions = list(NORMAL if actions is None else actions)
        self.plan = plan or {"title": "测试歌", "days": 7, "platforms": ["qq", "netease"], "question": None}
        self.requests = []
        self.bad_answer = bad_answer

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        pending = set()
        for message in body["messages"]:
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending, "工具结果没有对应的历史调用"
                pending.remove(message["tool_call_id"])
            else:
                assert not pending, "上一批工具结果尚未全部回复"
                if "tool_calls" in message:
                    assert message["tool_calls"], "历史调用不可因待执行队列出队而被清空"
                    pending = {c["id"] for c in message["tool_calls"]}
        assert not pending
        assert body["thinking"] == {"type": "disabled"}
        if body.get("tools"):
            name, args = self.actions.pop(0) if self.actions else action("finish_research")
            msg = {"content": None, "tool_calls": [{"id": "call_" + str(len(self.requests)), "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}
            reason = "tool_calls"
        elif "facts" in json.loads(body["messages"][1]["content"]):
            facts = json.loads(body["messages"][1]["content"])["facts"]
            answer = {"sections": [{"kind": "current_metrics", "observation_ids": [f["observation_id"] for f in facts[:24]]}]
                      if facts else [{"kind": "limitations", "observation_ids": []}], "next_action": "collect_more_history"}
            if self.bad_answer:
                answer["sections"] = [{"kind": "current_metrics", "observation_ids": ["invented"]}]
            msg, reason = {"content": json.dumps(answer)}, "stop"
        else:
            msg, reason = {"content": json.dumps(self.plan)}, "stop"
        msg.update(role="assistant", reasoning_content="PRIVATE_REASONING_MUST_NOT_BE_STORED")
        return httpx.Response(200, json={"model": "deepseek-flash", "choices": [{"message": msg, "finish_reason": reason}],
                                       "usage": {"total_tokens": 100}})


def setup_agent(store, registry, config, provider=None, **overrides):
    config = replace(config, deepseek_key="test-placeholder-not-a-real-key", **overrides)
    provider = provider or ScriptedProvider()
    model = DeepSeekClient(config, httpx.AsyncClient(transport=httpx.MockTransport(provider.respond)))
    return MusicAgent(store, registry, config, model), provider, model, config


async def test_native_tool_loop_and_grounded_answer(store, registry, config):
    agent, provider, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "completed" and q["plan"]["planner"] == "deepseek_agent"
        assert len(store.calls(qid)) == 6 and store.stats()["observations"] == 8
        assert q["report"]["answer"]["status"] == "verified"
        observed = {m["observation_id"] for s in store.results(qid) for m in s["metrics"]}
        assert {c["observation_id"] for c in q["report"]["answer"]["citations"]}.issubset(observed)
        assert any(m["role"] == "tool" for r in provider.requests for m in r["messages"])
        assert "PRIVATE_REASONING" not in json.dumps([q, store.agent_steps(qid)])
        assert "test-placeholder-not-a-real-key" not in json.dumps([q, store.agent_steps(qid)])
        for request in provider.requests:
            if request.get("tools"):
                snapshot = json.loads(request["messages"][1]["content"])["working_state"]
                assert "next_required_steps" in snapshot
                assert sum(m["role"] == "assistant" for m in request["messages"]) <= 1
                for song in snapshot["candidates"]:
                    if not song.get("detailed"):
                        assert song["numeric_id"] is None
    finally:
        await model.close()


async def test_finish_cannot_claim_complete_with_uncollected_versions(store, registry, config):
    actions = [NORMAL[0], action("netease__song_detail", ids=["1"]), action("netease__song_metrics", ids=["1"]),
               NORMAL[3], action("qq__song_detail", ids=["mid1"]), action("qq__song_metrics", ids=["1"]), action("finish_research")]
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider(actions))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "partial"
        assert {"qq.song_detail", "netease.song_detail"}.issubset(agent.missing_steps(store.get(qid)["checkpoint"]["agent_state"]))
    finally:
        await model.close()


async def test_metrics_require_all_requested_version_slots(store, registry, config):
    actions = [NORMAL[0], NORMAL[1], action("netease__song_metrics", ids=["1"]), *NORMAL[3:]]
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider(actions))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "partial"
        assert "netease.song_metrics" in agent.missing_steps(store.get(qid)["checkpoint"]["agent_state"])
        actions = agent.next_actions(store.get(qid)["checkpoint"]["agent_state"])
        assert any(a["tool"] == "netease__song_metrics" and a["arguments"] == {"ids": ["2"]} for a in actions)
    finally:
        await model.close()


def test_answer_cannot_omit_a_data_bearing_platform_recording():
    report = {"songs": [{"artists": ["甲"], "platform": p, "platform_id": "1", "url": "https://example.com/", "metrics": [
        {"observation_id": p, "status": "exact", "label": "评论", "value": 1, "collected_at": "now"}]} for p in ["qq", "netease"]]}
    with pytest.raises(ValueError, match="遗漏"):
        render_answer(EvidenceAnswer.model_validate({"sections": [{"kind": "current_metrics", "observation_ids": ["qq"]}]}), report)


async def test_no_tool_prose_receives_corrective_feedback_in_projected_context(store, registry, config):
    provider = ScriptedProvider()
    old = provider.respond
    replied = False
    def respond(request):
        nonlocal replied
        body = json.loads(request.content)
        if body.get("tools") and not replied:
            replied = True
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "已完成"}}],
                                             "usage": {"total_tokens": 10}})
        return old(request)
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "completed"
        assert any("请调用允许的工具" in m.get("content", "") for r in provider.requests for m in r["messages"])
    finally:
        await model.close()


async def test_id_error_returns_safe_known_ids_to_model(store, registry, config):
    provider = ScriptedProvider([NORMAL[0], action("netease__song_detail", ids=["invented"]), *NORMAL[1:]])
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        replies = [json.loads(m["content"]) for r in provider.requests for m in r["messages"] if m["role"] == "tool"]
        assert any(r.get("allowed_ids") == ["1", "2"] for r in replies)
        assert store.get(qid)["execution_status"] == "completed"
    finally:
        await model.close()


def test_configuration_repr_never_contains_secrets(config):
    config = replace(config, deepseek_key="SECRET_DEEPSEEK", planner_key="SECRET_PLANNER", api_token="SECRET_TOKEN",
                     database_url="postgresql://user:SECRET_PASSWORD@localhost/db")
    assert "SECRET_" not in repr(config)


@pytest.mark.parametrize("history", [
    [{"role": "assistant", "tool_calls": []}, {"role": "tool", "tool_call_id": "lost", "content": "{}"}],
    [{"role": "assistant", "tool_calls": [{"id": "pending"}]}],
])
async def test_invalid_protocol_never_sent_to_provider(config, history):
    requests = []
    client = DeepSeekClient(replace(config, deepseek_key="not-real"),
                            httpx.AsyncClient(transport=httpx.MockTransport(lambda r: requests.append(r))))
    try:
        with pytest.raises(ModelError):
            await client.complete(history)
        assert not requests
    finally:
        await client.close()


async def test_all_acquired_versions_persist_beyond_report_limit(store, registry, config):
    from musictracker.schemas import Metric, Song
    songs = [Song(platform="netease", platform_id=str(i), numeric_id=str(i), title="测试歌", artists=["甲"],
                  search_rank=i, url=f"https://music.163.com/song?id={i}") for i in range(1, 8)]
    async def search(**kwargs):
        return {"songs": [s.model_dump() for s in songs], "coverage": {"complete": False}}
    async def detail(ids):
        return {"songs": [s.model_dump() | {"search_rank": 0} for s in songs if s.platform_id in ids]}
    async def metrics(ids):
        return {"metrics": {i: [Metric(name="comment_count", label="评论", status="exact", value=int(i),
                                      source_url=f"https://music.163.com/song?id={i}").model_dump()] for i in ids}}
    registry.tools["netease.search_songs"].run = search
    registry.tools["netease.song_detail"].run = detail
    registry.tools["netease.song_metrics"].run = metrics
    actions = [action("netease__search_songs", keyword="测试歌"), action("netease__song_detail", ids=[str(i) for i in range(1, 8)]),
               action("netease__song_metrics", ids=[str(i) for i in range(1, 5)]),
               action("netease__song_metrics", ids=[str(i) for i in range(5, 8)]), action("finish_research")]
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider(actions, plan={"title": "测试歌", "platforms": ["netease"]}))
    qid, _ = store.create(QueryRequest(input="测试歌", platforms=["netease"], execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert len(store.results(qid)) == 7 and len(q["report"]["songs"]) == 4
        assert q["execution_status"] == "completed"
        assert [s["search_rank"] for s in q["checkpoint"]["agent_state"]["candidates"]] == list(range(1, 8))
    finally:
        await model.close()


async def test_explicit_artist_resolves_ambiguity_without_invented_id(store, registry, config):
    old = registry.tools["netease.song_detail"].run
    async def detail(**kwargs):
        output = await old(**kwargs)
        output["songs"][1]["origin_status"] = "platform_original"
        return output
    registry.tools["netease.song_detail"].run = detail
    provider = ScriptedProvider(plan={"title": "测试歌", "requested_artist": "歌手甲", "platforms": ["qq", "netease"]})
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="获取歌手甲《测试歌》数据", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "completed"
        assert store.get(qid)["plan"]["anchor_id"] == "netease:1"
    finally:
        await model.close()


async def test_model_cannot_invent_user_requested_artist(store, registry, config):
    provider = ScriptedProvider(plan={"title": "测试歌", "requested_artist": "歌手甲", "platforms": ["qq", "netease"]})
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["report"]["model_error"] == "invalid_plan" and not store.calls(qid)
    finally:
        await model.close()


async def test_missing_named_artist_never_silently_answers_another_artist(store, registry, config):
    provider = ScriptedProvider(plan={"title": "测试歌", "requested_artist": "未出现歌手", "platforms": ["qq", "netease"]})
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="获取未出现歌手《测试歌》数据", execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "waiting_input" and "未出现歌手" in q["checkpoint"]["question"]
        assert not q["report"].get("answer")
    finally:
        await model.close()


async def test_transient_failure_feedback_changes_next_action(store, registry, config):
    old = registry.tools["qq.search_songs"].run
    attempts = 0
    async def flaky(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PlatformError("timeout", "测试超时")
        return await old(**kwargs)
    registry.tools["qq.search_songs"].run = flaky
    actions = NORMAL[:4] + [NORMAL[3]] + NORMAL[4:]
    agent, provider, model, _ = setup_agent(store, registry, config, ScriptedProvider(actions))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert attempts == 2 and store.get(qid)["execution_status"] == "completed"
        feedback = [json.loads(m["content"]) for r in provider.requests for m in r["messages"] if m["role"] == "tool"]
        assert any(f.get("error") == "timeout" for f in feedback)
    finally:
        await model.close()


async def test_auth_cannot_be_bypassed_by_changing_arguments(store, registry, config):
    async def denied(**kwargs):
        raise PlatformError("authorization_required", "需要授权")
    registry.tools["qq.search_songs"].run = denied
    actions = NORMAL[:4] + [action("qq__search_songs", keyword="测试歌 歌手甲"), action("finish_research")]
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider(actions))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert len([c for c in store.calls(qid) if c["tool"].startswith("qq.")]) == 1
        assert store.get(qid)["execution_status"] == "partial"
    finally:
        await model.close()


@pytest.mark.parametrize("bad", [action("run_shell", command="delete everything"),
    action("qq__song_detail", ids=["inventedID"]), action("qq__search_songs", keyword="其他歌"),
    action("qq__search_songs", keyword="测试歌", cookie="private")])
async def test_tool_guard_rejects_model_injections(store, registry, config, bad):
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider([bad] + NORMAL))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "completed"
        assert len(store.calls(qid)) == 6
        assert any(s["payload"].get("ok") is False for s in store.agent_steps(qid) if s["kind"] == "tool_feedback")
    finally:
        await model.close()


async def test_bad_citations_rejected_without_discarding_facts(store, registry, config):
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider(bad_answer=True))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "partial" and q["persistence_status"] == "saved"
        assert q["report"]["model_error"] == "unverified_answer" and len(store.results(qid)) == 4
        assert q["report"]["answer"]["citations"] == []
        assert [s["payload"]["reason"] for s in store.agent_steps(qid) if s["kind"] == "answer_rejected"] == ["unknown_observation"] * 2
    finally:
        await model.close()


@pytest.mark.parametrize("structured_days,expected", [(None, 30), (90, 90)])
async def test_explicit_window_not_overridden_by_model_default(store, registry, config, structured_days, expected):
    agent, _, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="查询《测试歌》近30天的数据", days=structured_days, execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["plan"]["days"] == expected
    finally:
        await model.close()


async def test_answer_only_regeneration_preserves_observations_and_budget(store, registry, config):
    agent, provider, model, active_config = setup_agent(store, registry, config, ScriptedProvider(bad_answer=True))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        before_results, before_calls = store.results(qid), store.calls(qid)
        before_usage = store.get(qid)["checkpoint"]["agent_state"]["budget_tokens"]
        provider.bad_answer = False
        with TestClient(create_app(active_config, store, registry, run_worker=False)) as client:
            assert client.post(f"/api/v1/queries/{qid}/regenerate-answer").status_code == 202
            assert client.post(f"/api/v1/queries/{qid}/regenerate-answer").status_code == 409
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "completed" and q["report"]["answer"]["status"] == "verified"
        assert "model_error" not in q["report"]
        assert store.results(qid) == before_results and store.calls(qid) == before_calls
        assert q["checkpoint"]["agent_state"]["budget_tokens"] > before_usage
        assert q["checkpoint"]["answer_only_retry"] is False
        assert any(s["kind"] == "answer_regeneration" for s in store.agent_steps(qid))
        with Session(store.engine) as session:
            assert session.scalar(select(func.count()).select_from(ReportVersion).where(ReportVersion.query_id == qid)) == 2
    finally:
        await model.close()


def test_answer_regeneration_rejects_queries_without_saved_agent_facts(store, registry, config):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    store.finish(qid, {"songs": []}, "completed")
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        assert client.post(f"/api/v1/queries/{qid}/regenerate-answer").status_code == 409


async def test_answer_regeneration_respects_active_conversation(store, registry, config):
    agent, _, model, active_config = setup_agent(store, registry, config)
    cid = store.create_conversation()
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"), conversation_id=cid)
    try:
        await agent.run(qid)
        store.create(QueryRequest(input="另一个查询", execution_mode="agent"), conversation_id=cid)
        with TestClient(create_app(active_config, store, registry, run_worker=False)) as client:
            assert client.post(f"/api/v1/queries/{qid}/regenerate-answer").status_code == 409
        assert store.get(qid)["execution_status"] == "completed"
    finally:
        await model.close()


async def test_answer_regeneration_never_resets_exhausted_budget(store, registry, config):
    agent, provider, model, active_config = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        calls, results = store.calls(qid), store.results(qid)
        usage = store.get(qid)["checkpoint"]["agent_state"]["budget_tokens"]
        previous_requests = len(provider.requests)
        store.queue_answer_regeneration(qid)
        limited = MusicAgent(store, registry, replace(active_config, agent_token_budget=usage), model)
        await limited.run(qid)
        assert store.get(qid)["report"]["model_error"] == "model_budget"
        assert len(provider.requests) == previous_requests
        assert store.calls(qid) == calls and store.results(qid) == results
    finally:
        await model.close()


async def test_answer_regeneration_does_not_hide_upstream_partial_failure(store, registry, config):
    async def denied(**kwargs):
        raise PlatformError("authorization_required", "需要授权")
    registry.tools["qq.search_songs"].run = denied
    agent, _, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "partial"
        failures, calls = q["report"]["failures"], store.calls(qid)
        store.queue_answer_regeneration(qid)
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "partial" and q["report"]["answer"]["status"] == "verified"
        assert q["report"]["failures"] == failures and store.calls(qid) == calls
    finally:
        await model.close()


async def test_budget_stops_model_loop(store, registry, config):
    agent, _, model, _ = setup_agent(store, registry, config, agent_max_rounds=1)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert len(store.calls(qid)) == 1
        assert store.get(qid)["execution_status"] == "failed"
        assert store.get(qid)["checkpoint"]["agent_state"]["stop_reason"] == "budget_exhausted"
    finally:
        await model.close()


async def test_unconfigured_agent_is_not_rule_fallback(store, registry, config):
    model = DeepSeekClient(replace(config, deepseek_key=""))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await MusicAgent(store, registry, config, model).run(qid)
        assert store.get(qid)["execution_status"] == "failed" and not store.calls(qid)
        assert store.get(qid)["report"]["model_error"] == "model_unconfigured"
    finally:
        await model.close()


async def test_conversation_memory_supplied_to_planner(store, registry, config):
    cid = store.create_conversation()
    previous, _ = store.create(QueryRequest(input="先研究测试歌"), conversation_id=cid)
    store.update(previous, plan={"title": "测试歌", "days": 7, "platforms": ["qq", "netease"]})
    store.finish(previous, {"summary": "历史研究报告"}, "completed")
    agent, provider, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="刚才那首歌，继续看评论", execution_mode="agent"), parent=previous, conversation_id=cid)
    try:
        await agent.run(qid)
        context = json.loads(provider.requests[0]["messages"][1]["content"])["conversation"]
        assert context[0]["title"] == "测试歌" and context[0]["summary"] == "历史研究报告"
        assert len(store.conversation(cid)["turns"]) == 2
    finally:
        await model.close()


async def test_open_ended_clarification_saved(store, registry, config):
    provider = ScriptedProvider(plan={"title": None, "days": 7, "platforms": ["qq", "netease"], "question": "你想查询哪首歌？"})
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="获取它的数据", execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "waiting_input" and q["checkpoint"]["question"] == "你想查询哪首歌？"
        assert not store.calls(qid)
    finally:
        await model.close()


def test_conversation_api_and_idempotency(store, registry, config):
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        cid = client.post("/api/v1/conversations").json()["conversation_id"]
        url = f"/api/v1/conversations/{cid}/messages"
        headers = {"Idempotency-Key": "conversation-turn"}
        first = client.post(url, json={"input": "测试歌"}, headers=headers)
        assert first.status_code == 202
        qid = first.json()["query_id"]
        assert store.get(qid)["request"]["execution_mode"] == "agent"
        assert client.post(url, json={"input": "测试歌"}, headers=headers).json()["query_id"] == qid
        assert client.post(url, json={"input": "另一条"}).status_code == 409
        assert client.get(f"/api/v1/conversations/{cid}").json()["turns"][0]["input"] == "测试歌"
        assert client.get("/api/v1/conversations/missing").status_code == 404
        assert client.get("/api/v1/agent/status").json()["configured"] is False


def test_followup_promotes_legacy_query_once_and_replays_idempotently(store, registry, config):
    original, _ = store.create(QueryRequest(input="测试歌"))
    store.update(original, plan={"title": "测试歌"})
    store.finish(original, {"summary": "规则报告"}, "completed")
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        headers = {"Idempotency-Key": "followup-once"}
        url = f"/api/v1/queries/{original}/follow-up"
        first = client.post(url, json={"input": "继续查看评论"}, headers=headers)
        assert first.status_code == 202
        second = client.post(url, json={"input": "继续查看评论"}, headers=headers)
        assert second.json()["query_id"] == first.json()["query_id"]
        cid = first.json()["conversation_id"]
        assert store.ensure_conversation(original) == cid
        assert len(store.conversation(cid)["turns"]) == 2


async def test_planned_clarification_cannot_request_credentials(store, registry, config):
    provider = ScriptedProvider(plan={"title": None, "question": "请提供Cookie: private", "platforms": ["qq", "netease"]})
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="查它的数据", execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "failed"
        assert store.get(qid)["report"]["model_error"] == "invalid_plan"
        assert not store.get(qid)["checkpoint"].get("question")
    finally:
        await model.close()


@pytest.mark.parametrize("status,code", [(401, "model_auth"), (403, "model_auth"), (402, "model_quota"), (429, "model_quota")])
async def test_deepseek_auth_quota_no_retry(config, status, code):
    requests = []
    def reply(request):
        requests.append(request)
        return httpx.Response(status)
    model = DeepSeekClient(replace(config, deepseek_key="test"), httpx.AsyncClient(transport=httpx.MockTransport(reply)))
    try:
        with pytest.raises(ModelError) as error:
            await model.complete([{"role": "user", "content": "test"}])
        assert error.value.code == code and len(requests) == 1
    finally:
        await model.close()


async def test_key_never_sent_to_other_host(config):
    sent = []
    model = DeepSeekClient(replace(config, deepseek_key="test", deepseek_base_url="https://evil.example"),
                          httpx.AsyncClient(transport=httpx.MockTransport(lambda r: sent.append(r))))
    try:
        with pytest.raises(ModelError) as error:
            await model.complete([])
        assert error.value.code == "unsafe_model_url" and not sent
    finally:
        await model.close()


@pytest.mark.parametrize("section", [{"kind": "current_metrics", "observation_ids": ["fake"]},
    {"kind": "snapshot_changes", "observation_ids": ["a"]}, {"kind": "versions", "observation_ids": ["a"]}])
def test_evidence_validator_rejects_unsupported_claims(section):
    report = {"songs": [{"artists": ["甲"], "platform": "qq", "url": "https://y.qq.com/", "metrics": [
        {"observation_id": "a", "status": "exact", "label": "评论", "value": 1, "collected_at": "now"}]}]}
    with pytest.raises(ValueError):
        render_answer(EvidenceAnswer.model_validate({"sections": [section]}), report)


@pytest.mark.parametrize("anchor_id", [None, "netease:1"])
async def test_durable_resume_does_not_repeat_successful_tool(store, registry, config, anchor_id):
    provider = ScriptedProvider([NORMAL[0]])
    original = provider.respond
    outage = True
    def flaky(request):
        if len(provider.requests) == 2 and outage:
            return httpx.Response(503)
        return original(request)
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(flaky))
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent", anchor_id=anchor_id))
    try:
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "failed" and len(store.calls(qid)) == 1
        outage = False
        provider.actions = list(NORMAL[1:])
        store.update(qid, status="queued")
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "completed" and len(store.calls(qid)) == 6
        assert len([s for s in store.agent_steps(qid) if s["kind"] == "plan"]) == 1
    finally:
        await model.close()


async def test_user_selection_resumes_without_repeating_acquisition(store, registry, config):
    detail = registry.tools["netease.song_detail"].run
    async def ambiguous(**kwargs):
        data = await detail(**kwargs)
        data["songs"][1]["origin_status"] = "platform_original"
        return data
    registry.tools["netease.song_detail"].run = ambiguous
    agent, _, model, _ = setup_agent(store, registry, config)
    qid, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"))
    try:
        await agent.run(qid)
        q = store.get(qid)
        assert q["execution_status"] == "waiting_input" and len(q["checkpoint"]["choices"]) == 2
        assert not store.results(qid)
        store.update(qid, plan=q["plan"] | {"anchor_id": "netease:1"}, status="queued")
        await agent.run(qid)
        assert store.get(qid)["execution_status"] == "completed" and len(store.calls(qid)) == 6
    finally:
        await model.close()


async def test_model_cannot_expand_selected_platforms(store, registry, config):
    agent, _, model, _ = setup_agent(store, registry, config, ScriptedProvider(plan={"title": "测试歌", "platforms": ["qq"]}))
    qid, _ = store.create(QueryRequest(input="测试歌", platforms=["netease"], execution_mode="agent"))
    try:
        await agent.run(qid)
        assert store.get(qid)["report"]["model_error"] == "invalid_plan" and not store.calls(qid)
    finally:
        await model.close()


def test_unconfigured_connection_check_saved_as_failure(store, registry, config):
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        result = client.post("/api/v1/agent/check").json()
        assert result["connected"] is False
        assert store.get(result["query_id"])["execution_status"] == "failed"


async def test_clarification_reply_uses_only_user_named_existing_choice(store, registry, config):
    cid = store.create_conversation()
    prior, _ = store.create(QueryRequest(input="测试歌", execution_mode="agent"), conversation_id=cid)
    store.update(prior, status="waiting_input", plan={"title": "测试歌"}, checkpoint={"choices": [
        {"key": "netease:1", "artists": ["歌手甲"], "title": "测试歌", "album": "测试", "url": "https://music.163.com/song?id=1"}]})
    provider = ScriptedProvider(plan={"title": "测试歌", "days": 7, "platforms": ["qq", "netease"], "anchor_id": "netease:1"})
    agent, _, model, _ = setup_agent(store, registry, config, provider)
    qid, _ = store.create(QueryRequest(input="我指歌手甲", execution_mode="agent"), conversation_id=cid, parent=prior)
    try:
        await agent.run(qid)
        assert store.get(qid)["plan"]["anchor_id"] == "netease:1"
        assert store.get(qid)["execution_status"] == "completed"
    finally:
        await model.close()
