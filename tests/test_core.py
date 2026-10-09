from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from musictracker.models import ReportVersion
from musictracker.schemas import DetailInput, Metric, QueryRequest, Song, now
from musictracker.transport import PlatformError
from musictracker.workflow import Cancelled, OverviewWorkflow, base_title, rule_intent


@pytest.mark.parametrize("text,title,days", [
    ("我希望获取《空心》的近期数据", "空心", 7),
    ("获取《七里香》近30天数据", "七里香", 30),
    ("空心", "空心", 7),
    ("查询空心的近期数据", "空心", 7),
    ("请帮我查询“测试歌”的近一个月数据", "测试歌", 30),
])
def test_intent(text, title, days):
    plan = rule_intent(QueryRequest(input=text))
    assert plan["title"] == title and plan["days"] == days
    assert plan["window"]["timezone"] == "Asia/Shanghai"


@pytest.mark.parametrize("input", ["", " ", "Cookie: private", "Authorization: secret", "sk-1234567890123"])
def test_credentials_or_empty_rejected(input):
    with pytest.raises(ValidationError):
        QueryRequest(input=input)


def test_cache_requires_age():
    with pytest.raises(ValidationError):
        QueryRequest(input="空心", mode="prefer_cache")


@pytest.mark.parametrize("sid", ["http://127.0.0.1", "../private", "a b", "x?cookie=1"])
def test_tool_ids_are_not_urls(sid):
    with pytest.raises(ValidationError):
        DetailInput(ids=[sid])


@pytest.mark.parametrize("kwargs", [
    {"status": "unavailable", "value": 0, "reason": "missing"},
    {"status": "error", "value": None},
    {"status": "approximate", "value": 1000001, "display": "100w+"},
    {"status": "exact", "value": -1},
])
def test_invalid_metrics_rejected(kwargs):
    with pytest.raises(ValidationError):
        Metric(name="comment_count", label="评论", source_url="https://music.163.com/", **kwargs)


def test_zero_is_valid():
    assert Metric(name="comment_count", label="评论", status="exact", value=0, source_url="https://music.163.com/").value == 0


def test_title_matching_is_not_substring():
    assert base_title("空心 (Live)") == base_title("空心")
    assert base_title("空心之城") != base_title("空心")


def test_idempotency(store):
    req = QueryRequest(input="测试歌")
    q1, created = store.create(req, "same")
    q2, created2 = store.create(req, "same")
    assert q1 == q2 and created and not created2
    with pytest.raises(ValueError):
        store.create(QueryRequest(input="另一首"), "same")


def test_durable_lease(store):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    assert store.claim("one", 120) == qid
    assert store.claim("two", 120) is None
    store.update(qid, lease_until=now() - timedelta(seconds=1))
    assert store.claim("two", 120) == qid


async def test_empty_database_full_query(store, registry, config):
    qid, _ = store.create(QueryRequest(input="获取《测试歌》的近期数据"))
    await OverviewWorkflow(store, registry, config).run(qid)
    q = store.get(qid)
    assert q["execution_status"] == "completed"
    assert q["persistence_status"] == "saved"
    assert len(q["report"]["groups"]) == 2
    assert store.stats()["entities"] == 4 and store.stats()["observations"] == 8
    assert q["report"]["original"]["artists"] == ["歌手甲"]
    assert all(c["observation_id"] for c in q["report"]["claims"])
    assert all(m["source_url"] == song["url"] for song in store.results(qid) for m in song["metrics"])
    assert any("历史不足" in limitation for limitation in q["report"]["limitations"])


async def test_one_platform_failure_is_saved(store, registry, config):
    async def failed(**kwargs):
        raise PlatformError("authorization_required", "需要授权")
    registry.tools["qq.search_songs"].run = failed
    qid, _ = store.create(QueryRequest(input="测试歌"))
    await OverviewWorkflow(store, registry, config).run(qid)
    q = store.get(qid)
    assert q["execution_status"] == "partial" and q["persistence_status"] == "saved"
    assert len(store.results(qid)) == 2
    assert q["report"]["failures"]["qq"]["code"] == "authorization_required"


async def test_retry_only_failed_platform(store, registry, config):
    original_search = registry.tools["qq.search_songs"].run
    async def failed(**kwargs):
        raise PlatformError("timeout", "超时")
    registry.tools["qq.search_songs"].run = failed
    qid, _ = store.create(QueryRequest(input="测试歌"))
    workflow = OverviewWorkflow(store, registry, config)
    await workflow.run(qid)
    old = store.results(qid)
    previous_netease_calls = [c for c in store.calls(qid) if c["tool"].startswith("netease.")]
    registry.tools["qq.search_songs"].run = original_search
    store.update(qid, status="queued")
    await workflow.run(qid)
    assert store.get(qid)["execution_status"] == "completed"
    assert [s for s in store.results(qid) if s["platform"] == "netease"] == old
    assert [c for c in store.calls(qid) if c["tool"].startswith("netease.")] == previous_netease_calls
    assert store.stats()["observations"] == 8


async def test_empty_search_is_not_platform_failure(store, registry, config):
    async def empty(**kwargs):
        return {"songs": [], "coverage": {"method": "mock", "complete": False}}
    for platform in ["qq", "netease"]:
        registry.tools[platform + ".search_songs"].run = empty
    qid, _ = store.create(QueryRequest(input="测试歌"))
    await OverviewWorkflow(store, registry, config).run(qid)
    q = store.get(qid)
    assert q["execution_status"] == "completed" and q["report"]["songs"] == []
    assert not q["report"]["failures"] and store.stats()["observations"] == 0


async def test_both_fail_history_retained(store, registry, config):
    async def failed(**kwargs):
        raise PlatformError("timeout", "超时")
    for p in ["qq", "netease"]:
        registry.tools[p + ".search_songs"].run = failed
    qid, _ = store.create(QueryRequest(input="测试歌"))
    await OverviewWorkflow(store, registry, config).run(qid)
    assert store.get(qid)["execution_status"] == "failed"
    assert len(store.calls(qid)) == 2 and store.stats()["observations"] == 0


async def test_cache_reuses_observation_ids(store, registry, config):
    workflow = OverviewWorkflow(store, registry, config)
    q1, _ = store.create(QueryRequest(input="测试歌"))
    await workflow.run(q1)
    before = store.stats()["observations"]
    q2, _ = store.create(QueryRequest(input="测试歌", mode="prefer_cache", max_age_seconds=3600))
    await workflow.run(q2)
    assert store.stats()["observations"] == before
    assert store.get(q2)["report"]["origin"] == "cache"
    assert store.get(q2)["report"]["reused_from_query_id"] == q1
    assert store.calls(q2) == []
    assert store.results(q1)[0]["metrics"][0]["observation_id"] == store.results(q2)[0]["metrics"][0]["observation_id"]


async def test_cache_only_miss_no_network(store, registry, config):
    qid, _ = store.create(QueryRequest(input="测试歌", mode="cache_only"))
    await OverviewWorkflow(store, registry, config).run(qid)
    assert store.get(qid)["report"]["songs"] == [] and not store.calls(qid)


async def test_live_refresh_new_facts_and_comparison(store, registry, config):
    workflow = OverviewWorkflow(store, registry, config)
    req = QueryRequest(input="测试歌")
    q1, _ = store.create(req)
    await workflow.run(q1)
    old = store.results(q1)
    q2, _ = store.create(req, parent=q1)
    await workflow.run(q2)
    assert store.stats()["observations"] == 16
    new = store.results(q2)
    assert old[0]["metrics"][0]["observation_id"] != new[0]["metrics"][0]["observation_id"]
    assert new[0]["metrics"][0]["comparison"]["label"] == "相对上次有效观测变化"
    assert store.results(q1) == old


async def test_replay_preserves_committed_facts(store, registry, config):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    workflow = OverviewWorkflow(store, registry, config)
    await workflow.run(qid)
    old = store.results(qid)
    await workflow.run(qid)
    assert store.stats()["observations"] == 8 and store.results(qid) == old
    with Session(store.engine) as s:
        assert s.scalar(select(func.count()).select_from(ReportVersion)) == 2


async def test_cancel_before_tools(store, registry, config):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    store.update(qid, status="cancelled")
    with pytest.raises(Cancelled):
        await OverviewWorkflow(store, registry, config).run(qid)
    assert store.get(qid)["execution_status"] == "cancelled" and not store.calls(qid)


async def test_ambiguity_needs_selection(store, registry, config):
    async def detail(ids):
        return {"songs": [Song(platform="netease", platform_id=str(i), numeric_id=str(i), title="测试歌",
            artists=["不同作者" + str(i)], url="https://music.163.com/song?id=" + str(i),
            origin_status="platform_original").model_dump() for i in [1, 2]]}
    registry.tools["netease.song_detail"].run = detail
    qid, _ = store.create(QueryRequest(input="测试歌", platforms=["netease"]))
    await OverviewWorkflow(store, registry, config).run(qid)
    q = store.get(qid)
    assert q["execution_status"] == "waiting_input" and len(q["checkpoint"]["choices"]) == 2
    assert store.stats()["observations"] == 0


def test_metadata_snapshot_immutable(store):
    q1, _ = store.create(QueryRequest(input="测试歌"))
    song = Song(platform="netease", platform_id="1", numeric_id="1", title="旧标题", artists=["旧歌手"], url="https://music.163.com/song?id=1")
    store.persist_songs(q1, [song])
    q2, _ = store.create(QueryRequest(input="测试歌"))
    song.title = "新标题"
    store.persist_songs(q2, [song])
    assert store.results(q1)[0]["title"] == "旧标题"
    assert store.library()[0]["title"] == "新标题"
