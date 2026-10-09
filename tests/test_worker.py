import asyncio

import pytest
from sqlalchemy.exc import OperationalError

from musictracker.schemas import QueryRequest, Song
from musictracker.worker import Worker


@pytest.mark.parametrize("has_results,expected", [(False, "failed"), (True, "partial")])
async def test_database_write_error_is_not_saved(store, registry, config, monkeypatch, has_results, expected):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    if has_results:
        store.persist_songs(qid, [Song(platform="netease", platform_id="1", numeric_id="1",
            title="测试歌", artists=["歌手甲"], url="https://music.163.com/song?id=1")])
    class BrokenWorkflow:
        async def run(self, query_id):
            raise OperationalError("insert", {}, RuntimeError("test storage failure"))
    done = asyncio.Event()
    original_event = store.event
    def observe_event(query_id, kind, payload):
        original_event(query_id, kind, payload)
        if kind == "finished":
            done.set()
    monkeypatch.setattr(store, "event", observe_event)
    task = asyncio.create_task(Worker(store, BrokenWorkflow(), config).loop())
    try:
        await asyncio.wait_for(done.wait(), timeout=2)
        q = store.get(qid)
        assert q["execution_status"] == "failed" and q["persistence_status"] == expected
        assert "未确认保存" in q["error"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
