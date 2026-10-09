from dataclasses import replace

from fastapi.testclient import TestClient

from musictracker.api import create_app
from musictracker.schemas import Metric, QueryRequest, Song


def test_api_query_history_export(store, registry, config):
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        r = client.post("/api/v1/queries", json={"input": "测试歌"}, headers={"Idempotency-Key": "first"})
        assert r.status_code == 202
        qid = r.json()["query_id"]
        assert client.post("/api/v1/queries", json={"input": "测试歌"}, headers={"Idempotency-Key": "first"}).json()["query_id"] == qid
        assert client.post("/api/v1/queries", json={"input": "另一首"}, headers={"Idempotency-Key": "first"}).status_code == 409
        assert len(client.get("/api/v1/queries").json()["queries"]) == 1
        assert client.get(f"/api/v1/queries/{qid}/export?format=json").status_code == 200
        assert "results" in client.get(f"/api/v1/queries/{qid}/export?format=json").json()
        assert client.get("/api/v1/queries/no-such-id").status_code == 404
        assert len(client.get("/api/v1/tools").json()["tools"]) == 6


def test_api_origin_boundary(store, registry, config):
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        assert client.post("/api/v1/queries", json={"input": "测试歌"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert store.stats()["queries"] == 0


def test_api_token_boundary(store, registry, config):
    with TestClient(create_app(replace(config, api_token="test-only-token"), store, registry, run_worker=False)) as client:
        assert client.get("/api/v1/queries").status_code == 401
        assert client.get("/api/v1/queries", headers={"Authorization": "Bearer test-only-token"}).status_code == 200


def test_csv_formula_injection(store, registry, config):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    store.persist_songs(qid, [Song(platform="netease", platform_id="1", numeric_id="1", title="=danger()", artists=["+artist"], url="https://music.163.com/song?id=1",
        metrics=[Metric(name="comment_count", label="评论", status="exact", value=10, source_url="https://music.163.com/song?id=1")])])
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        text = client.get(f"/api/v1/queries/{qid}/export?format=csv").text
        assert "'=danger()" in text and "'+artist" in text


def test_cancel_and_sse_replay(store, registry, config):
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        qid = client.post("/api/v1/queries", json={"input": "测试歌"}).json()["query_id"]
        assert client.post(f"/api/v1/queries/{qid}/cancel").json()["execution_status"] == "cancelled"
        events = client.get(f"/api/v1/queries/{qid}/events").text
        assert "accepted" in events and "stream_end" in events
        assert client.get(f"/api/v1/queries/{qid}/events", headers={"Last-Event-ID": "broken"}).status_code == 422


def test_tool_diagnostic_saved(store, registry, config):
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        result = client.post("/api/v1/tools/invoke", json={"name": "netease.search_songs", "arguments": {"keyword": "测试歌"}}).json()
        assert result["persistence_status"] == "saved"
        q = store.get(result["query_id"])
        assert q["report"]["output"]["songs"] and len(store.calls(q["query_id"])) == 1
        assert client.post("/api/v1/tools/invoke", json={"name": "bad", "arguments": {}}).status_code == 404


def test_retry_event_cursor_does_not_replay_old_completion(store, registry, config):
    qid, _ = store.create(QueryRequest(input="测试歌"))
    store.finish(qid, {"failures": {"qq": {"code": "timeout"}}}, "failed")
    with TestClient(create_app(config, store, registry, run_worker=False)) as client:
        assert client.post(f"/api/v1/queries/{qid}/retry").status_code == 202
        q = client.get(f"/api/v1/queries/{qid}").json()
        assert q["execution_status"] == "queued"
        assert not store.events(qid, q["last_event_id"])
        store.event(qid, "new_attempt", {"message": "新的尝试"})
        assert [e["type"] for e in store.events(qid, q["last_event_id"])] == ["new_attempt"]
