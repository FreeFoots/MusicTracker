import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    AgentStep,
    Conversation,
    Entity,
    Event,
    Observation,
    Query,
    QueryResult,
    ReportVersion,
    ToolCall,
    uid,
)
from .schemas import QueryRequest, Song, now

TERMINAL = {"completed", "partial", "failed", "cancelled", "waiting_input"}


def utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


class Store:
    def __init__(self, database_url: str):
        if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"connect_args": {"check_same_thread": False}} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, pool_pre_ping=True, **kwargs)

    def initialize(self):
        from alembic.config import Config

        from alembic import command

        root = Path(__file__).resolve().parent.parent
        config = Config(str(root / "alembic.ini"))
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    def create(self, request: QueryRequest, key: str | None = None, parent: str | None = None,
               conversation_id: str | None = None) -> tuple[str, bool]:
        payload = request.model_dump()
        digest = hashlib.sha256(json.dumps({"request": payload, "conversation_id": conversation_id}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        def matches(prior):
            if prior.request_hash == digest:
                return True
            # Preserve pre-v0.6 idempotency keys without rewriting immutable historical requests.
            legacy = hashlib.sha256(json.dumps(prior.request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            return (conversation_id is None and prior.conversation_id is None and prior.request_hash == legacy
                    and QueryRequest.model_validate(prior.request).model_dump() == payload)
        with Session(self.engine) as s:
            if key:
                prior = s.scalar(select(Query).where(Query.idempotency_key == key))
                if prior:
                    if not matches(prior):
                        raise ValueError("同一幂等键对应不同请求")
                    return prior.id, False
            if conversation_id:
                conversation = s.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
                if conversation is None:
                    raise ValueError("会话不存在")
                active = s.scalar(select(Query).where(Query.conversation_id == conversation_id, Query.status.not_in(TERMINAL)).limit(1))
                if active:
                    raise ValueError("同一会话有未结束查询，请等待或取消后追问")
                if conversation.title == "音乐研究会话":
                    conversation.title = request.input[:100]
            query = Query(input=request.input, request=payload, request_hash=digest, idempotency_key=key,
                          parent_id=parent, conversation_id=conversation_id)
            s.add(query)
            try:
                s.flush()
                s.add(Event(query_id=query.id, type="accepted", payload={"message": "查询已创建并保存"}))
                s.commit()
                return query.id, True
            except IntegrityError:
                s.rollback()
                if not key:
                    raise
                prior = s.scalar(select(Query).where(Query.idempotency_key == key))
                if prior is None or not matches(prior):
                    raise ValueError("幂等请求冲突") from None
                return prior.id, False

    def get(self, qid: str) -> dict | None:
        with Session(self.engine) as s:
            q = s.get(Query, qid)
            if not q:
                return None
            return {"query_id": q.id, "input": q.input, "request": q.request, "parent_id": q.parent_id,
                    "conversation_id": q.conversation_id,
                    "execution_status": q.status, "persistence_status": q.persistence_status,
                    "plan": q.plan, "checkpoint": q.checkpoint, "report": q.report,
                    "created_at": utc(q.created_at).isoformat(),
                    "finished_at": utc(q.finished_at).isoformat() if q.finished_at else None, "error": q.error}

    def history(self, limit: int = 50) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.scalars(select(Query).order_by(Query.created_at.desc()).limit(limit)).all()
            return [{"query_id": q.id, "input": q.input, "execution_status": q.status,
                     "created_at": utc(q.created_at).isoformat(), "persistence_status": q.persistence_status,
                     "title": q.plan.get("title"), "result_count": len(q.report.get("songs", []))} for q in rows]

    def event(self, qid: str, kind: str, payload: dict):
        with Session(self.engine) as s, s.begin():
            s.add(Event(query_id=qid, type=kind, payload=payload))

    def events(self, qid: str, after: int = 0) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.scalars(select(Event).where(Event.query_id == qid, Event.id > after).order_by(Event.id)).all()
            return [{"event_id": e.id, "type": e.type, "payload": e.payload,
                     "created_at": utc(e.created_at).isoformat()} for e in rows]

    def event_cursor(self, qid: str) -> int:
        with Session(self.engine) as s:
            return s.scalar(select(func.max(Event.id)).where(Event.query_id == qid)) or 0

    def queue_answer_regeneration(self, qid: str):
        with Session(self.engine) as s, s.begin():
            cid = s.scalar(select(Query.conversation_id).where(Query.id == qid))
            if cid:
                s.scalar(select(Conversation).where(Conversation.id == cid).with_for_update())
            q = s.scalar(select(Query).where(Query.id == qid).with_for_update())
            if (q is None or q.status not in {"completed", "partial", "failed"}
                    or q.request.get("execution_mode") != "agent"
                    or not q.report.get("songs") or not q.checkpoint.get("agent_state")):
                raise ValueError("只可为已保存音乐事实的Agent查询重新生成回答")
            if cid and s.scalar(select(Query.id).where(Query.conversation_id == cid,
                    Query.id != qid, Query.status.not_in(TERMINAL)).limit(1)):
                raise ValueError("同一会话有未结束查询，请等待或取消后重新生成回答")
            q.status, q.lease_owner, q.lease_until, q.error = "queued", None, None, None
            q.checkpoint = q.checkpoint | {"answer_only_retry": True}
            s.add(Event(query_id=qid, type="answer_regeneration", payload={
                "message": "已入队，仅使用已保存证据重新生成回答，不重新请求音乐平台"}))

    def create_conversation(self, title: str = "音乐研究会话") -> str:
        with Session(self.engine) as s, s.begin():
            conversation = Conversation(title=title[:100])
            s.add(conversation)
            s.flush()
            return conversation.id

    def ensure_conversation(self, qid: str) -> str:
        with Session(self.engine) as s, s.begin():
            query = s.scalar(select(Query).where(Query.id == qid).with_for_update())
            if query is None:
                raise ValueError("查询不存在")
            if query.conversation_id:
                return query.conversation_id
            conversation = Conversation(title=(query.plan.get("title") or "音乐研究会话")[:100])
            s.add(conversation)
            s.flush()
            query.conversation_id = conversation.id
            return conversation.id

    def conversation(self, cid: str) -> dict | None:
        with Session(self.engine) as s:
            conversation = s.get(Conversation, cid)
            if not conversation:
                return None
            queries = s.scalars(select(Query).where(Query.conversation_id == cid)
                .order_by(Query.created_at.desc()).limit(20)).all()
            return {"conversation_id": cid, "title": conversation.title, "turns": [
                {"query_id": q.id, "input": q.input, "execution_status": q.status, "plan": q.plan,
                 "report": q.report, "question": q.checkpoint.get("question"),
                 "choices": q.checkpoint.get("choices", [])} for q in reversed(queries)]}

    def conversations(self) -> list[dict]:
        with Session(self.engine) as s:
            return [{"conversation_id": c.id, "title": c.title} for c in s.scalars(
                select(Conversation).order_by(Conversation.created_at.desc()).limit(50))]

    def agent_step(self, qid: str, kind: str, payload: dict):
        with Session(self.engine) as s, s.begin():
            s.add(AgentStep(query_id=qid, kind=kind, payload=payload))

    def agent_steps(self, qid: str) -> list[dict]:
        with Session(self.engine) as s:
            return [{"step_id": a.id, "kind": a.kind, "payload": a.payload,
                     "created_at": utc(a.created_at).isoformat()} for a in s.scalars(
                select(AgentStep).where(AgentStep.query_id == qid).order_by(AgentStep.id))]

    def update(self, qid: str, **values):
        with Session(self.engine) as s, s.begin():
            q = s.get(Query, qid)
            if q is None:
                raise KeyError(qid)
            # Cancellation always wins over a late-running node.
            if q.status == "cancelled" and values.get("status") not in {None, "cancelled"}:
                values.pop("status", None)
            for key, value in values.items():
                setattr(q, key, value)

    def audit(self, qid: str, tool: str, arguments: dict, status: str, error: str | None, duration: int):
        with Session(self.engine) as s, s.begin():
            s.add(ToolCall(query_id=qid, tool=tool, arguments=arguments, status=status, error_code=error, duration_ms=duration))

    def calls(self, qid: str) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.scalars(select(ToolCall).where(ToolCall.query_id == qid).order_by(ToolCall.created_at)).all()
            return [{"id": r.id, "tool": r.tool, "arguments": r.arguments, "status": r.status,
                     "error_code": r.error_code, "duration_ms": r.duration_ms} for r in rows]

    def claim(self, owner: str, lease_seconds: int) -> str | None:
        with Session(self.engine) as s, s.begin():
            query = s.scalar(select(Query).where(or_(Query.status == "queued",
                (Query.status.not_in(TERMINAL)) & (Query.lease_until < now())))
                .order_by(Query.created_at).with_for_update(skip_locked=True).limit(1))
            if query is None:
                return None
            query.status, query.lease_owner = "resolving", owner
            query.lease_until = now() + timedelta(seconds=lease_seconds)
            return query.id

    def heartbeat(self, qid: str, owner: str, seconds: int):
        with Session(self.engine) as s, s.begin():
            q = s.get(Query, qid)
            if q and q.lease_owner == owner and q.status not in TERMINAL:
                q.lease_until = now() + timedelta(seconds=seconds)

    def persist_songs(self, qid: str, songs: list[Song]) -> list[dict]:
        """Per-platform atomic commit. Existing committed facts survive replay unchanged."""
        out = []
        with Session(self.engine) as s, s.begin():
            for song in songs:
                entity = s.scalar(select(Entity).where(Entity.platform == song.platform, Entity.platform_id == song.platform_id))
                metadata = song.model_dump(exclude={"metrics", "entity_id"})
                if entity is None:
                    entity = Entity(platform=song.platform, platform_id=song.platform_id, title=song.title,
                                    artists=song.artists, metadata_json=metadata)
                    s.add(entity)
                    s.flush()
                result = s.scalar(select(QueryResult).where(QueryResult.query_id == qid, QueryResult.entity_id == entity.id))
                if result:
                    out.append(result.snapshot)
                    continue
                entity.title, entity.artists, entity.metadata_json, entity.updated_at = song.title, song.artists, metadata, now()
                song.entity_id = entity.id
                attempt = s.scalar(select(ToolCall).where(ToolCall.query_id == qid,
                    ToolCall.tool == f"{song.platform}.song_metrics").order_by(ToolCall.created_at.desc()).limit(1))
                for metric in song.metrics:
                    metric.observation_id = uid()
                    collected = datetime.fromisoformat(metric.collected_at)
                    if metric.status == "exact":
                        old = s.scalar(select(Observation).where(Observation.entity_id == entity.id,
                            Observation.metric == metric.name, Observation.status == "exact",
                            Observation.collected_at < collected, Observation.query_id != qid)
                            .order_by(Observation.collected_at.desc()).limit(1))
                        if old and old.data.get("scope") == metric.scope:
                            metric.comparison = {"delta": metric.value - old.value,
                                "from": utc(old.collected_at).isoformat(), "to": collected.isoformat(),
                                "label": "相对上次有效观测变化", "observation_id": old.id}
                    s.add(Observation(id=metric.observation_id, query_id=qid, entity_id=entity.id,
                        attempt_id=attempt.id if attempt else None, metric=metric.name, value=metric.value,
                        status=metric.status, data=metric.model_dump(), collected_at=collected))
                snapshot = song.model_dump()
                s.add(QueryResult(query_id=qid, entity_id=entity.id, snapshot=snapshot))
                out.append(snapshot)
            s.add(Event(query_id=qid, type="platform_saved", payload={"platform": songs[0].platform if songs else None,
                                                                       "count": len(out), "message": "平台结果已提交"}))
        return out

    def results(self, qid: str) -> list[dict]:
        with Session(self.engine) as s:
            return [r.snapshot | {"origin": r.origin} for r in s.scalars(select(QueryResult).where(QueryResult.query_id == qid)).all()]

    def finish(self, qid: str, report: dict, status: str):
        with Session(self.engine) as s, s.begin():
            q = s.get(Query, qid)
            if q.status == "cancelled":
                status = "cancelled"
            q.report, q.status, q.persistence_status, q.finished_at = report, status, "saved", now()
            q.lease_owner, q.lease_until = None, None
            s.add(ReportVersion(query_id=qid, report=report))
            s.add(Event(query_id=qid, type="finished", payload={"status": status, "persistence_status": "saved"}))

    def find_cache(self, qid: str, plan: dict, max_age: int | None) -> str | None:
        with Session(self.engine) as s:
            rows = s.scalars(select(Query).where(Query.id != qid, Query.status == "completed")
                             .order_by(Query.created_at.desc()).limit(100)).all()
            for q in rows:
                p = q.plan
                if any(p.get(k) != plan.get(k) for k in ["title", "days", "platforms", "max_versions", "anchor_id", "version"]):
                    continue
                if not q.report.get("songs") or q.report.get("origin") == "cache":
                    continue
                fetched = min(datetime.fromisoformat(song["fetched_at"]) for song in q.report["songs"])
                if max_age is None or (now() - utc(fetched)).total_seconds() <= max_age:
                    return q.id
            return None

    def reuse_cache(self, qid: str, source: str, plan: dict, finalize: bool = True) -> dict:
        with Session(self.engine) as s, s.begin():
            old = s.get(Query, source)
            report = dict(old.report) | {"origin": "cache", "reused_from_query_id": source, "query_id": qid,
                                       "window": plan["window"], "cache_note": "复用历史观测，非本次实时采集"}
            rows = s.scalars(select(QueryResult).where(QueryResult.query_id == source)).all()
            for r in rows:
                s.add(QueryResult(query_id=qid, entity_id=r.entity_id, snapshot=r.snapshot,
                                  origin="cache", reused_from_query_id=source))
        if finalize:
            self.finish(qid, report, "completed")
        return report

    def library(self) -> list[dict]:
        with Session(self.engine) as s:
            return [e.metadata_json | {"entity_id": e.id} for e in s.scalars(select(Entity).order_by(Entity.updated_at.desc()).limit(100))]

    def stats(self) -> dict:
        with Session(self.engine) as s:
            return {"queries": s.scalar(select(func.count()).select_from(Query)),
                    "entities": s.scalar(select(func.count()).select_from(Entity)),
                    "observations": s.scalar(select(func.count()).select_from(Observation)),
                    "database": self.engine.dialect.name}
