import asyncio
import csv
import io
import json
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .agent import MusicAgent, WorkflowRouter
from .config import Settings, settings
from .deepseek import DeepSeekClient, ModelError
from .schemas import QueryRequest, ToolInvocation, now
from .store import TERMINAL, Store
from .tools import ToolRegistry
from .transport import PlatformError, Transport
from .worker import Worker
from .workflow import OverviewWorkflow

ALLOWED_ORIGINS = {"http://localhost:4321", "http://127.0.0.1:4321"}


def create_app(config: Settings = settings, store: Store | None = None,
               tools: ToolRegistry | None = None, run_worker: bool = True, model: DeepSeekClient | None = None):
    database = store or Store(config.database_url)
    transport = None if tools else Transport(config)
    registry = tools or ToolRegistry(transport)
    deepseek = model or DeepSeekClient(config)
    rules = OverviewWorkflow(database, registry, config)
    workflow = WorkflowRouter(rules, MusicAgent(database, registry, config, deepseek))
    worker = Worker(database, workflow, config)

    @asynccontextmanager
    async def lifespan(app):
        database.initialize()
        task = asyncio.create_task(worker.loop()) if run_worker else None
        yield
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if transport:
            await transport.close()
        await deepseek.close()
        database.engine.dispose()

    app = FastAPI(title="MusicTracker", version=__version__, lifespan=lifespan)
    app.state.store, app.state.registry, app.state.workflow = database, registry, workflow
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.add_middleware(CORSMiddleware, allow_origins=list(ALLOWED_ORIGINS), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Idempotency-Key", "Authorization", "Last-Event-ID"])

    @app.middleware("http")
    async def security(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and origin not in ALLOWED_ORIGINS:
            return Response("Untrusted origin", status_code=403)
        if config.api_token and request.url.path not in {"/health"}:
            if not secrets.compare_digest(request.headers.get("authorization", ""), "Bearer " + config.api_token):
                return Response("Unauthorized", status_code=401)
        try:
            return await call_next(request)
        except SQLAlchemyError:
            return Response(json.dumps({"detail": "数据库暂不可用，未确认保存；请稍后重试"}), status_code=503, media_type="application/json")

    def get_query(qid):
        query = database.get(qid)
        if not query:
            raise HTTPException(404, "查询不存在")
        return query

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "scope": "single_user_local", **database.stats()}

    @app.get("/api/v1/platforms")
    def platforms():
        return {"platforms": [
            {"id": "qq", "name": "QQ 音乐", "search": "公开搜索建议（有限）", "verified_at": "2026-10-08",
             "metrics": {"comment_count": "exact", "favorite_count": "approximate", "play_count": "unavailable"}},
            {"id": "netease", "name": "网易云音乐", "search": "公开歌曲搜索", "verified_at": "2026-10-08",
             "metrics": {"comment_count": "exact", "favorite_count": "exact", "play_count": "unavailable"}},
        ], "note": "本机低频验证，不保证所有对象、地区或未来请求可用；商用需另行核验平台许可"}

    @app.get("/api/v1/tools")
    def tool_manifest():
        return {"tools": registry.manifest()}

    @app.get("/api/v1/agent/status")
    def agent_status():
        return {"provider": "deepseek", "configured": deepseek.configured, "model": config.deepseek_model,
                "max_rounds": config.agent_max_rounds, "max_tools": config.agent_max_tools,
                "token_budget": config.agent_token_budget, "deadline_seconds": config.agent_timeout,
                "note": "configured只表示本机已配置，不代表Key有效或真实模型评测通过",
                "privacy": "Agent模式会把本次查询、最近会话及公开工具结果发送给DeepSeek；Key只在服务器环境变量"}

    @app.post("/api/v1/conversations", status_code=201)
    def new_conversation():
        return {"conversation_id": database.create_conversation()}

    @app.post("/api/v1/agent/check")
    async def check_model():
        qid, _ = database.create(QueryRequest(input="DeepSeek 连接诊断", title="模型连接诊断"))
        database.update(qid, status="fetching", plan={"intent": "tool_diagnostic", "provider": "deepseek"})
        try:
            result = await deepseek.complete([{"role": "system", "content": '只输出JSON {"connected":true}，不调用工具。'},
                {"role": "user", "content": "MusicTracker模型连接测试"}], json_mode=True)
            if json.loads(result["message"]["content"]).get("connected") is not True:
                raise ModelError("model_schema", "连接响应未通过JSON契约校验")
            database.agent_step(qid, "connection_check", {"model": result["model"], "usage": result["usage"]})
            database.finish(qid, {"summary": "DeepSeek真实连接测试成功；不代表音乐任务评测通过", "model": result["model"]}, "completed")
            return {"query_id": qid, "connected": True, "model": result["model"], "usage": result["usage"]}
        except (ModelError, ValueError, AttributeError) as exc:
            message = exc.message if isinstance(exc, ModelError) else "模型连接响应无效"
            code = exc.code if isinstance(exc, ModelError) else "model_schema"
            database.finish(qid, {"summary": message, "model_error": code}, "failed")
            return {"query_id": qid, "connected": False, "error": message}

    @app.get("/api/v1/conversations")
    def conversations():
        return {"conversations": database.conversations()}

    @app.get("/api/v1/conversations/{cid}")
    def conversation(cid: str):
        result = database.conversation(cid)
        if result is None:
            raise HTTPException(404, "会话不存在")
        return result

    @app.post("/api/v1/conversations/{cid}/messages", status_code=202)
    def conversation_message(cid: str, body: QueryRequest, idempotency_key: str | None = Header(default=None)):
        convo = conversation(cid)
        if idempotency_key and len(idempotency_key) > 128:
            raise HTTPException(422, "幂等键超过长度上限")
        request = QueryRequest.model_validate(body.model_dump() | {"execution_mode": "agent"})
        try:
            qid, created = database.create(request, idempotency_key,
                parent=convo["turns"][-1]["query_id"] if convo["turns"] else None, conversation_id=cid)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"query_id": qid, "conversation_id": cid, "created": created}

    @app.post("/api/v1/tools/invoke")
    async def invoke_tool(invocation: ToolInvocation):
        if invocation.name not in registry.tools:
            raise HTTPException(404, "工具未注册")
        try:
            validated = registry.tools[invocation.name].schema.model_validate(invocation.arguments).model_dump()
        except ValueError:
            raise HTTPException(422, "工具参数不符合注册契约") from None
        qid, _ = database.create(QueryRequest(input="工具诊断：" + invocation.name, title="工具诊断", platforms=[invocation.name.split(".")[0]]))
        database.update(qid, status="fetching", plan={"intent": "tool_diagnostic"})
        try:
            result = await registry.invoke(invocation.name, validated, lambda *args: database.audit(qid, *args))
            report = {"tool": invocation.name, "arguments": validated, "output": result, "summary": "工具诊断结果已保存", "query_id": qid}
            database.finish(qid, report, "completed")
            return {"query_id": qid, "persistence_status": "saved", "output": result}
        except PlatformError as exc:
            database.finish(qid, {"error_code": exc.code, "summary": exc.message, "query_id": qid}, "failed")
            return {"query_id": qid, "persistence_status": "saved", "error": {"code": exc.code, "message": exc.message}}

    @app.post("/api/v1/queries", status_code=202)
    def create_query(body: QueryRequest, idempotency_key: str | None = Header(default=None)):
        if idempotency_key and len(idempotency_key) > 128:
            raise HTTPException(422, "幂等键超过长度上限")
        try:
            qid, created = database.create(body, idempotency_key)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"query_id": qid, "created": created, "execution_status": get_query(qid)["execution_status"]}

    @app.get("/api/v1/queries")
    def history():
        return {"queries": database.history()}

    @app.get("/api/v1/queries/{qid}")
    def query_detail(qid: str):
        # Read the cursor before the snapshot: events produced during the read remain replayable.
        cursor = database.event_cursor(qid)
        return get_query(qid) | {"results": database.results(qid), "tool_calls": database.calls(qid),
                                "last_event_id": cursor, "agent_steps": database.agent_steps(qid)}

    @app.post("/api/v1/queries/{qid}/follow-up", status_code=202)
    def follow_up(qid: str, body: QueryRequest, idempotency_key: str | None = Header(default=None)):
        q = get_query(qid)
        if q["execution_status"] not in TERMINAL:
            raise HTTPException(409, "请等待当前查询结束或取消后再追问")
        if q["plan"].get("intent") == "tool_diagnostic":
            raise HTTPException(409, "诊断记录不能作为研究会话，请创建新查询")
        cid = database.ensure_conversation(qid)
        return conversation_message(cid, body, idempotency_key)

    @app.get("/api/v1/queries/{qid}/events")
    async def events(qid: str, request: Request, after: int = 0, last_event_id: str | None = Header(default=None)):
        get_query(qid)
        try:
            cursor = max(after, int(last_event_id or 0))
        except ValueError:
            raise HTTPException(422, "无效事件游标") from None
        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                try:
                    rows = database.events(qid, cursor)
                    for event in rows:
                        cursor = event["event_id"]
                        yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if get_query(qid)["execution_status"] in TERMINAL:
                        yield 'data: {"type":"stream_end"}\n\n'
                        break
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.6)
                except SQLAlchemyError:
                    yield 'data: {"type":"database_unavailable","payload":{"message":"数据库暂不可用，保存状态未确认"}}\n\n'
                    break
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/v1/queries/{qid}/refresh", status_code=202)
    def refresh(qid: str):
        q = get_query(qid)
        if q["plan"].get("intent") == "tool_diagnostic":
            raise HTTPException(409, "工具诊断请从工具接口重新执行")
        body = QueryRequest.model_validate(q["request"] | {"mode": "live", "anchor_id": q["plan"].get("anchor_id")})
        new_id, _ = database.create(body, parent=qid, conversation_id=q.get("conversation_id"))
        return {"query_id": new_id}

    @app.post("/api/v1/queries/{qid}/retry", status_code=202)
    def retry(qid: str):
        q = get_query(qid)
        if q["execution_status"] not in {"failed", "partial"} or q["plan"].get("intent") == "tool_diagnostic":
            raise HTTPException(409, "当前查询不可重试")
        if q["request"].get("execution_mode") == "agent" and q["checkpoint"].get("agent_state"):
            raise HTTPException(409, "Agent已执行反馈循环；请实时刷新或在会话中追问以创建新的研究尝试，保留旧轨迹")
        # Successful committed platform batches are preserved; metric-only failures need refresh.
        if q["execution_status"] == "partial" and not q["report"].get("failures"):
            raise HTTPException(409, "该查询需要刷新以重新获取字段，不重复写入已提交批次")
        database.update(qid, status="queued", lease_owner=None, lease_until=None, error=None)
        database.event(qid, "retry", {"message": "已入队重试；保留已提交结果"})
        return {"query_id": qid}

    @app.post("/api/v1/queries/{qid}/regenerate-answer", status_code=202)
    def regenerate_answer(qid: str):
        get_query(qid)
        try:
            database.queue_answer_regeneration(qid)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"query_id": qid}

    @app.post("/api/v1/queries/{qid}/select", status_code=202)
    async def select_work(qid: str, request: Request):
        q = get_query(qid)
        body = await request.json()
        choice = body.get("anchor_id") if isinstance(body, dict) else None
        if q["execution_status"] != "waiting_input" or choice not in {c["key"] for c in q["checkpoint"].get("choices", [])}:
            raise HTTPException(409, "选择不在待确认候选中")
        database.update(qid, plan=q["plan"] | {"anchor_id": choice}, status="queued", lease_owner=None, lease_until=None)
        database.event(qid, "identity_selected", {"anchor_id": choice})
        return {"query_id": qid}

    @app.post("/api/v1/queries/{qid}/cancel")
    def cancel(qid: str):
        q = get_query(qid)
        if q["execution_status"] in TERMINAL:
            raise HTTPException(409, "查询已结束")
        database.update(qid, status="cancelled", persistence_status="saved", finished_at=now())
        database.event(qid, "finished", {"status": "cancelled", "message": "取消记录已保存，已提交数据保留"})
        return {"query_id": qid, "execution_status": "cancelled"}

    @app.get("/api/v1/library")
    def library():
        return {"songs": database.library()}

    @app.get("/api/v1/stats")
    def stats():
        return database.stats()

    @app.get("/api/v1/queries/{qid}/export")
    def export(qid: str, format: str = "json"):
        q = get_query(qid)
        if format == "json":
            document = q | {"results": database.results(qid), "tool_calls": database.calls(qid),
                            "agent_steps": database.agent_steps(qid)}
            return Response(json.dumps(document, ensure_ascii=False, indent=2), media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="musictracker-{qid}.json"'})
        if format != "csv":
            raise HTTPException(422, "支持 json 或 csv")
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["query_id", "title", "artists", "platform", "metric", "value", "display", "status", "scope", "collected_at", "source", "observation_id"])
        def safe(value):
            text = "" if value is None else str(value)
            return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text
        for song in database.results(qid):
            for metric in song["metrics"]:
                writer.writerow([safe(v) for v in [qid, song["title"], " / ".join(song["artists"]), song["platform"],
                    metric["name"], metric["value"], metric["display"], metric["status"], metric["scope"],
                    metric["collected_at"], metric["source_url"], metric["observation_id"]]])
        return Response("\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="musictracker-{qid}.csv"'})
    return app
