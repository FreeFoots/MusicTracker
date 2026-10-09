import argparse
import asyncio
import json
import os


def main():
    parser = argparse.ArgumentParser(description="MusicTracker local research workspace")
    parser.add_argument("command", choices=["start", "serve", "probe", "init-db", "eval-agent"], nargs="?", default="serve")
    parser.add_argument("--sqlite", action="store_true", help="Use isolated local development SQLite")
    parser.add_argument("--keyword", default="空心")
    parser.add_argument("--live", action="store_true", help="Run real DeepSeek/platform evaluations; requires local key and incurs API usage")
    parser.add_argument("--limit", type=int, default=3, help="Live evaluation cases, 1–8")
    parser.add_argument("--cases", help="Optional local JSON behavior corpus for live evaluation")
    args = parser.parse_args()
    from pathlib import Path
    envfile = Path(__file__).resolve().parent.parent / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if args.sqlite:
        os.environ["DATABASE_URL"] = "sqlite:///./data/musictracker.db"
    from .config import settings
    if args.command == "eval-agent":
        from .evaluation import evaluate
        evaluate(args.live, args.limit, args.cases)
    elif args.command == "start":
        from .launcher import start
        start()
    elif args.command == "serve":
        import uvicorn

        from .api import create_app
        uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("MUSICTRACKER_PORT", "8100")))
    elif args.command == "init-db":
        from .store import Store
        store = Store(settings.database_url)
        store.initialize()
        print(json.dumps(store.stats(), ensure_ascii=False))
    else:
        from .api import create_app
        from .schemas import QueryRequest
        from .store import Store
        from .tools import ToolRegistry
        from .transport import Transport
        async def probe():
            store, transport = Store(settings.database_url), Transport(settings)
            store.initialize()
            registry = ToolRegistry(transport)
            app = create_app(settings, store, registry, run_worker=False)
            qid, _ = store.create(QueryRequest(input=f"获取《{args.keyword}》的近期数据"))
            try:
                await app.state.workflow.run(qid)
                q = store.get(qid)
                print(json.dumps({"query_id": qid, "status": q["execution_status"], "persistence": q["persistence_status"],
                    "summary": q["report"].get("summary"), "choices": q["checkpoint"].get("choices", []),
                    "songs": [{"platform": s["platform"], "title": s["title"], "artists": s["artists"],
                               "metrics": s["metrics"]} for s in store.results(qid)], "tool_calls": store.calls(qid)}, ensure_ascii=False, indent=2))
            except Exception:
                store.update(qid, status="failed", error="诊断执行异常；详细原因见终端", persistence_status="partial")
                raise
            finally:
                await transport.close()
                store.engine.dispose()
        asyncio.run(probe())


if __name__ == "__main__":
    main()
