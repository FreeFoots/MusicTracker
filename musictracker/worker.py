import asyncio
import logging

from sqlalchemy.exc import SQLAlchemyError

from .config import Settings
from .models import uid
from .schemas import now
from .store import Store
from .workflow import Cancelled, OverviewWorkflow

log = logging.getLogger("musictracker.worker")


class Worker:
    def __init__(self, store: Store, workflow: OverviewWorkflow, settings: Settings):
        self.store, self.workflow, self.settings = store, workflow, settings
        self.owner = uid()
        self.active: str | None = None

    async def heartbeat(self, qid: str):
        while True:
            await asyncio.sleep(20)
            self.store.heartbeat(qid, self.owner, self.settings.lease_seconds)

    async def loop(self):
        while True:
            try:
                qid = self.store.claim(self.owner, self.settings.lease_seconds)
            except Exception:
                log.warning("Queue database unavailable; retrying without accepting new work")
                await asyncio.sleep(2)
                continue
            if qid is None:
                await asyncio.sleep(0.4)
                continue
            self.active = qid
            beat = asyncio.create_task(self.heartbeat(qid))
            try:
                await self.workflow.run(qid)
            except Cancelled:
                self.store.event(qid, "finished", {"status": "cancelled", "message": "已停止后续请求；已提交结果保留"})
            except asyncio.CancelledError:
                # Durable queue owns the job; shutdown does not discard accepted requests.
                self.store.update(qid, status="queued", lease_owner=None, lease_until=None)
                raise
            except Exception as exc:
                try:
                    error = str(exc) if isinstance(exc, ValueError) else "查询执行异常，请查看工具状态后重试"
                    persisted = "saved"
                    if isinstance(exc, SQLAlchemyError):
                        error = "数据库写入异常，获取结果未确认保存；可重试，不显示保存成功"
                        persisted = "partial" if self.store.results(qid) else "failed"
                    self.store.update(qid, status="failed", error=error, persistence_status=persisted,
                                      finished_at=now(), lease_owner=None, lease_until=None)
                    self.store.event(qid, "finished", {"status": "failed", "message": error})
                except Exception:
                    log.error("Result persistence unavailable for query %s; lease recovery required", qid)
            finally:
                beat.cancel()
                await asyncio.gather(beat, return_exceptions=True)
                self.active = None
