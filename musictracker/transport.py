import asyncio
import json
import time
from urllib.parse import urlparse

import httpx

from .config import Settings


class PlatformError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Transport:
    """Bounded public metadata access; no cookies, signing bypass or user-supplied URLs."""

    allowed_hosts = {"music.163.com", "u.y.qq.com", "c.y.qq.com"}

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            timeout=settings.request_timeout, follow_redirects=False, trust_env=False,
            headers={"User-Agent": "MusicTracker/0.5 (local music metadata research)"},
        )
        self.locks: dict[str, asyncio.Lock] = {}
        self.last: dict[str, float] = {}

    async def json(self, method: str, url: str, **kwargs) -> dict:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise PlatformError("unsafe_url", "平台域名不在准入名单中")
        host = parsed.hostname
        async with self.locks.setdefault(host, asyncio.Lock()):
            delay = self.settings.request_interval - (time.monotonic() - self.last.get(host, 0))
            if delay > 0:
                await asyncio.sleep(delay)
            self.last[host] = time.monotonic()
            try:
                async with self.client.stream(method, url, **kwargs) as response:
                    if response.status_code in {401, 403}:
                        raise PlatformError("authorization_required", "平台要求授权，未绕过访问限制")
                    if response.status_code == 429:
                        raise PlatformError("rate_limited", "平台限流，请稍后重试")
                    response.raise_for_status()
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > self.settings.max_response_bytes:
                            raise PlatformError("response_too_large", "平台响应超过安全上限")
                    data = json.loads(body)
                    if not isinstance(data, dict):
                        raise PlatformError("schema_changed", "平台返回格式异常")
                    return data
            except httpx.TimeoutException as exc:
                raise PlatformError("timeout", "平台请求超时") from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise PlatformError("upstream_error", "平台请求失败或返回非 JSON 内容") from exc

    async def close(self):
        await self.client.aclose()
