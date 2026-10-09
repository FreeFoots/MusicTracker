import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import BaseModel

from .platforms import QQ, Netease
from .schemas import DetailInput, SearchInput
from .transport import PlatformError, Transport


@dataclass
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    run: Callable[..., Awaitable[dict]]
    version: str = "1.0.0"


class ToolRegistry:
    """One allowlisted registry used by workflow, diagnostics and LangChain tool adapters."""

    def __init__(self, transport: Transport):
        self.tools: dict[str, Tool] = {}
        for platform, provider in [("qq", QQ(transport)), ("netease", Netease(transport))]:
            for action, description, schema, fn in [
                ("search_songs", "搜索公开候选；QQ 为有限搜索建议", SearchInput, provider.search),
                ("song_detail", "获取公开歌曲元数据及平台版本声明", DetailInput, provider.detail),
                ("song_metrics", "获取当前评论/收藏；缺失字段不写成零", DetailInput, provider.metrics),
            ]:
                name = f"{platform}.{action}"
                if action == "song_detail":
                    description += "；ids须逐字复制搜索结果platform_id（QQ是MID字符串，绝不是numeric_id）" if platform == "qq" else "；ids须复制搜索结果platform_id数字字符串"
                elif action == "song_metrics":
                    description += "；ids须复制已完成详情的numeric_id数字字符串，每次最多6个；QQ不能使用MID"
                self.tools[name] = Tool(name, description, schema, fn)

    def manifest(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "version": t.version,
                 "read_only": True, "input_schema": t.schema.model_json_schema()} for t in self.tools.values()]

    async def invoke(self, name: str, arguments: dict, audit: Callable | None = None) -> dict:
        if name not in self.tools:
            raise PlatformError("unknown_tool", "工具不在注册表中")
        tool = self.tools[name]
        args = tool.schema.model_validate(arguments).model_dump()
        start = time.monotonic()
        try:
            result = await tool.run(**args)
            if audit:
                audit(name, args, "completed", None, round((time.monotonic() - start) * 1000))
            return result
        except PlatformError as exc:
            if audit:
                audit(name, args, "failed", exc.code, round((time.monotonic() - start) * 1000))
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            if audit:
                audit(name, args, "failed", "schema_changed", round((time.monotonic() - start) * 1000))
            raise PlatformError("schema_changed", "平台结构与已验证契约不符，未猜测字段") from exc

    def langchain_tools(self):
        from langchain_core.tools import StructuredTool

        return [StructuredTool.from_function(name=t.name, description=t.description,
                args_schema=t.schema, coroutine=t.run) for t in self.tools.values()]
