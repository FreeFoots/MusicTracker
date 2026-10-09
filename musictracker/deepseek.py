"""Official DeepSeek transport. Never logs credentials or private reasoning_content."""
import json
import re
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings


class ModelError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class PlannedIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, max_length=60)
    days: int = Field(default=7, ge=1, le=365)
    platforms: list[str] = Field(default_factory=lambda: ["qq", "netease"], min_length=1, max_length=2)
    question: str | None = Field(default=None, max_length=300)
    anchor_id: str | None = Field(default=None, pattern=r"^(qq|netease):[A-Za-z0-9]+$")
    requested_artist: str | None = Field(default=None, min_length=1, max_length=80)


class AnswerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(pattern=r"^(current_metrics|snapshot_changes|versions|coverage|limitations)$")
    observation_ids: list[str] = Field(default_factory=list, max_length=24)


class EvidenceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: list[AnswerSection] = Field(min_length=1, max_length=5)
    next_action: str = Field(default="none", pattern=r"^(none|collect_more_history|refresh_failed_fields|confirm_work)$")


class DeepSeekClient:
    def __init__(self, config: Settings, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=config.model_timeout, follow_redirects=False, trust_env=False)

    @property
    def configured(self):
        return bool(self.config.deepseek_key.strip() and self.config.deepseek_model.strip())

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def validate_history(messages: list[dict]):
        pending = set()
        for message in messages:
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    raise ModelError("invalid_tool_history", "工具反馈与历史调用不匹配，未发送模型请求")
                pending.remove(call_id)
            else:
                if pending:
                    raise ModelError("invalid_tool_history", "历史工具调用缺少反馈，未发送模型请求")
                if "tool_calls" in message:
                    calls = message["tool_calls"]
                    if message.get("role") != "assistant" or not calls:
                        raise ModelError("invalid_tool_history", "历史工具调用无效，未发送模型请求")
                    pending = {c["id"] for c in calls}
                    if len(pending) != len(calls):
                        raise ModelError("invalid_tool_history", "历史调用ID重复，未发送模型请求")
        if pending:
            raise ModelError("invalid_tool_history", "历史工具调用未完成，未发送模型请求")

    async def complete(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> dict:
        if not self.configured:
            raise ModelError("model_unconfigured", "DeepSeek 尚未配置：请在项目 .env 设置 DEEPSEEK_API_KEY 后重启；未切换为规则冒充 Agent")
        try:
            parsed = urlparse(self.config.deepseek_base_url)
            port = parsed.port
        except ValueError:
            raise ModelError("unsafe_model_url", "模型入口格式无效，未发送密钥") from None
        if (parsed.scheme != "https" or parsed.hostname != "api.deepseek.com" or port not in {None, 443}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/", "/v1", "/v1/"}):
            raise ModelError("unsafe_model_url", "只允许 DeepSeek 官方 HTTPS 入口，未发送密钥")
        if len(json.dumps(messages, ensure_ascii=False)) > 60000:
            raise ModelError("context_budget", "会话与工具上下文超过本次安全预算，请缩小查询范围")
        self.validate_history(messages)
        body = {"model": self.config.deepseek_model, "messages": messages, "stream": False,
                "thinking": {"type": "disabled"}, "max_tokens": 2048}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            async with self.client.stream("POST", self.config.deepseek_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + self.config.deepseek_key}, json=body) as response:
                if response.status_code in {401, 403}:
                    raise ModelError("model_auth", "DeepSeek 授权失败，请检查本机 Key；未自动重试")
                if response.status_code in {402, 429}:
                    raise ModelError("model_quota", "DeepSeek 余额或限流错误；未自动重试")
                response.raise_for_status()
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 2_000_000:
                        raise ModelError("model_response_size", "模型响应超过上限")
            data = json.loads(raw)
            choice = data["choices"][0]
            if choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise ModelError("model_incomplete", "模型输出未正常结束，不执行截断的计划")
            message = choice["message"]
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []
            if not isinstance(content, str) or len(content) > 12000 or not isinstance(calls, list) or len(calls) > 4:
                raise ModelError("model_schema", "模型输出不符合受控契约")
            clean_calls = []
            call_ids = set()
            for call in calls:
                name, arguments = call["function"]["name"], call["function"]["arguments"]
                if (not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) or not isinstance(arguments, str)
                        or len(arguments) > 3000 or not isinstance(call["id"], str) or not call["id"]
                        or len(call["id"]) > 128 or call["id"] in call_ids):
                    raise ModelError("model_schema", "模型工具调用结构无效")
                call_ids.add(call["id"])
                clean_calls.append({"id": call["id"], "type": "function", "function": {"name": name, "arguments": arguments}})
            usage = data.get("usage") or {}
            total = usage.get("total_tokens")
            reported = isinstance(total, int) and not isinstance(total, bool) and total >= 0
            charged = total if reported else ((len(json.dumps(messages, ensure_ascii=False))
                                               + len(json.dumps(tools or [], ensure_ascii=False))) // 2 + 2048)
            # No response IDs, raw provider errors or hidden chain-of-thought are persisted.
            return {"message": {"role": "assistant", "content": content, **({"tool_calls": clean_calls} if clean_calls else {})},
                    "usage": {"total_tokens": total if reported else None, "budget_tokens": charged,
                              "estimated": not reported}, "model": data.get("model", self.config.deepseek_model)}
        except httpx.TimeoutException as exc:
            raise ModelError("model_timeout", "DeepSeek 请求超时，查询记录与已获取结果保留") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"model_http_{exc.response.status_code}", f"DeepSeek 请求失败（HTTP {exc.response.status_code}）；不保存服务原始错误或请求凭据") from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ModelError("model_response", "DeepSeek 返回异常，未采用未经验证的输出") from exc
