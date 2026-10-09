import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Platform = Literal["qq", "netease"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso() -> str:
    return now().isoformat()


class QueryRequest(BaseModel):
    input: str = Field(min_length=1, max_length=500)
    title: str | None = Field(default=None, max_length=100)
    days: int | None = Field(default=None, ge=1, le=365)
    platforms: list[Platform] = Field(default_factory=lambda: ["qq", "netease"], min_length=1)
    max_versions: int = Field(default=4, ge=1, le=6)
    mode: Literal["live", "prefer_cache", "cache_only"] = "live"
    max_age_seconds: int | None = Field(default=None, ge=0, le=604800)
    anchor_id: str | None = Field(default=None, pattern=r"^(qq|netease):[A-Za-z0-9]+$")
    execution_mode: Literal["rules", "agent"] = "rules"

    @model_validator(mode="after")
    def validate_query(self):
        self.input = self.input.strip()
        if not self.input:
            raise ValueError("查询内容不能为空")
        if re.search(r"(?i)(cookie\s*[:=]|authorization\s*[:=]|bearer\s+|sk-[A-Za-z0-9]{12})", self.input + " " + (self.title or "")):
            raise ValueError("请勿在查询中提交凭据")
        self.platforms = list(dict.fromkeys(self.platforms))
        if self.mode == "prefer_cache" and self.max_age_seconds is None:
            raise ValueError("prefer_cache 必须明确 max_age_seconds")
        return self


class Metric(BaseModel):
    name: str
    label: str
    status: Literal["exact", "approximate", "unavailable", "error"]
    value: int | float | None = None
    display: str | None = None
    reason: str | None = None
    scope: str = "current_cumulative"
    source_url: str
    collected_at: str = Field(default_factory=iso)
    observed_at: str | None = None
    observation_id: str | None = None
    comparison: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_metric(self):
        if self.status in {"unavailable", "error"}:
            if self.value is not None or not self.reason:
                raise ValueError("不可用指标必须有原因且数值为空")
        if self.status == "exact" and (self.value is None or self.value < 0):
            raise ValueError("精确指标必须有有效非负数值")
        if self.status == "approximate" and (not self.display or self.value is not None):
            raise ValueError("区间展示不能写成精确值")
        return self


class Song(BaseModel):
    platform: Platform
    platform_id: str
    numeric_id: str
    title: str
    artists: list[str]
    album: str = ""
    duration_ms: int = 0
    release_date: str | None = None
    url: str
    search_rank: int = 0
    version: str = "studio"
    origin_status: str = "unknown"
    origin_song_id: str | None = None
    credits: dict[str, list[str]] = Field(default_factory=dict)
    metrics: list[Metric] = Field(default_factory=list)
    entity_id: str | None = None
    relation: str = "unconfirmed"
    relation_reason: str = "仅歌名相同，作品关系待核对"
    fetched_at: str = Field(default_factory=iso)

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.platform_id}"


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keyword: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=12, ge=1, le=20)


class DetailInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def ids_are_safe(self):
        if any(not re.fullmatch(r"[A-Za-z0-9]{1,30}", v) for v in self.ids):
            raise ValueError("仅接受平台 ID，不接受任意 URL")
        return self


class ToolInvocation(BaseModel):
    name: str
    arguments: dict[str, Any]
