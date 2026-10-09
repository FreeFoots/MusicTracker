import asyncio
from datetime import datetime, timezone

from .schemas import Metric, Song
from .transport import PlatformError, Transport

QQ_URL = "https://u.y.qq.com/cgi-bin/musicu.fcg"
NETEASE_URL = "https://music.163.com"


def version_of(title: str, duration: int) -> str:
    lower = title.lower()
    if "live" in lower or "现场" in lower:
        return "live"
    if "dj" in lower or "remix" in lower:
        return "remix"
    if "伴奏" in title:
        return "instrumental"
    if 0 < duration < 90000:
        return "excerpt"
    return "studio"


def unavailable(name: str, label: str, url: str, reason: str) -> Metric:
    return Metric(name=name, label=label, status="unavailable", source_url=url, reason=reason)


def count_metric(name: str, label: str, value, url: str) -> Metric:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return unavailable(name, label, url, "field_missing_or_invalid")
    return Metric(name=name, label=label, status="exact", value=value, source_url=url)


class Netease:
    def __init__(self, transport: Transport):
        self.http = transport

    async def call(self, method: str, path: str, **kwargs) -> dict:
        data = await self.http.json(method, NETEASE_URL + path,
                                    headers={"Referer": NETEASE_URL + "/"}, **kwargs)
        if data.get("code") != 200:
            code = "authorization_required" if data.get("code") in {301, 302, 401, 403} else "upstream_error"
            raise PlatformError(code, f"网易云入口返回状态 {data.get('code', 'missing')}")
        return data

    def song(self, data: dict, rank: int = 0) -> Song:
        artist_data = data.get("artists") or data.get("ar") or []
        album = data.get("album") or data.get("al") or {}
        duration = int(data.get("duration") or data.get("dt") or 0)
        origin = data.get("originSongSimpleData") or {}
        timestamp = album.get("publishTime") or data.get("publishTime")
        release_date = None
        if isinstance(timestamp, int) and timestamp > 0:
            release_date = datetime.fromtimestamp(timestamp / 1000, timezone.utc).date().isoformat()
        sid = str(data["id"])
        return Song(
            platform="netease", platform_id=sid, numeric_id=sid, title=str(data["name"]),
            artists=[str(a["name"]) for a in artist_data if a.get("name")],
            album=str(album.get("name") or ""), duration_ms=duration, release_date=release_date,
            url=f"{NETEASE_URL}/song?id={sid}", search_rank=rank,
            version=version_of(str(data["name"]), duration),
            origin_status={1: "platform_original", 2: "platform_cover"}.get(data.get("originCoverType"), "unknown"),
            origin_song_id=str(origin["songId"]) if origin.get("songId") else None,
        )

    async def search(self, keyword: str, limit: int = 12) -> dict:
        data = await self.call("POST", "/api/search/get/web",
                               data={"s": keyword, "type": 1, "limit": limit, "offset": 0})
        result = data.get("result") or {}
        if not isinstance(result, dict):
            raise PlatformError("region_or_schema_restricted", "网易云未返回可解析候选；可能受网络地区或入口限制，未尝试解密或绕过")
        return {"songs": [self.song(s, i + 1).model_dump() for i, s in enumerate(result.get("songs") or [])],
                "coverage": {"method": "public_search", "limit": limit,
                             "total_reported": result.get("songCount"), "complete": False,
                             "selection_signal": "platform_search_order"}}

    async def detail(self, ids: list[str]) -> dict:
        if any(not i.isdigit() for i in ids):
            raise PlatformError("invalid_id", "网易云歌曲 ID 必须为数字")
        data = await self.call("GET", "/api/song/detail", params={"ids": "[" + ",".join(ids) + "]"})
        return {"songs": [self.song(s).model_dump() for s in data.get("songs") or []]}

    async def metrics(self, ids: list[str]) -> dict:
        result = {}
        blocked = None
        for sid in ids:
            if not sid.isdigit():
                raise PlatformError("invalid_id", "网易云歌曲 ID 必须为数字")
            metrics = []
            url = f"{NETEASE_URL}/song?id={sid}"
            for name, label, path, method, kwargs, extract in [
                ("comment_count", "评论总数", f"/api/v1/resource/comments/R_SO_4_{sid}", "GET",
                 {"params": {"limit": 1, "offset": 0}}, lambda d: d.get("total")),
                ("favorite_count", "红心数", "/api/song/red/count", "POST",
                 {"data": {"songId": sid}}, lambda d: (d.get("data") or {}).get("count")),
            ]:
                if blocked:
                    metrics.append(Metric(name=name, label=label, status="error", reason=blocked, source_url=url))
                    continue
                try:
                    data = await self.call(method, path, **kwargs)
                    metrics.append(count_metric(name, label, extract(data), url))
                except PlatformError as exc:
                    metrics.append(Metric(name=name, label=label, status="error", reason=exc.code, source_url=url))
                    if exc.code in {"authorization_required", "rate_limited"}:
                        blocked = exc.code
            metrics.append(unavailable("play_count", "播放量", url, "no_verified_public_endpoint"))
            result[sid] = [m.model_dump() for m in metrics]
        return {"metrics": result}


class QQ:
    def __init__(self, transport: Transport):
        self.http = transport

    async def call(self, calls: dict) -> dict:
        import json

        data = await self.http.json("GET", QQ_URL, params={"data": json.dumps(calls, ensure_ascii=False, separators=(",", ":"))},
                                    headers={"Referer": "https://y.qq.com/"})
        if data.get("code") != 0:
            raise PlatformError("upstream_error", "QQ 音乐顶层响应异常")
        return data

    @staticmethod
    def block(data: dict, key: str) -> dict:
        block = data.get(key) or {}
        if block.get("code") != 0:
            code = "authorization_required" if block.get("code") == 2001 else "upstream_error"
            raise PlatformError(code, f"QQ {key} 返回状态 {block.get('code', 'missing')}")
        if not isinstance(block.get("data"), dict):
            raise PlatformError("schema_changed", "QQ 数据结构变化")
        return block["data"]

    async def search(self, keyword: str, limit: int = 12) -> dict:
        # Verified public autocomplete only. It is deliberately NOT labelled full search.
        data = await self.http.json("GET", "https://c.y.qq.com/splcloud/fcgi-bin/smartbox_new.fcg",
                                   params={"key": keyword, "format": "json"}, headers={"Referer": "https://y.qq.com/"})
        if data.get("code") != 0:
            raise PlatformError("upstream_error", "QQ 搜索建议不可用")
        items = ((data.get("data") or {}).get("song") or {}).get("itemlist") or []
        songs = [Song(platform="qq", platform_id=str(s["mid"]), numeric_id=str(s["id"]),
                      title=str(s["name"]), artists=[str(s.get("singer") or "未知")],
                      search_rank=i + 1, url=f"https://y.qq.com/n/ryqq/songDetail/{s['mid']}").model_dump()
                 for i, s in enumerate(items[:limit]) if s.get("mid") and s.get("id")]
        return {"songs": songs, "coverage": {"method": "public_suggestion", "limit": len(items),
                "complete": False, "selection_signal": "platform_suggestion_order",
                "warning": "QQ 全量搜索未通过匿名验证；仅搜索建议及定向建议，非完整曲库"}}

    async def detail(self, ids: list[str]) -> dict:
        calls = {f"song{i}": {"module": "music.pf_song_detail_svr", "method": "get_song_detail_yqq",
                             "param": {"song_type": 0, "song_mid": sid}} for i, sid in enumerate(ids)}
        data = await self.call(calls)
        songs = []
        for i, sid in enumerate(ids):
            block = self.block(data, f"song{i}")
            track = block.get("track_info") or {}
            if not track.get("mid"):
                continue
            info = block.get("info") or {}
            credits = {}
            for key in ["composer", "lyric", "lyricist"]:
                values = [str(c["value"]) for c in (info.get(key) or {}).get("content", []) if c.get("value")]
                if values:
                    credits[key] = values
            duration = int(track.get("interval") or 0) * 1000
            title = str(track.get("title") or track.get("name"))
            songs.append(Song(platform="qq", platform_id=str(track["mid"]), numeric_id=str(track["id"]),
                              title=title, artists=[str(a["name"]) for a in track.get("singer", []) if a.get("name")],
                              album=str((track.get("album") or {}).get("name") or ""),
                              release_date=track.get("time_public") or None, duration_ms=duration,
                              credits=credits, version=version_of(title, duration),
                              url=f"https://y.qq.com/n/ryqq/songDetail/{sid}").model_dump())
        return {"songs": songs}

    async def metrics(self, ids: list[str]) -> dict:
        if any(not i.isdigit() for i in ids):
            raise PlatformError("invalid_id", "指标接口需要 QQ 数字歌曲 ID")
        data = await self.call({
            "comments": {"module": "music.globalComment.GlobalCommentRead", "method": "GetCommentCount",
                         "param": {"request_list": [{"biz_type": 1, "biz_id": sid, "biz_sub_type": 0} for sid in ids]}},
            "favorites": {"module": "music.musicasset.SongFavRead", "method": "GetSongFansNumberById",
                          "param": {"v_songId": [int(sid) for sid in ids]}},
        })
        blocks, errors = {}, {}
        for key in ["comments", "favorites"]:
            try:
                blocks[key] = self.block(data, key)
            except PlatformError as exc:
                errors[key] = exc.code
        comments = {str(x.get("biz_id")): x.get("count") for x in blocks.get("comments", {}).get("response_list", [])}
        result = {}
        for sid in ids:
            # This tool only knows numeric IDs, not the MID needed by the canonical page.
            # Keep the actual API entry; the overview enriches it using verified song details.
            url = QQ_URL
            comment = count_metric("comment_count", "评论总数", comments.get(sid), url)
            display = blocks.get("favorites", {}).get("m_show", {}).get(sid)
            favorite = (Metric(name="favorite_count", label="收藏展示", status="approximate", display=str(display),
                               reason="platform_display_bucket_not_exact", source_url=url)
                        if display else unavailable("favorite_count", "收藏展示", url, "field_missing"))
            if "comments" in errors:
                comment = Metric(name="comment_count", label="评论总数", status="error", reason=errors["comments"], source_url=url)
            if "favorites" in errors:
                favorite = Metric(name="favorite_count", label="收藏展示", status="error", reason=errors["favorites"], source_url=url)
            result[sid] = [m.model_dump() for m in [comment, favorite, unavailable("play_count", "播放量", url, "no_verified_public_endpoint")]]
        return {"metrics": result}


async def close_all(*transports):
    await asyncio.gather(*(t.close() for t in transports))
