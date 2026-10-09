import httpx
import pytest

from musictracker.config import Settings
from musictracker.schemas import Metric, Song
from musictracker.store import Store
from musictracker.tools import ToolRegistry
from musictracker.transport import Transport


@pytest.fixture
def store(tmp_path):
    db = Store("sqlite:///" + str(tmp_path / "test.db"))
    db.initialize()
    yield db
    db.engine.dispose()


@pytest.fixture
def config():
    return Settings(database_url="sqlite://", request_interval=0, deepseek_key="", planner_key="", api_token="",
                    planner_base_url="", planner_model="")


@pytest.fixture
def registry(config):
    transport = Transport(config, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    tools = ToolRegistry(transport)
    for platform in ["qq", "netease"]:
        def songs_for(platform):
            return [Song(platform=platform, platform_id=("mid" + str(i)) if platform == "qq" else str(i),
                numeric_id=str(i), title="测试歌", artists=["歌手甲" if i == 1 else "歌手乙"],
                album="测试专辑", duration_ms=180000 if i == 1 else 200000,
                url=("https://y.qq.com/n/ryqq/songDetail/mid" + str(i)) if platform == "qq" else ("https://music.163.com/song?id=" + str(i)),
                origin_status="platform_original" if platform == "netease" and i == 1 else "platform_cover" if platform == "netease" else "unknown",
                origin_song_id="1" if platform == "netease" and i == 2 else None,
                search_rank=i) for i in [1, 2]]
        sample = songs_for(platform)
        async def search(keyword, limit=12, sample=sample):
            return {"songs": [s.model_dump() for s in sample[:limit]], "coverage": {"method": "mock", "complete": False}}
        async def detail(ids, sample=sample):
            return {"songs": [s.model_dump() for s in sample if s.platform_id in ids]}
        async def metrics(ids, sample=sample):
            return {"metrics": {s.numeric_id: [Metric(name="comment_count", label="评论总数", status="exact", value=100 + int(s.numeric_id), source_url=s.url).model_dump(),
                         Metric(name="favorite_count", label="收藏", status="exact", value=5, source_url=s.url).model_dump()] for s in sample if s.numeric_id in ids}}
        tools.tools[f"{platform}.search_songs"].run = search
        tools.tools[f"{platform}.song_detail"].run = detail
        tools.tools[f"{platform}.song_metrics"].run = metrics
    return tools
