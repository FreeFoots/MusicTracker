import httpx
import pytest

from musictracker.platforms import QQ, Netease, count_metric, version_of
from musictracker.transport import PlatformError, Transport


def test_missing_not_zero():
    assert count_metric("comment_count", "评论", None, "https://music.163.com").value is None
    assert count_metric("comment_count", "评论", False, "https://music.163.com").status == "unavailable"


@pytest.mark.parametrize("title,duration,expected", [("空心 (Live)", 180000, "live"), ("空心 (DJ)", 180000, "remix"), ("空心", 39000, "excerpt"), ("空心", 280000, "studio")])
def test_version_types(title, duration, expected):
    assert version_of(title, duration) == expected


async def test_qq_bucket_is_approximate(config):
    def handler(request):
        return httpx.Response(200, json={"code": 0, "comments": {"code": 0, "data": {"response_list": [{"biz_id": "1", "count": 0}]}},
            "favorites": {"code": 0, "data": {"m_numbers": {"1": 1000001}, "m_show": {"1": "100w+"}}}})
    transport = Transport(config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await QQ(transport).metrics(["1"])
    metrics = result["metrics"]["1"]
    assert metrics[0]["status"] == "exact" and metrics[0]["value"] == 0
    assert metrics[1]["status"] == "approximate" and metrics[1]["value"] is None
    assert metrics[2]["status"] == "unavailable"
    assert metrics[0]["source_url"] == "https://u.y.qq.com/cgi-bin/musicu.fcg"
    await transport.close()


async def test_region_payload_reported_not_decoded(config):
    transport = Transport(config, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"code": 200, "result": "encoded", "abroad": True}))))
    with pytest.raises(PlatformError, match="地区"):
        await Netease(transport).search("空心")
    await transport.close()


async def test_ssrf_blocked(config):
    transport = Transport(config)
    with pytest.raises(PlatformError):
        await transport.json("GET", "http://127.0.0.1/private")
    await transport.close()


@pytest.mark.parametrize("status,code", [(401, "authorization_required"), (403, "authorization_required"), (429, "rate_limited")])
async def test_no_retry_auth_or_rate_limit(config, status, code):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(status)
    transport = Transport(config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(PlatformError) as error:
        await transport.json("GET", "https://music.163.com/api/search")
    assert error.value.code == code and count == 1
    await transport.close()


async def test_response_size_limit(config):
    from dataclasses import replace
    config = replace(config, max_response_bytes=20)
    transport = Transport(config, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 100))))
    with pytest.raises(PlatformError) as error:
        await transport.json("GET", "https://music.163.com/api/search")
    assert error.value.code == "response_too_large"
    await transport.close()
