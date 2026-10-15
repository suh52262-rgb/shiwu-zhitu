"""Regression from real browser: uploaded file must not enable unsafe reupload."""
import io
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("url_only", [False, True])
def test_unknown_upload_disables_submit_until_404(monkeypatch, url_only):
    outcome = [503]

    def get(url, **kwargs):
        status = outcome[0] if "/by-request/" in url else 200
        payload = ({"items": [{"problem_id": "p1", "title": "测试题"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(status, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: pytest.fail("Unexpected duplicate POST"))
    monkeypatch.setattr(st, "file_uploader", lambda *_args, **_kwargs: io.BytesIO(b"synthetic"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py")).run()
    app.session_state.media_mode = "📹 传一段视频"
    if url_only:
        app.query_params["video_request"] = "11111111-2222-3333-4444-555555555555"
    else:
        app.session_state.video_job_request_id = "11111111-2222-3333-4444-555555555555"
    app.run()
    assert not app.exception
    assert app.button(key="video_go").disabled
    app.button(key="recover_video_job").click().run()
    assert app.button(key="video_go").disabled
    outcome[0] = 404
    app.button(key="recover_video_job").click().run()
    assert not app.exception
    assert not app.button(key="video_go").disabled
    assert app.session_state.video_job_request_id == "11111111-2222-3333-4444-555555555555"
    assert "video_request" in app.query_params
    assert any("同一编号" in item.value for item in app.success)
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["data"]["client_request_id"])
        return httpx.Response(503, json={}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    upload = io.BytesIO(b"synthetic")
    upload.name, upload.type = "sample.mp4", "video/mp4"
    monkeypatch.setattr(st, "file_uploader", lambda *_args, **_kwargs: upload)
    app.button(key="video_go").click().run()
    assert calls == ["11111111-2222-3333-4444-555555555555"]
    assert app.session_state.video_job_request_absent_confirmed
    assert app.session_state.video_job_request_id == calls[0]
    assert not app.button(key="video_go").disabled


# ================================================================ 契约 ⑤：「放弃这次上传」的出口
#
# ⚠️ **为什么必须有它**（契约 T0 不变式⑤ 的原话）：
#   编号复用只对"同一次上传"成立；没有出口的话，用户**换一个视频就永远 409** ——
#   编号被钉住 → 服务端按内容判冲突 → 409 → 提示"用同一编号重试" → 再试还是 409（死循环）。
#   实测确认过：编号同时写进 URL，**刷新也甩不掉**，`clear_media` 也不清它。

def _pending_app(monkeypatch, rid="11111111-2222-3333-4444-555555555555"):
    """把页面带到"有编号、结果未知"的状态（上传按钮禁用）。"""
    def get(url, **kwargs):
        payload = ({"items": [{"problem_id": "p1", "title": "测试题"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: pytest.fail("不该发出 POST"))
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: io.BytesIO(b"synthetic"))
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py")).run()
    app.session_state.media_mode = "📹 传一段视频"
    app.session_state.video_job_request_id = rid
    app.query_params["video_request"] = rid
    app.run()
    assert not app.exception
    return app


def test_abandoning_this_upload_clears_the_pinned_id(monkeypatch):
    """**契约 ⑤ 的出口**：点了之后编号（会话 + URL）都要清掉，否则换视频永远 409。"""
    app = _pending_app(monkeypatch)
    assert app.button(key="video_go").disabled, "夹具没进到「结果未知」的状态"
    assert app.button(key="abandon_video_request") is not None, "契约 ⑤ 要求的出口不见了"
    app.button(key="abandon_video_request").click().run()
    assert not app.exception
    assert "video_job_request_id" not in app.session_state, "会话里的编号没清掉"
    assert "video_request" not in app.query_params, "URL 里的编号没清掉（刷新会捡回来）"
    assert not app.button(key="video_go").disabled, "清掉之后应该能上传新视频了"


def test_abandon_exit_states_the_cost(monkeypatch):
    """⚠️ 放弃 = **承认那次结果不再找回** —— 提示必须说清代价，不能轻描淡写成"重新上传"。"""
    app = _pending_app(monkeypatch)
    app.button(key="abandon_video_request").click().run()
    texts = " ".join(item.value for item in app.success)
    assert "不会" in texts and "找回" in texts, f"没说清代价：{texts!r}"

