"""Student-facing release boundaries use actual API status."""
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


@pytest.mark.parametrize("status,has_graph,expected", [
    ("legacy_unreviewed", True, "尚未完成复核"),
    ("legacy_unreviewed", False, "尚未发布"),
    ("draft", False, "待复核草稿"),
    ("reviewed", False, "已复核但尚未发布"),
])
def test_release_status_is_visible(monkeypatch, status, has_graph, expected):
    def get(url, **kwargs):
        payload = {"items": [{"problem_id": "p", "title": "题", "graph_status": status,
                               "has_graph": has_graph}]} if url.endswith("/problems") else {"status": "ok"}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    visible = [item.value for item in at.warning] + [item.value for item in at.caption]
    assert any(expected in value for value in visible)
    at.session_state["tutor"] = {"hint": "", "graph_binding_status": "unknown"}
    at.run()
    assert any("旧作答未记录" in item.value for item in at.caption)
