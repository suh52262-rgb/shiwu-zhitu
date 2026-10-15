"""题目列表摘要不能替代学生实际看到的完整原题。"""
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


def install_api(monkeypatch, *, fail_second=False, wrong_id=False):
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        status = 200
        if url.endswith("/problems"):
            data = {"items": [{"problem_id": pid, "title": pid} for pid in ("p1", "p2")]}
        elif "/problems/" in url:
            pid = url.rsplit("/", 1)[-1]
            data = {"problem_id": "other" if wrong_id else pid,
                    "statement": f"{pid} 完整原题：条件全部保留。\n\n(1) 求速度。\n(2) 求加速度。",
                    "answer": "禁止显示的参考答案", "figure_urls": []}
            if pid == "p1":
                data["figure_urls"] = ["/problem-assets/p1/figures/question.png"]
            if pid == "p2" and fail_second:
                status, data = 503, {"detail": "暂不可用"}
        else:
            data = {"status": "ok"}
        return httpx.Response(status, json=data, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    return calls


def text(at):
    return "\n".join(item.value for item in at.markdown)


def test_full_question_and_original_figure_are_visible(monkeypatch):
    calls = install_api(monkeypatch)
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    assert "p1 完整原题" in text(at) and "(2) 求加速度。" in text(at)
    assert "禁止显示的参考答案" not in text(at)
    assert any(url.endswith("/problems/p1") for url in calls)
    images = at.get("image")
    assert len(images) == 1
    assert images[0].proto.imgs[0].url.endswith("/problem-assets/p1/figures/question.png")


def test_switch_replaces_statement_and_figures(monkeypatch):
    install_api(monkeypatch)
    at = AppTest.from_file(str(APP)).run()
    at.selectbox(key="problem_select").select("p2").run()
    assert not at.exception
    assert "p2 完整原题" in text(at) and "p1 完整原题" not in text(at)
    assert len(at.get("image")) == 0


def test_detail_failure_does_not_reuse_previous_question(monkeypatch):
    install_api(monkeypatch, fail_second=True)
    at = AppTest.from_file(str(APP)).run()
    at.selectbox(key="problem_select").select("p2").run()
    assert not at.exception
    assert any("完整原题读取失败" in item.value for item in at.error)
    assert "p1 完整原题" not in text(at) and "p2 完整原题" not in text(at)


def test_foreign_detail_is_rejected(monkeypatch):
    install_api(monkeypatch, wrong_id=True)
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    assert any("题目详情编号不匹配" in item.value for item in at.error)
    assert "完整原题：" not in text(at)

