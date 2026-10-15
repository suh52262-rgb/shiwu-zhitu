"""Student OCR corrections replace projections and retain edits on failure."""
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from frontend.ui_logic import apply_ocr_correction, correction_unavailable, switch_session

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


def initial_state():
    return dict(attempt_id="a", attempt_status="in_progress", tutor={"hint": "原分析"},
                timeline=[{"type": "submit", "timestamp": 0, "content": "F=4"}],
                video_result={"attempt_id": "a", "media_type": "image", "submitted": 1,
                              "review": {"timeline": "过期复盘"},
                              "rounds": [{"round": 1, "answer": "F=4", "source_id": "src",
                                          "source_revision": 0, "hint": "原分析"}]})


def corrected_payload():
    return dict(attempt_id="a", source_id="src", source_revision=1, original_text="F=4",
                corrected_text="F=49", analysis={"attempt_id": "a", "source_id": "src",
                    "source_revision": 1, "hint": "更正后的分析", "finished": False},
                state={}, timeline=[{"event_type": "submit", "timestamp": 0, "content": "F=4"},
                                    {"event_type": "ocr_correction", "timestamp": 0,
                                     "content": "source=src; revision=1; supersedes=0"}],
                idempotent=False)


def test_projection_replaces_timeline_and_retains_original_ocr():
    state = initial_state()
    state["recommendations"] = {"weak_concepts": ["过期诊断"]}
    apply_ocr_correction(state, corrected_payload(), "src")
    apply_ocr_correction(state, corrected_payload(), "src")  # replay never appends a submission
    assert len(state["timeline"]) == 2
    assert sum(e["type"] == "submit" for e in state["timeline"]) == 1
    assert state["video_result"]["review"] is None
    assert "recommendations" not in state
    assert state["video_result"]["rounds"][0]["original_answer"] == "F=4"
    assert state["video_result"]["rounds"][0]["answer"] == "F=49"


def test_foreign_response_preserves_progress():
    state = initial_state()
    before = deepcopy(state)
    with pytest.raises(ValueError, match="不匹配"):
        apply_ocr_correction(state, {**corrected_payload(), "attempt_id": "other"}, "src")
    assert state == before


def test_old_round_terminal_and_switch_guards():
    state = initial_state()
    result = state["video_result"]
    source = result["rounds"][0]
    assert not correction_unavailable(state, result, source)
    result["rounds"].append({"source_id": "new"})
    assert "仅可更正最近" in correction_unavailable(state, result, source)
    state["attempt_status"] = "finished"
    assert "已结束" in correction_unavailable(state, result, source)
    state["active_session"] = ("s", "p1")
    state["ocr_edit_src"] = "编辑草稿"
    state["ocr_error_src"] = "失败"
    switch_session(state, "s", "p2", 1)
    assert not any(k.startswith("ocr_") for k in state)


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest
    def get(url, **kwargs):
        payload = {"items": [{"problem_id": "p1", "title": "题一"},
                             {"problem_id": "p2", "title": "题二"}]} if url.endswith("/problems") else {"status": "ok"}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    at = AppTest.from_file(str(APP)).run()
    for key, value in initial_state().items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def test_app_correction_posts_text_only_and_replaces_visible_analysis(app, monkeypatch):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json=corrected_payload(), request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    app.text_area(key="ocr_edit_src").set_value("F=49").run()
    app.button(key="ocr_submit_src").click().run()
    assert not app.exception
    assert calls[0][0].endswith("/video/correct_ocr")
    assert calls[0][1]["json"] == dict(student_id="stu_demo", problem_id="p1", attempt_id="a",
                                          source_id="src", base_revision=0, corrected_text="F=49")
    assert "files" not in calls[0][1]
    assert app.session_state["tutor"]["hint"] == "更正后的分析"
    assert len(app.session_state["timeline"]) == 2
    assert "F=4" in [t.value for t in app.text]
    assert not any("过期复盘" in m.value for m in app.markdown)
    assert any("原复盘已失效" in i.value for i in app.info)
    assert any("**识别更正**" in m.value for m in app.markdown)
    assert not any("source=src" in m.value for m in app.markdown)


@pytest.mark.parametrize("status", [409, 502])
def test_app_failed_correction_retains_editor_and_old_progress(app, monkeypatch, status):
    def post(url, **kwargs):
        return httpx.Response(status, json={"detail": "来源已过期"}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    before = deepcopy(app.session_state["timeline"])
    app.text_area(key="ocr_edit_src").set_value("F=49").run()
    app.button(key="ocr_submit_src").click().run()
    assert not app.exception
    assert app.session_state["timeline"] == before
    assert app.session_state["tutor"]["hint"] == "原分析"
    assert app.text_area(key="ocr_edit_src").value == "F=49"
    assert any("原进度与编辑内容已保留" in e.value for e in app.error)
    app.selectbox(key="problem_select").select("p2").run()
    assert "ocr_edit_src" not in app.session_state


def test_app_terminal_attempt_has_explanation_without_editor(app):
    app.session_state["attempt_status"] = "finished"
    app.session_state["tutor"] = {"hint": "已完成", "finished": True}
    app.run()
    assert not app.exception
    assert not any(t.key == "ocr_edit_src" for t in app.text_area)
    assert any("作答已结束" in c.value for c in app.caption)


def test_correction_retains_old_review_and_idempotent_replay_keeps_new_review():
    state = initial_state()
    state["video_result"].update(input_revision=0, generation_fingerprint="old-model-prompt")
    apply_ocr_correction(state, corrected_payload(), "src")
    result = state["video_result"]
    assert result["review_history"][0]["review"]["timeline"] == "过期复盘"
    assert result["review_history"][0]["generation_fingerprint"] == "old-model-prompt"
    assert result["correction_generation"] == 1
    result.update(review={"timeline": "新版"}, input_revision=1, review_refresh_required=False)
    before = deepcopy(state)
    apply_ocr_correction(state, corrected_payload(), "src")
    assert state == before


def test_stale_correction_cannot_overwrite_newer_text():
    state = initial_state()
    apply_ocr_correction(state, corrected_payload(), "src")
    before = deepcopy(state)
    with pytest.raises(ValueError):
        apply_ocr_correction(state, {**corrected_payload(), "source_revision": 0}, "src")
    assert state == before
