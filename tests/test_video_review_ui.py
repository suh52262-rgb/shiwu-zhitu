"""On-demand review never replaces the teaching state or crosses sessions."""
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from frontend.ui_logic import (apply_video_review, can_generate_review, switch_session,
                               video_review_request)


def state():
    return dict(active_session=("stu_demo", "p1"), attempt_id="a", attempt_status="completed",
                tutor={"finished": True, "hint": "已完成"},
                timeline=[{"type": "submit", "timestamp": 0, "content": "作答"}],
                ocr_edit_src="编辑草稿", video_result={"media_type": "video", "student_id": "stu_demo",
                    "problem_id": "p1", "attempt_id": "a", "run_id": "b" * 32,
                    "review": None, "rounds": [{"answer": "F=49", "finished": True}],
                    "timings": {"② 感知层": 5, "④ 驱动教学链路": 8}})


def response():
    return dict(attempt_id="a", run_id="b" * 32, review_status="ready",
                input_revision=0, generation_fingerprint="gen0", review_input_fingerprint="input0",
                review_source_revisions={}, review={"timeline": "复盘结论"}, review_scope="video_run_snapshot", review_elapsed_sec=3)


def test_completed_attempt_can_review_without_mutating_teaching_or_editor():
    current = state()
    before = deepcopy(current)
    request = video_review_request(current, "stu_demo", "p1")
    assert request == dict(student_id="stu_demo", problem_id="p1", attempt_id="a", run_id="b" * 32,
                           input_revision=0, refresh_input=False, _ui_correction_generation=0,
                           _ui_source_revisions={})
    apply_video_review(current, request, response())
    for key in ("tutor", "timeline", "ocr_edit_src", "attempt_status"):
        assert current[key] == before[key]
    assert current["video_result"]["review"] == {"timeline": "复盘结论"}
    assert current["video_result"]["timings"] == before["video_result"]["timings"]


@pytest.mark.parametrize("change", ["correction", "run", "attempt", "student", "problem"])
def test_late_review_does_not_change_new_or_corrected_session(change):
    current = state()
    request = video_review_request(current, "stu_demo", "p1")
    if change == "correction":
        current["video_result"]["ocr_corrected"] = True
        current["video_result"]["correction_generation"] = 1
    elif change == "run":
        current["video_result"]["run_id"] = "c" * 32
    elif change == "attempt":
        current["attempt_id"] = "new"
    else:
        switch_session(current, "other" if change == "student" else "stu_demo",
                       "p2" if change == "problem" else "p1", 2)
    before = deepcopy(current)
    with pytest.raises(ValueError, match="不匹配"):
        apply_video_review(current, request, response())
    assert current == before


@pytest.mark.parametrize("field,value", [("run_id", "c" * 32), ("attempt_id", "other"),
                                         ("review_status", "not_requested"),
                                         ("review_status", "failed"), ("review", None)])
def test_invalid_response_preserves_current_review_and_progress(field, value):
    current = state()
    request = video_review_request(current, "stu_demo", "p1")
    before = deepcopy(current)
    with pytest.raises(ValueError, match="不匹配"):
        apply_video_review(current, request, {**response(), field: value})
    assert current == before


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest
    def get(url, **kwargs):
        data = {"items": [{"problem_id": "p1", "title": "题一"}]} if url.endswith("/problems") else {"status": "ok"}
        return httpx.Response(200, json=data, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py")).run()
    for key, value in state().items():
        test.session_state[key] = value
    test.run()
    assert not test.exception
    return test


def test_button_calls_only_review_and_displays_result(app, monkeypatch):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json=response(), request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    app.button(key="generate_video_review").click().run()
    assert not app.exception
    assert len(calls) == 1 and calls[0][0].endswith("/video/review")
    assert calls[0][1]["json"] == {k: v for k, v in video_review_request(state(), "stu_demo", "p1").items()
                                      if not k.startswith("_ui_")}
    assert app.session_state.video_result["review"]["timeline"] == "复盘结论"
    assert app.session_state.ocr_edit_src == "编辑草稿"
    assert any("复盘结论" in item.value for item in app.markdown)


def test_failed_review_keeps_rounds_and_retry_button(app, monkeypatch):
    before = deepcopy(app.session_state.video_result)
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(
        502, json={"detail": "模型超时"}, request=httpx.Request("POST", url)))
    app.button(key="generate_video_review").click().run()
    assert not app.exception
    assert app.session_state.video_result == before
    assert app.session_state.ocr_edit_src == "编辑草稿"
    assert app.button(key="generate_video_review")
    assert "模型超时" in app.session_state.video_review_error


def test_corrected_video_can_generate_new_revision(app, monkeypatch):
    result = dict(app.session_state.video_result, ocr_corrected=True)
    app.session_state.video_result = result
    app.run()
    assert not app.exception
    assert app.button(key="generate_video_review").label == "根据更正内容生成新版复盘"
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, json={**response(), "input_revision": 1},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    app.button(key="generate_video_review").click().run()
    assert not app.exception
    assert calls[0]["refresh_input"] is True and calls[0]["input_revision"] == 0
    assert app.session_state.video_result["input_revision"] == 1
    assert app.session_state.video_result["review_refresh_required"] is False
    assert any("当前复盘已基于更正内容" in item.value for item in app.info)


def test_second_correction_rejects_late_refresh_with_same_input_revision():
    current = state()
    current["video_result"].update(ocr_corrected=True, correction_generation=1,
        rounds=[{"source_id": "src", "source_revision": 1}])
    request = video_review_request(current, "stu_demo", "p1")
    current["video_result"]["rounds"][0]["source_revision"] = 2
    current["video_result"]["correction_generation"] = 2
    before = deepcopy(current)
    with pytest.raises(ValueError):
        apply_video_review(current, request, {**response(), "input_revision": 1,
            "review_source_revisions": {"src": 1}})
    assert current == before


@pytest.mark.parametrize("field,value", [("input_revision", 7), ("generation_fingerprint", ""),
    ("review_input_fingerprint", ""), ("review_source_revisions", {"other": 1})])
def test_revision_identity_mismatch_preserves_editor_and_teaching(field, value):
    current = state()
    request = video_review_request(current, "stu_demo", "p1")
    before = deepcopy(current)
    with pytest.raises(ValueError):
        apply_video_review(current, request, {**response(), field: value})
    assert current == before


def test_corrected_review_failure_keeps_refresh_retry_and_history(app, monkeypatch):
    app.session_state.video_result = dict(app.session_state.video_result,
        ocr_corrected=True, correction_generation=1, review_refresh_required=True,
        review_history=[{"input_revision": 0, "review": {"timeline": "原报告"}}])
    app.run()
    before = deepcopy(app.session_state.video_result)
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(
        409, json={"detail": "材料版本冲突"}, request=httpx.Request("POST", url)))
    app.button(key="generate_video_review").click().run()
    assert not app.exception
    assert app.session_state.video_result == before
    assert app.session_state.ocr_edit_src == "编辑草稿"
    assert app.button(key="generate_video_review").label == "根据更正内容生成新版复盘"
    # 旧报告**按原版本保留并标明过期**，且**不得作为当前结论展示**（2026-10-09 决定）。
    assert any("历史复盘" in item.label and "过期" in item.label for item in app.expander)
    assert "材料版本冲突" in app.session_state.video_review_error


# ---- 生成入口的判定（决定 1 / 3）------------------------------------------------

def photo_state():
    """一张照片的结果：**没有 run_id**，服务端说 `review_available=false`。"""
    return dict(active_session=("stu_demo", "p1"), attempt_id="a", attempt_status="completed",
                tutor={"finished": True, "hint": "下一步：检查方向"},
                video_result={"media_type": "image", "student_id": "stu_demo",
                              "problem_id": "p1", "attempt_id": "a",
                              "recognized_text": "F=49", "review": None,
                              "review_status": "not_requested", "review_available": False,
                              "review_refresh_required": True,   # 旧报告**需要更新**……
                              "review_history": [{"input_revision": 0, "superseded": True,
                                                  "review": {"timeline": "原报告"}}]})


def test_review_button_needs_a_server_side_generation_path():
    """没有 `review_available` 的旧响应 ⇒ 回落原判断（不改变既有行为）。"""
    current = state()
    assert can_generate_review(current, "stu_demo", "p1") is True
    current["video_result"]["review_available"] = False
    assert can_generate_review(current, "stu_demo", "p1") is False


def test_photo_gets_no_generation_button_even_when_a_report_is_stale():
    """**决定 1 的核心**：`review_refresh_required` ≠ 可以生成。

    照片更正后正是「旧报告需要更新、却无处可生成」⇒ **只提示，不给按钮**。
    若这里返回 True，界面就会挂出一个按不动的「重新生成」。
    """
    current = photo_state()
    assert current["video_result"]["review_refresh_required"] is True
    assert can_generate_review(current, "stu_demo", "p1") is False


def test_photo_ready_report_still_has_no_later_entry_point():
    """**决定 3**：`review_available` 说的是「当前材料能否由后续接口请求复盘」，
    与「本次是否已生成报告」**独立** ⇒ 兼容调用 `include_review=true` 拿到 `ready`，
    它照样可以是 `false`。"""
    current = photo_state()
    current["video_result"].update(review_status="ready", review={"timeline": "报告"})
    assert can_generate_review(current, "stu_demo", "p1") is False
