"""Background video jobs retain scope across reruns and never resubmit on polling."""
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from frontend.ui_logic import (invalidate_video_job, video_job_context, retain_video_job,
    current_video_job, apply_video_job_result, switch_session, rollback_video_job_start)


def state():
    return {"active_session": ("stu_demo", "p1"), "attempt_id": "old", "timeline": [],
            "tutor": {"hint": "旧提示"}, "start_ts": 0, "ocr_edit_src": "编辑草稿"}


def job(status="queued"):
    return {"job_id": "j1", "student_id": "stu_demo", "problem_id": "p1",
            "status": status, "phase": "teaching", "progress": {"completed": 1, "total": 2},
            "result": {"student_id": "stu_demo", "problem_id": "p1", "attempt_id": "new",
                "media_type": "video", "submitted": 1, "rounds": [{"answer": "F=49", "hint": "新提示"}]}}


def test_switch_back_can_resume_pending_job_but_new_input_invalidates():
    current = state()
    context = video_job_context(current, "stu_demo", "p1")
    retain_video_job(current, context, job())
    switch_session(current, "stu_demo", "p2", 1)
    assert current_video_job(current, "stu_demo", "p1") is None
    switch_session(current, "stu_demo", "p1", 2)
    assert current_video_job(current, "stu_demo", "p1")["job_id"] == "j1"
    invalidate_video_job(current)
    before = deepcopy(current)
    with pytest.raises(ValueError):
        apply_video_job_result(current, context, job("succeeded"), 3)
    assert current == before


def test_success_applies_once_and_preserves_editor():
    current = state()
    context = video_job_context(current, "stu_demo", "p1")
    retain_video_job(current, context, job())
    assert apply_video_job_result(current, context, job("succeeded"), 3)
    assert current["attempt_id"] == "new" and current["tutor"]["hint"] == "新提示"
    assert current["ocr_edit_src"] == "编辑草稿"
    assert not apply_video_job_result(current, context, job("succeeded"), 4)
    assert len(current["timeline"]) == 1


@pytest.mark.parametrize("status", ["queued", "running", "failed", "interrupted"])
def test_non_success_keeps_previous_teaching(status):
    current = state()
    before = deepcopy(current)
    assert not apply_video_job_result(current, video_job_context(current, "stu_demo", "p1"), job(status), 3)
    for key in before:
        assert current[key] == before[key]


def test_foreign_job_rejected_without_state_changes():
    current = state()
    before = deepcopy(current)
    with pytest.raises(ValueError):
        retain_video_job(current, video_job_context(current, "stu_demo", "p1"), dict(job(), student_id="other"))
    assert current == before


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest
    def get(url, **kwargs):
        payload = {"items": [{"problem_id": "p1", "title": "题一"}]} if url.endswith("/problems") else {"status": "ok"}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py")).run()
    for key, value in state().items():
        app.session_state[key] = value
    app.session_state.pending_video_jobs = {("stu_demo", "p1"): {**job(), "generation": 0}}
    app.run()
    assert not app.exception
    return app


def test_manual_refresh_get_only_and_success_updates_view(app, monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        payload = job("succeeded") if "/video/jobs/" in url else (
            {"items": [{"problem_id": "p1", "title": "题一"}]} if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Polling resubmitted a paid job"))
    app.button(key="refresh_video_job").click().run()
    assert not app.exception
    requests = [item for item in calls if "/video/jobs/" in item[0]]
    assert len(requests) == 1
    assert requests[0][1]["params"] == {"student_id": "stu_demo", "problem_id": "p1"}
    assert app.session_state.attempt_id == "new"
    assert app.session_state.ocr_edit_src == "编辑草稿"
    assert not any(button.key == "refresh_video_job" for button in app.button)


def test_interrupted_task_does_not_retry_upload(app, monkeypatch):
    app.session_state.pending_video_jobs = {("stu_demo", "p1"): {**job("interrupted"), "generation": 0}}
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Interrupted task restarted automatically"))
    app.run()
    assert not app.exception
    assert any("不会自动重新处理" in item.value for item in app.caption)
    assert app.session_state.tutor["hint"] == "旧提示"


def test_url_job_reference_survives_fresh_page_session_without_post(app, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Reload resubmitted a paid job"))
    fresh = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py"))
    fresh.query_params["video_job"] = "j1"
    fresh.query_params["video_job_problem"] = "p1"
    fresh.query_params["video_job_student"] = "stu_demo"
    fresh.run()
    assert not fresh.exception
    assert fresh.button(key="refresh_video_job")
    assert fresh.session_state.pending_video_jobs[("stu_demo", "p1")]["job_id"] == "j1"
    assert fresh.session_state.attempt_id is None


def test_failed_job_start_restores_prior_queryable_task_without_undoing_newer_action():
    current = state()
    old_context = video_job_context(current, "stu_demo", "p1")
    retain_video_job(current, old_context, job())
    invalidate_video_job(current)
    failed_context = video_job_context(current, "stu_demo", "p1")
    assert current_video_job(current, "stu_demo", "p1") is None
    rollback_video_job_start(current, failed_context)
    assert current_video_job(current, "stu_demo", "p1")["job_id"] == "j1"
    invalidate_video_job(current)
    failed_context = video_job_context(current, "stu_demo", "p1")
    invalidate_video_job(current)  # Another successful input invalidates prior task.
    before = deepcopy(current)
    rollback_video_job_start(current, failed_context)
    assert current == before


@pytest.mark.parametrize("mutation", ["owner", "attempt", "rounds"])
def test_bad_success_does_not_replace_pending_job_or_teaching(mutation):
    current = state()
    context = video_job_context(current, "stu_demo", "p1")
    retain_video_job(current, context, job())
    bad = job("succeeded")
    if mutation == "owner":
        bad["result"]["student_id"] = "other"
    elif mutation == "attempt":
        bad["attempt_id"] = "unrelated"
    else:
        bad["result"]["rounds"] = None
    before = deepcopy(current)
    with pytest.raises(ValueError):
        apply_video_job_result(current, context, bad, 3)
    assert current == before


def test_late_progress_cannot_revert_terminal_or_newer_snapshot():
    current = state()
    context = video_job_context(current, "stu_demo", "p1")
    retain_video_job(current, context, {**job("running"), "updated_at": "2026-10-09T02:00:00+00:00"})
    before = deepcopy(current)
    with pytest.raises(ValueError):
        retain_video_job(current, context, {**job(), "updated_at": "2026-10-09T01:00:00+00:00"})
    assert current == before
    apply_video_job_result(current, context, job("succeeded"), 3)
    before = deepcopy(current)
    with pytest.raises(ValueError):
        retain_video_job(current, context, job("running"))
    assert current == before


def test_foreign_student_url_reference_is_not_restored(app, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Reload resubmitted a paid job"))
    fresh = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py"))
    fresh.query_params["video_job"] = "j1"
    fresh.query_params["video_job_problem"] = "p1"
    fresh.query_params["video_job_student"] = "other"
    fresh.run()
    assert not fresh.exception
    assert not any(button.key == "refresh_video_job" for button in fresh.button)
    assert not fresh.session_state.get("pending_video_jobs", {})


def test_failed_text_submission_keeps_pending_video_queryable(app, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Response(
        503, json={"detail": "unavailable"}, request=httpx.Request("POST", a[0])))
    app.text_area[0].set_value("F=ma").run()
    next(button for button in app.button if button.label == "提交当前步骤").click().run()
    assert not app.exception
    assert app.button(key="refresh_video_job")
    assert app.session_state.pending_video_jobs[("stu_demo", "p1")]["job_id"] == "j1"


# ================================================================ 自动查询（契约 T3）
#
# 契约原文：「默认按 **3 秒间隔**自动查询运行中的任务，终态停止；保留手动刷新。
# 查询失败**不重新上传、不再次调用模型**。」
#
# ⚠️ AppTest 不会推进时间，所以这里验的是**分支选择**（注册不注册定时器），
# 而不是"真的每 3 秒跳一次" —— 后者要在真浏览器里看。

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


def captions(at) -> str:
    return "\n".join(str(item.value) for item in at.caption)


def build_app(monkeypatch, get, job_state):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(httpx, "get", get)
    at = AppTest.from_file(str(APP)).run()
    for key, value in state().items():
        at.session_state[key] = value
    at.session_state.pending_video_jobs = {("stu_demo", "p1"): {**job_state, "generation": 0}}
    at.run()
    assert not at.exception, at.exception
    return at


def job_get(record, payload_for):
    def get(url, **kwargs):
        record.append((url, kwargs))
        payload = payload_for(url)
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    return get


def test_a_running_task_is_polled_and_the_page_says_so(monkeypatch):
    record = []
    get = job_get(record, lambda url: job("running") if "/video/jobs/" in url
                  else ({"items": [{"problem_id": "p1", "title": "题一"}]}
                        if url.endswith("/problems") else {"status": "ok"}))
    at = build_app(monkeypatch, get, job("running"))
    assert "每 3 秒自动查询一次" in captions(at), captions(at)
    # 自动查询**确实发了一次 GET**（片段在首次脚本运行时就执行）
    assert any("/video/jobs/" in url for url, _ in record)


def test_a_terminal_task_stops_the_auto_query(monkeypatch):
    """**终态停止** —— 这句话本身也要让学生看到，否则他不知道还要不要等。"""
    record = []
    get = job_get(record, lambda url: job("succeeded") if "/video/jobs/" in url
                  else ({"items": [{"problem_id": "p1", "title": "题一"}]}
                        if url.endswith("/problems") else {"status": "ok"}))
    at = build_app(monkeypatch, get, job("succeeded"))
    assert "自动查询已停止" in captions(at), captions(at)


def test_the_auto_query_can_be_turned_off(monkeypatch):
    """留了开关（运维可能想关轮询）；关掉时要说清现在是手动。"""
    monkeypatch.setenv("VIDEO_JOB_POLL_SECONDS", "0")
    record = []
    get = job_get(record, lambda url: job("running") if "/video/jobs/" in url
                  else ({"items": [{"problem_id": "p1", "title": "题一"}]}
                        if url.endswith("/problems") else {"status": "ok"}))
    at = build_app(monkeypatch, get, job("running"))
    assert "自动查询已关闭" in captions(at), captions(at)
    assert not any("/video/jobs/" in url for url, _ in record), "关掉了还在轮询"


def test_auto_query_only_issues_get(monkeypatch):
    """**只发 GET** —— 这条路径上没有任何"重新提交"的可能。"""
    record = []
    get = job_get(record, lambda url: job("running") if "/video/jobs/" in url
                  else ({"items": [{"problem_id": "p1", "title": "题一"}]}
                        if url.endswith("/problems") else {"status": "ok"}))
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **kw: pytest.fail("自动查询重新提交了付费任务"))
    at = build_app(monkeypatch, get, job("running"))
    assert not at.exception
    assert any("/video/jobs/" in url for url, _ in record)


def test_a_failed_auto_query_pauses_instead_of_hammering_or_resubmitting(monkeypatch):
    """查询失败 ⇒ **暂停自动查询**（改回手动），并且**不重新上传、不重跑模型**。

    不暂停的话，后端不通时这里会每 3 秒敲一次。
    """
    job_calls = []
    def get(url, **kwargs):
        if "/video/jobs/" in url:
            job_calls.append(url)
            raise httpx.ConnectError("backend down")
        payload = ({"items": [{"problem_id": "p1", "title": "题一"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **kw: pytest.fail("查询失败后重新上传了"))
    at = build_app(monkeypatch, get, job("running"))
    assert at.session_state.get("video_job_poll_paused") is True
    assert "自动查询已暂停" in captions(at), captions(at)
    # 任务编号与已有作答都还在 —— 暂停 ≠ 丢任务
    assert at.session_state.pending_video_jobs[("stu_demo", "p1")]["job_id"] == "j1"
    assert at.button(key="refresh_video_job"), "暂停后手动刷新必须还在"


# ================================================================ 结果未知时的补救入口
#
# ⚠️ **有编号、没任务 = 结果仍然未知**（不是"没建成"）。
# 这时页面上**只能给"按编号再查"**，**不能给"重新上传"** ——
# 后者是重复付费的入口。

PENDING_RID = "11111111-2222-3333-4444-555555555555"


def pending_at(monkeypatch, get):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **kw: pytest.fail("结果未知时重新上传了付费任务"))
    at = AppTest.from_file(str(APP)).run()
    for key, value in state().items():
        at.session_state[key] = value
    at.session_state.video_job_request_id = PENDING_RID
    at.run()
    assert not at.exception, at.exception
    return at


def test_an_unresolved_upload_offers_a_lookup_and_never_a_reupload(monkeypatch):
    def get(url, **kwargs):
        payload = ({"items": [{"problem_id": "p1", "title": "题一"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    at = pending_at(monkeypatch, get)
    assert at.button(key="recover_video_job"), "没有提供按编号查询的入口"
    text = captions(at) + "\n".join(str(b.label) for b in at.button)
    assert "结果还没确认" in text
    assert "可以重新上传" not in text, "把'未知'说成了'可以重传'"


def test_the_lookup_button_says_when_the_server_confirms_there_is_nothing(monkeypatch):
    """404 ⇒ **服务端确认没有** ⇒ 这时才可以说"可以放心重新上传"。"""
    def get(url, **kwargs):
        if "/video/jobs/by-request/" in url:
            return httpx.Response(404, json={"detail": "没有这个编号对应的视频任务"},
                                  request=httpx.Request("GET", url))
        payload = ({"items": [{"problem_id": "p1", "title": "题一"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    at = pending_at(monkeypatch, get)
    at.button(key="recover_video_job").click().run()
    assert not at.exception
    assert "同一编号" in "\n".join(str(i.value) for i in at.success)


def test_the_lookup_button_stays_cautious_when_it_cannot_confirm(monkeypatch):
    """连不上 / 5xx ⇒ **仍未知**，措辞必须说"请不要重新上传"。"""
    def get(url, **kwargs):
        if "/video/jobs/by-request/" in url:
            raise httpx.ConnectError("backend down")
        payload = ({"items": [{"problem_id": "p1", "title": "题一"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    at = pending_at(monkeypatch, get)
    at.button(key="recover_video_job").click().run()
    assert not at.exception
    warning = "\n".join(str(i.value) for i in at.warning)
    assert "没能确认" in warning
    assert "请不要重新上传" in warning
