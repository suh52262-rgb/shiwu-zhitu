"""Behavior regressions for student/problem isolation and media feedback."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from frontend.ui_logic import (clear_media, current_attempt_id, media_notices,
                               reset_session, restore_attempt, switch_session,
                               # ⚠️ `app_function` 手工搭 exec 的命名空间，
                               # **产品代码用到什么，这里就得补什么** ——
                               # 否则报的是 `NameError` 而看起来像产品 bug。
                               invalidate_video_job, retain_video_job,
                               video_job_context, current_video_job,
                               apply_video_job_result, apply_ocr_correction,
                               correction_unavailable, video_review_request,
                               apply_video_review, rollback_video_job_start,
                               record_photo_failure, remember_request_id,
                               RECOVER_ABSENT, RECOVER_FOUND, RECOVER_UNKNOWN)

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


def test_switch_restores_only_its_own_progress_and_clears_uploads():
    state = {}
    switch_session(state, "alice", "p1", 1)
    state.update(attempt_id="a1", timeline=[{"type": "submit", "content": "F=ma", "timestamp": 2}],
                 tutor={"finished": False}, video_result={"submitted": 1}, recommendations={"x": 1})
    version = state["upload_version"]
    switch_session(state, "alice", "p2", 3)
    assert state["attempt_id"] is None and state["timeline"] == []
    assert "video_result" not in state and "recommendations" not in state
    assert state["upload_version"] > version
    switch_session(state, "bob", "p1", 4)
    assert state["attempt_id"] is None
    switch_session(state, "alice", "p1", 5)
    assert current_attempt_id(state) == "a1"
    assert state["timeline"][0]["content"] == "F=ma"
    reset_session(state, 6)
    switch_session(state, "alice", "p2", 7)
    switch_session(state, "alice", "p1", 8)
    assert state["attempt_id"] is None and state["timeline"] == []


def test_restore_rebuilds_timeline_and_rejects_foreign_or_terminal_records():
    state = {"tutor": {"graph_mode": True}}
    record = dict(attempt_id="a", student_id="s", problem_id="p", status="in_progress",
                  current_node="S2", events=[dict(event_type="submit", timestamp=12, content="F=ma")],
                  hints=[dict(hint_level=2, hint="检查受力")])
    assert restore_attempt(state, record, "s", "p")
    assert state["timeline"] == [dict(type="submit", timestamp=12, content="F=ma")]
    assert state["tutor"]["hint"] == "检查受力"
    record["state"] = {"current_step": "受力分析", "diagnosed_issue": "新诊断", "hint": "", "graph_mode": False}
    assert restore_attempt(state, record, "s", "p")
    assert state["tutor"]["issue"] == "新诊断"
    assert state["tutor"]["current_step"] == "受力分析"
    assert state["tutor"]["hint"] == ""  # The latest L0 must not replay an old hint.
    assert not restore_attempt(state, record, "other", "p")
    assert not restore_attempt(state, {**record, "status": "completed"}, "s", "p")
    state["tutor"]["finished"] = True
    assert current_attempt_id(state) is None
    state["attempt_status"] = "completed"
    state["tutor"] = {"hint": "请求失败"}
    assert current_attempt_id(state) is None


def test_storage_and_partial_coverage_are_truthful():
    result = dict(media_type="video", upload_removed=False,
                  coverage=dict(truncated=True, covered_until_sec=60, video_total_sec=180,
                                truncate_reason="frame_limit", failed_frame_count=1),
                  teaching_stop_reason="completed", remaining_snapshots=3)
    view = media_notices(result)
    assert len(view["warnings"]) == 3
    assert "60" in view["warnings"][0] and "180" in view["warnings"][0]
    assert "尚未确认删除" in view["storage"]
    assert "已从服务器删除" in media_notices({"upload_removed": True})["storage"]
    assert "未知" in media_notices({})["storage"]
    assert "原图保留状态未提供" in media_notices({"media_type": "image"})["storage"]
    assert "原图已保留" in media_notices({"media_type":"image", "rounds":[{"image_evidence_status":"retained"}]})["storage"]


class Session(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value


def app_function(name, state, post, recover=None):
    """Execute the real request handler with controlled UI/network boundaries.

    `recover` 是"按上传编号找回"的结果（默认**没找到**）——
    它自己做 GET 也碰 `st.query_params`，所以在这一层替换掉，
    让被测的是**产品 handler 的分支逻辑**而不是网络。
    """
    import time
    import uuid as _uuid
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = dict(st=SimpleNamespace(session_state=state, spinner=lambda *a: nullcontext(),
                                  balloons=lambda: None, query_params={}),
               httpx=SimpleNamespace(post=post), BACKEND_URL="http://test", STUDENT_ID="s", problem_id="p",
               clear_media=clear_media, current_attempt_id=current_attempt_id, time=time, uuid=_uuid,
               # ⚠️ 产品 handler 用到、而 hand-built env 里缺的，一律 NameError。
               # 新加功能时**这份清单要跟着长** —— 否则失败信息指向产品，实际在测试。
               invalidate_video_job=invalidate_video_job,
               rollback_video_job_start=rollback_video_job_start,
               retain_video_job=retain_video_job,
               video_job_context=video_job_context,
               current_video_job=current_video_job,
               apply_video_job_result=apply_video_job_result,
               apply_ocr_correction=apply_ocr_correction,
               correction_unavailable=correction_unavailable,
               video_review_request=video_review_request,
               apply_video_review=apply_video_review,
               record_photo_failure=record_photo_failure,
               remember_request_id=remember_request_id,
               RECOVER_ABSENT=RECOVER_ABSENT, RECOVER_FOUND=RECOVER_FOUND,
               RECOVER_UNKNOWN=RECOVER_UNKNOWN,
               recover_video_job_by_request=recover or (lambda _rid: RECOVER_ABSENT),
               add_event=lambda kind, text: state["timeline"].append({"type": kind, "content": text}))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), env)
    return env[name]


def test_photo_failure_clears_old_success_and_sends_active_attempt():
    state = Session(attempt_id="a", tutor={}, timeline=[], video_result={"submitted": 99})
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(502, json={"detail": "识别服务离线"}, request=httpx.Request("POST", url))
    upload = SimpleNamespace(name="x", type="image/png", getvalue=lambda: b"data")
    app_function("call_image_analyze", state, post)(upload)
    assert calls[0]["data"]["attempt_id"] == "a"
    assert "video_result" not in state
    # ⚠️ 2026-10-09 决定：失败原因**不再写进 `tutor["hint"]`**。
    # `hint` 是分级提示的载体（契约 §1.1 不变式①），把错误塞进去，
    # 学生就会把「处理失败」当成一条教学提示读。
    assert "识别服务离线" in state["photo_error"]
    assert "hint" not in state["tutor"]
    # 重试要能**原样重发**同一张照片，学生不该再选一次文件。
    assert state["photo_retry"] == {"name": "x", "data": b"data", "mime": "image/png"}


def test_empty_photo_recognition_is_a_retryable_failure_not_a_hint():
    """「没识别到内容」也是失败，要走同一条路 —— 有原因、有重试。"""
    state = Session(attempt_id="a", tutor={}, timeline=[], video_result={"submitted": 99})
    def post(url, **kwargs):
        return httpx.Response(200, json={"rounds": [], "review_status": "not_requested"},
                              request=httpx.Request("POST", url))
    upload = SimpleNamespace(name="x", type="image/png", getvalue=lambda: b"data")
    app_function("call_image_analyze", state, post)(upload)
    assert "没有识别到作答内容" in state["photo_error"]
    assert state["photo_retry"]["data"] == b"data"


def test_successful_photo_clears_the_retry_entry():
    state = Session(attempt_id="a", tutor={}, timeline=[], video_result={},
                    active_session=("s", "p"), attempt_status="in_progress")
    def post(url, **kwargs):
        return httpx.Response(200, json={"attempt_id": "a", "recognized_text": "F=49",
                                         "review_status": "not_requested", "review_available": False,
                                         "rounds": [{"answer": "F=49", "hint": "检查方向"}]},
                              request=httpx.Request("POST", url))
    upload = SimpleNamespace(name="x", type="image/png", getvalue=lambda: b"data")
    app_function("call_image_analyze", state, post)(upload)
    assert "photo_error" not in state and "photo_retry" not in state
    assert state["video_result"]["review_status"] == "not_requested"


def test_video_submission_unknown_preserves_existing_result_and_does_not_resubmit():
    state = Session(active_session=("s", "p"), attempt_id="a", tutor={"hint": "已有提示"},
                    timeline=[{"type": "submit", "content": "F=49"}], video_result={"submitted": 99})
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        raise httpx.ReadTimeout("上传后连接中断")
    upload = SimpleNamespace(name="x", type="video/mp4", getvalue=lambda: b"data")
    # 默认夹具的找回结果是 **ABSENT（404 确认没有）**
    app_function("call_video_analyze", state, post)(upload)
    assert calls[0][0].endswith("/video/jobs") and calls[0][1]["data"]["attempt_id"] == "a"
    assert len(calls) == 1
    assert state["video_result"] == {"submitted": 99}
    assert state["tutor"] == {"hint": "已有提示"}
    assert state["timeline"] == [{"type": "submit", "content": "F=49"}]
    assert "创建未确认" in state["video_job_error"]
    assert "按编号" in state["video_job_error"]
    assert "当前未找到任务" in state["video_job_error"]
    assert "原请求仍可能稍后完成" in state["video_job_error"]
    assert "同一编号" in state["video_job_error"]
    assert state["video_job_request_id"] == calls[0][1]["data"]["client_request_id"]


def test_an_unconfirmed_lookup_must_not_say_you_may_reupload():
    """⚠️ **查不通 ≠ 没建成** —— 这是本轮修的一个真实缺陷。

    旧版把"网络异常 / 服务端 500"与"404 确认没有"一律当成 False，
    于是都显示成「没有找到任务 ⇒ 可以重新上传，不会重复」——
    **而那时任务可能已经建成并开始付费处理了。**
    """
    state = Session(active_session=("s", "p"), attempt_id="a", tutor={}, timeline=[])
    def post(url, **kwargs):
        raise httpx.ReadTimeout("上传后连接中断")
    upload = SimpleNamespace(name="x", type="video/mp4", getvalue=lambda: b"data")
    app_function("call_video_analyze", state, post,
                 recover=lambda _rid: RECOVER_UNKNOWN)(upload)
    message = state["video_job_error"]
    assert "没能确认" in message or "仍然未知" in message
    assert "请不要直接重新上传" in message
    # ⚠️ **绝不能出现"可以重新上传"** —— 那是重复付费的入口
    assert "可以重新上传" not in message


def test_the_request_id_is_kept_so_it_can_be_queried_later():
    """编号要**留在会话里**：结果未知时最需要过后再查一次，而刷新会把局部变量冲掉。"""
    state = Session(active_session=("s", "p"), attempt_id="a", tutor={}, timeline=[])
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        raise httpx.ReadTimeout("上传后连接中断")
    upload = SimpleNamespace(name="x", type="video/mp4", getvalue=lambda: b"data")
    app_function("call_video_analyze", state, post,
                 recover=lambda _rid: RECOVER_UNKNOWN)(upload)
    sent = calls[0]["data"]["client_request_id"]
    assert state["video_job_request_id"] == sent, "编号没有留在会话里"
    assert sent in state["video_job_error"], "编号也没写进提示"


def test_a_recovered_submission_clears_the_pending_request_id():
    """找回成功 ⇒ 编号使命结束，不该再提示"有一次结果没确认"。"""
    state = Session(active_session=("s", "p"), attempt_id="a", tutor={}, timeline=[],
                    video_job_request_id="old")
    def post(url, **kwargs):
        raise httpx.ReadTimeout("上传后连接中断")
    upload = SimpleNamespace(name="x", type="video/mp4", getvalue=lambda: b"data")
    app_function("call_video_analyze", state, post,
                 recover=lambda _rid: RECOVER_FOUND)(upload)
    assert "video_job_request_id" not in state
    assert "查到了任务" in state["video_job_error"]


def test_video_submission_carries_a_client_request_id():
    """契约 §2.2：页面在**一次上传动作开始时**生成编号并带上。"""
    state = Session(active_session=("s", "p"), attempt_id="a", tutor={}, timeline=[])
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"job_id": "j" * 32, "student_id": "s",
                                         "problem_id": "p", "status": "queued",
                                         "phase": "queued"},
                              request=httpx.Request("POST", url))
    upload = SimpleNamespace(name="x", type="video/mp4", getvalue=lambda: b"data")
    app_function("call_video_analyze", state, post)(upload)
    sent = calls[0]["data"]["client_request_id"]
    assert len(sent) == 36 and sent.count("-") == 4, f"不是 UUID：{sent}"


def test_a_timed_out_submission_that_actually_landed_is_taken_over():
    """**超时 ≠ 没建成。** 服务端可能已经收下并开始处理了。

    契约 §2.2 把动作定死了：**先按编号查询，不得直接重传** ——
    重传就是第二次付费，还可能多出一条作答记录。
    """
    state = Session(active_session=("s", "p"), attempt_id="a", tutor={"hint": "已有提示"},
                    timeline=[], video_result={"submitted": 99},
                    video_job_generations={("s", "p"): 1})
    def post(url, **kwargs):
        raise httpx.ReadTimeout("上传后连接中断")
    upload = SimpleNamespace(name="x", type="video/mp4", getvalue=lambda: b"data")
    app_function("call_video_analyze", state, post,
                 recover=lambda _rid: RECOVER_FOUND)(upload)
    # ⚠️ 代数语义：`invalidate_video_job` 在动作开始时就 +1（作废此前接管的任务）；
    # **找回成功只是不回滚** ⇒ 停在 2。若这里回滚成 1，那才是 bug：
    # 那意味着"刚接管的任务"又被判定为过期。
    assert state["video_job_generations"] == {("s", "p"): 2}, "找回成功却回滚了代数"
    assert "查到了任务" in state["video_job_error"]
    assert "不会重复上传" in state["video_job_error"]


@pytest.mark.parametrize("handler", ["call_image_analyze", "call_video_analyze"])
def test_completed_media_request_does_not_reuse_terminal_id(handler):
    state = Session(attempt_id="done", tutor={"finished": True}, timeline=[])
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"rounds": []}, request=httpx.Request("POST", url))
    upload = SimpleNamespace(name="x", type="image/png", getvalue=lambda: b"data")
    app_function(handler, state, post)(upload)
    assert "attempt_id" not in calls[0]["data"]


def test_streamlit_switch_resume_reset_and_complete_text_request(monkeypatch):
    from streamlit.testing.v1 import AppTest
    requests = []
    def get(url, **kwargs):
        if url.endswith("/problems"):
            payload = {"items": [{"problem_id": "p1", "title": "一"}, {"problem_id": "p2", "title": "二"}]}
        elif url.endswith("/health"):
            payload = {"status": "ok"}
        elif "/attempt/" in url:
            requests.append(("GET", url, kwargs))
            payload = dict(attempt_id="a1", student_id="stu_demo", problem_id="p1", status="in_progress",
                           events=[dict(timestamp=3, event_type="submit", content="服务器恢复的步骤")])
        else:
            return httpx.Response(404, json={}, request=httpx.Request("GET", url))
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    def post(url, **kwargs):
        requests.append(("POST", url, kwargs))
        payload = {"attempt_id": "a1", "finished": False, "hint": "继续", "graph_mode": False}
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    def click(label):
        next(b for b in at.button if b.label == label).click().run()
        assert not at.exception
    at.text_area[0].set_value("F=ma")
    click("提交当前步骤")
    first = next(r for r in requests if r[0] == "POST")
    from app.api.attempts import AnalyzeRequest
    assert first[2]["json"]["attempt_id"] == ""
    AnalyzeRequest.model_validate(first[2]["json"])
    at.selectbox[0].select("p2").run()
    assert at.session_state["timeline"] == []
    at.selectbox[0].select("p1").run()
    assert at.session_state["timeline"][0]["content"] == "服务器恢复的步骤"
    assert not any("/reset" in r[1] for r in requests)
    click("提交当前步骤")
    assert requests[-1][2]["json"]["attempt_id"] == "a1"
    at.session_state["tutor"] = {"finished": True}
    click("提交当前步骤")
    assert requests[-1][2]["json"]["attempt_id"] == ""
    AnalyzeRequest.model_validate(requests[-1][2]["json"])
    saved_timeline = list(at.session_state["timeline"])
    def fail_reset(url, **kwargs):
        return httpx.Response(503, json={}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fail_reset)
    click("重新开始本题")
    assert at.session_state["timeline"] == saved_timeline
    assert any("重新开始失败" in e.value for e in at.error)
    monkeypatch.setattr(httpx, "post", post)
    click("重新开始本题")
    assert requests[-1][1].endswith("/attempt/reset")
    assert at.session_state["timeline"] == []


def test_streamlit_renders_full_ocr_and_coverage_without_false_deletion(monkeypatch):
    from streamlit.testing.v1 import AppTest
    def get(url, **kwargs):
        payload = {"items": [{"problem_id": "p", "title": "题"}]} if url.endswith("/problems") else {"status": "ok"}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    at = AppTest.from_file(str(APP)).run()
    full = "这是一段超过七十字的识别原文。" * 12 + "完整尾部49N"
    at.session_state["tutor"] = {"hint": "继续", "current_method": "能量方法", "graph_mode": False}
    at.session_state["video_result"] = dict(media_type="video", upload_removed=False, submitted=1,
        rounds=[dict(round=1, answer=full, current_method="能量方法")],
        coverage=dict(truncated=True, covered_until_sec=30, video_total_sec=100))
    at.run()
    assert not at.exception
    assert full in [t.value for t in at.text]
    assert at.dataframe[0].value.iloc[0]["**识别到的作答**"] == full
    assert any("仅分析了部分范围" in w.value for w in at.warning)
    assert any("尚未确认删除" in c.value for c in at.caption)
    assert not any("已从服务器删除" in c.value for c in at.caption)
    assert any("能量方法" in c.value for c in at.caption)
