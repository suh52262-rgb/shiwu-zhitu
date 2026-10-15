"""契约 §2.1 的不变式：`review_status` 三值 与 `review_available`。

契约 §四⑤ 写的是「**没有测试的不变式等于没有**」。
2026-10-09 的 T0 裁决把「学生没要复盘」与「要了但生成失败」从同一个
`available` 里拆开 —— 本文件就是那条裁决的落点，**逐条钉住，防止再被压回一个值**。

为什么值得单独一个文件：这两个状态看起来只差一个字符串，
但**处置完全相反**（前者正常，后者该提示"可重试"）。
把它们混起来时，前端既没法提示失败，也没法在成功时放心。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import video
from app.main import app
from app.services import review_builder
from app.vision import service
from fakes import install_fake_llm

BASE = {"student_id": "contract_probe", "problem_id": "phy_001"}

#: 契约 §2.1 冻结的**全部**取值。任何别的字符串都是回归。
FROZEN_STATUSES = {"not_requested", "ready", "failed"}


def _report(timeline="report"):
    return SimpleNamespace(model_dump=lambda: {
        "timeline": timeline, "sticking_points": [], "key_concept": "机械能守恒条件",
        "next_step": "复核守恒条件", "encouragement": "继续", "answer_check": None})


@pytest.fixture
def client(monkeypatch, tmp_path):
    """真实 HTTP + 真实 LangGraph，只替换外部 IO（视觉与复盘模型）。"""
    monkeypatch.setattr(video, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(video, "RUNS_DIR", tmp_path / "video_runs")
    monkeypatch.setattr(video, "UPLOAD_DIR", tmp_path / "video_runs" / "uploads")
    monkeypatch.setattr(service, "call_vision", lambda *a, **kw: (
        json.dumps({"text": "mgh = mv^2/2", "has_content": True}), {}))

    def pipeline(path, out_dir, options, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps({"frames": [
            {"frame_type": "keyframe", "timestamp_sec": ts,
             "text": "mgh" if ts == 0 else "mgh = mv^2/2"} for ts in (0.0, 25.0)
        ]}), encoding="utf-8")
        return {"coverage": {"truncated": False, "video_total_sec": 30.0,
                             "covered_until_sec": 30.0}, "failed": 0}

    monkeypatch.setattr(service, "run_pipeline", pipeline)
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs.get("llm"))
        return _report()

    monkeypatch.setattr(review_builder, "review_trajectory", generate)
    install_fake_llm(monkeypatch)
    # generation_context also pins provider identity before calling our review
    # substitute. Give it a complete IO fake; no operator .env or key is needed.
    from app.services import llm as llm_service
    identity_client = SimpleNamespace(
        name="unit", kind="fake", model="unit-review", base_url="https://unit.invalid/v1",
        temperature=0, timeout=1, max_retries=0, max_tokens=100, thinking="off")
    monkeypatch.setattr(llm_service, "get_llm", lambda *args, **kwargs: identity_client)
    return TestClient(app), calls


def _post_photo(client, **form):
    return client.post("/video/analyze_image", data={**BASE, **form},
                       files={"file": ("work.png", b"image")})


def _post_video(client, **form):
    return client.post("/video/analyze", data={**BASE, "every_sec": 20, **form},
                       files={"file": ("work.mp4", b"video")})


# ---------------------------------------------------------------- 照片路径

def test_photo_defaults_to_requesting_the_review(client):
    """**默认 true 是为了旧调用方** —— 契约 §2.1 明写，不能悄悄改成 false。"""
    http, calls = client
    body = _post_photo(http).json()
    assert body["review_status"] == "ready"
    assert body["review"] is not None
    assert len(calls) == 1


def test_photo_skip_is_not_requested_not_failed(client):
    """`include_review=false` ≠ 失败。**这正是被压掉的那个区别。**"""
    http, calls = client
    body = _post_photo(http, include_review="false").json()
    assert body["review_status"] == "not_requested"
    assert body["review"] is None
    # 没请求就不该有模型调用 —— 这是"提速"的实际含义。
    assert calls == []


def test_photo_has_no_later_generation_path(client):
    """照片**不落盘、没有 run_id** ⇒ 如实说 `false`，别挂一个按不动的按钮。"""
    http, _ = client
    assert _post_photo(http, include_review="false").json()["review_available"] is False
    assert _post_photo(http).json()["review_available"] is False


def test_photo_review_failure_does_not_destroy_the_analysis(client, monkeypatch):
    """契约 §2.1 不变式①：复盘失败**不得击穿已经算好的批改结果**。

    旧实现让它抛成 502 —— 学生连「识别到了什么、下一步做什么」都拿不到，
    而那时教学其实**已经跑完了**。
    """
    http, _ = client

    def explode(*_args, **_kwargs):
        raise RuntimeError("review backend down")

    monkeypatch.setattr(review_builder, "review_trajectory", explode)
    response = _post_photo(http)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["review_status"] == "failed"
    assert body["review"] is None          # 不得残留上一次的
    assert body["recognized_text"] == "mgh = mv^2/2"
    assert body["rounds"][0]["answer"] == "mgh = mv^2/2"


# ---------------------------------------------------------------- 视频路径

def test_video_not_requested_and_failed_are_different_values(client, monkeypatch):
    """**这一条就是本次裁决本身。** 两种情况必须给出不同的值。"""
    http, _ = client
    skipped = _post_video(http).json()                 # 默认不请求复盘
    assert skipped["review_status"] == "not_requested"

    def explode(*_args, **_kwargs):
        raise RuntimeError("review backend down")

    monkeypatch.setattr(review_builder, "review_trajectory", explode)
    broken = _post_video(http, include_review="true").json()
    assert broken["review_status"] == "failed"

    # 两者都要**照常返回教学轨迹**，不能因为复盘失败就整体失败。
    assert skipped["rounds"] and broken["rounds"]
    assert skipped["review"] is None and broken["review"] is None


def test_review_available_claims_are_truthful(client):
    """`review_available` 说「可以稍后生成」，就得**真的**能生成。

    这是它区别于 `review_status` 的地方：它**不是**生成状态，
    所以「没请求」（not_requested）时它照样可以是 `true`。
    """
    http, _ = client
    run = _post_video(http).json()
    assert run["review_status"] == "not_requested"
    assert run["review_available"] is True

    response = http.post("/video/review", json={
        "student_id": BASE["student_id"], "problem_id": BASE["problem_id"],
        "attempt_id": run["attempt_id"], "run_id": run["run_id"]})
    assert response.status_code == 200, response.text
    generated = response.json()
    assert generated["review_status"] == "ready"
    assert generated["review"] is not None


# ---------------------------------------------------------------- 全局不变式

@pytest.mark.parametrize("flow", ["photo_default", "photo_skip", "photo_failed",
                                  "video_skip", "video_failed"])
def test_status_is_never_outside_the_frozen_set(client, monkeypatch, flow):
    """`available` 曾经是第四个值 —— **它现在是 `review_available`，不再是状态**。"""
    http, _ = client
    if flow == "photo_failed":
        monkeypatch.setattr(review_builder, "review_trajectory",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
        body = _post_photo(http).json()
    elif flow == "video_failed":
        monkeypatch.setattr(review_builder, "review_trajectory",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
        body = _post_video(http, include_review="true").json()
    elif flow.startswith("photo"):
        body = _post_photo(http, include_review=str(flow == "photo_default").lower()).json()
    else:
        body = _post_video(http).json()
    assert body["review_status"] in FROZEN_STATUSES, body["review_status"]
    assert "available" not in body["review_status"]


def test_client_side_invalidation_stays_three_valued():
    """更正后旧报告失效：客户端**不得**引入第四个值（曾用过 `invalidated`）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
    from ui_logic import apply_ocr_correction
    state = {"active_session": ("s", "p"), "attempt_id": "a", "attempt_status": "in_progress",
             "tutor": {"status": "in_progress"},
             "video_result": {"media_type": "video", "run_id": "b" * 32, "input_revision": 0,
                              "rounds": [{"source_id": "src", "source_revision": 0,
                                          "answer": "old", "weak_concepts": []}],
                              "review": {"timeline": "旧报告"}, "review_status": "ready"}}
    apply_ocr_correction(state, {
        "attempt_id": "a", "source_id": "src", "source_revision": 1,
        "original_text": "old", "corrected_text": "new",
        "analysis": {"attempt_id": "a", "weak_concepts": [], "finished": False},
        "timeline": []}, "src")
    result = state["video_result"]
    assert result["review_status"] in FROZEN_STATUSES
    assert result["review"] is None                      # 旧报告不得继续冒充当前结论
    assert result["review_refresh_required"] is True     # 「过期」由**独立字段**表达
    history = result["review_history"]
    assert history                                       # 旧报告仍可查
    # 保留的是**原版本**（版本号在），并且**明确标了过期**。
    assert history[0]["input_revision"] == 0
    assert history[0]["superseded"] is True
    assert history[0]["review"] == {"timeline": "旧报告"}


def test_a_generated_report_can_still_have_no_later_entry_point():
    """**决定 3**：`review_available` 与「本次是否已生成报告」**独立**。

    它统一解释为「当前材料能否由**后续接口**请求复盘」——
    所以照片即使走兼容调用（`include_review=true`）拿到 `ready`，它照样是 `false`。
    反方向也成立：视频没请求生成（`not_requested`），它照样是 `true`。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
    from ui_logic import can_generate_review

    photo = {"active_session": ("s", "p"), "attempt_id": "a",
             "video_result": {"media_type": "image", "student_id": "s", "problem_id": "p",
                              "attempt_id": "a", "review_status": "ready",
                              "review": {"timeline": "报告"}, "review_available": False}}
    assert can_generate_review(photo, "s", "p") is False          # ready 但**不能**再生成

    video = {"active_session": ("s", "p"), "attempt_id": "a",
             "video_result": {"media_type": "video", "student_id": "s", "problem_id": "p",
                              "attempt_id": "a", "run_id": "b" * 32,
                              "review_status": "not_requested", "review": None,
                              "review_available": True}}
    assert can_generate_review(video, "s", "p") is True           # 未生成但**能**生成


def test_photo_page_asks_for_a_hint_without_a_review():
    """契约 §2.1 + T3：新照片页面必须**显式**发这两个参数。

    只断言源码字面量是刻意的 —— 这一条防的是"文档说改了、页面没改"，
    而页面到后端的真实调用已由上面的 HTTP 用例覆盖。
    """
    code = (Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py").read_text(
        encoding="utf-8")
    start = code.index("def call_image_analyze")
    window = code[start:code.index("/video/analyze_image", start) + 400]
    assert '"request_hint": "true"' in window, "照片请求没显式要提示"
    assert '"include_review": "false"' in window, "照片请求没显式关掉复盘"
