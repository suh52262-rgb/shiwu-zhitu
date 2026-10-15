"""T3：复盘 v3 的三类展示（`observed_error` / `uncertain` / `suggestion`）。

## 这一条防的是什么
契约 §2.3 把 `findings.category` 固定三选一，三类要学生做**不同的事**：
```
observed_error   去看那处原文，判断自己是不是真错了
uncertain        去核对识别文本（**我可能看错了**），先别信下面的结论
suggestion       **不是错**，是可以做得更好
```
把三类都写成「做错了」，就等于把「我可能看错了」和「你可以写得更好」
一并说成了「你错了」—— **错误归因是本项目的红线。**

## 为什么分两层测
```
纯函数   review_categories()：分组对不对、引用指不指得回原文 —— 这里才会错
渲染层   AppTest：三类**真的出现在页面上**，而不是函数算对了没人用
⇒ 只测其中一层都会漏：纯函数测试挡不住"算了不渲染"，
  渲染测试又说不清"分组错在哪"。
```
"""
from pathlib import Path

import httpx
import pytest

from app.services.review_builder import (AnswerCheck, ReviewReport, StickingPoint,
                                         build_evidence_catalog, build_findings,
                                         build_validation)
from frontend.ui_logic import review_categories

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE = [
    {"id": "E5.1", "source": "student_answer", "revision": None, "round": 5,
     "ts": 41.0, "text": "I = 1.98e4 N·s"},
    {"id": "E7.1", "source": "student_answer", "revision": 2, "round": 7,
     "ts": 63.0, "text": "F = I/Δt = 1.65e5 N"},
]


def v3_review(**overrides) -> dict:
    review = {
        "timeline": "他先算了冲量，再算平均力。",
        "review_schema_version": "v3",
        "evidence": [dict(e) for e in EVIDENCE],
        "findings": [
            {"category": "observed_error", "where": "方向", "what": "冲量方向写反了",
             "why": "Δv 取的是碰撞后减碰撞前", "downstream": "平均力方向跟着错",
             "evidence_ids": ["E5.1", "E7.1"]},
            {"category": "uncertain", "where": "识别", "what": "`1.98e4` 可能是 `1.98×10^4`",
             "why": "", "downstream": "", "evidence_ids": []},
            {"category": "suggestion", "where": "下一步", "what": "先规定正方向再代入",
             "why": "", "downstream": "", "evidence_ids": []},
        ],
        "validation": {"valid": True, "scope": "结构与引用校验；**不代表物理语义已验证**",
                       "validator_version": "v3", "issues": [],
                       "evidence_count": 2, "value_count": 1},
    }
    review.update(overrides)
    return review


# ---------------------------------------------------------------- 纯函数

def test_v1_v2_reports_fall_back_to_the_old_renderer():
    """历史报告**不伪装**成带证据报告（契约 §2.3）—— 没有版本号就不走三类。"""
    assert review_categories({"timeline": "x", "sticking_points": []})["is_v3"] is False
    assert review_categories({"review_schema_version": ""})["is_v3"] is False
    assert review_categories(None)["is_v3"] is False


def test_empty_findings_is_still_a_valid_v3_report():
    """契约⑥：无明确问题时**允许空列表**。

    ⚠️ 判据必须是**结构版本号**，不是"有没有 findings" ——
    否则"没发现问题"会被当成"不是 v3"，悄悄退回老渲染。
    """
    view = review_categories(v3_review(findings=[]))
    assert view["is_v3"] is True
    assert [g["items"] for g in view["groups"]] == [[], [], []]


def test_three_categories_are_grouped_and_ordered_for_the_student():
    """`uncertain` 排最前 —— 它在给**整份报告**打折。

    排在具体问题后面，学生读完一堆指控才看到"这些可能都不算数"，
    **等于先用错误结论说服了他。**
    """
    view = review_categories(v3_review())
    assert [g["category"] for g in view["groups"]] == \
        ["uncertain", "observed_error", "suggestion"]
    assert [g["label"] for g in view["groups"]] == ["需要核查", "观察到的问题", "可选练习"]
    assert [i["where"] for g in view["groups"] for i in g["items"]] == ["识别", "方向", "下一步"]


def test_evidence_is_resolved_to_the_server_side_original_text():
    """契约②：引用原文由**服务端**填充。

    所以视图里出现的每一句原文，都必须是证据目录里那一份 ——
    模型不能改写，前端也不能替它改写。
    """
    view = review_categories(v3_review())
    observed = next(g for g in view["groups"] if g["category"] == "observed_error")
    cited = observed["items"][0]["evidence"]
    assert cited == [
        {"id": "E5.1", "found": True, "round": 5, "revision": None,
         "text": "I = 1.98e4 N·s"},
        {"id": "E7.1", "found": True, "round": 7, "revision": 2,
         "text": "F = I/Δt = 1.65e5 N"},
    ]


def test_a_dangling_reference_is_shown_not_swallowed():
    """契约⑦说这种报告不该被缓存；但前端**不能假设**后端永远拦得住。

    看不见的坏引用比看得见的更危险 —— 它会让一条没有依据的判断
    看起来像有依据。
    """
    review = v3_review(findings=[{"category": "observed_error", "where": "方向",
                                  "what": "x", "why": "", "downstream": "",
                                  "evidence_ids": ["E9.9"]}])
    cited = review_categories(review)["groups"][1]["items"][0]["evidence"]
    assert cited[0]["id"] == "E9.9" and cited[0]["found"] is False


def test_a_fourth_category_is_surfaced_not_stuffed_into_a_group():
    """契约说 category 固定三选一。真出现第四种值 = 有一边错了。

    静默塞进三类里，就等于**替它选了一个类别** —— 而那正是契约禁止的。
    """
    review = v3_review(findings=[{"category": "praise", "where": "草稿",
                                  "what": "写得很整齐", "why": "", "downstream": "",
                                  "evidence_ids": []}])
    view = review_categories(review)
    assert all(not g["items"] for g in view["groups"])
    assert [u["category"] for u in view["unknown"]] == ["praise"]


def test_validation_issues_are_translated_without_claiming_physics():
    """契约 §2.3⑧：`validation` **只代表结构与引用校验**。

    返回里带着 `scope` 那句话就是为了这件事 —— 免得前端拿到
    `valid: true` 就显示成「物理语义已验证」。
    """
    review = v3_review(validation={
        "valid": False, "scope": "结构与引用校验；**不代表物理语义已验证**",
        "validator_version": "v3",
        "issues": [{"kind": "model_written_count", "detail": "自写次数：四轮"}],
        "evidence_count": 2})
    view = review_categories(review)
    assert "模型自己写的次数" in view["validation_issues"][0]["text"]
    assert "物理语义" in view["validation"]["scope"]


# ---------------------------------------------------------------- 与后端的接线

def test_the_real_backend_v3_payload_maps_into_the_three_categories():
    """**这一条是前后端之间那道缝的钉子。**

    `review_categories` 的字段名是照着 `review_builder` 写的；
    后端一旦改名，纯前端测试全都照样通过 —— 只有跨过这条缝才测得出来。
    """
    report = ReviewReport(
        timeline="先算冲量，再算平均力。",
        sticking_points=[StickingPoint(where="方向", what="冲量方向写反",
                                       why="Δv 取反了", downstream="平均力方向跟着错",
                                       evidence_ids=["E5.1"])],
        key_concept="动量定理",
        next_step="先规定正方向",
        encouragement="思路是对的",
        answer_check=AnswerCheck(student_answer="F = 1.65e5",
                                 derived_from_work="F = I/Δt", consistent="一致",
                                 note="结论与推导一致", ocr_caution="`1.98e4` 可能读漏了乘号"),
    )
    rounds = [{"round": 5, "ts": 41.0, "answer": "I = 1.98e4 N·s"},
              {"round": 7, "ts": 63.0, "answer": "F = I/Δt = 1.65e5 N"}]
    evidence = build_evidence_catalog(rounds)
    findings = build_findings(report, {})
    payload = {"review_schema_version": "v3", "evidence": evidence, "findings": findings,
               "validation": build_validation(evidence, findings)}

    view = review_categories(payload)
    assert view["is_v3"] is True
    got = {g["category"]: g["items"] for g in view["groups"]}
    assert [i["where"] for i in got["observed_error"]] == ["方向"]
    assert [i["where"] for i in got["uncertain"]] == ["识别"]
    assert [i["where"] for i in got["suggestion"]] == ["下一步", "鼓励"]
    # 模型填的引用被解析回**服务端原文**
    assert got["observed_error"][0]["evidence"][0]["text"] == "I = 1.98e4 N·s"


# ---------------------------------------------------------------- 渲染层

def ui_state(review: dict) -> dict:
    return dict(active_session=("stu_demo", "p1"), attempt_id="a", attempt_status="completed",
                tutor={"finished": True, "hint": "已完成"},
                timeline=[{"type": "submit", "timestamp": 0, "content": "作答"}],
                video_result={"media_type": "video", "student_id": "stu_demo",
                              "problem_id": "p1", "attempt_id": "a", "run_id": "b" * 32,
                              "review": review, "review_status": "ready",
                              "review_available": True, "submitted": 2,
                              "rounds": [{"round": 5, "answer": "I = 1.98e4 N·s"}]})


def render(monkeypatch, review: dict):
    from streamlit.testing.v1 import AppTest

    def get(url, **kwargs):
        payload = ({"items": [{"problem_id": "p1", "title": "题一"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    app = AppTest.from_file(str(ROOT / "frontend" / "streamlit_app.py")).run()
    for key, value in ui_state(review).items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, app.exception
    return app


def markdown_text(app) -> str:
    return "\n".join(str(item.value) for item in app.markdown)


def caption_text(app) -> str:
    return "\n".join(str(item.value) for item in app.caption)


def alert_text(app) -> str:
    """`st.info` / `st.warning` / `st.error` / `st.success` **不是** markdown 或 caption。

    ⚠️ 不把这一层收进来，断言就会"看着通过、其实什么都没检查到" ——
    我第一版把 `st.info` 的内容拿去 `markdown` 里找，就是这么假失败的。
    """
    parts = []
    for name in ("info", "warning", "error", "success"):
        parts += [str(item.value) for item in getattr(app, name)]
    return "\n".join(parts)


def visible_text(app) -> str:
    return "\n".join((markdown_text(app), caption_text(app), alert_text(app)))


def test_the_three_labels_actually_reach_the_page(monkeypatch):
    """函数算对了 ≠ 学生看得见。**这一条才是 T3 的验收点。**"""
    app = render(monkeypatch, v3_review())
    text = markdown_text(app)
    for label in ("需要核查", "观察到的问题", "可选练习"):
        assert label in text, f"页面上没有「{label}」"


def test_v3_does_not_call_everything_an_error(monkeypatch):
    """**T3 的原话**：不统一写成「做错了」。

    `suggestion` 只是"可以更好"，把它说成"做错了"就是错误归因。
    """
    app = render(monkeypatch, v3_review())
    text = markdown_text(app)
    assert "做错了" not in text, "三类又被统一写成「做错了」"
    # 但**非 v3** 的老渲染保持原样（那里确实是"卡点"，措辞不改）。
    legacy = render(monkeypatch, {"timeline": "t", "sticking_points": [
        {"where": "方向", "what": "写反了", "why": "Δv 取反", "downstream": ""}],
        "review_schema_version": ""})
    assert "做错了" in markdown_text(legacy)


def test_the_student_can_trace_a_judgement_back_to_the_original_text(monkeypatch):
    """引用要能点回原文 —— 这是"报告中的判断可以找到依据"的落点。"""
    app = render(monkeypatch, v3_review())
    text = caption_text(app)
    assert "E5.1" in text and "I = 1.98e4 N·s" in text
    assert "第 5 轮" in text
    # 修订号只在**有**的时候才显示（没有就不编）。
    assert "修订 2" in text


def test_a_dangling_reference_is_visible_on_the_page(monkeypatch):
    app = render(monkeypatch, v3_review(findings=[
        {"category": "observed_error", "where": "方向", "what": "x", "why": "",
         "downstream": "", "evidence_ids": ["E9.9"]}]))
    assert "不存在的证据" in caption_text(app)


def test_no_findings_says_so_without_calling_the_student_wrong(monkeypatch):
    """契约⑥：**信息不足 ≠ 学生做错**。"""
    app = render(monkeypatch, v3_review(findings=[]))
    text = visible_text(app)
    assert "没有发现需要指出的问题" in text
    assert "看不出来不等于做错" in text


def test_a_fourth_category_is_visible_on_the_page(monkeypatch):
    app = render(monkeypatch, v3_review(findings=[
        {"category": "praise", "where": "草稿", "what": "写得很整齐", "why": "",
         "downstream": "", "evidence_ids": []}]))
    assert "无法归类" in visible_text(app)


def test_validation_never_reads_as_physics_verified(monkeypatch):
    """契约 §2.3⑧：**不得**显示成「物理语义已验证」。"""
    app = render(monkeypatch, v3_review())
    text = caption_text(app)
    assert "不代表物理语义已验证" in text
    assert "通过 ≠ 物理正确" in text


def test_the_rebuild_is_attributed_to_the_system_not_the_student(monkeypatch):
    """T3：把「用你自己的方程推出来」改成「系统核对时重建」。

    旧措辞把**系统的补算**说成**学生写过的步骤** ——
    学生看到会以为"我明明没写这一步"。
    """
    review = v3_review(answer_check={"student_answer": "F = 1.65e5",
                                     "derived_from_work": "F = I/Δt",
                                     "consistent": "一致", "note": "一致",
                                     "ocr_caution": ""})
    app = render(monkeypatch, review)
    text = caption_text(app)
    assert "系统核对时重建" in text
    assert "用你自己的方程推出来" not in text
