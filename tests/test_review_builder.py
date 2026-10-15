"""离线复盘的测试。

## 为什么这个模块需要单独测
它存在的理由是**换一种说话方式**（提问→陈述），而这**只能靠看提示词内容**来保证。
另外两件容易悄悄坏掉的事：
· **学生实际写了什么必须进输入** —— 第一版我猜错字段名，输入里学生作答是空的，
  **产出"看着依然通顺"**，只是卡点描述全变成"或者…或者…"（在猜）。
· **复盘失败不能让整条视频链路挂掉** —— 没有复盘，逐轮轨迹照样能看。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.services.review_builder import (  # noqa: E402
    ReviewReport,
    StickingPoint,
    build_material,
    build_review_user,
    review_trajectory,
)

PROMPT = Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v1.md'))


class _FakeProblem:
    problem_id = "demo_03"
    statement = "汽车质量 1200 kg，以 15.0 m/s 撞墙后以 1.50 m/s 弹回……"
    knowledge_points = ["冲量", "正方向约定", "动量定理（冲量—动量定理）"]


ROUNDS = [
    {"round": 1, "current_node": "S1", "hint_level": 0, "stuck_score": 0.3,
     "answer": "I = mΔv = 1200×(15.0−1.50) = 1.62×10⁴ N·s",
     "hint": "", "issue": "未考虑碰撞前后速度方向相反"},
    {"round": 2, "current_node": "S2a", "hint_level": 1, "stuck_score": 0.3,
     "answer": "取碰撞前方向为正，I = m(v′−v)", "hint": "先规定正方向",
     "issue": ""},
]


# ================================================================ 输入必须含学生作答

def test_user_prompt_contains_the_students_actual_answer():
    """**这是本模块最容易悄悄坏掉的地方。**

    第一版我取了 `current_answer`（不存在），实际字段是 `student_input` ——
    输入里学生作答是空的，**而产出看着依然通顺**。
    复盘要判"他是真会还是凑对"，**唯一依据就是学生写了什么**。
    """
    u = build_review_user(_FakeProblem.statement, _FakeProblem.knowledge_points, ROUNDS)
    assert "1200×(15.0−1.50)" in u, "学生的作答没进输入"
    assert "取碰撞前方向为正" in u, "第二轮的作答没进输入"
    assert "第1轮" in u and "第2轮" in u


def test_user_prompt_accepts_student_input_field_name():
    """另一种字段名（库里 traces 用的就是 `student_input`）也要能吃。"""
    r = [{"student_input": "我写了这个"}, {"answer": "我写了那个"}]
    u = build_review_user("题干", [], r)
    assert "我写了这个" in u and "我写了那个" in u


def test_user_prompt_lists_the_knowledge_points():
    """`key_concept` 要求逐字取自词表 —— 词表得在输入里。"""
    u = build_review_user(_FakeProblem.statement, _FakeProblem.knowledge_points, ROUNDS)
    for k in _FakeProblem.knowledge_points:
        assert k in u


def test_user_prompt_handles_nested_dicts():
    """轨迹里的字段可能是嵌套 dict（`tutor_output` 就是），**两种都要能吃**。"""
    u = build_review_user("题干", [], [{"answer": {"text": "嵌套里的作答"}}])
    assert "嵌套里的作答" in u


# ================================================================ 失败不能打挂链路

def test_returns_none_on_empty_rounds():
    assert review_trajectory(_FakeProblem, []) is None


def test_returns_none_when_llm_fails():
    """**复盘是锦上添花，不能把整条视频链路带崩。**"""
    class _Bad:
        def chat_structured_resilient(self, *a, **k):
            class _O:
                value = None
                reasons = ["模型没返回"]
            return _O()

    assert review_trajectory(_FakeProblem, ROUNDS, llm=_Bad()) is None


def test_swallows_exceptions():
    """连抛异常也不许穿透。"""
    class _Boom:
        def chat_structured_resilient(self, *a, **k):
            raise RuntimeError("炸了")

    assert review_trajectory(_FakeProblem, ROUNDS, llm=_Boom()) is None


def test_returns_report_on_success():
    class _Good:
        def chat_structured_resilient(self, system, user, schema):
            class _O:
                value = ReviewReport(
                    timeline="他先算冲量……",
                    sticking_points=[StickingPoint(
                        where="第(1)问 求冲量", what="用速率相减",
                        why="冲量是矢量，要先定正方向",
                        downstream="第(2)问方向跟着错")],
                    key_concept="正方向约定",
                    next_step="重做第(1)问",
                    encouragement="你第(2)问的公式用对了")
                reasons = []
            return _O()

    r = review_trajectory(_FakeProblem, ROUNDS, llm=_Good())
    assert isinstance(r, ReviewReport)
    assert r.key_concept == "正方向约定"
    assert len(r.sticking_points) == 1


@pytest.mark.parametrize('version',['v3','v4'])
def test_review_render_policy_preserves_raw_and_does_not_claim_semantic_validation(monkeypatch,version):
    from types import SimpleNamespace
    monkeypatch.setenv('PROMPT_REVIEW_VERSION',version)
    original='未展示的后续步骤被模型推断'
    report=ReviewReport(timeline='材料有作答记录',sticking_points=[StickingPoint(
        where='冲量',what='使用速率相减',why='冲量需按速度方向计算',downstream=original,
        evidence_ids=['R1'])],key_concept='正方向约定',next_step='核对方向',encouragement='可继续核对')
    class Client:
        def chat_structured_resilient(self,*args):return SimpleNamespace(value=report,degraded=False,reasons=[])
    result=review_trajectory(_FakeProblem,ROUNDS,llm=Client())
    assert result is not None
    assert result._raw_generation['sticking_points'][0]['downstream']==original
    assert '_raw_generation' not in result.model_dump()
    assert result.sticking_points[0].why=='冲量需按速度方向计算'
    if version=='v4':
        assert result.sticking_points[0].downstream=='实际后续过程与认知原因未由材料确认。'
        assert result.render_policy=='physical_check_only_v1'
        assert result.canonicalized_fields==['sticking_points[0].downstream']
    else:
        assert result.sticking_points[0].downstream==original
        assert result.render_policy=='legacy'


# ================================================================ 提示词本身

def test_prompt_exists():
    assert PROMPT.is_file(), "review 提示词不存在"


def test_prompt_forbids_the_live_tutoring_style():
    """**这是整个改动的要点** —— 不许提问、不许六级递进、不许藏答案。"""
    t = PROMPT.read_text(encoding="utf-8")
    assert "不要问学生问题" in t, "没禁止提问（那会退回实时风格）"
    assert "不要六级递进" in t
    assert "不要藏着答案" in t


def test_prompt_requires_merging_sticking_points():
    """**1~3 处，不是 15 条** —— 不归并就退回原形态了。"""
    t = PROMPT.read_text(encoding="utf-8")
    assert "1~3 处" in t
    assert "归并" in t


def test_prompt_requires_concrete_encouragement():
    """**不许空话**（"你很棒""继续加油"）。"""
    t = PROMPT.read_text(encoding="utf-8")
    assert "空话" in t


# ================================================================ 接线

from test_media_lifecycle_integration import media_client, BASE


def test_endpoint_records_the_students_answer(media_client, monkeypatch):
    """The deferred review receives the actual submitted text, not a hint."""
    import json
    from app.api import video
    from fakes import install_fake_llm
    install_fake_llm(monkeypatch)
    response = media_client.post("/video/analyze", data=BASE,
                                 files={"file": ("work.mp4", b"mock-media")})
    assert response.status_code == 200, response.text
    result = response.json()
    expected = ["mgh", "mgh = mv^2/2"]
    assert [r["answer"] for r in result["rounds"]] == expected
    saved = json.loads((video.RUNS_DIR / result["run_id"] / "review_input.json").read_text(encoding="utf-8"))
    assert [r["answer"] for r in saved["input"]["rounds"]] == expected
    assert result["review"] is None and result["review_status"] == "not_requested"


def test_ui_renders_review_above_the_round_table():
    """复盘要在逐轮表格**上面** —— 学生先看结论。

    顺序反了，学生就得自己从 15 行里找规律 —— **那正是原形态的问题**。
    """
    ui = Path(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).read_text(encoding="utf-8")
    i_rev = ui.find('_rev = _vr.get("review")')
    # ⚠️ **不能只找 `逐轮记录` 四个字。**
    # 我在 ② 上传处加了一句指路文案"往下滚到 ④ AI Tutor 看复盘与**逐轮记录**"，
    # 于是 `find("逐轮记录")` 找到的是那句文字，而不是表格 ——
    # **测试因此假失败**（它以为表格跑到了复盘前面）。
    # 改用 expander 的标题（带 emoji，不会和正文撞）。
    i_tbl = ui.find("📹 逐轮记录")
    assert i_rev >= 0, "UI 没渲染复盘"
    assert i_tbl >= 0, "找不到逐轮记录那张表"
    assert i_rev < i_tbl, "复盘跑到逐轮表格下面去了"

# ================================================================ UI 显示（v2 的两件事）

def test_ui_shows_the_answer_check():
    """**结论与推导的自洽性要给学生看。**

    「凑对」（答案对、理由错）是自动批改天生抓不到的一类错误 ——
    而这个检查不需要知道正确答案，只需要内部一致性。
    """
    ui = Path(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).read_text(encoding="utf-8")
    assert "结论核对" in ui, "UI 没显示结论核对"
    assert "answer_check" in ui, "没读 answer_check 字段"
    assert "derived_from_work" in ui and "student_answer" in ui


def test_ui_shows_the_ocr_caution_prominently():
    """⚠️ **识别可疑不能藏在折叠区里。**

    复盘基于"识别到的文本"判断，**它自己无法察觉识别错误，只有学生能**。
    实测确认过一次：学生写 `49`、读成 `4`，旧复盘说成"没有物理意义"。
    """
    ui = Path(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).read_text(encoding="utf-8")
    assert "ocr_caution" in ui, "UI 没读 ocr_caution"
    assert "我可能看错了" in ui or "可能看错" in ui, "没给学生看得懂的提示"
    # 用 st.error（最显眼的一档），而不是 caption
    i = ui.find("ocr_caution")
    assert "st.error" in ui[max(0, i - 400):i + 400],\
        "识别可疑没用醒目样式，会被学生忽略"


def test_ui_tells_the_student_to_trust_their_own_paper():
    """最后还是那句：**以你的原稿为准**。"""
    ui = Path(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).read_text(encoding="utf-8")
    assert "以你的原稿为准" in ui or "以你的原稿" in ui


def test_degraded_nonempty_report_cannot_be_treated_as_success():
    from types import SimpleNamespace
    report = ReviewReport(timeline="Even a nonempty fallback is not model evidence", sticking_points=[],
                          key_concept="momentum", next_step="retry", encouragement="retry")
    client = SimpleNamespace(chat_structured_resilient=lambda *_: SimpleNamespace(
        value=report, degraded=True, reasons=["three timeouts exhausted"]))
    assert review_trajectory(_FakeProblem, ROUNDS, llm=client) is None


def test_schema_safe_default_report_is_failure_even_without_degraded_flag():
    from types import SimpleNamespace
    for timeline in ("", "   "):
        report = ReviewReport(timeline=timeline, sticking_points=[], key_concept="", next_step="", encouragement="")
        client = SimpleNamespace(chat_structured_resilient=lambda *_: SimpleNamespace(
            value=report, degraded=False, reasons=[]))
        assert review_trajectory(_FakeProblem, ROUNDS, llm=client) is None


def test_review_input_marks_machine_inference_and_incomplete_ocr():
    user = build_review_user("题干", ["冲量"], [{
        "answer": "x" * 301, "ts": 17.5, "stuck_score": 0.9, "diagnosis": "机器诊断专属占位"}])
    assert "非停顿标注" in user and "视频时刻=17.5秒" in user
    # ⚠️ 301 字**在 12,000 的预算内** ⇒ 这一轮**根本不截断**，也就不该出现省略标记。
    # （旧版每轮硬截 300 字，所以这条断言以前是反的。）
    assert "因预算未展示" not in user
    assert "总预算 12000 字符" in user
    assert "完整轨迹" not in user and "大小加明确物理方向" in user
    assert "机器诊断专属占位" not in user and "0.9" not in user


def test_review_input_marks_omitted_material_without_calling_it_missing_work():
    """契约⑨：未发送部分要**明确记录为省略**，且**不得据此判断遗漏推理**。"""
    rounds = [{"round": i, "ts": float(i), "answer": f"第{i}轮" + "字" * 400}
              for i in range(1, 40)]
    user = build_review_user("题干", [], rounds)
    assert "总预算 12000 字符" in user
    assert "整轮未展示" in user, "整轮被省略必须说出来"
    assert "未展示不代表这些轮没有内容" in user
    assert "不得据此推断学生漏了推理" in user


def test_a_partially_omitted_round_reports_how_much_was_dropped():
    """单轮自己就超预算时：按**完整行**截，并**如实报出**丢了几行几个字。

    ⚠️ 不切断行 —— 半行交给模型，等于让它把"没写完的半句"当成证据。
    """
    rounds = [{"round": 1, "ts": 1.0, "answer": "\n".join("行" * 80 for _ in range(300))}]
    material = build_material(rounds)
    item = material["items"][0]
    assert item["omitted_lines"] > 0 and item["omitted_chars"] > 0
    assert len(item["text"]) <= material["budget"]
    assert item["text"].endswith("行") and not item["text"].endswith("\n"), "切在了行边界"
    user = build_review_user("题干", [], rounds)
    assert "因预算未展示" in user and "未展示不代表学生未写" in user


def test_review_input_prefers_the_latest_answers_when_over_budget():
    """契约⑨：**优先保留最新作答** —— 从最后一轮往前分配，不是从第一轮往后。"""
    rounds = [{"round": i, "ts": float(i), "answer": f"R{i}-" + "字" * 900}
              for i in range(1, 30)]
    material = build_material(rounds)
    kept = [it["round"] for it in material["items"]]
    assert 29 in kept, "最新一轮被丢了"
    assert 1 not in kept, "预算不够时最老的轮次应当让位"
    assert material["omitted_rounds"], "整轮省略必须记下来"
    assert material["chars"] <= material["budget"]


def test_review_input_never_repeats_the_same_material_twice():
    """**本轮修掉的那个洞**：轨迹正文与证据目录曾是两次独立构造 ——
    正文截 300、目录发全文 ⇒ 预算净增，而且**模型看到的正文和它能引用的原文不是同一份**。
    """
    long = "第一行\n" + "第二行" + "字" * 400
    rounds = [{"round": 1, "ts": 1.0, "answer": long, "source_id": "s1", "source_revision": 0}]
    user = build_review_user("题干", [], rounds)
    material = build_material(rounds)
    shown = material["items"][0]["text"]
    # 正文与证据目录里出现的是**同一份文本**，各一次。
    assert user.count(shown) >= 2, "两处都要用同一份文本"
    assert "【后文截断" not in user, "旧的 300 字硬截已经取消"


def test_unlisted_model_concept_is_not_exposed_as_curriculum_label():
    from types import SimpleNamespace
    report = ReviewReport(timeline="按大小和方向作答", sticking_points=[],
                          key_concept="模型自造概念", next_step="核查原稿", encouragement="公式成立")
    client = SimpleNamespace(chat_structured_resilient=lambda *_: SimpleNamespace(
        value=report, degraded=False, reasons=[]))
    result = review_trajectory(_FakeProblem, ROUNDS, llm=client)
    assert result is not None and result.key_concept == "" and result.sticking_points == []


def test_model_reconstructed_quote_is_replaced_by_actual_final_text():
    from types import SimpleNamespace
    from app.services.review_builder import AnswerCheck
    final = "I=1.98×10^4 N·s\n方向与初速度相反"
    report = ReviewReport(timeline="给出冲量大小和方向", sticking_points=[],
        key_concept="冲量", next_step="可选练习", encouragement="方向明确",
        answer_check=AnswerCheck(student_answer="学生写了一个未出现的负号",
            derived_from_work="核对时重建", consistent="无法判断"))
    client = SimpleNamespace(chat_structured_resilient=lambda *_: SimpleNamespace(
        value=report, degraded=False, reasons=[]))
    result = review_trajectory(_FakeProblem, [{"answer": final}], llm=client)
    assert result.answer_check.student_answer == final and result.key_concept == ""
