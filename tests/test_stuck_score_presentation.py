"""T3：系统卡顿分的呈现 —— **收进技术详情，并标明它是启发式指标**。

## 这一条防的是什么
`stuck_score` 是**规则型**打分（空闲时长 / 错误次数 / 请求提示次数）。
它**没有校准过**与「学生真的卡住」的关系 —— 文档里反复写过，
无图题（`phy_001`）上还实测偏低 **+0.35**（口径问题，不是 bug）。

⇒ 把它摆在学生主视图里、还配上「建议回顾薄弱知识点」，
  就是在**替他下认知结论**。数字要留着（排障与教师端要用），
  **但得放在它该在的位置，并说清它是什么、不是什么。**

## 两个反向的失效都要挡
```
删过头  分数整段消失 ⇒ 排障时没人找得到它（"计算了但没人渲染"是本项目的老毛病）
删不够  主视图里还留着「卡顿度 0.62」 ⇒ 等于没改
⇒ 所以既断言"显眼处没有了"，也断言"技术详情里还在、且带了免责说明"。
```
"""
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "streamlit_app.py"


def ui_state() -> dict:
    return dict(
        active_session=("stu_demo", "p1"), attempt_id="a", attempt_status="in_progress",
        tutor={"hint_level": 2, "hint": "先看方向", "stuck_score": 0.62,
               "stuck_reasons": ["空闲 50 秒", "连续 2 次未推进"],
               "hint_reasons": ["连续 2 次失误 → 升到 L2"],
               "status": "in_progress", "issue": "方向未定", "issue_type": "concept"},
        timeline=[{"type": "submit", "timestamp": 0, "content": "作答"}],
        video_result={"media_type": "video", "student_id": "stu_demo", "problem_id": "p1",
                      "attempt_id": "a", "run_id": "b" * 32, "submitted": 1,
                      "rounds": [{"round": 1, "answer": "F = 49 N", "hint_level": 2,
                                  "stuck_score": 0.62, "ts": 0.0}]},
    )


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def get(url, **kwargs):
        payload = ({"items": [{"problem_id": "p1", "title": "题一"}]}
                   if url.endswith("/problems") else {"status": "ok"})
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    test = AppTest.from_file(str(APP)).run()
    for key, value in ui_state().items():
        test.session_state[key] = value
    test.run()
    assert not test.exception, test.exception
    return test


def visible_text(app) -> str:
    """markdown + caption + 四类 alert。**渲染层的断言必须走这一层** ——
    `st.info` / `st.warning` 不是 markdown 也不是 caption，只查后两者会假通过。"""
    parts = [str(item.value) for item in app.markdown]
    parts += [str(item.value) for item in app.caption]
    for name in ("info", "warning", "error", "success"):
        parts += [str(item.value) for item in getattr(app, name)]
    return "\n".join(parts)


# ---------------------------------------------------------------- 该走的走了

def test_the_prominent_score_wording_is_gone(app):
    """「卡顿度 / 卡顿指数」这两个**把启发式说成结论**的措辞必须消失。"""
    text = visible_text(app)
    assert "卡顿度" not in text, "提示等级旁边还挂着卡顿分"
    assert "卡顿指数" not in text, "⑤ 里还把它当成「学习分析」的结论"


def test_the_round_table_no_longer_mixes_a_heuristic_into_the_record(app):
    """逐轮记录那张表的用途是**让学生核对识别对不对**。

    卡顿分混在里面，会显得它和「识别到的作答」一样是**事实**。
    """
    assert app.dataframe, "逐轮记录表没渲染出来，这条断言就没意义了"
    columns = list(app.dataframe[0].value.columns)
    assert "卡顿" not in columns, f"表里还留着卡顿列：{columns}"
    assert "**识别到的作答**" in columns, "识别原文那列不见了（它是学生唯一的核对入口）"


def test_the_score_is_still_reachable_as_technical_detail(app):
    """**删过头也是失效**：分数整段消失，排障时就没人找得到它。"""
    labels = [str(item.label) for item in app.expander]
    assert any("卡顿分" in label for label in labels), f"没有卡顿分的技术详情区：{labels}"
    text = visible_text(app)
    assert "0.62" in text, "分数本身没了"
    assert "空闲 50 秒" in text, "触发信号没了"


# ---------------------------------------------------------------- 该标的标了

def test_the_technical_detail_says_what_the_number_is_not(app):
    """**没有这句话，它还是会被当成"你卡住了"。**

    契约与文档对"不冒充认知判断"的要求是硬的 ——
    所以免责说明必须是**内容**，不是标题里的一个词。
    """
    text = visible_text(app)
    assert "不是对你「卡没卡住」的判断" in text
    assert "没有校准过" in text
    # 还要说清提示级别不是它一个人决定的 —— 否则学生会把 L2 直接归因到卡顿分。
    assert "提示级别也不只看它" in text


def test_the_expander_label_flags_it_as_heuristic(app):
    label = next((str(i.label) for i in app.expander if "卡顿分" in str(i.label)), "")
    assert "启发式" in label, f"折叠标题没标启发式：{label!r}"


# ---------------------------------------------------------------- 接线（静态）

def test_the_detail_renderer_is_actually_called():
    """**防「写了没接」**：本项目栽过这个跟头（存了没人读、算了没人渲染）。

    纯函数测试挡不住它 —— 所以这里直接查源码里有没有调用点。
    """
    code = APP.read_text(encoding="utf-8")
    assert code.count("stuck_technical_detail(") >= 2, "只定义没调用"


def test_the_hint_level_line_no_longer_carries_the_score():
    """提示等级那行的措辞是学生**每轮都会看到**的，单独钉一下。"""
    code = APP.read_text(encoding="utf-8")
    i = code.index('_line = f"提示等级')
    window = code[i:i + 700]
    assert "stuck_score" not in window, "提示等级那行又把卡顿分加回来了"
