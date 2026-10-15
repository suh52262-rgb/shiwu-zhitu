"""知识点归位（`snap_concept`）的测试。

## 它要解决的问题
`diagnose` 产出的 `concept` 写进**学生画像**，而**推荐的判据是
「画像知识点 ∩ 题库知识点」**。实测提示词没有封闭词表，模型把举例当前缀
自己加限定，同一个知识点被记成三条：
```
"机械能守恒条件（只有重力或弹力做功）"
"机械能守恒条件（除重力外是否有其他力做功）"
"机械能守恒条件（是否只有重力/弹力做功）"
```
⇒ 画像越滚越长、推荐匹配不上、教师端统计失真。

## 为什么两头都要
· `diagnosis/v2.md` 里**要求**照抄词表 —— 那是"请求对齐"
· `snap_concept` 在代码里**保证**对齐 —— 这轮的教训是
  **提示词约束不住输出**（P14 那条"硬性要求一"就没管住）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.services.concept_normalizer import (  # noqa: E402
    snap_concept,
    snap_many,
)

VOCAB = ["机械能守恒", "摩擦力做功", "动能定理", "牛顿第二定律"]


# ================================================================ 核心：归位

def test_snaps_the_embellished_variants():
    """**这就是实际发生的那三条** —— 都要归到「机械能守恒」。"""
    for bad in ("机械能守恒条件（只有重力或弹力做功）",
                "机械能守恒条件（除重力外是否有其他力做功）",
                "机械能守恒条件（是否只有重力/弹力做功）"):
        assert snap_concept(bad, VOCAB) == "机械能守恒", bad


def test_exact_match_returns_vocabulary_text():
    assert snap_concept("动能定理", VOCAB) == "动能定理"


def test_normalizes_whitespace():
    assert snap_concept(" 动能 定理 ", VOCAB) == "动能定理"


def test_three_variants_collapse_to_one():
    """**归位之后必须是同一条** —— 这是修这个问题的全部意义。"""
    got = snap_many(["机械能守恒条件（只有重力或弹力做功）",
                     "机械能守恒条件（除重力外是否有其他力做功）",
                     "机械能守恒条件（是否只有重力/弹力做功）"], VOCAB)
    assert got == ["机械能守恒"]


# ================================================================ 宁缺勿滥

def test_unmatched_returns_empty_not_the_original():
    """**对不上就丢弃，不要保留原值。**

    保留原值 = 画像里又混进词表外的标签，前功尽弃。
    """
    assert snap_concept("量子纠缠", VOCAB) == ""


def test_empty_input():
    assert snap_concept("", VOCAB) == ""
    assert snap_concept("   ", VOCAB) == ""
    assert snap_concept(None, VOCAB) == ""


def test_no_vocabulary_returns_raw():
    """**没有词表就不做加工** —— 原样返回，总比凭空造一个好。"""
    assert snap_concept("机械能守恒条件（只有重力做功）", None) ==\
        "机械能守恒条件（只有重力做功）"
    assert snap_concept("某个概念", []) == "某个概念"


def test_does_not_over_match():
    """**别把不相干的也归进来** —— 误归位比不归位更糟（画像会记错）。"""
    assert snap_concept("简谐运动的振幅", VOCAB) == ""


def test_prefers_the_longest_match():
    """**多个候选命中时取最长的那个。**

    ⚠️ 实测踩到：`'动量定理的矢量性与速度方向变化'` **同时**包含
    `动量` 与 `矢量性`，而原来的实现"按词表顺序返回第一个命中"，
    于是归成了 `动量` —— **太粗，丢掉了"矢量性"这个真正的考点**。

    最长匹配最贴切，这是 substring 匹配的标准做法。
    """
    v = ["动量", "冲量", "矢量性", "正方向约定"]
    assert snap_concept("动量定理的矢量性与速度方向变化", v) == "矢量性"
    assert snap_concept("冲量的矢量性", v) == "矢量性"
    # 词表里没有 `矢量性` 时才退而求其次
    assert snap_concept("动量定理的矢量性与速度方向变化", ["动量"]) == "动量"


def test_longest_match_applies_to_the_head_rule_too():
    """主干命中那条路也要取最长。"""
    v = ["简谐运动", "简谐运动条件"]
    assert snap_concept("简谐运动条件（回复力与位移成正比）", v) == "简谐运动条件"


# ================================================================ 批处理

def test_snap_many_dedups_and_keeps_order():
    got = snap_many(["摩擦力做功", "机械能守恒条件（…）", "动能定理",
                     "机械能守恒条件（另一种说法）"], VOCAB)
    assert got == ["摩擦力做功", "机械能守恒", "动能定理"]


def test_snap_many_drops_unmatched():
    assert snap_many(["量子纠缠", "动能定理", "相对论"], VOCAB) == ["动能定理"]


def test_snap_many_handles_empty():
    assert snap_many([], VOCAB) == []
    assert snap_many(None, VOCAB) == []


# ================================================================ 接线

def test_diagnose_uses_the_normalizer():
    """**防"写了没接"** —— 归位必须在 diagnose 节点里真的用上。"""
    import inspect

    from app.agent import graph as G

    src = inspect.getsource(G.diagnose)
    assert "snap_concept" in src, "diagnose 没调用归位函数"
    assert "weak_concepts.append(_concept)" in src, "归位结果没写回"


def test_diagnose_feeds_vocabulary_into_prompt():
    """B1 那半也要在：**词表得喂进提示词**，否则模型没法照抄。"""
    import inspect

    from app.agent import graph as G

    src = inspect.getsource(G.diagnose)
    assert "_knowledge_vocabulary" in src
    assert "逐字照抄" in src, "提示词块里没有约束措辞"


def test_vocabulary_comes_from_problem_and_graph():
    """词表要**同时**取 Problem 与 ProblemGraph 的 —— 图上节点也标了知识点。"""
    import inspect

    from app.agent import graph as G

    src = inspect.getsource(G._knowledge_vocabulary)
    assert 'deps.get("problem_graph")' in src
    assert 'deps.get("problem")' in src


def test_diagnosis_v2_prompt_exists_and_constrains():
    p = Path(str(Path(__file__).resolve().parents[1] / 'prompts/diagnosis/v2.md'))
    assert p.is_file(), "v2 提示词不存在"
    t = p.read_text(encoding="utf-8")
    assert "逐字照抄" in t, "没有要求照抄"
    assert "留空" in t, "没有给「对不上就留空」的出口"
    # v1 必须原样保留（项目惯例：不改旧版）
    assert Path(str(Path(__file__).resolve().parents[1] / 'prompts/diagnosis/v1.md')).is_file(), "v1 被删了"


def test_registry_points_at_a_version_that_has_the_concept_rule():
    """默认版本必须**带着 `concept` 封闭词表的要求**（本文件的测试对象）。

    ⚠️ **2026-10-11 改**：原来断言 `== "v2"`。
    v3 加入**诊断措辞的纸面口径**（契约 §2.10）后默认切到 v3，
    而 **v3 自带 v2 的全部内容**（`concept` 那节逐字保留）✓
    ⇒ 这里断言的是**意图**（默认版本含该要求），不是**具体版本号** ——
      否则每次升版都要改这条测试，而它本来不该关心版本号 ✗
    """
    from app.agent.prompt_registry import DEFAULT_VERSIONS, get_prompt

    assert DEFAULT_VERSIONS["diagnosis"] in {"v2", "v3", "v4"};\
        "默认版本必须是有 concept 要求的那几版之一"
    prompt = get_prompt("diagnosis")
    assert "必须原样取自" in prompt, "默认提示词必须带着 concept 封闭词表要求"
    assert "不要改写、不要加括号补充、不要合并" in prompt
