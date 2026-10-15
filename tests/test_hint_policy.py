"""Hint Policy 测试（P5，任务书第 8 节）：显式规则决策矩阵。"""
import pytest

from app.agent.hint_policy import HintPolicyInput, decide_hint_level


def _inp(**kw) -> HintPolicyInput:
    return HintPolicyInput(**kw)


# ---- L0：不提示 ----

def test_correct_answer_no_request_is_level0():
    d = decide_hint_level(_inp(correct=True))
    assert d.level == 0


def test_off_graph_input_is_level0():
    d = decide_hint_level(_inp(off_graph=True, correct=False))
    assert d.level == 0


# ---- 学生主动请求 ----

def test_first_request_gives_concept_level():
    """首次主动请求：至少给到 L2 概念提示。"""
    d = decide_hint_level(_inp(request_hint=True, correct=False))
    assert d.level == 2


def test_request_escalates_from_current():
    d = decide_hint_level(_inp(request_hint=True, correct=False, current_hint_level=2))
    assert d.level == 3


def test_request_caps_at_4_automatically():
    d = decide_hint_level(_inp(request_hint=True, correct=False, current_hint_level=3))
    assert d.level == 4


def test_request_with_repeated_failures_gives_full_explanation():
    """多次提示仍未推进 + 主动请求 → L5 完整讲解。"""
    d = decide_hint_level(
        _inp(request_hint=True, correct=False, current_hint_level=4, consecutive_failures=4)
    )
    assert d.level == 5


def test_request_after_many_failures_gives_full_explanation():
    d = decide_hint_level(_inp(request_hint=True, correct=False, consecutive_failures=4))
    assert d.level == 5


def test_correct_answer_with_request_gives_reinforcement():
    """答对后仍请求提示 → **从 L2 起步，并可逐级升级**。

    ## ⚠️ 这条断言改过（2026-10-07，产品决定）
    **原断言是 `d.level == 1`** —— 设计意图是"判对时给一句巩固性说明"。
    但实测下来是：**反复点请求提示永远停在 L1，点了没变化**。

    产品方选择了另一种解释：**做对的学生点"请求提示"，意思往往是
    "我想多懂一点"**（为什么是这个方向？还有没有别的解法？），
    而不是"我卡住了"。固定的 L1 巩固语太浅。

    **改成与判错时同一套升级阶梯**（`_hint_for_request`），
    只是**理由措辞不同** —— 判对说"请求更深入的解释"。

    **不是把测试放宽了事**：下面 `test_correct_request_reason_mentions_depth`
    把新的措辞也钉住了。
    """
    d = decide_hint_level(_inp(request_hint=True, correct=True))
    assert d.level == 2, "判对时请求提示应当升到 L2（产品决定）"


def test_correct_request_reason_mentions_depth():
    """判对时的理由要说**"更深入的解释"**，不是"卡住了"。

    同样是升级，但语义不同 —— 理由要能区分，否则 trace 里看不出来。
    """
    d = decide_hint_level(_inp(request_hint=True, correct=True))
    joined = " ".join(d.reasons)
    assert "更深入" in joined or "深入" in joined, d.reasons


def test_correct_request_escalates_step_by_step():
    """**判对时反复点也要逐级升**，不能停在 L2。"""
    seen, lv = [], 2
    for _ in range(3):
        d = decide_hint_level(_inp(request_hint=True, correct=True,
                                   current_hint_level=lv))
        seen.append(d.level)
        lv = d.level
    assert seen == [3, 4, 5], f"判对时升级阶梯不对：{seen}"


# ---- 错误作答的自动升级（级别由连续失败轮数驱动） ----

@pytest.mark.parametrize("failures,expected", [(1, 1), (2, 2), (3, 3), (4, 4), (6, 4)])
def test_error_escalates_with_failures(failures, expected):
    d = decide_hint_level(_inp(correct=False, consecutive_failures=failures))
    assert d.level == expected


def test_never_auto_goes_above_4():
    """L5 永不自动下发。"""
    d = decide_hint_level(
        _inp(correct=False, consecutive_failures=9, stuck_score=0.95, is_key_node=True)
    )
    assert d.level == 4


# ---- 修正项 ----

def test_high_stuck_score_adds_one():
    d = decide_hint_level(_inp(correct=False, consecutive_failures=2, stuck_score=0.80))
    assert d.level == 3  # 2 + 1


def test_key_node_with_repeated_failures_adds_one():
    d = decide_hint_level(_inp(correct=False, consecutive_failures=3, is_key_node=True))
    assert d.level == 4  # 3 + 1


def test_same_hint_before_adds_one():
    d = decide_hint_level(_inp(correct=False, consecutive_failures=2, same_hint_before=True))
    assert d.level == 3  # 2 + 1


def test_decision_reasons_are_recorded():
    d = decide_hint_level(
        _inp(correct=False, consecutive_failures=3, stuck_score=0.8, is_key_node=True)
    )
    assert d.level == 4
    assert any("连续未推进" in r for r in d.reasons)
    assert any("卡顿分" in r for r in d.reasons)
    assert any("关键节点" in r for r in d.reasons)
