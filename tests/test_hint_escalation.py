"""学生主动请求提示的升级行为 —— **判对与判错共用一套阶梯**。

## 用户实测报的现象
> 无论点击多少次「请求提示」按钮，都只会给出 L1 级提示。

## 查清的过程与结论
```python
if inp.correct:                      # ← 判对时先短路
    if inp.request_hint:
        return HintDecision(1, ["本轮推进正确，按请求给巩固性方向说明"])
```
· 既有测试 `test_correct_answer_with_request_gives_reinforcement`
  **断言的就是 `level == 1`** —— 所以**这是有意的设计，不是疏忽**
· 升级逻辑（L2→L5）**只在判错时才走得到**

## ⚠️ 中间我一度判定"这是设计、只改 UI 文案" —— 然后产品方选了另一种解释
**做对的学生点"请求提示"，意思往往是"我想多懂一点"**
（为什么是这个方向？还有没有别的解法？），而不是"我卡住了"。
**固定的 L1 巩固语太浅，点了没变化。**

**⇒ 决定（2026-10-07）：判对时也逐级升级，与判错共用同一套阶梯**，
只是**理由措辞不同** —— 判对说"请求更深入的解释"。

**这条改动的依据在 `test_hint_policy.py` 里也记了**
（那条旧断言不是被放宽，而是被**替换成新意图**并补了措辞测试）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.agent.hint_policy import (  # noqa: E402
    FAILURES_FOR_FULL_EXPLANATION,
    HintPolicyInput,
    decide_intervention,
)


def _inp(**kw) -> HintPolicyInput:
    base = dict(correct=False, off_graph=False, degraded=False,
                request_hint=True, current_hint_level=0,
                consecutive_failures=0)
    base.update(kw)
    return HintPolicyInput(**base)


# ================================================================ 设计取舍（正面钉住）

def test_correct_round_request_escalates_to_L2():
    """**判对时请求提示也要升到 L2 —— 产品决定（2026-10-07）。**

    原来固定给 L1 巩固性说明，实测"无论点多少次都只给 L1"。
    产品方解释：**做对的学生点请求提示，意思是"我想多懂一点"**
    （为什么是这个方向？还有没有别的解法？），而不是"我卡住了"。
    **固定的 L1 巩固语太浅。**
    """
    # 注意：**请求提示的下限是 L2**（`min(max(lv+1, 2), 4)`）——
    # 请求了就不该只给 L1。所以 lv=0 → 2，不是 1。
    assert decide_intervention(
        _inp(correct=True, current_hint_level=0)).level == 2
    assert decide_intervention(
        _inp(correct=True, current_hint_level=1)).level == 2
    assert decide_intervention(
        _inp(correct=True, current_hint_level=2)).level == 3


def test_correct_round_request_escalates_step_by_step():
    """判对时反复点也要**逐级升到 L5**。"""
    seen, lv = [], 2
    for _ in range(3):
        d = decide_intervention(_inp(correct=True, current_hint_level=lv))
        seen.append(d.level)
        lv = d.level
    assert seen == [3, 4, 5], f"判对时升级阶梯不对：{seen}"


def test_explicit_request_escalates_even_with_missing_reasoning():
    assert decide_intervention(_inp(correct=True, current_hint_level=2,
                                   missing_reasoning=["未展示正方向约定"])).level == 3
    assert decide_intervention(_inp(correct=True, request_hint=False,
                                   missing_reasoning=["未展示正方向约定"])).level == 1


def test_correct_round_reason_says_depth_not_being_stuck():
    """判对时的理由要说**"更深入的解释"**，而不是"主动请求/卡住了"。

    同样是升级，**语义不同** —— 理由要能区分，否则 trace 里看不出
    学生是"卡住求助"还是"做对了想更深"。
    """
    d = decide_intervention(_inp(correct=True, current_hint_level=1))
    joined = " ".join(d.reasons)
    assert "深入" in joined, f"判对的理由没说清是深入解释：{d.reasons}"
    assert "已判对" in joined, d.reasons


# ================================================================ 判错时**确实会升级**

def test_wrong_round_request_escalates_step_by_step():
    """**判错时反复点要 L2 → L3 → L4 → L5。**

    这才是"学生真卡住"的情形 —— 升级逻辑就是为它写的。
    """
    seen, lv = [], 1
    for _ in range(4):
        d = decide_intervention(_inp(correct=False, current_hint_level=lv))
        seen.append(d.level)
        lv = d.level
    assert seen == [2, 3, 4, 5], f"升级阶梯不对：{seen}"


def test_first_request_on_wrong_round_jumps_to_two():
    """判错 + 还没给过提示（lv=0）→ 至少 L2。请求了就不该只给 L1。"""
    d = decide_intervention(_inp(correct=False, current_hint_level=0))
    assert d.level >= 2


def test_never_exceeds_five():
    assert decide_intervention(_inp(current_hint_level=4)).level == 5
    assert decide_intervention(_inp(current_hint_level=9)).level == 5


def test_caps_at_four_before_full_explanation():
    """逐级升，**不直接跳到 L5** —— 只是想要点提示时不该直接给答案。"""
    assert decide_intervention(_inp(current_hint_level=2)).level == 3


def test_consecutive_failures_trigger_full_explanation():
    """连续失败够多 → L5 完整讲解。"""
    d = decide_intervention(_inp(current_hint_level=1,
                                 consecutive_failures=FAILURES_FOR_FULL_EXPLANATION))
    assert d.level == 5


# ================================================================ 不抑制原则

def test_wrong_round_request_is_never_suppressed():
    """**"主动请求永远不抑制"** —— 判错时各种级别下都必须给出 > L0。"""
    for lv in (0, 1, 2, 3, 4, 5):
        for fails in (0, 3):
            d = decide_intervention(_inp(correct=False, current_hint_level=lv,
                                         consecutive_failures=fails))
            assert d.level > 0, f"请求被压制：lv={lv} fails={fails}"


def test_reasons_show_the_transition():
    """理由要能看出**从哪一级升到哪一级**（可解释性是本模块的要求）。"""
    d = decide_intervention(_inp(correct=False, current_hint_level=1))
    joined = " ".join(d.reasons)
    assert "L1" in joined and "L2" in joined, d.reasons


# ================================================================ 别的情形没被影响

def test_correct_without_request_still_observes():
    d = decide_intervention(_inp(correct=True, request_hint=False,
                                 current_hint_level=0))
    assert d.level == 0


def test_degraded_still_withholds():
    """降级时一律不提示 —— 优先级最高。

    （降级意味着**我们不知道**学生这步对不对，给提示就是编造。）
    """
    assert decide_intervention(_inp(degraded=True, correct=True)).level == 0


def test_off_graph_still_observes():
    d = decide_intervention(_inp(off_graph=True, correct=False, request_hint=False))
    assert d.level == 0
