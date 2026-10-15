"""P21 教学干预控制器。

## 这一版相对旧版的核心变化

旧版 `decide_hint_level` 把"是否干预"和"给几级"耦合在一起：
`level == 0` 既表示"没事发生"，也表示"故意不给"，**调用方分不清**。
而且它**只会加压**——失败越多级别越高，没有一条"现在不该打断学生"
的反向逻辑。

P21 补的三样：`action` 字段、补全的输入、以及**抑制**这条反向逻辑。
"""
import pytest

from app.agent.hint_policy import (
    ACTION_EXPLAIN,
    ACTION_GUIDE,
    ACTION_NUDGE,
    ACTION_OBSERVE,
    ACTION_WITHHOLD,
    GRACE_FAILS_ON_NEW_NODE,
    HintPolicyInput,
    action_for_level,
    decide_hint_level,
    decide_intervention,
)


# ---------------------------------------------------------------- action 语义

def test_observe_and_withhold_are_distinct():
    """**`level=0` 有两种完全不同的含义**——这是本模块存在的理由。

    · `observe`  = 没事发生，继续看
    · `withhold` = **本来能给，故意不给**（有明确理由）

    旧版只有一个 `level`，调用方无法区分，也就无法把"给学生留思考空间"
    这件事如实记录下来。
    """
    assert ACTION_OBSERVE != ACTION_WITHHOLD

    observe = decide_intervention(HintPolicyInput(correct=False, off_graph=True))
    assert observe.action == ACTION_OBSERVE
    assert observe.level == 0
    assert not observe.is_withheld

    withhold = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=1,
        current_node="S3", just_entered_node=True))
    assert withhold.action == ACTION_WITHHOLD
    assert withhold.level == 0
    assert withhold.is_withheld
    assert withhold.withheld_reason, "抑制必须给出理由"


def test_action_tracks_level():
    assert action_for_level(0) == ACTION_OBSERVE
    assert action_for_level(1) == ACTION_NUDGE
    assert action_for_level(2) == ACTION_NUDGE
    assert action_for_level(3) == ACTION_GUIDE
    assert action_for_level(4) == ACTION_GUIDE
    assert action_for_level(5) == ACTION_EXPLAIN


def test_should_intervene_property():
    """注意：**空输入不等于"没有信号"**。

    `HintPolicyInput()` 的默认值是 `correct=False, consecutive_failures=0`——
    这是一个**矛盾输入**（答错了却说 0 轮失败），策略用
    `max(consecutive_failures, 1)` 防御性地给 L1。这是有意为之。
    真正的"没有信号"要用 `correct=True` 或 `off_graph=True` 表达。
    """
    assert decide_intervention(HintPolicyInput()).level >= 1, "矛盾输入应防御性给 L1"
    assert not decide_intervention(HintPolicyInput(correct=True)).should_intervene
    assert not decide_intervention(HintPolicyInput(off_graph=True)).should_intervene
    d = decide_intervention(HintPolicyInput(correct=False, consecutive_failures=2))
    assert d.should_intervene


# ---------------------------------------------------------------- 反向逻辑（新增）

def test_holds_back_when_just_entered_new_node():
    """**反向逻辑**：刚走到新节点、首次失误 → 不提示。

    旧版只会加压：失败 1 轮立刻给 L1。但学生刚到新步骤时，
    一次失误就介入会**剥夺他自我纠错的机会**，
    还会让他形成"一卡就等提示"的依赖。
    """
    d = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=1,
        current_node="S3", just_entered_node=True))
    assert d.is_withheld
    assert d.level == 0
    assert "S3" in d.reasons[0]


def test_suppression_does_not_apply_after_grace():
    """**抑制不是放任**：连着错超过宽限轮数，还是按常规加压。"""
    d = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=GRACE_FAILS_ON_NEW_NODE + 1,
        current_node="S3", just_entered_node=True))
    assert d.should_intervene, "宽限轮数之后仍不提示，就是把学生晾着"
    assert d.level >= 1


def test_suppression_disabled_if_already_hinted_on_node():
    """该节点上已经提示过 → 不再抑制（否则会与"已提示过要升级"自相矛盾）。"""
    d = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=1, current_node="S3",
        just_entered_node=True, same_hint_before=True))
    assert d.should_intervene


def test_idle_alone_never_triggers_a_hint():
    """**单纯发呆不提示**——学生可能正在思考。

    这是 `idle_time` 唯一正确的用法：**只在结合失败时**作为加压依据。
    """
    d = decide_intervention(HintPolicyInput(
        correct=True, idle_time=300.0, consecutive_failures=0))
    assert d.action == ACTION_OBSERVE
    assert d.level == 0


def test_long_idle_plus_repeated_errors_overrides_suppression():
    """但"空转很久 + 已经错过不止一次"要区分开——那时候别再晾着他。"""
    d = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=1, current_node="S3",
        just_entered_node=True, idle_time=120.0, error_count=3))
    assert d.should_intervene, "明显卡住时不该继续抑制"


def test_student_request_always_wins_over_suppression():
    """**主动请求永不抑制**——那是明确的求助信号，压制它等于让学生白问。"""
    d = decide_intervention(HintPolicyInput(
        request_hint=True, correct=False, consecutive_failures=1,
        current_node="S3", just_entered_node=True))
    assert d.should_intervene
    assert d.level >= 2


# ---------------------------------------------------------------- 输入补全

def test_hint_history_counts_per_node():
    """`hint_history` 比 `same_hint_before` 一个布尔细：

    知道"这个节点提示过 2 次"和"1 次"，对是否升级的判断不一样。
    """
    inp = HintPolicyInput(
        correct=False, consecutive_failures=2, current_node="S3",
        hint_history=[{"node": "S3", "level": 1}, {"node": "S2", "level": 1}])
    d = decide_intervention(inp)
    assert any("已提示 1 次" in r or "该节点" in r for r in d.reasons) or d.level >= 2


def test_many_hints_on_same_node_escalate():
    inp = HintPolicyInput(
        correct=False, consecutive_failures=2, current_node="S3",
        hint_history=[{"node": "S3", "level": 1}, {"node": "S3", "level": 2}])
    escalated = decide_intervention(inp)
    baseline = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=2, current_node="S3"))
    assert escalated.level >= baseline.level


def test_history_entries_without_node_are_ignored():
    """历史项格式异常不能崩——它来自持久化，可能被旧版本写过。"""
    d = decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=1, current_node="S3",
        hint_history=[{"level": 1}, "garbage", None]))
    assert d.level >= 1


# ---------------------------------------------------------------- 既有行为不回退

def test_previous_behaviours_unchanged():
    """P21 是**增量**：原有决策不能变。"""
    # 降级 → 不提示（P20）
    d = decide_intervention(HintPolicyInput(degraded=True))
    assert d.level == 0 and d.is_withheld

    # 正确 + 缺推理 → L1
    d = decide_intervention(HintPolicyInput(
        correct=True, missing_reasoning=["没说明 N=0"]))
    assert d.level == 1

    # 正确 → L0
    assert decide_intervention(HintPolicyInput(correct=True)).level == 0

    # 无关输入 → L0
    assert decide_intervention(HintPolicyInput(off_graph=True)).action == ACTION_OBSERVE

    # 连续失败递增
    levels = [decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=n)).level for n in (1, 2, 3, 4)]
    assert levels == sorted(levels), "级别应随失败轮数单调不减"

    # 自动提示不超过 L4
    assert decide_intervention(HintPolicyInput(
        correct=False, consecutive_failures=99)).level <= 4


def test_legacy_alias_still_works():
    """`decide_hint_level` 保留为兼容名（新代码用 `decide_intervention`）。"""
    inp = HintPolicyInput(correct=False, consecutive_failures=2)
    assert decide_hint_level(inp).level == decide_intervention(inp).level


# ---------------------------------------------------------------- 接线

def test_graph_passes_new_inputs():
    """**防"写了没接"**：图里必须真的把 P21 的上下文传进去。"""
    import inspect

    from app.agent import graph

    src = inspect.getsource(graph.hint_policy)
    for field in ("just_entered_node", "idle_time", "hint_history",
                  "current_node", "error_count"):
        assert field in src, f"图里没有传 {field}"
    assert "hint_action" in src, "没有把 action 回传（前端/Trace 无法区分 observe/withhold）"
