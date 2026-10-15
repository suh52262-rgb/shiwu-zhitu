"""P22 Hint Effectiveness：判定"上一次提示有没有用"。

## 这组测试盯的三件事

1. **三态**：`effective` 是 `True`/`False`/**`None`**。`None` 不是"无效"，
   而是"无法判定"——把它当 `False` 会**系统性地低估**提示效果。
2. **判据是状态差分**，不是"下一轮对不对"。学生换节点、退回去补步骤
   也算推进。
3. **汇总时分母用可判定条数**，不是总数。
"""
import pytest

from app.agent.hint_effect import HintEffect, judge_hint_effect, summarize_effects


# ---------------------------------------------------------------- 有效

def test_effective_when_completed_nodes_grow():
    """判据核心：**已完成节点集合增长** = 推进了。"""
    e = judge_hint_effect(
        hint_node="S3",
        visited_before=["S1", "S2"],
        visited_after=["S1", "S2", "S3"],
        current_node_after="S3",
        correct_after=True,
    )
    assert e.effective is True
    assert "S3" in e.evidence


def test_effective_when_student_left_the_node():
    """离开了原节点也算推进——学生可能换了思路，或退回去补前面的步骤。"""
    e = judge_hint_effect(
        hint_node="S3",
        visited_before=["S1", "S2"],
        visited_after=["S1", "S2"],
        current_node_after="S5",
        correct_after=False,
    )
    assert e.effective is True
    assert "S3" in e.evidence and "S5" in e.evidence


def test_effective_when_round_correct_without_new_node():
    """本轮判对但没新增完成节点（如"结论对、补了推理"）——**算有效**。"""
    e = judge_hint_effect(
        hint_node="S3",
        visited_before=["S1"], visited_after=["S1"],
        current_node_after="S3", correct_after=True,
    )
    assert e.effective is True


# ---------------------------------------------------------------- 无效

def test_ineffective_when_stuck_on_same_node():
    e = judge_hint_effect(
        hint_node="S3",
        visited_before=["S1", "S2"], visited_after=["S1", "S2"],
        current_node_after="S3",
        correct_after=False, consecutive_failures_after=2,
    )
    assert e.effective is False
    assert "S3" in e.evidence


# ---------------------------------------------------------------- 无法判定

def test_undecidable_when_degraded():
    """降级时**不能判为无效**——我们根本不知道学生推进没有。"""
    e = judge_hint_effect(
        hint_node="S3", visited_before=[], visited_after=[],
        current_node_after="S3", correct_after=False,
        consecutive_failures_after=1, degraded_after=True,
    )
    assert e.effective is None
    assert not e.decidable
    assert "降级" in e.evidence


def test_undecidable_when_off_graph():
    """与本题无关的输入不能作为效果依据。"""
    e = judge_hint_effect(
        hint_node="S3", visited_before=["S1"], visited_after=["S1"],
        current_node_after="S3", correct_after=False,
        consecutive_failures_after=1, off_graph_after=True,
    )
    assert e.effective is None


def test_undecidable_when_no_change():
    e = judge_hint_effect(
        hint_node="S3", visited_before=["S1"], visited_after=["S1"],
        current_node_after="S3", correct_after=False,
        consecutive_failures_after=0,
    )
    assert e.effective is None


def test_none_is_not_false():
    """**这是本模块最容易搞错的一点。**"""
    e = judge_hint_effect(
        hint_node="S3", visited_before=[], visited_after=[],
        current_node_after="S3", degraded_after=True)
    assert e.effective is not False
    assert e.effective != False  # noqa: E712  刻意用 != 强调三态
    assert e.effective is None


# ---------------------------------------------------------------- 汇总

def test_summary_excludes_undecidable_from_denominator():
    """**分母用可判定条数，不是总数。**

    把"无法判定"算进分母会把提示效果系统性地算低——
    等于因为"没测到"而扣分。
    """
    s = summarize_effects([True, True, False, None, None])
    assert s["total"] == 5
    assert s["decided"] == 3
    assert s["undecidable"] == 2
    assert s["effective"] == 2
    assert s["ineffective"] == 1
    assert s["rate"] == pytest.approx(2 / 3), "分母应是可判定条数 3，而不是总数 5"


def test_summary_rate_none_when_nothing_decidable():
    """一条都判不了时 `rate` 是 `None`，**不是 0%**。"""
    s = summarize_effects([None, None])
    assert s["rate"] is None
    assert s["decided"] == 0


def test_summary_empty():
    s = summarize_effects([])
    assert s["total"] == 0 and s["rate"] is None


# ---------------------------------------------------------------- 接线

def test_persist_and_resolve_are_wired():
    """**防"写了没接"**：表字段、落库、回看三处都要真的接上。"""
    import inspect

    from app.api import attempts
    from app.db import persistence

    api_src = inspect.getsource(attempts)
    assert "resolve_pending_hints" in api_src, "每轮没有回看判定上一条提示"
    assert "state_before" in api_src, "下发提示时没有记提示前快照"

    db_src = inspect.getsource(persistence.persist_hint)
    assert "return int(row.id)" in db_src, "persist_hint 没有返回 id"


def test_snapshot_is_taken_before_hint_not_after():
    """快照必须是**提示前**的状态。

    记成提示后的话，"之后的变化"就永远是空的——判定会全部落到
    `undecidable`，而且**看不出来是错的**。
    """
    import inspect

    from app.api import attempts

    src = inspect.getsource(attempts)
    i_resolve = src.index("resolve_pending_hints")
    i_persist = src.index("persist_hint(")
    assert i_resolve < i_persist, \
        "判定与落库的顺序反了：本轮的新提示会参与判定自己"
