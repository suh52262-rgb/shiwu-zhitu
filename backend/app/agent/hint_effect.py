"""P22 Hint Effectiveness：判定"上一次提示到底有没有用"。

## 为什么单独一个模块

P21 让系统能**记录**发了什么提示；P22 要回答的是**那个提示有没有起作用**。
这是两件事：前者是事实，后者是**判断**——而判断必须可解释、可复核。

## 三态，不是布尔

`effective` 取 `True` / `False` / **`None`**：

  · `True`  —— 提示后学生在**同一个节点**上推进了
  · `False` —— 提示后学生**还在原地**，又失败了
  · `None`  —— **无法判定**（换了节点、输入与题无关、本轮分析降级…）

`None` 不是"没效果"，而是"我们不知道"。把它当成 `False` 会低估提示效果，
当成 `True` 会高估——**两种都是编造**。项目里 `correct` 早就是三态，
这里沿用同一条原则。

## 判定的难点：怎么算"推进了"

不能只看"下一轮对不对"——学生可能换了节点（那说明推进了），
也可能退回去重做（那也是推进）。所以用**状态差分**：
比较提示前快照与当前状态里"已完成节点集合"是否增长。

这也是为什么要存 `state_before` 快照：**没有提示前的状态，就无法判断之后的变化。**
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HintEffect:
    """一次提示的效果判定结果。"""

    #: True=有效 / False=无效 / None=无法判定（**不是"无效"**）
    effective: bool | None = None
    #: 判定依据（人可读，写进库供复核）
    evidence: str = ""

    @property
    def decidable(self) -> bool:
        return self.effective is not None


def judge_hint_effect(
    *,
    hint_node: str,
    visited_before: list[str],
    visited_after: list[str],
    current_node_after: str = "",
    correct_after: bool = False,
    consecutive_failures_after: int = 0,
    off_graph_after: bool = False,
    degraded_after: bool = False,
) -> HintEffect:
    """判定上一次提示是否有效。

    参数全部是**显式传入的状态**，本函数不访问数据库也不调模型——
    这样它可测、可复核，也能被 P24 教师端复用。
    """
    before = set(visited_before or [])
    after = set(visited_after or [])

    # ---- 先排除"无法判定"的情形 ----
    #
    # 顺序很重要：这些情形即使看起来"没推进"，也不能判为无效。
    if degraded_after:
        return HintEffect(None, "本轮分析降级，无法判断提示后学生是否推进")
    if off_graph_after:
        return HintEffect(None, "本轮输入与本题无关，不能作为提示效果的依据")

    # ---- 有效：已完成节点集合增长 ----
    gained = after - before
    if gained:
        return HintEffect(True, f"提示后推进：新完成 {sorted(gained)}")

    # ---- 有效：没新增完成节点，但**离开了原节点** ----
    # （学生可能换了个思路重做，或退回去补前面的步骤——都算推进）
    if current_node_after and hint_node and current_node_after != hint_node:
        return HintEffect(
            True,
            f"提示后离开原节点 {hint_node} → {current_node_after}",
        )

    # ---- 无效：仍在原节点，且本轮仍未推进 ----
    if not correct_after and consecutive_failures_after > 0:
        return HintEffect(
            False,
            f"提示后仍在节点 {hint_node or '（未知）'} 且连续未推进 "
            f"{consecutive_failures_after} 轮",
        )

    # ---- 其余：本轮判对了但没新增节点 ----
    # 例如"结论对但缺推理"被指出后学生补了推理，节点集合不变。
    # 这算不算有效？**算**——`correct_after=True` 说明确实往前走了。
    if correct_after:
        return HintEffect(True, "提示后本轮判定为正确推进（未新增完成节点）")

    return HintEffect(None, "状态没有可判定的变化")


def summarize_effects(effects: list[bool | None]) -> dict:
    """汇总一批判定结果。

    **`None` 单独计数**，不计入分母——否则"无法判定"会被当成"无效"，
    把提示效果系统性地算低。
    """
    decided = [e for e in effects if e is not None]
    return {
        "total": len(effects),
        "decided": len(decided),
        "undecidable": len(effects) - len(decided),
        "effective": sum(1 for e in decided if e),
        "ineffective": sum(1 for e in decided if not e),
        # 分母用**可判定的条数**，不是总数
        "rate": (sum(1 for e in decided if e) / len(decided)) if decided else None,
    }
