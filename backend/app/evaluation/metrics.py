"""P18 评测指标：六个纯函数。

## 为什么放在 `backend/app/` 而不是 `scripts/`

指标要能被**测试**、被 **P24 教师端**复用、被**报告持久化**调用。
放在脚本里只能从命令行用一次。

## 口径必须显式（这是 P18 前置的兑现）

`docs/排查-节点轨迹三种口径.md` 记的教训：全项目对"学生到达了哪些节点"
曾有三种口径（完成 / 碰到 / 试了没成），数字互相矛盾。

所以这里的每个指标都在 docstring 里**写明它用哪个口径**，
并且**只接受调用方传进来的集合**——口径由调用方从
`app.agent.node_semantics` 取，本模块不自己判断。

## 每个指标都返回 None 表示"无法计算"

而不是返回 0。**0 和"没有数据"是两个完全不同的结论**——
把"没有负样本"报成"假诊断率 0%"，会让一份没有依据的报告看起来很漂亮。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rate:
    """一个比率指标：值 + 分母 + 明细。

    **带上分母**：`1/1` 和 `100/100` 同为 100%，可信度完全不同。
    只报百分比是评测报告最常见的不诚实。
    """

    value: float | None
    numerator: int = 0
    denominator: int = 0
    note: str = ""
    detail: list[str] = field(default_factory=list)
    is_count: bool = True

    @property
    def computable(self) -> bool:
        return self.value is not None

    def render(self) -> str:
        if not self.computable:
            return f"n/a（{self.note or '无数据'}）"
        pct = f"{self.value:.0%}"
        # 计数指标即使分子为 0、有排除说明，也必须保留完整分子/分母。
        counts = (f"（{self.numerator}/{self.denominator}）"
                  if self.is_count else f"（{self.denominator} 个样本）")
        return f"{pct}{counts}" + (f"  ⚠ {self.note}" if self.note else "")


# ---------------------------------------------------------------- 节点类

def node_recall(gold_nodes: list[str], touched: set[str]) -> Rate:
    """该覆盖的节点覆盖到了多少。

    **口径：`touched`（碰到就算，不区分对错）** —— 来自
    `app.agent.node_semantics.touched_nodes`。

    为什么不用"正确完成"：这个指标问的是"学生**走到**了没有"（覆盖度），
    而"完成"是掌握度，两者对同一个学生可以差很多。
    要掌握度请另用 `completed_nodes` 并**在报告里写明**。
    """
    gold = set(gold_nodes)
    if not gold:
        return Rate(None, note="用例未声明 gold 节点集")
    hit = gold & touched
    return Rate(len(hit) / len(gold), len(hit), len(gold),
                detail=[f"缺失：{sorted(gold - touched)}"] if gold - touched else [])


def alternative_path_acceptance(labeled: list[tuple[str, bool]]) -> Rate:
    """走替代解法的学生被判为正确的比例。

    **为什么单列一个指标**：`path_inconsistency` 那条校验规则
    （"给了替代解法但图上只有一条路径"）实测在**每一张自动生成的图上都触发**。
    后果就是：学生用另一种正确方法解题，系统可能判他错。

    参数是 `[(case_id, accepted)]`——调用方负责判断"这个替代路径有没有被接受"。
    """
    if not labeled:
        return Rate(None, note="没有声明替代路径的用例")
    ok = sum(1 for _, accepted in labeled if accepted)
    return Rate(ok / len(labeled), ok, len(labeled),
                detail=[f"被拒：{cid}" for cid, a in labeled if not a])


# ---------------------------------------------------------------- 诊断类

def false_diagnosis_rate(labeled: list[tuple[str, bool, bool | None]]) -> Rate:
    """**系统说是错的，其中有多少其实是对的。**

    参数是 `[(case_id, system_says_error, is_really_error)]`。

    这是教学系统里**代价最高**的一类错误：冤枉学生会打击信心，
    而且比"漏报一个错误"严重得多。所以单列指标盯着它。

    ⚠️ **必须有负样本**（`is_really_error=False` 的用例）。
    全是错误样本时无法计算——此时返回 `n/a` 而不是 0%，
    因为"没测"和"测得很好"是两回事。
    """
    # `real is None` = 用例没标注 → **排除**，不当成"学生是对的"
    negatives = [(cid, said) for cid, said, real in labeled if real is False]
    if not negatives:
        return Rate(None, note="用例集里没有'学生其实是对的'的负样本，无法计算")
    fp = sum(1 for _, said in negatives if said)
    return Rate(fp / len(negatives), fp, len(negatives),
                detail=[f"误判：{cid}" for cid, said in negatives if said])


def diagnosis_accuracy(labeled: list[tuple[str, bool, bool | None]]) -> Rate:
    """诊断整体准确率（对错都算）。与假诊断率**一起看**才有意义。

    只看准确率会被"全判为对"刷高——那种策略准确率等于负样本占比。
    """
    # 未标注的用例一并排除——否则"准确率"里混进了没依据的样本
    usable = [(c, s, r) for c, s, r in labeled if r is not None]
    if not usable:
        return Rate(None, note="无已标注用例")
    ok = sum(1 for _, said, real in usable if said == real)
    return Rate(ok / len(usable), ok, len(usable),
                note=f"另有 {len(labeled) - len(usable)} 条未标注，已排除"
                if len(usable) < len(labeled) else "")


# ---------------------------------------------------------------- 检索类

def hit_rate(per_query: list[tuple[str, bool]]) -> Rate:
    """`[(query_id, 是否有相关结果进入 top_k)]` 中命中的比例。"""
    if not per_query:
        return Rate(None, note="无检索用例")
    ok = sum(1 for _, hit in per_query if hit)
    return Rate(ok / len(per_query), ok, len(per_query),
                detail=[f"未命中：{q}" for q, hit in per_query if not hit])


def mrr(per_query: list[tuple[str, int | None]]) -> Rate:
    """平均倒数排名。`[(query_id, 首个相关结果的 1-based 排名 或 None)]`。

    **为什么要 MRR 而不只看 Hit Rate**：Hit Rate 只看"有没有进 top-K"，
    把答案排在第 1 位和第 5 位当成一样好。而 Tutor 只把前几条喂给模型
    ——排名直接决定学生看到什么。
    """
    if not per_query:
        return Rate(None, note="无检索用例")
    total = 0.0
    for _, rank in per_query:
        if rank and rank > 0:
            total += 1.0 / rank
    # **numerator 留 0**：MRR 不是计数比，报成 "46%（4/4）" 会被读成
    # "4 条里中了 4 条"——正好相反（4 条里 2 条排第 1、2 条落空才得 0.5）。
    # 分母是"查询数"，分子没有对应含义。
    return Rate(total / len(per_query), 0, len(per_query),
                note="倒数排名的均值，非计数比",
                is_count=False,
                detail=[f"未命中：{q}" for q, r in per_query if not r])


# ---------------------------------------------------------------- 结构类

def structure_valid_rate(results: list[tuple[str, bool, list[str]]]) -> Rate:
    """生成图通过结构校验的比例。

    参数是 `[(graph_id, 校验通过, 问题列表)]`。

    **与"物理正确率"分开**：结构合法（有起止节点、无孤儿、路径连通）
    不代表物理正确。合在一起报会掩盖"结构全过但物理全错"的情况。
    """
    if not results:
        return Rate(None, note="没有可校验的图")
    ok = sum(1 for _, valid, _ in results if valid)
    bad = [gid for gid, valid, _ in results if not valid]
    return Rate(ok / len(results), ok, len(results),
                detail=[f"结构不合法：{gid}" for gid in bad])


# ---------------------------------------------------------------- 汇总

#: 指标名 → (中文名, 口径说明)。报告里必须带口径，否则数字无法解读。
METRIC_SPECS: dict[str, tuple[str, str]] = {
    "node_recall": ("Node Recall", "节点口径 = touched（碰到就算）"),
    "alternative_path_acceptance": ("Alternative Path Acceptance", "按用例声明的替代路径"),
    "false_diagnosis_rate": ("False Diagnosis Rate", "降序指标，越低越好"),
    "diagnosis_accuracy": ("Diagnosis Accuracy", "与假诊断率一起看"),
    "hit_rate": ("Hit Rate", "top_k 内有相关结果"),
    "mrr": ("MRR", "首个相关结果的倒数排名"),
    "structure_valid_rate": ("Structure Valid Rate", "只判结构，不含物理"),
}
