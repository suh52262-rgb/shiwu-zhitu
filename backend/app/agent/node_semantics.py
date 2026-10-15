"""节点归属的**唯一口径定义处**。

## 为什么需要这个模块

"学生到达了哪些节点"有三种都合理的答案，取决于问的是什么问题：

| 口径 | 回答的问题 | 用途 |
|---|---|---|
| **`completed_nodes`** | 学生**正确建立**了哪些步骤？ | 掌握进度、路径单调性 |
| **`touched_nodes`** | 学生这轮**碰到**了哪些步骤？ | 覆盖度（Node Recall） |
| **`attempted_only_nodes`** | 哪些步骤**试了但没成**？ | 卡点识别、错误诊断 |

问题从来不是"有三种"——而是**三处消费方各用各的、谁也不说用的是哪一种**。

实测踩过的坑：全链冒烟测试里 3 条节点断言失败，排查后确认
**不是 bug**——匹配器按"只承认被论证成立的步骤"给了 `[]`，
而用例期望按"碰到了就算"写。两边都对，只是**口径不同**。
见 `docs/排查-节点轨迹三种口径.md`。

## 用法

**不要自己写 `set(match.matched_nodes) | ...`。** 从下面三个函数里挑一个，
挑的时候问自己："我要的是'碰过'还是'完成'？"

P18 做指标时尤其要注意：**每个指标都要显式声明口径**。
Node Recall 通常是"碰过"（覆盖度），而路径单调性是"完成"。
混用会得出互相矛盾的数字。
"""
from __future__ import annotations

from typing import Any

from app.agent.step_matcher import StepMatch


def _get(match: StepMatch | dict[str, Any], field: str) -> list[str]:
    """同时接受 `StepMatch` 对象与 API 返回的 dict。

    两种来源都要支持：图内直接用对象，而模拟器/评测拿到的是
    HTTP 响应里的 `step_match` 字典。**让口径函数适配两种来源，
    好过让每个调用方各写一遍提取逻辑**——那正是三种口径的由来。
    """
    if isinstance(match, dict):
        return list(match.get(field) or [])
    return list(getattr(match, field, None) or [])


def _correct(match: StepMatch | dict[str, Any]) -> bool:
    if isinstance(match, dict):
        return bool(match.get("correct"))
    return bool(getattr(match, "correct", False))


def completed_nodes(match: StepMatch | dict[str, Any]) -> set[str]:
    """学生这轮**正确建立**的节点。

    与 `state_tracker.apply_step_match` 里 `visited_nodes` 的口径**完全一致**
    （那里写的是 `match.matched_nodes if match.correct else []`）。

    ⚠️ 语义是"**论证成立**"而不是"提到了"。学生只给结论不论证时，
    匹配器**有意**不给节点信用——这是教学上正确的设计，
    不要为了用例好过而放宽。
    """
    return set(_get(match, "matched_nodes")) if _correct(match) else set()


def touched_nodes(match: StepMatch | dict[str, Any]) -> set[str]:
    """学生这轮**碰到**的节点：对应上的 ∪ 尝试过的。

    用于**覆盖度**类判断（"这个知识点学生碰到没有"），不区分对错。
    `scripts/run_eval.py` 原本自己拼了 `matched | attempted`，现在从这里取。
    """
    return set(_get(match, "matched_nodes")) | set(_get(match, "attempted_nodes"))


def attempted_only_nodes(match: StepMatch | dict[str, Any]) -> set[str]:
    """**试了但没完成**的节点：尝试过且不在对应上的集合里。

    用于定位卡点与错误诊断——"学生在 S5 上反复失败"
    比"学生完成了 S1、S2"更能说明问题。
    """
    return set(_get(match, "attempted_nodes")) - set(_get(match, "matched_nodes"))


#: 口径名 → 取值函数。给需要**按名选用**的地方（如评测配置）用。
SEMANTICS = {
    "completed": completed_nodes,
    "touched": touched_nodes,
    "attempted_only": attempted_only_nodes,
}


def resolve(name: str):
    """按名取口径函数。名字不认识时**抛错**而不是给默认值——
    静默用错口径正是这个模块要消灭的问题。"""
    if name not in SEMANTICS:
        raise KeyError(
            f"未知的节点口径 {name!r}；可选：{sorted(SEMANTICS)}。"
            "请显式选择'碰过'还是'完成'。"
        )
    return SEMANTICS[name]
