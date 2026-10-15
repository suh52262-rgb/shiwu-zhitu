"""路径扩展：把 `alternative_methods` 建成图上的支路节点。

## 为什么单独做一步（而不是让 P15 自己改）
P15 是**只读**校验器。**一个既报警又自己修的组件，报的问题还有谁复核？**
它连 `needs_review` 都不肯替人清（docstring 原话：「机器说没问题」与
「人确认过」是两件事），自然不该去改图。

所以：

    P14 生成 → P15（只读）报警 → **本模块（独立一步）** → 新草稿（仍是 needs_review）

**P15 保持只读**，改图的是一个独立组件，**产物仍是草稿** —— 机器依旧不自我批准。

## 为什么该做（实测依据）
`prompts/graph_generator/v2.md` 里已经写明了：

> 只把另一种解法写进 `alternative_methods` 是不够的。Step Matcher 靠图上的
> 节点判断学生走的是哪条路；替代解法不在图里，学生用另一种正确解法时系统就
> 认不出来，**会把他误判成"没按标准路径走"**。
> `alternative_methods` 仍然要填，但它是**给人看的补充说明**…**不能替代支路节点**。

**但实测 P14 仍不照做**：跑 `demo_03`（成员B 明确写了两条方法）两次，
产出都是**单路径**，替代解法只落在 `alternative_methods` 文字里。

⇒ **问题不是"提示词没想到"，是"模型在那条大提示词里不照做"。**
所以单独一步，只让它干这一件事。

## 安全设计
1. **硬不变量**：已有节点 id、边、路径**必须全部保留**（`_check_preserved`）。
   违反了就整次拒绝 —— 防止它把已经对的部分改坏。
2. **不循环**：只试一次。失败就退回原图，并如实报告。
3. **仍是草稿**：`needs_review=True`，并记 `expanded_by="auto"` 与原始
   `alternative_methods`，好让人看出"这段支路是机器加的"。
4. **拒绝也要留痕**：模型判为物理上不成立的解法进 `rejected_methods`。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from app.agent.prompt_registry import get_prompt
from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.llm import get_llm
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

ROLE = "graph_expander"
PROMPT_VERSION = "v1"


class RejectedMethod(BaseModel):
    """模型判为物理上不成立、因此没建支路的解法。"""

    method: str = ""
    reason: str = ""


class ExpansionOutput(BaseModel):
    """扩展步骤的输出。

    **字段全部给默认值**，因为模型可能漏填某一项——
    漏填该由下面的硬不变量去判，而不是让 pydantic 直接抛异常。
    一个字段缺失就整次失败，代价太大。
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)
    valid_paths: list[list[str]] = Field(default_factory=list)
    rejected_methods: list[RejectedMethod] = Field(default_factory=list)
    notes: str = ""


class ExpansionError(RuntimeError):
    """扩展失败（模型不可用、输出不可解析、违反硬不变量）。"""


def _brief(graph: ProblemGraph) -> dict:
    """送给模型的图（不含内部元数据，去掉噪声）。"""
    return {
        "nodes": [{"id": n.id, "name": n.name, "type": n.type,
                   "description": n.description,
                   "knowledge_points": list(n.knowledge_points),
                   "common_errors": list(n.common_errors)}
                  for n in graph.nodes],
        "edges": [list(e) for e in graph.edges],
        "valid_paths": [list(p) for p in graph.valid_paths],
        "alternative_methods": list(
            (graph.metadata or {}).get("alternative_methods") or []),
    }


def _check_preserved(old: ProblemGraph, new_nodes: list[dict],
                     new_edges: list[list[str]]) -> Optional[str]:
    """**硬不变量**：已有节点 id、边、路径必须全部保留。

    返回 `None` 表示通过，否则返回拒绝原因。

    **这是本模块最重要的防线。** 模型的任务是"只增加支路"，
    若它顺手改了已有节点名或删了一条边，那是**把已经对的部分改坏**——
    比不做更糟。
    """
    old_ids = {n.id for n in old.nodes}
    new_ids = {n.get("id") for n in new_nodes}
    missing = old_ids - new_ids
    if missing:
        return f"已有节点被删除：{sorted(missing)}"

    old_edges = {tuple(e) for e in old.edges}
    new_edgeset = {tuple(e) for e in new_edges}
    lost = old_edges - new_edgeset
    if lost:
        return f"已有边被删除：{sorted(lost)}"
    return None


def _check_paths_preserved(old: ProblemGraph,
                           new_paths: list[list[str]]) -> Optional[str]:
    old_norm = {tuple(p) for p in old.valid_paths}
    new_norm = {tuple(p) for p in new_paths}
    lost = old_norm - new_norm
    if lost:
        return f"已有合法路径被删除或改动：{sorted(lost)}"
    return None


def _check_paths_valid(new_nodes: list[dict], new_edges: list[list[str]],
                       new_paths: list[list[str]]) -> Optional[str]:
    """每条路径：节点都存在、相邻步都有边、同起同终。"""
    ids = {n.get("id") for n in new_nodes}
    eset = {tuple(e) for e in new_edges}
    if not new_paths:
        return "没有产出任何路径"
    starts, ends = set(), set()
    for p in new_paths:
        if not p:
            return "有一条空路径"
        miss = [x for x in p if x not in ids]
        if miss:
            return f"路径引用了不存在的节点：{miss}"
        for a, b in zip(p, p[1:]):
            if (a, b) not in eset:
                return f"路径中有不存在的步：{a}→{b}"
        starts.add(p[0])
        ends.add(p[-1])
    if len(starts) > 1:
        return f"路径的入口不一致：{sorted(starts)}"
    if len(ends) > 1:
        return f"路径的终点不一致：{sorted(ends)}"
    return None


def expand_alternative_paths(
    graph: ProblemGraph,
    *,
    problem_statement: str = "",
    llm: Optional[LLMProvider] = None,
) -> tuple[ProblemGraph, dict]:
    """把 `alternative_methods` 建成支路节点。

    返回 `(新图, 报告)`。**报告里必须能看出成功还是失败、以及为什么。**

    失败时**原样返回传入的图**（不抛异常）——调用方不该因为扩展失败而拿不到图。
    `ExpansionError` 只在参数明显不对（比如没有替代解法）时抛。
    """
    alts = list((graph.metadata or {}).get("alternative_methods") or [])
    if not alts:
        raise ExpansionError("这张图没有 `alternative_methods`，没有可扩展的东西")

    report: dict = {"ok": False, "added_nodes": [], "added_paths": 0,
                    "rejected_methods": [], "reason": ""}
    llm = llm or get_llm(ROLE)

    payload = _brief(graph)
    if problem_statement:
        payload["problem_statement"] = problem_statement[:1200]

    try:
        data = llm.chat_structured(get_prompt(ROLE, PROMPT_VERSION),
                                   json.dumps(payload, ensure_ascii=False),
                                   ExpansionOutput)
        new_nodes = [n.model_dump() for n in data.nodes]
        new_edges = [list(e) for e in data.edges]
        new_paths = [list(p) for p in data.valid_paths]
    except Exception as exc:  # noqa: BLE001
        report["reason"] = f"模型调用或解析失败：{type(exc).__name__}: {exc}"
        logger.warning("路径扩展失败：%s", exc)
        return graph, report

    report["rejected_methods"] = [r.model_dump() for r in data.rejected_methods]
    report["notes"] = str(data.notes or "")

    # ---- 硬不变量 ----
    for check, why in (
        (_check_preserved(graph, new_nodes, new_edges), "已有节点/边未保留"),
        (_check_paths_preserved(graph, new_paths), "已有路径未保留"),
        (_check_paths_valid(new_nodes, new_edges, new_paths), "新路径不合法"),
    ):
        if check:
            report["reason"] = f"{why}：{check}"
            logger.warning("路径扩展被拒绝：%s", report["reason"])
            return graph, report

    added = sorted({n["id"] for n in new_nodes}
                   - {n.id for n in graph.nodes})
    if not added:
        report["reason"] = "没有新增任何节点——替代解法没有被建成支路"
        return graph, report

    # ---- 组装新图 ----
    out = graph.model_copy(deep=True)
    out.nodes = [GraphNode(**n) for n in new_nodes]
    out.edges = new_edges
    out.valid_paths = new_paths
    out.metadata = dict(out.metadata or {})
    # **仍是草稿**：机器加的支路要能被认出来
    out.metadata["needs_review"] = True
    out.metadata["expanded_by"] = "auto"
    out.metadata["expanded_from_alternative_methods"] = alts
    out.metadata["branch_kind"] = (
        "替代方法（由 `expand_alternative_paths` 机器补建，"
        "**未经人复核**）")
    out.metadata["note"] = (
        f"支路节点 {added} 是机器根据 `alternative_methods` 补的，"
        f"请人工确认：① 该解法物理上成立吗 ② 它真的与主路径等价吗")

    report.update(ok=True, added_nodes=added,
                  added_paths=len(new_paths) - len(graph.valid_paths),
                  reason="")
    return out, report
