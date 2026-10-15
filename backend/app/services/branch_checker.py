"""假分叉检查：**"两个都要做"被建成了"二选一"**。

## 为什么需要
P14 会把题面要求的**并列输出**建成两条可选路径。实测（`tb_pilot_xiache16`）：
```
题面：求此二粒子间的静电力**和**万有引力各为多大？   ← 两个都要算
建图：S2 → {S3 库仑定律 | S4 万有引力定律} → S5      ← 建成了二选一
```
**结构完全合法，P15 抓不到**（和 `demo_05` 同一个错误类）。

## 规则
对每个**出度 > 1** 的分叉点，看它的各后继分别命中题面要求的哪个量：
· 两个后继命中**不同的量** → **两者都是题目要的 → 假分叉**
· 各后继命中**同一个量** → 是同一件事的不同做法 → **真分叉**

前提：要知道"题面要求哪些量"。**由 VL 抽取那一步顺便给出**
（边际成本几乎为 0）—— 见 `prompts/example_extractor/v1.md`。
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

#: 命中判定时忽略的虚词
_STOP = ("的", "为", "是", "各", "多", "大", "小", "值", "大小", "多少")


def _norm(s: str) -> str:
    return re.sub(r"[\s，。、？：；（）()\[\]]", "", s or "")


def _core(q: str) -> str:
    """取一个量的"核心词"—— 去掉'为/大小/各'这类尾巴，便于在描述里命中。"""
    t = _norm(q)
    for w in _STOP:
        t = t.replace(w, "")
    return t or _norm(q)


def _text_of(node) -> str:
    return _norm(f"{getattr(node, 'name', '')}{getattr(node, 'description', '')}")


def find_false_branches(graph, quantities_asked: Iterable[str]) -> list[dict]:
    """找出疑似"假分叉"（把并列输出建成了二选一）。

    返回 `[{"node", "successors", "hit": {后继: 命中的量}, "why"}]`。
    """
    asked = [(q, _core(q)) for q in (quantities_asked or []) if _core(q)]
    if len(asked) < 2:
        return []          # 题面只要一个量 → 不存在"并列输出"的问题

    by_id = {n.id: n for n in graph.nodes}
    out: dict[str, list[str]] = {}
    for e in graph.edges:
        # ⚠️ **必须容错。** 题库是数据，畸形边不能让检查崩 ——
        # 我在 `graph_validator` 里刚修过这个，转头在新文件里又写了一遍
        # （两边都是 `for a, b in graph.edges`），是测试把我拦住的。
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.setdefault(e[0], []).append(e[1])

    found = []
    for nid, succ in out.items():
        if len(succ) < 2:
            continue
        hit: dict[str, list[str]] = {}
        for s in succ:
            node = by_id.get(s)
            if node is None:
                continue
            t = _text_of(node)
            hit[s] = [q for q, c in asked if c and c in t]
        # 只看后继**自身**（再往下的子树常会汇合，会误判）
        # 两个后继各自命中**不同的**量 → 假分叉
        distinct = set()
        for s, qs in hit.items():
            if qs:
                distinct.add(qs[0])
        if len(distinct) >= 2:
            found.append({
                "node": nid,
                "successors": list(succ),
                "hit": hit,
                "why": (f"分叉点 {nid} 的各后继分别命中题面要求的**不同**量 "
                        f"{sorted(distinct)} —— 这两件事题目**都要**，"
                        f"不该建成二选一"),
            })
    return found


def check_and_explain(graph, quantities_asked: Iterable[str]) -> str:
    """给 trace / 校验报告用的一句话。"""
    bad = find_false_branches(graph, quantities_asked)
    if not bad:
        return ""
    return "；".join(b["why"] for b in bad)
