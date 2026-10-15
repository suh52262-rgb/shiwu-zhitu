"""LLM 重排：让模型同时看到 query 与候选段落，重新判断相关性。

## 为什么这个有效，而前面三次无效（实测）

| 方法 | 净变化 | p | 引入新信息？ |
|---|---|---|---|
| 查询扩展（口语→术语） | +2.5% | 0.18 | 换 query，仍是同一套向量打分 |
| 长度惩罚加强 | +0.7% | — | ❌ 只缩放既有分数 |
| 标题结构降权（×0.5） | **+0.0%** | — | ❌ 同上 |
| **LLM rerank** | **+5.3%** | **0.0009** | ✅ **query-passage 交互** |

前三轮证明了「缩放既有分数」改不动 top-5 的**成员**——
分数动态范围 0.6~0.9，而重排要翻越 0.15 量级的分差，
一旦加大到能翻转，就会**同时作用于正确答案**（覆盖率高就抵消）。

**rerank 是在改成员**：它引入模型对「这段能不能回答这个问题」的判断，
那是以往排序里**没有的信息**。

## 实测（300 条 golden set，配对检验）
```
基线        228/300 = 76.0%
LLM rerank  244/300 = 81.3%    净 +16
翻转：变好 19 / 变差 3 / 不变 278
McNemar 精确检验 p = 0.0009  → 显著
```

## 代价（也是实测的）
一轮教学里检索可能被调用多次，所以代价要**按次**算：

| | 中位 | P90 |
|---|---|---|
| 重排关闭 | 90 ms | 120 ms |
| **重排开启** | **583 ms** | **919 ms** |

**每次检索多约 0.5 秒（P90 约 0.8 秒）+ 一次模型调用。**

对照：一次教学轮次本身含多次模型调用（`analyze_step` / `diagnosis` / `hint`
各是一次），单次几秒量级。**+0.5 秒约占一轮的 5~15%。**

**判断：相对 +5.3% 的检索命中率提升，这个代价可接受**——
但不是免费的：留 `RAG_RERANK=0` 这个开关，
而且在检索被调用多次的路径上**会累加**。
"""
from __future__ import annotations

import logging
import os
import re

from app.services.llm import get_llm

logger = logging.getLogger(__name__)

#: 送入重排的候选数。**不是越大越好**：候选越多提示词越长、越贵，
#: 而实测 top-10 已能覆盖大部分正确答案。
#: 送进重排的候选数。**可用环境变量覆盖**（2026-10-10 第 4 项）：
#: ```
#: RAG_RERANK_DEPTH=6  ⇒ 提示词更短、延迟更低，但**排在 7 名以后的原句重排看不到**
#: ```
#: ⚠️ 离线对照脚本 `scripts/compare_rerank_cost.py` 的结论（实测，300 条）：
#: ```
#: 深度 10 ⇒ 严格命中 220 ｜ 深度 8 ⇒ 215（**丢 5**）｜ 深度 6 ⇒ 205（**丢 15**）｜ 深度 4 ⇒ 190（丢 30）
#: ```
#: ⇒ 降深度是有代价的，**别只看速度**；而且"降到某深度的延迟"**尚未实测**（要付费）。
RERANK_DEPTH = int(os.getenv("RAG_RERANK_DEPTH", "10"))

#: 让模型挑几条。挑 3 条足够覆盖 top-5 的头部——
#: 挑更多会让"其余原序"部分变短，反而削弱重排的作用。
PICK = 3

#: 每个候选截断多少字。**要截断**：完整段落会让提示词爆炸，
#: 而判断"能不能回答"通常前 120 字就够。
SNIPPET = 120

#: 默认**开启**。与查询扩展不同——那个的收益在噪声内（p=0.18）不值得默认开；
#: 这个 p=0.0009，是三轮尝试里唯一显著的。
#: 关掉：`RAG_RERANK=0`。
ENABLED = os.getenv("RAG_RERANK", "1") not in ("0", "false", "False")

PROMPT = """学生在问一个大学物理问题。下面是教材里检索到的若干段落，**顺序不代表相关度**。

学生的问题：{q}

候选段落：
{items}

请挑出**最能回答这个问题**的 {pick} 段，按相关度从高到低。
只输出段落编号，用逗号分隔，例如：`3,7,1`
不要解释。如果都不相关，输出 `无`。"""


def _fmt(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        t = (r.get("text") or "").replace("\n", " ")[:SNIPPET]
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def parse_picks(text: str, n: int, *, pick: int = PICK) -> list[int]:
    """从模型输出里取出段落编号（1-based）。

    容错很重要：模型可能输出 `3,7,1`、`3、7、1`、`第3段`、
    甚至带解释。**取不到就返回空**——调用方退回原序，不猜测。
    """
    nums = [int(x) for x in re.findall(r"\d+", str(text or ""))]
    out: list[int] = []
    for x in nums:
        if 1 <= x <= n and x not in out:
            out.append(x)
        if len(out) >= pick:
            break
    return out


def rerank(query: str, results: list[dict], *, llm=None,
           depth: int = RERANK_DEPTH, status: dict | None = None) -> list[dict]:
    """用 LLM 重排候选。**任何失败都退回原序**——检索不能因重排而失败。

    返回长度与 `results` 相同（重排，不增删）：
    模型挑中的按它的顺序在前，其余保持原相对顺序在后。

    ## 🔴 2026-10-10 修正：「模型说无相关」与「重排失败」**必须分开**
    ```
    原实现把两者**合并成同一条路**（都 `return results`）：
        if not picks:
            # 模型说"都不相关"**或**输出无法解析——保持原序
            return results
    ⇒ 后果（核查报告实测）：酒店 query 的重排模型**明确输出「无」**，
      代码却回退原序 ⇒ **无关问题照样被塞进三条物理教材结果**。
    ⇒ 现在：
       · 明确「无」        ⇒ **返回空**（这才是"拒答"），状态 `none`
       · 超时/异常/解析不了 ⇒ 保留原序回退，状态 `failed` / `unparsed`
                              —— **不许伪装成"重排成功"或"无结果"**
    ```
    `status` 是**可选出参**（调用方传一个 dict 进来，这里填）——
    这样不改变返回值形状，既有调用方不受影响。
    """
    def _mark(kind: str, reason: str = "") -> None:
        if status is not None:
            status.clear()
            status.update({"rerank_status": kind, "rerank_reason": reason})

    if not ENABLED or not results or not (query or "").strip():
        _mark("skipped", "未启用 / 无候选 / 无查询")
        return results
    head = results[:depth]
    if len(head) < 2:
        _mark("skipped", "候选不足两条，无需重排")
        return results

    try:
        if llm is None:
            # **模块级导入**（项目惯例）：这样测试能 patch `rerank.get_llm`。
            # 在函数内 `from ... import` 会让 fake 打不到，
            # 测试里就会**打真 API**——实测踩过。
            llm = get_llm("rerank")
        resp = llm.chat("你是大学物理教材检索助手。",
                        PROMPT.format(q=query.strip(), items=_fmt(head), pick=PICK))
        picks = parse_picks(resp, len(head))
    except Exception as exc:  # noqa: BLE001 重排失败不该影响检索
        logger.info("重排失败，退回原序：%s: %s", type(exc).__name__, exc)
        _mark("failed", f"{type(exc).__name__}: {exc}")
        return results

    if not picks:
        if _is_explicit_none(resp):
            # ✅ **模型明确判断"都不相关"** ⇒ 拒答（返回空），而不是回退原序。
            logger.info("重排判断无相关材料，本轮不提供教材参考")
            _mark("none", "重排模型明确判断候选里没有相关材料")
            return []
        # ⚠️ 输出**没法解析** ⇒ 保守回退原序，但**如实记状态**——
        #    这与上面的"明确无相关"是**两回事**，不许混。
        _mark("unparsed", f"重排输出无法解析：{resp[:120]!r}")
        return results

    chosen = [head[k - 1] for k in picks]
    rest = [r for j, r in enumerate(head, 1) if j not in picks]
    _mark("ok", f"重排选中 {len(picks)} 条")
    return chosen + rest + results[depth:]


def _is_explicit_none(text: str) -> bool:
    """模型是不是**明确说了"无"**（而不是输出乱码/空）。

    ⚠️ 判据要**窄**：只有"整条回复基本上就是那个否定词"才算。
    否则一句"第 2 条不相关"会被误读成"全都无关" ⇒ **把好结果也拒了**。
    """
    t = (text or "").strip().strip("。.！!`*_ \n\t")
    return t in {"无", "無", "没有", "无相关", "无相关材料", "none", "None", "NONE"}
