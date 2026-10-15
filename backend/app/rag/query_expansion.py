"""查询扩展：跨越"学生口语"与"教材术语"之间的词汇鸿沟。

## 为什么需要（有实测依据）

`docs/检索失败归因分析.md` 的结论：golden set 60 条里 10 条真未命中，
**根因不是索引、也不是向量质量**，而是**用词不匹配**：

| 学生怎么说 | 教材怎么写 |
|---|---|
| 重量和高度**乘起来** | 重量 w 和高度 h 的**乘积**、（3.1）式 |
| **二分之一 mv 平方** | 相对论的**质点动能公式**、**泰勒级数展开** |
| 只拿**六分之一**的分子 | 分子**速度方向分布** |
| 刚体位置需要**六个量** | 三个**平动** + 三个**转动**自由度 |

决定性证据：**用 gold 原文去查，两条都排第 1 名**——段落就在索引里，
检索得动，只是"学生的话"和"书上的话"对不上。

## 两路一起召回，而不是赌一路

改写可能跑偏（把学生的问题理解成别的话题），而原文有时反而准。
所以**两份结果融合**，而不是只用改写后的。

融合用 **RRF（Reciprocal Rank Fusion）**：它只看排名、不看分数，
因此不需要在两路分数之间做归一化——这正是它的用途。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: RRF 的平滑常数（文献常用 60）。越小则头部排名的影响越大。
RRF_K = 60

#: 生成几个改写变体。**多了没用**：每多一路就多一次向量检索，
#: 而边际收益衰减很快。
MAX_VARIANTS = 2

#: 总开关。**默认关闭**（`RAG_QUERY_EXPANSION=1` 打开）。
#:
#: ## 为什么默认关
#: 三条实测理由，缺一条都不足以否决它：
#:
#: 1. **收益在噪声内**：golden set 60 条上，严格命中 75% → 77%（+1 条），
#:    同页 83% → 87%（+2 条）。**n=60 时这个差异不显著。**
#: 2. **代价是每次检索一次 LLM 调用**（延迟 + 费用），
#:    而检索在一轮教学里可能被调用多次。
#: 3. **它破坏了一条既有契约**：`retrieve_knowledge` 原本不调模型，
#:    项目里有测试依赖"绕过 LLM 层时不应产生 token 用量"
#:    （`test_trace_usage_is_empty_when_llm_layer_is_bypassed`）。
#:    是那条测试抓到了这个改动，**它是对的**。
#:
#: ## 什么时候该打开
#: 当 golden set 扩到几百条、能把 +2 与噪声分开时；
#: 或者当"学生口语查不到"成为主要投诉时。
#: **改默认值前请先用 `--suite retrieval_golden` 量一次。**
ENABLED = os.getenv("RAG_QUERY_EXPANSION", "0") not in ("0", "false", "False")

PROMPT = """学生在问一个大学物理问题，用的是自己的口语。请把它改写成**教材里会用的说法**，便于检索教材原文。

学生的问题：{query}

要求：
1. 输出 {n} 条改写，每行一条，**不要编号、不要解释**
2. 用教材的**术语**，例如：
   - "重量和高度乘起来" → "重力做功 重力和高度的乘积"
   - "二分之一mv平方" → "动能公式 低速极限 泰勒级数展开"
   - "刚体要几个量确定位置" → "刚体 平动自由度 转动自由度"
3. 保留原问题的物理含义，**不要改变它在问什么**
4. 每条不超过 30 字"""


def expand_query(query: str, *, llm=None, n: int = MAX_VARIANTS) -> list[str]:
    """把学生口语改写成教材术语，返回若干变体。

    **任何失败都返回空列表**（调用方退回原始查询）——
    检索路径上的东西不能让整条链挂掉，这是项目一贯的降级原则。
    """
    if not ENABLED or not (query or "").strip():
        return []
    try:
        if llm is None:
            from app.services.llm import get_llm

            llm = get_llm("tutor")
        text = llm.chat("你是大学物理教材检索助手。", PROMPT.format(query=query.strip(), n=n))
    except Exception as exc:  # noqa: BLE001 扩展失败不该影响检索本身
        logger.info("查询扩展失败，退回原始查询：%s: %s", type(exc).__name__, exc)
        return []

    out: list[str] = []
    for line in str(text or "").splitlines():
        s = line.strip().lstrip("0123456789.、) ").strip().strip('"“”')
        # 太短的多半是"好的""改写如下"这类废话；太长的不像检索词
        if 4 <= len(s) <= 60 and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out


def merge_rrf(result_lists: list[list[dict]], *, k: int = RRF_K) -> list[dict]:
    """RRF 融合多路检索结果（保留，但**默认不用**，见 `merge_interleave`）。

    每一路的贡献是 `1/(k + rank)`。**只看排名不看分数**——不需要在
    "原查询分数"与"改写查询分数"之间做归一化。

    ⚠️ **为什么它不是默认**：实测在 golden set（60 条）上，
    RRF(k=60) 与交错法在严格命中上差 5 个百分点（75% vs 80%），
    但 **n=60 时 3 个用例的差异在噪声内**，不足以据此选择。
    真正让交错法胜出的是**稳健性**（见其 docstring）。
    """
    lists = [lst for lst in result_lists if lst]
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])

    fused: dict[tuple, dict] = {}
    for lst in lists:
        for rank, item in enumerate(lst, 1):
            key = _ident(item)
            contribution = 1.0 / (k + rank)
            slot = fused.get(key)
            if slot is None:
                fused[key] = {**item, "rrf": contribution,
                              "score": float(item.get("score") or 0.0)}
            else:
                slot["rrf"] += contribution
                slot["score"] = max(slot["score"], float(item.get("score") or 0.0))
    return sorted(fused.values(), key=lambda x: -x["rrf"])


def _ident(item: dict) -> tuple:
    """块身份。与检索去重同一口径 `(来源, 页, 文本)`。"""
    return (item.get("source"), item.get("page"), (item.get("text") or "").strip())


def merge_interleave(result_lists: list[list[dict]]) -> list[dict]:
    """**默认融合法**：各路按名次交错 —— base[0], alt[0], base[1], alt[1], …

    ## 为什么选它（判据是稳健性，不是那 5 个百分点）

    实测（golden set 60 条）：

    | 策略 | 严格 | 同页 | MRR |
    |---|---|---|---|
    | 只用原文 | 75% | 83% | 59% |
    | **只用改写** | **55%** | 72% | 41% |
    | RRF k=60 | 75% | 92% | 59% |
    | **交错** | **80%** | 90% | **60%** |

    交错法在严格命中与 MRR 上最好，但 **n=60 时 3 个用例的差异在噪声内**，
    不足以据此宣称更优。

    **真正的理由是它降级得更体面**：原查询的结果固定占据第 1、3、5…位，
    所以**即使改写完全跑偏，top-5 里仍有三条来自原查询**。
    RRF 没这个保证——一个坏改写能把原查询的好结果压下去。

    > 「只用改写」实测 55%（比基线 75% 低 20 个百分点）说明改写**确实会跑偏**，
    > 所以这个保证不是理论担忧，是有实测支撑的。
    """
    lists = [lst for lst in result_lists if lst]
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    out: list[dict] = []
    seen: set[tuple] = set()
    depth = max(len(lst) for lst in lists)
    for rank in range(depth):
        for lst in lists:
            if rank >= len(lst):
                continue
            key = _ident(lst[rank])
            if key in seen:
                continue
            seen.add(key)
            out.append(lst[rank])
    return out
