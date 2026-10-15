"""真 rerank：LLM 对候选做 query-passage 交互打分。

## 与前三次的**本质区别**

| 方法 | 引入新信息？ |
|---|---|
| 查询扩展 | 换了 query，但仍是同一套向量打分 |
| 长度惩罚 | ❌ 只在既有分数上做缩放 |
| 标题降权 | ❌ 同上 |
| **LLM rerank** | ✅ **让模型同时看到 query 与 passage，重新判断相关性** |

前三轮证明了"缩放既有分数"无效——因为 top-5 的成员由向量相似度决定，
缩放改不动成员。**rerank 是在改成员。**

## 设计

每条查询：检索 top-10 → 把 10 条候选（截断到 120 字）列给模型 →
让它挑出最相关的 3 条 → 按"模型选中的顺序 + 其余原序"重排 → 取 top-5 评 Hit@5。

**一次调用一条查询**（不是一条候选一次），成本可控。
"""
from pathlib import Path
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(Path(__file__).resolve().parents[1] / '.env'))

from app.rag.retriever import retrieve  # noqa: E402

DEPTH = 10
TOPK = 5
cases = json.load(open(str(Path(__file__).resolve().parents[1] / 'evaluation/datasets/retrieval_golden_cases.json'),
                       encoding="utf-8"))["cases"]

PROMPT = """学生在问一个大学物理问题。下面是教材里检索到的若干段落，**顺序不代表相关度**。

学生的问题：{q}

候选段落：
{items}

请挑出**最能回答这个问题**的 3 段，按相关度从高到低。
只输出段落编号，用逗号分隔，例如：`3,7,1`
不要解释。如果都不相关，输出 `无`。"""


def _fmt(res):
    lines = []
    for i, r in enumerate(res, 1):
        t = (r.get("text") or "").replace("\n", " ")[:120]
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def _parse(text, n):
    nums = [int(x) for x in re.findall(r"\d+", str(text or ""))]
    out = []
    for x in nums:
        if 1 <= x <= n and x not in out:
            out.append(x)
        if len(out) >= 3:
            break
    return out


def main() -> int:
    from app.services.llm import get_llm

    llm = get_llm("tutor")

    print(f"  检索 {len(cases)} 条 top-{DEPTH} 并缓存……")
    cache = []
    for i, c in enumerate(cases, 1):
        res = retrieve(c["query"], top_k=DEPTH)
        cache.append({"case": c, "res": res})
        if i % 100 == 0:
            print(f"    … {i}/{len(cases)}")

    def hit_rank(res, gold):
        for i, r in enumerate(res[:TOPK], 1):
            if r.get("source") == gold["doc_id"] and r.get("page") == gold["page"]:
                return i
        return None

    base_hits = sum(1 for e in cache if hit_rank(e["res"], e["case"]["gold"]))
    print(f"\n  基线（现行重排）Hit@{TOPK}：{base_hits}/{len(cache)} = "
          f"{base_hits/len(cache):.1%}")

    print(f"\n  LLM rerank（{len(cache)} 次调用）……")
    ok = 0
    flips = {"better": 0, "worse": 0, "same": 0}
    for i, e in enumerate(cache, 1):
        res, gold = e["res"], e["case"]["gold"]
        b = hit_rank(res, gold)
        try:
            resp = llm.chat("你是大学物理教材检索助手。",
                            PROMPT.format(q=e["case"]["query"], items=_fmt(res)))
            order = _parse(resp, len(res))
        except Exception:  # noqa: BLE001 单条失败退回原序
            order = []
        if order:
            chosen = [res[k - 1] for k in order]
            rest = [r for j, r in enumerate(res, 1) if j not in order]
            reranked = chosen + rest
        else:
            reranked = res
        a = hit_rank(reranked, gold)
        ok += a is not None
        if (b is None) and (a is not None):
            flips["better"] += 1
        elif (b is not None) and (a is None):
            flips["worse"] += 1
        else:
            flips["same"] += 1
        if i % 50 == 0:
            print(f"    … {i}/{len(cache)}  当前 {ok}/{i}")

    rate = ok / len(cache)
    print(f"\n{'=' * 66}")
    print(f"  基线        {base_hits}/{len(cache)} = {base_hits/len(cache):.1%}")
    print(f"  LLM rerank  {ok}/{len(cache)} = {rate:.1%}")
    print(f"  净变化      {ok - base_hits:+d} 条")
    print(f"  翻转：变好 {flips['better']} / 变差 {flips['worse']} / "
          f"不变 {flips['same']}")
    import math
    b, c_ = flips["worse"], flips["better"]
    n = b + c_
    if n:
        k = min(b, c_)
        p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
        print(f"  McNemar 精确检验 p = {p:.4f}"
              f"  → {'**显著**' if p < 0.05 else '不显著'}")
    else:
        print("  ❌ 零翻转——实验很可能没生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
