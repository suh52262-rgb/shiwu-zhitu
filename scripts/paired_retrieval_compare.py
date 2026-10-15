"""配对比较：查询扩展 开 vs 关（300 条，McNemar 检验）。

## 为什么要配对

独立比例比较要 ~1900 条才能分辨 2%（上面算过）。但**同一批问题跑两种条件**
是配对数据 —— McNemar 只看"翻转了的那几条"：

    ┌─────────────┬──────────┬──────────┐
    │             │ 扩展命中  │ 扩展未命中│
    ├─────────────┼──────────┼──────────┤
    │ 基线命中     │    a     │  b ← 变差 │
    │ 基线未命中   │  c ← 变好 │    d     │
    └─────────────┴──────────┴──────────┘

    H0 下 b ~ Binomial(b+c, 0.5)。**a 与 d 不参与检验**——
    它们没有提供"两种条件有差异"的信息。

功效高是因为它丢掉了大量无关信息（那些两种条件都对的题）。

用法：
    python scripts/paired_retrieval_compare.py            # 全量 300
    python scripts/paired_retrieval_compare.py --limit 60 # 只跑前 60
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.rag import query_expansion as qe  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402

GOLDEN = ROOT / "evaluation" / "datasets" / "retrieval_golden_cases.json"
OUT = ROOT / "evaluation" / "reports" / "paired-query-expansion.json"
TOP_K = 5


def _ident(item: dict) -> tuple:
    return (item.get("source"), item.get("page"), (item.get("text") or "").strip())


def _rank_topk(results: list[dict], gold: dict) -> int | None:
    """gold 所在页在 top_k 里的名次（宽松口径：同页即可）。"""
    for i, r in enumerate(results[:TOP_K], 1):
        if r.get("source") == gold["doc_id"] and r.get("page") == gold["page"]:
            return i
    return None


def _binom_two_sided(b: int, c: int) -> float:
    """McNemar 精确检验的 p 值（不依赖 scipy）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) for X ~ Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description="配对比较查询扩展")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"][:args.limit]

    # **强制打开扩展。** 上一轮把默认值改成了关闭（收益在噪声内、
    # 且每次检索多一次模型调用），而 `expand_query` 尊重那个开关。
    # 不显式打开的话，"实验组"跑的还是基线 —— 实测因此得到 b=c=0 的假结果。
    if not qe.ENABLED:
        qe.ENABLED = True
        print("  （已强制打开 RAG 查询扩展——本脚本就是来量它的）")
    print(f"  配对比较：{len(cases)} 条（查询扩展 开 vs 关）")
    print("  ⚠ 每条要一次改写调用，全量约十几分钟\n")

    rows = []
    t0 = time.perf_counter()
    for i, c in enumerate(cases, 1):
        gold = c["gold"]
        # ---- 条件 A：不开扩展 ----
        base = retrieve(c["query"], top_k=TOP_K)
        r_base = _rank_topk(base, gold)

        # ---- 条件 B：开扩展（原文 + 改写，交错融合）----
        variants = qe.expand_query(c["query"])
        if variants:
            depth = TOP_K * 2
            lists = [retrieve(c["query"], top_k=depth)]
            for v in variants:
                try:
                    lists.append(retrieve(v, top_k=depth))
                except Exception:  # noqa: BLE001
                    pass
            merged = qe.merge_interleave(lists)[:TOP_K]
        else:
            merged = base
        r_exp = _rank_topk(merged, gold)

        rows.append({"id": c["id"], "query": c["query"],
                     "base_rank": r_base, "exp_rank": r_exp,
                     "variants": variants})
        if i % 30 == 0:
            print(f"    … {i}/{len(cases)}  ({time.perf_counter()-t0:.0f}s)")

    # ---- McNemar ----
    a = sum(1 for r in rows if r["base_rank"] and r["exp_rank"])
    b = sum(1 for r in rows if r["base_rank"] and not r["exp_rank"])
    c_ = sum(1 for r in rows if not r["base_rank"] and r["exp_rank"])
    d = sum(1 for r in rows if not r["base_rank"] and not r["exp_rank"])
    n = len(rows)
    p = _binom_two_sided(b, c_)

    print(f"\n{'=' * 70}")
    print(f"  配对结果（宽松口径：gold 所在页进 top{TOP_K}）")
    print(f"{'=' * 70}")
    print(f"  {'':<16}{'扩展命中':>10}{'扩展未命中':>12}")
    print(f"  {'基线命中':<16}{a:>10}{b:>12}")
    print(f"  {'基线未命中':<16}{c_:>10}{d:>12}")
    print(f"\n  基线命中率      {a+b:>4}/{n} = {(a+b)/n:.1%}")
    print(f"  扩展命中率      {a+c_:>4}/{n} = {(a+c_)/n:.1%}")
    print(f"  净变化          {(a+c_)-(a+b):>+4} 条")
    if b + c_ == 0:
        print("\n  ❌ **零翻转**：两种条件结果完全相同。")
        print("     这几乎一定是实验没生效（如扩展没打开），")
        print("     **不是**'扩展无效'的证据。请先确认实验组真的跑了。")
        return 2
    print(f"\n  **McNemar 精确检验**（只看翻转的 {b+c_} 条）")
    print(f"    变好 {c_} 条 / 变差 {b} 条      p = {p:.4f}")
    if p < 0.05:
        verdict = ("扩展**显著更好**" if c_ > b else "扩展**显著更差**")
    else:
        verdict = "**无显著差异**——这个改动不该被当成'修好了'"
    print(f"    结论：{verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n": n, "top_k": TOP_K,
        "table": {"both_hit": a, "base_only": b, "exp_only": c_, "both_miss": d},
        "base_rate": (a + b) / n, "exp_rate": (a + c_) / n,
        "mcnemar_p": p, "verdict": verdict,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细已写入 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
