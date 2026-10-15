"""重排的**成本—召回**对照（RAG 核查第 4 项；**只读归档，不调模型**）。

## 为什么要有这个脚本
```
核查报告：24 次真实 Tool 调用**中位 13.85 秒、最大 93.64 秒**；
          而预热后的**向量检索只要 73.7 毫秒** ⇒ 延迟几乎全在**重排那一次 LLM 调用**上。
⇒ 要谈"优化成本"，先得知道**降深度会丢什么** —— 否则就是拿召回换速度而不自知。
```
## 只用归档算（零付费）
```
· `baseline.jsonl`：300 条的真实排名 ⇒ 能算**不同深度下的严格命中**
· `reranked.json` ：24 条配对的真实耗时 ⇒ 能算**延迟分布**与**重排收益**
```
## ⚠️ 哪些是**实测**、哪些是**推断**（不许混着报）
```
实测：300 条的严格命中（按归档排名算）；24 条的延迟分布；24 条的重排收益。
推断：**不同深度下的延迟** —— 那要真调模型（付费），本脚本**不测、也不假装测了**。
```
"""
import argparse
import json
import statistics as stat
from pathlib import Path

#: 现状参数（与 `app.rag.rerank` 保持一致；改那边要同步这里，否则对照就漂了）
CURRENT_DEPTH = 10
DEPTHS = (4, 6, 8, 10)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="重排成本—召回对照（只读归档）")
    ap.add_argument("audit_dir", help="核查产物目录（含 baseline.jsonl / reranked.json）")
    ap.add_argument("--out", default="", help="可选：把结果另存为 JSON")
    args = ap.parse_args()
    d = Path(args.audit_dir)

    baseline = _load_jsonl(d / "baseline.jsonl")
    result: dict = {"audit_dir": str(d), "baseline_n": len(baseline), "measured": {}, "inferred": {}}

    # ---------- ① 不同深度下的**严格命中**（实测：按归档排名算） ----------
    exact = [r.get("ranks", {}).get("exact_rank") for r in baseline]
    page = [r.get("ranks", {}).get("page_rank") for r in baseline]
    n = len(baseline) or 1
    by_depth = {}
    for dep in DEPTHS:
        by_depth[str(dep)] = {
            "exact_hits": sum(1 for x in exact if isinstance(x, int) and x <= dep),
            "page_hits": sum(1 for x in page if isinstance(x, int) and x <= dep),
            "exact_rate": round(sum(1 for x in exact if isinstance(x, int) and x <= dep) / n, 4),
        }
    result["measured"]["strict_hits_by_depth"] = by_depth

    # ---------- ② **退步案例**：降到 depth 会丢掉哪些（逐条列，可复核） ----------
    lost: dict = {}
    for dep in (4, 6, 8):
        rows = [{"id": r.get("id"), "exact_rank": r.get("ranks", {}).get("exact_rank"),
                 "query": (r.get("query") or "")[:60]}
                for r in baseline
                if isinstance(r.get("ranks", {}).get("exact_rank"), int)
                and r["ranks"]["exact_rank"] > dep]
        lost[f"depth_{dep}_loses"] = {"count": len(rows), "cases": rows[:20],
                                      "note": "这些题的原句排在 dep 名之外 ⇒ "
                                              "重排**看不到它**，无法提升进前三"}
    result["measured"]["regressions"] = lost

    # ---------- ③ 延迟分布（实测）与重排收益（实测） ----------
    reranked = json.loads((d / "reranked.json").read_text(encoding="utf-8")) \
        if (d / "reranked.json").is_file() else []
    rows = reranked if isinstance(reranked, list) else (reranked.get("rows") or [])
    lat = [r.get("elapsed_sec") for r in rows if isinstance(r.get("elapsed_sec"), (int, float))]
    if lat:
        lat_sorted = sorted(lat)
        result["measured"]["latency_sec"] = {
            "n": len(lat), "median": round(stat.median(lat), 2), "max": round(max(lat), 2),
            "min": round(min(lat), 2),
            "p90": round(lat_sorted[int(len(lat_sorted) * 0.9) - 1], 2),
            "over_30s": sum(1 for x in lat if x > 30), "over_60s": sum(1 for x in lat if x > 60),
        }
        # 重排**收益**：有重排 vs 无重排的严格命中（同一批 24 条）
        base_hit = sum(1 for r in rows
                       if isinstance((r.get("baseline_ranks") or {}).get("exact_rank"), int)
                       and r["baseline_ranks"]["exact_rank"] <= 5)
        rer_hit = sum(1 for r in rows
                      if isinstance((r.get("ranks") or {}).get("exact_rank"), int)
                      and r["ranks"]["exact_rank"] <= 5)
        result["measured"]["rerank_gain_on_paired_24"] = {
            "no_rerank_hit5": base_hit, "with_rerank_hit5": rer_hit, "delta": rer_hit - base_hit}

    # ---------- ④ **没测的**：明确写出来，不许拿推断当实测 ----------
    result["inferred"] = {
        "note": "以下**没有实测**：不同深度下的延迟、专用重排器的表现。",
        "why": "深度改了要真调模型（付费）；本轮零付费请求。",
        "current_depth": CURRENT_DEPTH,
        "prompt_size_ratio": {str(dep): round(dep / CURRENT_DEPTH, 2) for dep in DEPTHS},
        "prompt_size_note": "候选数之比 —— **只是提示词规模的比例，不是延迟的比例**。",
    }
    result["claim"] = ("Offline comparison only; the latency column for other depths is "
                       "NOT measured. No paid request in this run.")

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
