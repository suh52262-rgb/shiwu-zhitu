"""T4 批次前后对照：把**教学可用**与**复盘结构合规**分开报。

## 为什么必须分开
运行器把案例状态定成 `succeeded/failed`：**复盘被结构闸门拒掉（502）时整条案例记为 failed** ——
可那一轮**教学链路是成功的**（HTTP 200、有提示、有判步）。
⇒ 只报一个数会把两件事糊在一起：
```
教学可用率     教学调用成不成功
复盘结构合规率 复盘过不过结构闸门（引用 / 统计 / 归因）
案例状态率     运行器现在定义的那个数（≈ 两者都成功）
```
**下批的可用率一定要和"新判据回放旧输出"的基线一起看**
（`scripts/replay_new_judge_on_old_batch.py` 给出基线 24/72），
否则分不清是模型改好了还是尺子变严了。

用法：
    python scripts/compare_t4_batches.py 批次目录 [另一个批次目录 …]
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "backend")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from app.services.review_builder import classify_defects, structural_defects  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))


def survey(batch: Path) -> dict:
    out = {"dir": batch.name, "records": 0, "teaching_ok": 0, "review_ok": 0,
           "recorded_valid": 0,
           "status_ok": 0, "physical_requests": 0, "network_retries": 0,
           "structure_retries": 0, "tokens": 0, "new_judge_ok": 0,
           "by_category": {}, "kinds": Counter(), "elapsed": []}
    summary = batch / "summary.json"
    if summary.is_file():
        s = load(summary)
        out["dry_run"], out["smoke"] = s.get("dry_run"), s.get("smoke")
    records = sorted((batch / "runs").glob("*.json"))
    for path in records:
        r = load(path)
        out["records"] += 1
        teaching = r.get("teaching") or []
        t_ok = bool(teaching) and all(t.get("http_status") == 200 for t in teaching)
        out["teaching_ok"] += t_ok
        review = r.get("review") or {}
        r_ok = review.get("http_status") == 200
        out["review_ok"] += r_ok
        # ⚠️ **HTTP 200 ≠ 结构合规。** 上一批 72/72 都是 200，
        # 而响应体里的 `validation.valid` 只有 28 份为真 —— 因为当时统计类缺陷**不阻断**。
        # 三个数必须分开记，否则"合规 72/72"会被读成"报告都没问题"。
        payload = (review.get("response") or {}).get("review")
        if isinstance(payload, dict) and payload:
            out["recorded_valid"] += bool((payload.get("validation") or {}).get("valid"))
        out["status_ok"] += r.get("status") == "succeeded"
        out["physical_requests"] += r.get("physical_requests") or 0
        out["network_retries"] += r.get("network_retries") or 0
        out["structure_retries"] += r.get("structure_retries") or 0
        out["tokens"] += r.get("reported_tokens") or 0
        if r.get("elapsed_sec"):
            out["elapsed"].append(r["elapsed_sec"])
        cat = r.get("category") or "?"
        bucket = out["by_category"].setdefault(cat, {"n": 0, "teaching": 0, "review": 0})
        bucket["n"] += 1
        bucket["teaching"] += t_ok
        bucket["review"] += r_ok
        # 用**当前**判据重算这份复盘（与批次记录的状态无关）
        issues = structural_defects(payload) if isinstance(payload, dict) and payload else None
        if issues is not None:
            out["new_judge_ok"] += not issues
            for d in issues:
                out["kinds"][d["kind"]] += 1

    # 复盘被拒时，把闸门给的理由收集起来
    out["review_reject_reasons"] = Counter()
    for path in records:
        r = load(path)
        review = r.get("review") or {}
        if review.get("http_status") not in (200, None):
            detail = ((review.get("response") or {}).get("detail") or "")[:60]
            out["review_reject_reasons"][detail] += 1
    return out


def show(s: dict) -> None:
    n = s["records"]
    if not n:
        print(f"  {s['dir']}: 没有记录")
        return
    print(f"\n=== {s['dir']}  (dry_run={s.get('dry_run')} smoke={s.get('smoke')}) ===")
    print(f"  记录数              {n}")
    print(f"  教学可用            {s['teaching_ok']}/{n}")
    print(f"  复盘 HTTP 200       {s['review_ok']}/{n}   ← 只是「接口没报错」，**不等于合规**")
    print(f"  复盘 valid（当时记的）{s['recorded_valid']}/{n}   ← 当时**不阻断**，所以它低于上面那行")
    print(f"  复盘合规（**新判据重算**） {s['new_judge_ok']}/{n}   ← 与下批比较要用这一个")
    print(f"  案例状态 succeeded  {s['status_ok']}/{n}   ← 运行器定义的那个数（≈两者都成功）")
    print(f"  物理请求 {s['physical_requests']}  网络重试 {s['network_retries']}  "
          f"结构重试 {s['structure_retries']}  tokens {s['tokens']}")
    if s["elapsed"]:
        e = sorted(s["elapsed"])
        print(f"  单案例耗时 中位 {e[len(e)//2]:.1f}s  最小 {e[0]:.1f}s  最大 {e[-1]:.1f}s")
    print("  按类别（教学 / 复盘 / 新判据）：")
    for cat, b in sorted(s["by_category"].items()):
        print(f"    {cat:24s} 教学 {b['teaching']}/{b['n']}  复盘 {b['review']}/{b['n']}")
    if s["kinds"]:
        print("  新判据命中的缺陷：")
        for kind, c in s["kinds"].most_common():
            print(f"    {kind:32s} {c}")
    if s["review_reject_reasons"]:
        print("  复盘被拒的理由（闸门给的原话）：")
        for reason, c in s["review_reject_reasons"].most_common(6):
            print(f"    ×{c:3d}  {reason}")


if __name__ == "__main__":
    targets = sys.argv[1:] or [str(ROOT / "artifacts" / "t4_development_20261009")]
    print("T4 批次对照 —— 教学可用 / 复盘合规 **分开报**")
    for t in targets:
        p = Path(t)
        if not p.is_absolute():
            p = ROOT / t
        if not p.is_dir():
            print(f"  [跳过] {t} 不存在")
            continue
        show(survey(p))
