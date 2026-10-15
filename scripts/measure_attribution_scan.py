"""在 72 份归档包上量归因扫描器：召回 vs 真值（10 条复核确认），误报 vs 其余 62 份。

⚠️ **改了 `unsupported_claims` 的判据就必须重跑这个脚本**，
并把结果同步到 `tests/test_review_v3_structures.py` 里那两组内联引文
（测试**不读 `artifacts/`** —— 那是运行产物、已被 gitignore）。

实测结果（2026-10-09，本脚本）：
```
召回 10/10   （复核确认的 10 条全部命中）
误报  1/62   （Q005「说明串联电流和分压他已经掌握」——
              提示词第 9 条明令禁止的写法，**很可能是评审漏标的真违规**）
```
判据就是被这份量测**改出来**的：窗口 14→40、引号内跳过、删掉有歧义的 `没有出现…痕迹`。

前置：`artifacts/t4_development_20261009/` 与 `..._reviews_20261009/` 存在（不在版本库里）。
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "backend")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from app.services.review_builder import report_text_fields, unsupported_claims  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "artifacts" / "t4_development_20261009"
REV = ROOT / "artifacts" / "t4_development_reviews_20261009"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))

if not DEV.is_dir():
    print(f"[X] 找不到 {DEV} —— 这个脚本要在跑过开发集的机器上运行")
    raise SystemExit(2)

CONFIRMED = {"Q007", "Q008", "Q023", "Q026", "Q034", "Q036", "Q053", "Q056", "Q066", "Q071"}

hits: dict[str, list[tuple[str, str]]] = {}
for p in sorted((DEV / "review_packets").glob("*.json")):
    pkt = load(p)
    report = pkt.get("review") or {}
    found = []
    for where, text in report_text_fields(report):
        for h in unsupported_claims(text):
            found.append((where, h))
    if found:
        hits[p.stem] = found

print(f"=== 扫描结果：72 份里 {len(hits)} 份命中 ===")
tp = sorted(set(hits) & CONFIRMED)
fp = sorted(set(hits) - CONFIRMED)
fn = sorted(CONFIRMED - set(hits))

print(f"\n【召回】复核确认的 10 条里，命中 {len(tp)}：{tp}")
if fn:
    print(f"  ❌ 未命中：{fn}")
print(f"\n【误报】其余 62 份里，命中 {len(fp)}：{fp}")

print("\n=== 逐条看命中内容 ===")
for qid in sorted(hits):
    tag = "真值命中" if qid in CONFIRMED else "**未标为违规**"
    print(f"\n  {qid}  [{tag}]")
    for where, h in hits[qid]:
        print(f"      {where}: {h!r}")

print(f"\n===== 召回 {len(tp)}/10   误报 {len(fp)}/62 =====")
