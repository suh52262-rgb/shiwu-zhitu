"""清点 RAG 样本：**只算已复核可用**的，八格缺口照实算（用户 2026-10-11 第 1 步）。

口径：
```
· 只计**已复核且可用**的条目 ✓
· **未应用的修订、暂缓（held）案例不计入** ✓
· 维度：主题(电磁/光学) × 求助类型(正常/误概念) × 证据完整度(单块/须组合)
  ⇒ 8 格 × 3 = 24 ✓
```
"""
from __future__ import annotations

import json
import pathlib

DS = pathlib.Path("evaluation/datasets")
CELLS = [(t, h, e) for t in ("电磁", "光学") for h in ("正常", "误概念") for e in ("单块", "组合")]


def norm(s: str) -> str:
    s = str(s)
    return ("电磁" if "电磁" in s else ("光学" if "光学" in s else "?")) if s else "?"


def kind(s: str) -> str:
    s = str(s)
    if "误概念" in s:
        return "误概念"
    if "正常" in s:
        return "正常"
    return "?"


def shape(s: str) -> str:
    s = str(s)
    return "组合" if ("组合" in s or "须组合" in s) else ("单块" if "单块" in s else "?")


# ⚠️ 这 3 条的 `dimensions` 文本格式与其它文件不同，脚本自动解析不出 ✗
#    ⇒ **显式归类并写明依据**（不猜）：
#    · dc_01：v3 复核后证据收为 `{b31}` 单块 ⇒ 电磁·正常·**单块**
#    · gx_85：证据 `{b11, b20}` ⇒ 光学·正常·**组合**
#    · gx_03：证据 `{b14, b15}` ⇒ 光学·正常·**组合**
MANUAL_CELLS = {
    "recheck8_dc_01": ("电磁", "正常", "单块"),
    "recheck8_gx_85": ("光学", "正常", "组合"),
    "recheck8_gx_03": ("光学", "正常", "组合"),
}


def cell_of(c: dict, src: str) -> tuple:
    if c["id"] in MANUAL_CELLS:
        return MANUAL_CELLS[c["id"]]
    d = c.get("dimensions") or (c.get("old") or {}).get("dimensions") or {}
    return (norm(str(d) + str(c.get("topic", ""))),
            kind(str(d) + str(c.get("help_type", ""))),
            shape(str(d) + str(c.get("evidence_completeness", ""))))


rows = []
# ---- 1) recheck8（8 条，v3，已复核）----
v3 = json.loads((DS / "annotation_recheck8_v3.json").read_text(encoding="utf-8"))
have = {c["id"] for c in v3["drafts"]}
all8 = list(v3["drafts"])
for f in sorted(DS.glob("annotation_recheck8_part*.json")):
    for c in json.loads(f.read_text(encoding="utf-8")).get("drafts", []):
        if c["id"] not in have:
            all8.append(c); have.add(c["id"])
for c in all8:
    d = c.get("dimensions") or {}
    rows.append({"id": c["id"], "src": "recheck8", "ok": True,
                 "cell": (norm(c.get("topic", "") + str(d)),
                          kind(str(d) + c.get("help_type", "")),
                          shape(str(d) + str(c.get("evidence_completeness", ""))))})

# ---- 2) recheck24 三批（v2/v2/batch2）----
for f, ver in (("annotation_recheck24_six_v2.json", "v2"),
               ("annotation_recheck24_batch3_v2.json", "v2")):
    p = DS / f
    if not p.is_file():
        continue
    for c in json.loads(p.read_text(encoding="utf-8"))["drafts"]:
        held = str(c.get("status", "")).startswith("HELD")
        d = c.get("dimensions") or (c.get("old") or {}).get("dimensions") or {}
        rows.append({"id": c["id"], "src": f, "ok": not held,
                     "cell": (norm(str(d) + str(c.get("topic", ""))),
                              kind(str(d) + str(c.get("help_type", ""))),
                              shape(str(d) + str(c.get("evidence_completeness", ""))))})

print("=== 全部候选（含暂缓）===")
for r in rows:
    print(f"  {'✅' if r['ok'] else '⏸ 暂缓'} {r['id']:34s} {r['cell']}")

print("\n=== 八格清点（**只计已复核可用**）===")
cnt = {c: 0 for c in CELLS}
for r in rows:
    if r["ok"] and r["cell"] in cnt:
        cnt[r["cell"]] += 1
missing_total = 0
for c in CELLS:
    need = max(0, 3 - cnt[c])
    missing_total += need
    print(f"  {c[0]}·{c[1]}·{c[2]:2s}  已有 {cnt[c]}  ⇒ 需补 {need}")
print(f"\n⇒ **缺口合计 {missing_total} 条**（目标 24）")
unknown = [r["id"] for r in rows if r["ok"] and r["cell"] not in cnt]
if unknown:
    print("⚠️ 格位无法判定（需人工归类）：", unknown)
