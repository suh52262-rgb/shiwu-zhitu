"""清掉 `phy_003` 节点级混进来的"数学/过程技能"标签。

## 为什么
它们是 `phy_003` 的 **P14 自动生成图**产出的 ——
**把"做题技巧"标成了物理知识点**（如 `代数化简`、`递推关系`）。

而推荐的判据是「画像知识点 ∩ 题库知识点」——
**掺进这些泛泛的技能标签，匹配会变松、推荐会失准。**

## 口径
**只从 `graph.json` 的<节点级>移除，`problem.json` 的题级不动。**
移除的是"不是物理知识点"的那些；`惯性参考系`/`正方向约定`/`变质量系统` 这类
**是真物理的，留**。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 要移除的"数学/过程技能"标签。
#:
#: **判据**：它描述的是"怎么算"，不是"物理上要懂什么"。
REMOVE = {
    "不等式比较", "代数化简", "代数方程求解", "初始条件应用",
    "求和性质", "求和运算", "结果比较", "逐次过程分析", "递推关系",
}
#: 只处理这道题 —— 这些标签是它的自动生成图带进来的。
TARGET = "phy_003"


def process(*, apply: bool = False) -> dict:
    stat = {"nodes": 0, "removed": 0}
    det: list[str] = []
    gf = ROOT / "data" / "problems" / TARGET / "graph.json"
    g = json.loads(gf.read_text(encoding="utf-8"))
    for n in g.get("nodes") or []:
        old = list(n.get("knowledge_points") or [])
        new = [k for k in old if k not in REMOVE]
        stat["nodes"] += 1
        if new != old:
            gone = [k for k in old if k in REMOVE]
            stat["removed"] += len(gone)
            det.append(f"{n.get('id')}  移除 {gone}")
            if apply:
                n["knowledge_points"] = new
    if apply and det:
        gf.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stat": stat, "detail": det}


def main() -> int:
    ap = argparse.ArgumentParser(description="清理非物理的知识点标签")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    r = process(apply=a.apply)
    print(f"  === {TARGET}  {'已写入' if a.apply else 'DRY-RUN（未改动）'} ===")
    print(f"    扫描节点 {r['stat']['nodes']}   移除标签 {r['stat']['removed']} 个")
    for d in r["detail"]:
        print(f"      {d}")
    if not r["detail"]:
        print("    ✅ 无需改动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
