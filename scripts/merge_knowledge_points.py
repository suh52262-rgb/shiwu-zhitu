"""统一题库的知识点标注（受控词表的第一步）。

## 为什么
推荐的判据是「画像知识点 ∩ 题库知识点」—— **题库自己标注不统一，推荐就漏**。
而 `diagnose` 的归位也依赖词表，词表乱则归位结果不一致。

实测 72 条知识点里有 6 组同义 + 4 组待定，用户已拍板。

## 用法
```
python scripts/merge_knowledge_points.py            # dry-run
python scripts/merge_knowledge_points.py --apply
```
**默认 dry-run** —— 改题库是破坏性操作。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 合并映射：`旧名 → 规范名`。
#:
#: 分三类，都记清楚来源：
#: · **用户拍板**（机械能守恒族、简谐运动族、矢量性族）
#: · **高置信同义**（措辞不同、所指为一）
#: · **保留不动**：动量 / 冲量 / 动量定理 —— 用户明确说保留
MERGE: dict[str, str] = {
    # ---- 用户①：机械能守恒族 ----
    "机械能守恒": "机械能守恒条件",
    "机械能守恒的适用条件": "机械能守恒条件",
    # ---- 用户③：简谐运动族 ----
    "简谐运动振幅": "简谐运动",
    "简谐运动机械能": "简谐运动",
    # ---- 用户：三个"矢量性"合并 ----
    #  `矢量性` 是**新起的规范名** —— 三个原名的外延不完全一样
    #  （速度 / 冲量 / 动量定理），但都是"矢量要带方向"这一件事。
    "速度的矢量性": "矢量性",
    "冲量的矢量性": "矢量性",
    "动量定理的矢量性": "矢量性",
    # ---- 高置信同义 ----
    "动量守恒": "动量守恒定律",
    "系统动量守恒": "动量守恒定律",
    "动量守恒定律的应用": "动量守恒定律",
    "变质量系统（逐次跳下）": "变质量系统",
    "参考系与相对速度": "相对速度",
    "求和性质": "求和运算",
    "求和与不等式比较": "求和运算",
}


def _dedup(seq: list[str]) -> list[str]:
    out: list[str] = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


def merge_kp(kps) -> tuple[list[str], list[tuple[str, str]]]:
    """返回 `(新列表, [(旧,新)...])`。**保序去重。**"""
    old = [str(k) for k in (kps or []) if k]
    new = _dedup([MERGE.get(k, k) for k in old])
    changes = [(k, MERGE[k]) for k in old if k in MERGE]
    return new, changes


def process(*, apply: bool = False) -> dict:
    stat = {"files": 0, "changed": 0, "renames": 0, "dropped": 0}
    det: list[str] = []
    for pdir in sorted((ROOT / "data" / "problems").iterdir()):
        if not pdir.is_dir():
            continue

        # ---- problem.json ----
        pf = pdir / "problem.json"
        if pf.is_file():
            d = json.loads(pf.read_text(encoding="utf-8"))
            old = list(d.get("knowledge_points") or [])
            new, ch = merge_kp(old)
            stat["files"] += 1
            if new != old:
                stat["changed"] += 1
                stat["renames"] += len(ch)
                stat["dropped"] += len(old) - len(new)
                det.append(f"{pdir.name}/problem.json\n      改前 {old}\n      改后 {new}")
                if apply:
                    d["knowledge_points"] = new
                    pf.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

        # ---- graph.json（节点级）----
        gf = pdir / "graph.json"
        if gf.is_file():
            g = json.loads(gf.read_text(encoding="utf-8"))
            touched = False
            for n in g.get("nodes") or []:
                old = list(n.get("knowledge_points") or [])
                new, ch = merge_kp(old)
                if new != old:
                    touched = True
                    stat["renames"] += len(ch)
                    stat["dropped"] += len(old) - len(new)
                    det.append(f"{pdir.name}/graph.json {n.get('id')}\n"
                               f"      改前 {old}\n      改后 {new}")
                    if apply:
                        n["knowledge_points"] = new
            if touched:
                stat["changed"] += 1
                if apply:
                    gf.write_text(json.dumps(g, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    return {"stat": stat, "detail": det}


def main() -> int:
    ap = argparse.ArgumentParser(description="统一题库知识点标注")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()
    r = process(apply=a.apply)
    s = r["stat"]
    print(f"  === {'已写入' if a.apply else 'DRY-RUN（未改动任何数据）'} ===")
    print(f"    扫描文件 {s['files']}   需改 {s['changed']}")
    print(f"    重命名 {s['renames']} 处   合并掉 {s['dropped']} 条")
    if r["detail"]:
        print(f"\n  === 变更明细（前 {a.show}）===")
        for d in r["detail"][:a.show]:
            print(f"\n    {d}")
    else:
        print("\n    ✅ 无需改动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
