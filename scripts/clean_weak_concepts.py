"""知识点标签清洗：把库里"非规范形式"的 `weak_concepts` 归位到题库词表。

## 为什么需要
`diagnose` 之前产出自由措辞的 `concept`（如
`机械能守恒条件（只有重力或弹力做功）`），它被写进画像与作答记录。
`concept_normalizer` 修的是**新产生的**，**历史数据不会自动变**。

⇒ 不清洗的话，推荐对老学生照样漏（判据是「画像知识点 ∩ 题库知识点」）。

## 口径
· `attempts.weak_concepts` 是**逐题**的 → 词表取**那道题**的
· `student_profiles.weak_concepts` 是**跨题**的 → 词表取**全部题的并集**

## 用法
```
python scripts/clean_weak_concepts.py            # dry-run，只打印
python scripts/clean_weak_concepts.py --apply    # 真写
```
**默认 dry-run** —— 改画像是有副作用的操作，不该默认执行。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.persistence import get_engine  # noqa: E402
from app.db.tables import Attempt, StudentProfile  # noqa: E402
from app.services.concept_normalizer import snap_many  # noqa: E402
from app.services.problem_repository import get_problem_repository  # noqa: E402


def build_vocabularies() -> tuple[dict[str, list[str]], list[str]]:
    """返回 `(逐题词表, 全局并集)`。

    **两边都取** —— 无图题只有 Problem 上有知识点；有图题的节点上也标了。
    """
    repo = get_problem_repository()
    per: dict[str, list[str]] = {}
    for p in repo.list_problems():
        v = list(p.knowledge_points or [])
        g = repo.get_graph(p.problem_id)
        for n in getattr(g, "nodes", None) or []:
            v += [k for k in (n.knowledge_points or []) if k]
        per[p.problem_id] = sorted({k for k in v if k})
    union = sorted({k for v in per.values() for k in v})
    return per, union


def clean(*, apply: bool = False) -> dict:
    per, union = build_vocabularies()
    eng = get_engine()
    stat = {"attempts_scanned": 0, "attempts_changed": 0,
            "profiles_scanned": 0, "profiles_changed": 0,
            "dropped": 0, "before": 0, "after": 0}
    changes: list[tuple[str, str, list[str], list[str]]] = []

    # ⚠️ **必须用 Session，不能用 Connection。**
    # `conn.execute(select(ORM类)).scalars()` 会退化成"取第一列"，
    # 拿到的是字符串而不是实体 —— 实测报 `'str' object has no attribute
    # 'weak_concepts'`。
    with Session(eng) as sess:
        for row in sess.execute(select(Attempt)).scalars():
            stat["attempts_scanned"] += 1
            old = list(row.weak_concepts or [])
            if not old:
                continue
            new = snap_many(old, per.get(row.problem_id) or union)
            if new != old:
                stat["attempts_changed"] += 1
                stat["dropped"] += sum(
                    1 for x in old
                    if not snap_many([x], per.get(row.problem_id) or union))
                stat["before"] += len(old)
                stat["after"] += len(new)
                changes.append((f"attempt {row.id}", row.problem_id or "", old, new))
                if apply:
                    row.weak_concepts = new

        for row in sess.execute(select(StudentProfile)).scalars():
            stat["profiles_scanned"] += 1
            old = list(row.weak_concepts or [])
            if not old:
                continue
            new = snap_many(old, union)
            if new != old:
                stat["profiles_changed"] += 1
                stat["dropped"] += sum(
                    1 for x in old
                    if not snap_many([x], per.get(row.problem_id) or union))
                stat["before"] += len(old)
                stat["after"] += len(new)
                changes.append((f"profile {row.student_id}", "", old, new))
                if apply:
                    row.weak_concepts = new

        if apply:
            sess.commit()

    return {"stat": stat, "changes": changes}


def main() -> int:
    ap = argparse.ArgumentParser(description="清洗 weak_concepts 标签")
    ap.add_argument("--apply", action="store_true",
                    help="真写（默认只 dry-run 打印）")
    ap.add_argument("--show", type=int, default=20, help="打印多少条变更")
    args = ap.parse_args()

    r = clean(apply=args.apply)
    s = r["stat"]
    print(f"  === {'已写入' if args.apply else 'DRY-RUN（未改动任何数据）'} ===")
    print(f"    attempts  扫描 {s['attempts_scanned']}  需改 {s['attempts_changed']}")
    print(f"    profiles  扫描 {s['profiles_scanned']}  需改 {s['profiles_changed']}")
    # **三个互不重叠的口径** —— 前两版用了会重叠的分解，算出过负数。
    print(f"    标签：改动前 {s['before']} 条 → 改动后 {s['after']} 条"
          f"（净减少 {s['before'] - s['after']}）")
    print(f"    其中**归不上词表被丢弃** {s['dropped']} 条")

    if r["changes"]:
        print(f"\n  === 变更明细（前 {args.show} 条）===")
        for tag, pid, old, new in r["changes"][:args.show]:
            print(f"\n    {tag}  {pid}")
            print(f"      改前 {old}")
            print(f"      改后 {new}")
    else:
        print("\n    ✅ 没有需要改的 —— 库里已经全是规范形式")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
