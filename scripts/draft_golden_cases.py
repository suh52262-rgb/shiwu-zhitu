"""金标样本**起草器**（不自动定稿）—— 为扩到电磁/光学准备新样本。

## 为什么要有它、以及它**不做什么**
```
核查报告：现有 300 条只覆盖力学、热学（各 150）⇒ 要扩到电磁/光学。
而金标必须"**从真实内容块反写提问**"，不能"先编问题再找块" ——
后者极易造出"问题与块对不上"的**假金标**，而假金标会让下一轮基线**从一开始就是错的**。
⇒ 本脚本**只起草**：拉候选块、过滤明显不是知识点的块、输出草稿表。
   **逐条由人核过**才写进正式数据集 —— 它**不写 `retrieval_golden_cases*.json`**。
```
## 过滤哪些块（判据可复核）
```
· 前言/序/目录/出版信息：含「面向21世纪」「课程教材」「本卷共分」「目录」「ISBN」…
· 参考书目：含「出版社」「大学学报」「参考文献」或多个人名+书名号
· 页眉页脚与编号：过短、或几乎全是数字/符号
· 图注：以「图」开头且很短
⚠️ 判据是**启发式**，会漏也会误杀 ⇒ 所以它**只起草**。
"""
import argparse
import json
import re
from pathlib import Path

#: 明显不是知识点的块（前言 / 书目 / 版权 / 目录）
_NOISE = re.compile(
    r"面向21世纪|课程教材|本卷共分|出版社|大学学报|参考文献|图书在版|ISBN|版权所有"
    r"|^目录|^前言|^序\b|^附录|印次|开本|字数|定价")
#: 图注（很短且以"图"开头）
_CAPTION = re.compile(r"^图\s*\d")


def looks_like_knowledge(text: str) -> bool:
    """这一块**像不像**能当金标的知识点（启发式，**只用于起草**）。"""
    t = (text or "").strip()
    if len(t) < 100 or len(t) > 500:
        return False
    if _NOISE.search(t) or _CAPTION.match(t):
        return False
    # 几乎没有中文 ⇒ 多半是公式/符号排版残留或外文引用
    if sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff") < 40:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="金标样本起草（只输出草稿，不写数据集）")
    ap.add_argument("doc_id", help="教材 doc_id，如 jiaocai-5-dianci")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="", help="草稿 JSON 路径（可选）")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from sqlalchemy import text  # noqa: PLC0415

    from app.db.postgres import get_engine  # noqa: PLC0415

    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT id, page, block_index, text FROM document_blocks "
            "WHERE doc_id = :d AND indexable IS NOT FALSE AND type = 'text' "
            "ORDER BY page, block_index"), {"d": args.doc_id}).mappings().all()

    drafts = []
    for r in rows:
        if not looks_like_knowledge(r["text"]):
            continue
        drafts.append({"doc_id": args.doc_id, "page": r["page"], "block_index": r["block_index"],
                       "block_id": r["id"], "text": r["text"].strip(),
                       # ⭐ **提问要人来写** —— 这里只给方向提示，不生成问题。
                       "query": "", "note": "待人工提问（学生口吻，能被这一块回答）"})
        if len(drafts) >= args.limit:
            break

    out = {"doc_id": args.doc_id, "scanned": len(rows), "drafted": len(drafts),
           "claim": ("DRAFT ONLY —— 逐条人工核过才可进正式数据集；"
                     "本文件不得直接当作金标使用。"),
           "drafts": drafts}
    text_out = json.dumps(out, ensure_ascii=False, indent=2)
    print(text_out)
    if args.out:
        Path(args.out).write_text(text_out, encoding="utf-8")
        print(f"\nSaved draft to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
