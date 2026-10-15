"""构建检索 golden set：**先选源段落，再据它出题**。

## 为什么必须这个顺序

如果先跑检索、再把返回的东西标成"正确答案"，标的就是**检索器自己的输出**
——那样的评测永远 100%，也永远发现不了问题。

正确做法是**反过来的**：
  1. 从语料里采样一个**源段落**（gold 在此时就定了）
  2. 据这个段落写出一个**学生式的问题**（答案在该段落里）
  3. 评测时问："检索能不能找回**我出发的那个段落**"

源段落是**选出来的**，不是检索出来的——这样才非循环。

## 出题的要求

模型容易把问题写成"关键词检索"（太容易）或抄原文（太简单）。
所以提示词强调：**像学生真的会问的那样**，用口语、带疑问、不要照抄原句。

用法：
    python scripts/build_retrieval_golden.py --per-doc 30
    python scripts/build_retrieval_golden.py --per-doc 30 --sample-out   # 只看样本
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.rag import block_store  # noqa: E402

OUT = ROOT / "evaluation" / "datasets" / "retrieval_golden_cases.json"

#: 采样只取**可索引、够长**的正文块——太短的块承载不了问题，
#: 而且它们本来就是碎片化问题的来源，拿来出题会放大噪声。
MIN_CHARS = 40
MAX_CHARS = 400

#: 前置页只到版权页/前言，出不了物理题，跳过
SKIP_FIRST_PAGES = 20

CJK = re.compile(r"[\u4e00-\u9fff]")
#: 数学符号——占比过高说明这其实是**排版工具没排好的公式**，
#: 不是散文。实测采样里混进过 `'=Eo-U(R）=lU(R）/2.新增加的径向动能mu2/2=…'`
#: 这种 OCR 乱码块，拿它出题只会产出垃圾。
MATH = set("=+-*/^_∫∑√±×÷≈≠≤≥<>()[]{}")
#: 版权页/参考信息的特征
JUNK = re.compile(r"ISBN|http|www\.|出版社|印刷|开本|字数|定价|版权所有|教学成果奖")

#: **过渡句**：只说明"接下来讲什么"，本身回答不了问题。
#: 拿它当 gold，检索找不到是正常的，却会被记成失败（归因分析里记过这种）。
#: 过渡句的**开头标志**。这类句子只是"接下来要讲什么"。
TRANSITION_HEAD = re.compile(r"^(前面|以上|上述|下面|接下来|现在|这里|本节|本章)")
#: 过渡句里会出现的动词。与开头标志**分开判**——
#: 一开始把它们写在同一个正则里并限制 12 字距离，结果实测那条隔了 13 字就漏了。
TRANSITION_VERB = re.compile(r"(讨论|推导|介绍|说明|考虑|给出|来看|看另外)")
#: 习题片段：题号 + "试求/求/证明/计算"，或明显的两题粘连
EXERCISE = re.compile(r"[（(]?\d+[）)]?\s*[.、]?\s*(试求|试证|试分析|求|证明|计算|解[:：])|"
                      r"\d+\s*[-－]\s*\d+\s*\.")


def _is_prose(text: str) -> bool:
    """够不够格当"源段落"。

    **从垃圾源段落出的题，评测出来的数字也没意义**——所以这一步
    宁可严一点、少采几条。实测第一版没这个筛选，采到了版权页和
    OCR 乱码公式块。
    """
    t = text.strip()
    if not (MIN_CHARS <= len(t) <= MAX_CHARS):
        return False
    if JUNK.search(t):
        return False
    cjk = len(CJK.findall(t))
    if cjk < 15:
        return False
    # 汉字占比：真散文应过半；公式/表格则不是
    if cjk / len(t) < 0.5:
        return False
    # 至少要有句读——没有的话多半是残块
    if "。" not in t and "，" not in t:
        return False
    # 数学符号占比不能过高
    math_ratio = sum(1 for c in t if c in MATH) / len(t)
    if math_ratio > 0.12:
        return False
    # **过渡句不能当 gold**：它回答不了问题，拿它评测等于给检索埋必错的样本。
    # 两个判据分开：以过渡词开头 + 前 40 字内有过渡动词。
    if TRANSITION_HEAD.match(t) and TRANSITION_VERB.search(t[:40]):
        return False
    # **习题片段不能当 gold**：往往还是两道题 OCR 粘连的产物
    if EXERCISE.search(t[:60]):
        return False
    return True

PROMPT = """你是大学物理课程的助教。下面是一段教材原文。

请**模仿学生的口吻**提出一个该段落能够回答的问题。

要求：
1. **像学生真的会问的那样**——口语、带疑问语气，例如"为什么……？""……是怎么回事？"
2. **不要照抄原句**，也不要堆关键词（不要写成"机械能守恒 条件 摩擦力"这种）
3. 问题要**具体到能用这段话回答**，不要太空泛（避免"什么是物理"这类）
4. 只输出问题本身，一行，不要引号、不要解释

教材原文：
---
{text}
---

问题："""


def _pick_blocks(per_doc: int, seed: int,
                used_ids: set[str] | None = None) -> list[dict]:
    """分层采样源段落。"""
    rng = random.Random(seed)
    picked: list[dict] = []
    docs = [d for d in block_store.list_documents() if d["doc_id"] != "demo-textbook"]
    for d in docs:
        blocks = block_store.load_blocks(d["doc_id"])
        cand = [
            b for b in blocks
            # **只取可索引的**：不可索引的块检索本来就找不到，
            # 拿它出题等于给评测埋一个必错的样本。
            if b.get("indexable", True)
            and b["type"] == "text"
            and b["page"] > SKIP_FIRST_PAGES
            and not b["text"].strip().startswith("图")
            and _is_prose(b["text"])
            # **--append 时排除已用过的块**，避免同一段落出两道题
            and (not used_ids or b["id"] not in used_ids)
        ]
        if not cand:
            continue
        # 分层：按页均匀铺开，避免全挤在一章里
        cand.sort(key=lambda b: b["page"])
        step = max(1, len(cand) // per_doc)
        chosen = cand[::step][:per_doc]
        rng.shuffle(chosen)
        for b in chosen:
            picked.append({
                "doc_id": d["doc_id"],
                "page": b["page"],
                "block_id": b["id"],
                "text": b["text"],
            })
        print(f"  {d['doc_id']:<22}候选 {len(cand):>5} 条 → 取 {len(chosen)}")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description="构建检索 golden set")
    ap.add_argument("--per-doc", type=int, default=30, help="每本教材出多少题")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--sample-out", action="store_true", help="只打印样本，不调模型")
    ap.add_argument("--append", action="store_true",
                    help="保留已有条目，只补新的（保住旧基线可比）")
    args = ap.parse_args()

    existing: list[dict] = []
    used_ids: set[str] = set()
    if args.append and OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8")).get("cases", [])
        used_ids = {c["gold"]["block_id"] for c in existing if c.get("gold")}
        print(f"  已有 {len(existing)} 条，将保留并补充（seed 会偏移以取到新样本）")

    print("=== 第 1 步：采样源段落（gold 此刻就定了）===")
    picked = _pick_blocks(args.per_doc, args.seed + (1000 if args.append else 0),
                          used_ids)
    print(f"  新增 {len(picked)} 条源段落")

    if args.sample_out:
        for p in picked[:8]:
            print(f"\n  [{p['doc_id']} p{p['page']}]")
            print(f"    {p['text'][:110]}")
        return 0

    print("\n=== 第 2 步：据源段落出题（调模型）===")
    from app.services.llm import get_llm

    llm = get_llm("tutor")
    out = []
    for i, p in enumerate(picked, 1):
        try:
            resp = llm.chat("你是大学物理课程的助教。", PROMPT.format(text=p["text"][:400]))
            q = (resp if isinstance(resp, str) else str(resp)).strip()
            q = q.splitlines()[0].strip().strip('"“”') if q else ""
        except Exception as exc:  # noqa: BLE001 单条失败不该毁掉整批
            print(f"    ✗ 第 {i} 条出题失败：{type(exc).__name__}")
            continue
        if len(q) < 6:
            print(f"    ✗ 第 {i} 条问题过短，丢弃：{q!r}")
            continue
        out.append({
            "id": f"rg_{len(out)+1:03d}",
            "query": q,
            # **gold 是"我出发的那个块"**，不是"检索返回的块"
            "gold": {"doc_id": p["doc_id"], "page": p["page"],
                     "block_id": p["block_id"], "text": p["text"]},
        })
        if i % 10 == 0:
            print(f"    … {i}/{len(picked)}")

    # 已有条目的 id 保持不变；新条目从接着的编号开始
    offset = max((int(c["id"].split("_")[-1]) for c in existing), default=0)
    for i, c in enumerate(out, 1):
        c["id"] = f"rg_{offset + i:03d}"
    merged = existing + out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "metric": "retrieval_golden",
        "description": (
            "检索 golden set。**非循环构建**：先采样源段落（gold 在此确定），"
            "再据段落出学生式问题；评测问的是『能否找回出发的那个段落』，"
            "而不是『检索返回了什么』。"
            "gold 是该段落的 (doc_id, page, block_id)。"
        ),
        "cases": merged,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✓ 已写入 {OUT.relative_to(ROOT)}"
          f"（新增 {len(out)}，合计 {len(merged)} 条）")
    if existing:
        print(f"    原有 {len(existing)} 条保持不变——旧基线仍可直接对比")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

