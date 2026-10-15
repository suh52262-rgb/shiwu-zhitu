"""换教材时重跑隔离判据核对（docs/公式隔离率抽查报告.md 的三步）。

用法：
    python scripts/check_isolation.py jiaocai-4-rexue
    python scripts/check_isolation.py jiaocai-4-rexue --sample 60 --seed 7

## 为什么要重跑

`formula_filter` 的阈值是在**一本教材**上核对出来的（同一 OCR 引擎、
同一字体、同一扫描质量）。换教材前必须重新核对——这正是那份报告里
写下的警告，而不是"跑过一次就永久有效"。

## 三步里哪一步最关键

**第 2 步（全量汉字数分布）最便宜，也最能说明问题。**
中文教材里"一个汉字都没有"的短块不可能是散文——如果这个比例
在新教材上明显偏低，说明判据的适用前提变了，得重新调。

第 1 步（人工抽样）不可自动化，脚本只负责**打出来给你看**。
"""
import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.rag import block_store  # noqa: E402

CJK = re.compile(r"[\u4e00-\u9fff]")

#: 报告里的基线（教材-3 力学）
BASELINE = {
    "doc_id": "jiaocai-3-lixue",
    "isolated_rate": 10.3,
    "zero_cjk_share": 97.5,
    "ge3_cjk_count": 14,
    "vl_covered_share": 97.8,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="换教材时核对隔离判据")
    ap.add_argument("doc_id")
    ap.add_argument("--sample", type=int, default=60, help="人工抽样条数")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    blocks = block_store.load_blocks(args.doc_id)
    if not blocks:
        print(f"  ✗ 文档 {args.doc_id} 没有内容块（还没导入？）")
        return 1
    all_isolated = [b for b in blocks if not b.get("indexable", True)]
    # **按原因拆分。** `indexable=False` 有两种来源：
    #   · 公式过滤器 —— 坏公式（本脚本要核对的）
    #   · 图注去重   —— 与 figure_caption 重复的正文块（与判据无关）
    # 混在一起会把图注当成"误伤的公式碎片"，得出判据失效的错误结论。
    isolated = [b for b in all_isolated
                if "坏公式" in (b.get("note") or "")]
    deduped = [b for b in all_isolated if b not in isolated]
    rate = len(isolated) / len(blocks) * 100

    print(f"{'=' * 74}")
    print(f"  隔离判据核对 · {args.doc_id}")
    print(f"{'=' * 74}")
    print(f"  总块 {len(blocks)}")
    print(f"    公式隔离（本脚本核对对象）  {len(isolated):>5}（{rate:.1f}%）")
    print(f"    图注去重（与判据无关）      {len(deduped):>5}")
    print(f"  基线（{BASELINE['doc_id']}）：隔离率 {BASELINE['isolated_rate']}%")
    print("  隔离块类型：", dict(Counter(b["type"] for b in isolated)))

    # ============ 第 2 步：全量汉字数分布（最关键，也最便宜）============
    print(f"\n── 第 2 步：全量汉字数分布（最关键）{'─' * 30}")
    dist = Counter(len(CJK.findall(b["text"])) for b in isolated)
    for n in sorted(dist):
        print(f"    {n:>2} 个汉字：{dist[n]:>5} 条")
    zero = dist.get(0, 0)
    zero_share = zero / len(isolated) * 100 if isolated else 0.0
    print(f"\n    **零汉字占比 {zero_share:.1f}%**"
          f"（基线 {BASELINE['zero_cjk_share']}%）")

    ge3 = [b for b in isolated if len(CJK.findall(b["text"])) >= 3]
    print(f"    含 ≥3 汉字的 {len(ge3)} 条"
          f"（基线 {BASELINE['ge3_cjk_count']} 条 / 2369 块）")
    if ge3:
        print("    抽样看：")
        for b in ge3[:12]:
            print(f"      p{b['page']:<5}{b['text'][:62]!r}")

    # 判据是否仍成立
    print("\n    → 判据适用性判断：")
    if zero_share >= 90:
        print(f"      ✅ 零汉字占比 {zero_share:.1f}% ≥ 90%，"
              "'零汉字不可能是散文'这条前提仍然成立")
    elif zero_share >= 80:
        print(f"      ⚠ 零汉字占比 {zero_share:.1f}%，低于基线但仍在合理范围；"
              "建议加大人工抽样")
    else:
        print(f"      ❌ 零汉字占比仅 {zero_share:.1f}%，**判据前提可能已不成立**，"
              "必须人工逐条核对并重调阈值")

    # ============ 第 3 步：VL 公式覆盖 ============
    print(f"\n── 第 3 步：隔离页的 VL 公式覆盖 {'─' * 30}")
    vl_pages: dict[int, int] = {}
    for b in blocks:
        if b["type"] == "formula" and b.get("source_kind") == "vl":
            vl_pages[b["page"]] = vl_pages.get(b["page"], 0) + 1
    iso_pages = {b["page"] for b in isolated}
    covered = [p for p in iso_pages if vl_pages.get(p)]
    share = len(covered) / len(iso_pages) * 100 if iso_pages else 0.0
    print(f"    有隔离块的页 {len(iso_pages)} 页，其中有 VL 公式块 {len(covered)} 页"
          f"（**{share:.1f}%**，基线 {BASELINE['vl_covered_share']}%）")
    bare = [b for b in isolated if not vl_pages.get(b["page"])]
    print(f"    最坏情况（无 VL 公式覆盖，隔离内容是唯一表示）：{len(bare)} 条")
    for b in bare[:15]:
        print(f"      p{b['page']:<5}{b['text'][:62]!r}")
    print("      → 逐条看：若多为版权页/页码/索引，则隔离是**收益**；"
          "若出现教学内容，判据过严")

    # ============ 第 1 步：人工抽样（脚本只负责打出来）============
    print(f"\n── 第 1 步：分层随机抽样 {args.sample} 条（**需人工逐条判定**）{'─' * 12}")
    random.seed(args.seed)
    layers: dict[int, list] = {}
    for b in isolated:
        layers.setdefault(b["page"] // 50, []).append(b)
    per = max(1, args.sample // max(1, len(layers)))
    sample: list = []
    for k in sorted(layers):
        sample.extend(random.sample(layers[k], min(per, len(layers[k]))))
    print(f"    {len(layers)} 层 × {per} = {len(sample)} 条\n")
    for i, b in enumerate(sample, 1):
        t = (b["text"] or "").replace("\n", "⏎")
        print(f"    [{i:>3}] p{b['page']:<5}{b['type']:<8}{len(b['text']):>4}字  {t[:70]!r}")
    print("\n    ⚠ 这一步**不能自动化**：要人判断'这条拦得对不对'。")
    print("      重点找：像散文的、像答案的、像表头的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
