"""第三方**独立复现** `evidence_sha256` 的算式（RAG/T4 各批次哈希链的根）。

## 为什么要有这个脚本（2026-10-10，同一问题被**两次**提出）
```
v10 裁决者、v15 评审 A 都报「evidence_sha256 未能复现」。
A 试了 **98 种**拼装方案（名称/内容/逐包哈希/字段序列化…）**全部未命中**。
⇒ 查清后：**它不是不可复现，是没人知道该用哪个函数** ——
   算式在 `scripts/run_t4_batch.py:evidence_digest()` 里，但复核者只能反推。
⇒ 反推一个**两级哈希**是徒劳的。**把算式交出去**才是修法。
```

## 算式（逐字，可自行核对）
```python
files = (batch/"runs").glob("*.json") + (batch/"review_packets").glob("*") + batch/"restricted_mapping.json"
content = {str(p.relative_to(batch)): sha256(p.read_bytes()) for p in sorted(files) if p.is_file()}
evidence_sha256 = sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
```
⚠️ 两个易错点：**① 两级哈希**（先逐文件、再整体）；**② 第二级的输入是
`json.dumps(..., sort_keys=True)`**（不是把字节拼起来）。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence_digest(batch: Path) -> str:
    """与 `scripts/run_t4_batch.py:evidence_digest()` **同一个算式**（就地重写一份便于独立核对）。

    ⚠️ 本脚本**故意不 import 产线函数** —— 独立复现要能脱离产线代码算出同一个值，
    否则"复现"就变成了"再调用一次同一个函数"（那不是核验）。
    """
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    files = list((batch / "runs").glob("*.json")) + list((batch / "review_packets").glob("*"))
    mapping = batch / "restricted_mapping.json"
    if mapping.exists():
        files.append(mapping)
    content = {str(p.relative_to(batch)): sha(p) for p in sorted(files) if p.is_file()}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="独立复现 evidence_sha256")
    ap.add_argument("batch", type=Path, help="批次目录（含 runs/ 与 review_packets/）")
    ap.add_argument("--compare", default="", help="可选：与 summary.json 里记的值比对")
    args = ap.parse_args()

    if not args.batch.is_dir():
        print(f"❌ 目录不存在：{args.batch}")
        return 1
    got = evidence_digest(args.batch)
    print(f"evidence_sha256 = {got}")

    summary = args.batch / "summary.json"
    recorded = ""
    if summary.is_file():
        recorded = str(json.loads(summary.read_text(encoding="utf-8")).get("evidence_sha256") or "")
        print(f"summary 记的     = {recorded or '（无）'}")
    if args.compare:
        print(f"命令给的         = {args.compare}")
    ref = args.compare or recorded
    if ref:
        print("✅ 一致" if ref == got else "❌ **不一致**")
        return 0 if ref == got else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
