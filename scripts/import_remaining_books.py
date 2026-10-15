"""批量导入剩余教材，每本导完自动跑隔离判据核对。

## 为什么核对要挂在导入后面

`docs/公式隔离率抽查报告.md` 里写下的要求：阈值是在**一本教材**上核对出来的
（同一 OCR 引擎、同一字体、同一扫描质量），**换教材必须重新核对**。

挂成自动步骤，就不会因为"忘了"而漏掉——而漏掉的后果是
**判据在暗中失效，没人知道**。

## 用法
    python scripts/import_remaining_books.py            # 全部 6 本
    python scripts/import_remaining_books.py --only 1   # 只跑第 1 个
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

#: (pdf 文件名, doc_id)。doc_id 只用 ASCII——`validate_doc_id` 的白名单不含中文。
BOOKS = [
    ("教材-1-大学物理学(上册)第五版_可搜索.pdf", "jiaocai-1-shangce"),
    ("教材-2-大学物理学(下册)第五版_可搜索.pdf", "jiaocai-2-xiache"),
    ("教材-1-大学物理学(学习辅导与习题解答)第五版_可搜索.pdf", "jiaocai-1-jieda"),
    ("教材-5-新概念物理教程 电磁学_可搜索.pdf", "jiaocai-5-dianci"),
    ("教材-6-新概念物理教程 光学_可搜索.pdf", "jiaocai-6-guangxue"),
    ("教材-7-新概念物理教程 量子物理_可搜索.pdf", "jiaocai-7-liangzi"),
]

SUMMARY = ROOT / "evaluation" / "reports" / "import-batch-summary.json"


def run(cmd: list[str], *, quiet: bool = False) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if not quiet:
        print(out)
    return p.returncode, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0, help="只跑第 N 个（从 1 起）")
    ap.add_argument("--from", dest="from_n", type=int, default=1,
                    help="从第 N 个开始（跳过已完成的）")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    if args.only:
        books = BOOKS[args.only - 1: args.only]
    else:
        books = BOOKS[args.from_n - 1:]
    results = []
    t_all = time.perf_counter()

    for i, (name, doc_id) in enumerate(books, 1):
        pdf = ROOT / "data" / "textbooks" / name
        print(f"\n{'=' * 74}")
        print(f"  [{i}/{len(books)}] {doc_id}  ←  {name}")
        print(f"{'=' * 74}", flush=True)
        if not pdf.is_file():
            print(f"  ✗ 文件不存在：{pdf}")
            results.append({"doc_id": doc_id, "status": "missing"})
            continue

        t0 = time.perf_counter()
        code, out = run([str(PY), "scripts/import_pdf_batch.py", str(pdf),
                         "--doc-id", doc_id, "--batch", str(args.batch)])
        elapsed = time.perf_counter() - t0
        row = {"doc_id": doc_id, "exit": code, "seconds": round(elapsed, 1)}
        # 从输出里抠出规模信息
        for line in out.splitlines():
            if "库中：" in line:
                row["lib"] = line.split("库中：")[-1].strip()
            if "输入 token" in line:
                row["tokens"] = line.split("输入 token：")[-1].strip()
        row["status"] = "ok" if code == 0 else f"exit={code}"
        print(f"  → {row['status']}  用时 {elapsed/60:.1f} 分钟"
              f"  {row.get('lib', '')}", flush=True)

        # ---- 导完就跑隔离核对（报告里要求的步骤，挂成自动的就不会忘）----
        print(f"  ── 隔离判据核对 ──", flush=True)
        _, iso_out = run([str(PY), "scripts/check_isolation.py", doc_id,
                          "--sample", "0"], quiet=True)
        for line in iso_out.splitlines():
            s = line.strip()
            if (s.startswith("公式隔离") or s.startswith("图注去重")
                    or "零汉字占比" in s or "VL 公式块" in s
                    or "判据适用性" in s or s.startswith("✅") or s.startswith("⚠")
                    or s.startswith("❌")):
                print(f"    {s}", flush=True)
        row["isolation_check"] = iso_out
        results.append(row)

    print(f"\n{'=' * 74}")
    print(f"  全部完成，总耗时 {(time.perf_counter()-t_all)/60:.1f} 分钟")
    print(f"{'=' * 74}")
    for r in results:
        print(f"  {r['doc_id']:<22}{r.get('status', ''):<10}"
              f"{r.get('seconds', 0)/60:>6.1f} 分钟  {r.get('lib', '')}")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": [{k: v for k, v in r.items()
                            if k != "isolation_check"} for r in results]},
              open(SUMMARY, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  摘要已写入 {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
