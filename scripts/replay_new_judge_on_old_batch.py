"""对照：把**新判据**回放到**上一批 72 份旧输出**上。

## 这个数为什么必须有
下一批数字一定会变。要解释它，必须先把"**判据变严**"这一份单独量出来 ——
否则分不清"模型改好了"和"尺子变严了"。这里量的是：

    上一批那批**完全相同**的输出，放在**新尺子**下会是多少分？

⇒ 那就是下批的**基线**：即使模型一个字没改，也该是这个数。
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "backend")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from app.services.review_builder import classify_defects, structural_defects  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "artifacts" / "t4_development_20261009"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))

if not DEV.is_dir():
    print(f"[X] 找不到 {DEV}（上一批产物不在这台机器上）")
    raise SystemExit(2)

old_ok = new_ok = 0
new_kinds: Counter = Counter()
regressions = []
total = 0
for p in sorted((DEV / "review_packets").glob("*.json")):
    pkt = load(p)
    review = pkt.get("review") or {}
    if not review:
        continue
    total += 1
    recorded = (review.get("validation") or {}).get("valid")     # 旧尺子的结论（当时记下来的）
    issues = structural_defects(review)                          # 新尺子重算
    old_ok += bool(recorded)
    new_pass = not issues
    new_ok += new_pass
    for d in issues:
        new_kinds[d["kind"]] += 1
    if recorded and issues:                                      # 旧尺子放行、新尺子拦下
        regressions.append((p.stem, [d["kind"] for d in issues], classify_defects(issues)))

print(f"上一批输出共 {total} 份\n")
print(f"  旧尺子（当时记在 validation.valid 里）：**{old_ok}/{total}** 通过")
print(f"  新尺子（现在的 structural_defects 重算）：**{new_ok}/{total}** 通过")
print(f"""
  ⚠️ **不要把这两个数相减当成"判据变严"** —— 旧尺子那个数**本身是被误报压低的**：
     旧判据的裸 `N 次` 分支把中文普通副词「一次」判成计数，
     归档 72 份里 **42 份的唯一命中就是它**（独立复核重放实测）。
     ⇒ 修正后 model_written_count 从 44 掉到 {new_kinds.get('model_written_count', 0)}，
       新尺子反而**比旧尺子宽松**。
""")

print("  新尺子命中的缺陷种类（按份数）：")
for kind, n in new_kinds.most_common():
    print(f"    {kind:32s} {n}")

newly_blocked = [r for r in regressions]
print(f"\n  旧尺子放行、新尺子拦下的 {len(newly_blocked)} 份 —— **这一批才是本轮新增的严格度**：")
for qid, kinds, category in newly_blocked[:40]:
    print(f"    {qid:6s} {category:22s} {kinds}")
if len(newly_blocked) > 40:
    print(f"    … 其余 {len(newly_blocked)-40} 份同类")
if not newly_blocked:
    print("    （无）")

print(f"""
=== 这份对照怎么用 ===
  下一批跑完，把它的"结构通过率"和这里的 **{new_ok}/{total}** 比，而不是和上一批报告的 100% 比：
     高于 {new_ok}/{total}  ⇒ 模型确实改了（这部分才是修复的效果）
     等于 {new_ok}/{total}  ⇒ 模型没改
     低于 {new_ok}/{total}  ⇒ 需要解释（随机性；或者判据又动了 —— 先查判据）

  ⚠️ 另外要连着报两个数，否则会误读：
     · **判据覆盖面**：它抓得到什么、抓不到什么（裸量词形态抓不到）
     · **误报面**：本脚本与 `measure_attribution_scan.py` 量出来的那部分
""")
