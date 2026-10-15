"""重新冻结评测清单 —— **先把配置差异摊开给你看，再决定写不写**。

## 为什么不能直接跑 `prepare_eval.py`
`run_eval.py` 会校验冻结清单里一百多个文件的哈希，代码一动就报 `input_drift` 并拒跑。
而直接重跑 `prepare_eval.py` 会把清单**连配置一起重写** ——
万一模型 / 提示词版本被顺手改了，**旧数字就悄悄失去可比性**，而且没人会发现。

⇒ 这个脚本先按**旧清单里记录的参数**重建一份，逐字段对比
（`files` / `graphs` 两个哈希字段除外），
**确认"只改哈希、不动配置"之后**才允许写入。

用法：
    python scripts/refreeze_eval_manifest.py           # 只对比，不写
    python scripts/refreeze_eval_manifest.py --write   # 确认无误后重写
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_eval import prepare, verify  # noqa: E402

MF = ROOT / "evaluation" / "prepared_manifest.json"
old = json.loads(MF.read_text(encoding="utf-8"))

# 从旧清单反推原始参数
ARGS = dict(model=old["model_config"]["llm_model"],
            budget_cny=old["budget"]["planned_ceiling_cny"] or 0,
            limit=old["limit"],
            include_proposed=old["evaluation_config"]["label_scope"] == "proposed_exploratory",
            budget_approved=old["budget"]["approved"],
            budget_unlimited=old["budget"]["unlimited"])
print("按旧清单反推的参数：")
for k, v in ARGS.items():
    print(f"  {k} = {v}")

new = prepare(**ARGS)

print("\n=== 除 files/graphs（哈希）外的字段对比 ===")
changed = []
for key in sorted(set(old) | set(new)):
    if key in ("files", "graphs"):
        continue
    if old.get(key) != new.get(key):
        changed.append(key)
        print(f"  ⚠️ {key}")
        print(f"       旧: {json.dumps(old.get(key), ensure_ascii=False)[:150]}")
        print(f"       新: {json.dumps(new.get(key), ensure_ascii=False)[:150]}")
if not changed:
    print("  （全部一致 —— 重冻结只改哈希，不动配置）")

o, n = set(old.get("files", {})), set(new.get("files", {}))
print(f"\n=== 文件集合 ===")
print(f"  旧 {len(o)} 个 / 新 {len(n)} 个；新增 {sorted(n - o)}；移除 {sorted(o - n)}")

drift = verify(old, ROOT)
print(f"\n=== 待重写后就会消失的漂移：{len(drift)} 处 ===")
for d in drift:
    print("  · " + d)

if "--write" in sys.argv:
    MF.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ 已重写 {MF.relative_to(ROOT)}")
    left = verify(new, ROOT)
    print(f"   重写后校验：{'通过（无漂移）' if not left else left}")
else:
    print("\n（未写入；加 --write 才重写）")
