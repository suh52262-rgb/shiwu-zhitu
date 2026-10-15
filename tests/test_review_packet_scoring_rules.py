"""匿名评审包**必须携带计分规则**（2026-10-11 流程缺陷修复的回归测试）。

## 背景（用户裁定）
```
v16 留出集盲评里，一名评审把 `answer_leakage` 的「越高越差」方向误用到了另外三个
「越高越好」字段 ⇒ **36/36 方向颠倒** ✗
事后实测：36 个匿名包里「越高越好 / 越高越差 / 无泄题 / rubric」**全部 0 命中**，
`instructions` 也不含方向 ⇒ **规则只在 docs 表格里，没随包投递** ✗
⚠️ 但**不能**据此断定它是那次错误的**唯一根因** —— 只能说**流程缺陷 + 可能促因** ✓
⇒ 原则：**匿名只隐藏版本身份，不能隐藏评分规则** ✓
```
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "run_t4_batch.py"


def _source() -> str:
    return SRC.read_text(encoding="utf-8")


def test_packet_carries_scoring_rules():
    """包里必须有 `scoring_rules`（不是只有一句英文 instructions）。"""
    s = _source()
    assert '"scoring_rules"' in s, "匿名包未携带 scoring_rules ⇒ 计分规则又只留在 docs 里了"


def test_scoring_rules_include_direction_for_all_four_fields():
    """**方向必须逐字段写明** —— 这正是上一轮出错的地方 ✓"""
    s = _source()
    assert '"direction"' in s
    assert "越高越好（5 = 最好）" in s, "三个『越高越好』字段的方向未写明"
    assert "越高越差（1 = 最好）" in s, "泄漏字段的方向未写明"
    for f in ("correctness_1to5", "targeting_1to5", "actionability_1to5", "answer_leakage_1to5"):
        assert f in s, f"计分规则缺少字段 {f}"


def test_scoring_rules_include_the_other_five_required_items():
    """用户要求携带的六项：评分对象 / 方向 / 取值范围 / 锚点 / 泄题射程 / 判据版本。"""
    s = _source()
    assert '"unit_of_analysis"' in s, "缺『评分对象』"
    assert '"range"' in s, "缺『取值范围』"
    assert '"anchor"' in s, "缺『锚点』"
    assert '"leakage_scope"' in s, "缺『泄题射程』"
    assert '"criterion_version"' in s, "缺『判据版本』"


def test_leakage_scope_separates_hint_from_review_fields():
    """泄题射程必须写清：**只有 `teaching[].hint`**，事后复盘字段不在内 ✓"""
    s = _source()
    assert "只衡量即时提示" in s
    assert "not_measured" in s
    assert "核对时重建" in s, "缺『标注核对时重建者不计泄题』这一条判例"


def test_leakage_scope_warns_not_to_mix_with_ucc_scope():
    """⚠️ 两条判据射程不能混用（上一轮 A 正是混用了 ✗）。"""
    s = _source()
    assert "不管算不算泄题" in s, "未写明 §2.10（认知归因）判据不管泄题与否"


def test_criterion_version_is_recorded():
    """判据要有版本号，便于日后比对 ✓"""
    s = _source()
    assert "2026-10-11-v1" in s
