"""诊断措辞的**纸面口径**（契约 §2.10）必须有测试钉住。

## 背景
```
v6/v14/v15 的「隐藏历史结论复核」把 `teaching[].issue` 的
「误认为…」「未认识到…」型措辞判为**无依据的认知归因** ✗
⇒ 用户裁决 (c) 并冻结判据（契约 §2.10）：
   **错误关系、运算或结论可以支持对本轮作答的诊断；
     即使材料还有中间步骤，也不能因此把学生的理解状态、信念、动机
     或未展示的推导过程当成已知事实。**
⇒ 这组测试保证**提示词与字段说明都带着这条口径**，
  不会在后续改提示词时被悄悄改掉 ✗
```
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.prompt_registry import DEFAULT_VERSIONS, get_prompt  # noqa: E402
from app.agent.schemas import Diagnosis  # noqa: E402


def test_diagnosis_default_is_v3():
    """默认必须是 v3（带纸面口径的那一版）。"""
    assert DEFAULT_VERSIONS["diagnosis"] == "v4"


def test_diagnosis_prompt_carries_the_frozen_rule():
    """提示词里必须**逐字**带着判据与四条示例的要点。"""
    p = get_prompt("diagnosis")
    for key in (
        "错误关系、运算或结论可以支持",          # 判据前半
        "学生的理解状态、信念、动机",            # 判据后半（禁止对象）
        "不能因此把学生的理解状态",              # 读法锁定句
        "依据不足",                              # 示例 3
        "无依据的认知归因",                      # 示例 4
        "只描述作答错误就能达成同样的诊断目的",  # 硬约束②：不必推断心理
    ):
        assert key in p, f"诊断提示词缺少口径要点：{key}"


def test_issue_field_description_states_the_rule():
    """字段说明也要写明（改结构的人不必去读提示词）。"""
    desc = Diagnosis.model_fields["issue"].description
    assert "纸上写了什么" in desc
    assert "不得" in desc and "理解状态" in desc
    assert "误认为" in desc and "未认识到" in desc      # 明确点出禁用写法


def test_prompt_states_the_applicability_condition():
    """⚠️ 示例的**适用条件**必须在提示词里（用户 2026-10-11 冻结前检查②）。

    ```
    只有学生**确实写出**对应运算/关系时，才能作出示例里的纸面判断 ✓
    **不能由错误终值反推**他用了什么关系，更不能反推他怎么想 ✗
    ```
    """
    p = get_prompt("diagnosis")
    for key in ("不可省略", "只有学生确实写出了那个运算或关系时才成立",
                "不得由终值反推", "该终值与题设条件不符"):
        assert key in p, f"提示词缺少适用条件要点：{key}"


def test_v2_is_still_available_for_ab():
    """⚠️ 旧版必须保留（可切回做 A/B）—— 不许原地改旧版。"""
    assert (ROOT / "prompts" / "diagnosis" / "v2.md").is_file()
    assert (ROOT / "prompts" / "diagnosis" / "v3.md").is_file()
    # v3 是**独立的一份**，不是靠 v2 拼出来的（否则改 v2 会连带改掉 v3 ✗）
    v3 = (ROOT / "prompts" / "diagnosis" / "v3.md").read_text(encoding="utf-8")
    assert "### 反面例子" in v3, "v3 应自带 v2 的 `concept` 节（独立完整）"
