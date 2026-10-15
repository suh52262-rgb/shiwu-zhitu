"""Agent 节点的结构化输出 Schema（Phase 3）。

所有 LLM 输出都必须符合这里的 Pydantic Schema，
由 services.llm.chat_structured 强制约束并校验。
"""
from typing import Literal

from pydantic import BaseModel, Field

# 错误归因类型（对应开发文档第四节 diagnose 节点）
IssueType = Literal[
    "concept",        # 物理概念/规律理解错误
    "modeling",       # 物理建模/过程分析错误
    "method",         # 解题方法/思路选择不当
    "calculation",    # 数学运算/代入错误
    "verification",   # 缺少检验或检验方式错误
    "none",           # 无明显错误
]

Severity = Literal["low", "medium", "high"]


class StepAnalysis(BaseModel):
    """analyze_step 节点输出：学生当前作答的结构化分析。"""

    current_step: str = Field(description="学生当前进行到的解题步骤")
    method: str = Field(default="", description="学生采用的主要解题方法，未知则为空")
    has_obvious_error: bool = Field(description="是否存在明显物理错误")
    concepts: list[str] = Field(default_factory=list, description="涉及的知识点")
    is_complete: bool = Field(default=False, description="是否已完成整道题的解答")
    #: P23：结论正确但缺少关键推理时列出。有 Graph 时以 Step Matcher 的节点级结论为准
    #: （它更精确），无 Graph 时只能由本节点给出——否则 graphless 下这个字段永远为空。
    missing_reasoning: list[str] = Field(
        default_factory=list, description="结论正确但缺失的关键推理，最多 3 条；推理完整则留空"
    )
    recommended_next_step: str = Field(
        default="", description="建议学生下一步做什么（一句话，面向学生）"
    )


class Diagnosis(BaseModel):
    """diagnose 节点输出：错误归因。"""

    issue_type: IssueType = "none"
    issue: str = Field(default="", description=(
        "错误的一句话描述（面向教师），无错误则为空。"
        "⚠️ **只描述这份作答里纸上写了什么**（契约 §2.10，2026-10-10 用户裁决）："
        "可以指出错误的关系/运算/结论，**不得**断言学生的理解状态、信念、动机"
        "或未展示的推导过程（如「误认为…」「未认识到…」）。"
        "写法示例：❌『误认为摩擦力方向与拉力同向』→ ✅『把摩擦力与拉力相加得合力 14 N，与题设方向不符』"))
    concept: str = Field(default="", description="出错涉及的核心概念，无错误则为空")
    severity: Severity = "low"
    recommended_action: str = Field(default="", description="给辅导系统的一句话建议")
    finding_kind: Literal['observed_error', 'missing_information', 'uncertain', 'none'] = 'observed_error'
    missing_fields: list[str] = Field(default_factory=list, description=(
        '遗漏项只能从服务端result_fields选择key；present=true不得报遗漏。'
        'missing_information的文字由服务端生成，issue不得自由叙述遗漏。'))


class Hint(BaseModel):
    """generate_hint 节点输出：一条分级提示。"""

    hint: str = Field(description="给学生的提示文本")
