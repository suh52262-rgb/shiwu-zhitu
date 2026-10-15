"""LangGraph 工作流 State 定义。

直接复用 models.student.StudentState（Pydantic 模型）：
LangGraph 原生支持 Pydantic 模型作为 State schema，
节点只需返回需要更新的字段子集（dict），框架负责合并与校验。
"""
from app.models.student import StudentState, TimelineEvent

# 工作流 State 与对外数据模型保持同构，避免两套字段定义漂移
TutorState = StudentState

__all__ = ["TutorState", "StudentState", "TimelineEvent"]
