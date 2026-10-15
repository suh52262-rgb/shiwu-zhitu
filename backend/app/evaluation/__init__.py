"""P18 评测：数据集、指标、报告三层。

三层各自的职责：
  · `evaluation/datasets/`  —— 用例（数据，不含逻辑）
  · `app/evaluation/`       —— 指标实现（纯函数，可测、可被 P24 复用）
  · `evaluation/reports/`   —— 落盘的评测报告

**为什么指标不放 scripts/**：脚本只能从命令行用一次；
指标还要被测试、被教师端复用、被报告持久化调用。
"""
from app.evaluation.metrics import (  # noqa: F401
    METRIC_SPECS,
    Rate,
    alternative_path_acceptance,
    diagnosis_accuracy,
    false_diagnosis_rate,
    hit_rate,
    mrr,
    node_recall,
    structure_valid_rate,
)

__all__ = [
    "METRIC_SPECS",
    "Rate",
    "alternative_path_acceptance",
    "diagnosis_accuracy",
    "false_diagnosis_rate",
    "hit_rate",
    "mrr",
    "node_recall",
    "structure_valid_rate",
]
