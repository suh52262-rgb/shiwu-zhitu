"""多轮学生学习模拟器（P12）。

对外入口：
- `list_cases()` / `load_case()`：读取 `simulation/*.json` 剧本
- `run_case()`：驱动完整教学闭环，产出逐轮结果与汇总
- `offline` 子模块：`--offline` 用的脚本化 LLM

用法见 `scripts/run_simulation.py`。
"""
from app.simulation.cases import (
    SIMULATION_DIR,
    SimulationCase,
    SimulationStep,
    StepExpectation,
    list_cases,
    load_case,
)
from app.simulation.runner import (
    SimulationReport,
    StepResult,
    print_step,
    print_summary,
    run_case,
)

__all__ = [
    "SIMULATION_DIR",
    "SimulationCase",
    "SimulationStep",
    "StepExpectation",
    "SimulationReport",
    "StepResult",
    "list_cases",
    "load_case",
    "run_case",
    "print_step",
    "print_summary",
]
