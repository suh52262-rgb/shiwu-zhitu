"""Agent 提示词的兼容入口（Phase 3 遗留，P13 已外置）。

P13 起提示词正文放在项目根的 `prompts/<角色>/v<版本>.md`，由
`app.agent.prompt_registry` 负责加载与版本管理——**改提示词请改 .md 文件，
不要改这里**。

本模块只保留旧的模块级常量，供尚未迁移的调用方使用；常量在导入时求值一次，
之后若切换版本（改 `DEFAULT_VERSIONS` 或设 `PROMPT_<角色>_VERSION`）不会自动刷新。
节点代码应改为调用 `get_prompt(role)`，见 `agent/graph.py` 与 `agent/step_matcher.py`。
"""
from app.agent.prompt_registry import get_prompt

ANALYZE_STEP_SYSTEM = get_prompt("analyze_step")
DIAGNOSE_SYSTEM = get_prompt("diagnosis")
GENERATE_HINT_SYSTEM = get_prompt("tutor")
STEP_MATCH_SYSTEM = get_prompt("step_matcher")

__all__ = [
    "ANALYZE_STEP_SYSTEM",
    "DIAGNOSE_SYSTEM",
    "GENERATE_HINT_SYSTEM",
    "STEP_MATCH_SYSTEM",
]
