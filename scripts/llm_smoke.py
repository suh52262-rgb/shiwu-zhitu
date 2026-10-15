"""LLM 统一接口真实调用冒烟测试（Phase 2）。

前置条件：项目根目录 .env 中已配置对应 provider 的 API Key。
运行：.venv/Scripts/python scripts/llm_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pydantic import BaseModel  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.llm import get_llm  # noqa: E402


class StepAnalysis(BaseModel):
    """演示用结构化输出 Schema（Phase 3 将扩展为完整的 analyze 输出）。"""

    current_step: str
    has_obvious_error: bool
    concepts: list[str]


def main() -> None:
    settings = get_settings()
    print(f"provider={settings.llm_provider}  model={settings.llm_model or '(provider 默认)'}")

    llm = get_llm()

    print("\n== 1. 普通对话 chat() ==")
    reply = llm.chat(
        "你是一个大学物理助教，回答保持简洁。",
        "用一句话说明机械能守恒定律的条件。",
    )
    print(reply)

    print("\n== 2. 结构化输出 chat_structured() ==")
    result = llm.chat_structured(
        "你是大学物理助教，负责分析学生的解题步骤。",
        "题目：物体从粗糙斜面顶端由静止滑下，判断下滑过程中机械能是否守恒。"
        "学生作答：mgh = 1/2·mv²，所以机械能守恒。",
        StepAnalysis,
    )
    print(result.model_dump_json(indent=2, ensure_ascii=False))

    print("\nSMOKE_OK")


if __name__ == "__main__":
    main()
