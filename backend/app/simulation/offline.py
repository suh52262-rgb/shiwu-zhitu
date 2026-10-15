"""离线脚本化 LLM（P12 的 `--offline` 模式）。

用途：在没有真实模型 / 没有网络时跑通整条链路，做**冒烟与回归**——
验证模拟器本身、State 演化、Hint Policy 升级、Attempt 落库是否正常。

**它不代表模型准确率**：脚本化 LLM 直接按用例里的 `expected` 作答，
所以节点匹配的断言在离线模式下必然成立。真实准确率必须用真实 LLM 跑
（`scripts/run_simulation.py <case>`，不带 `--offline`）。

实现要点：按【学生当前输入】定位当前步，因此不依赖节点调用顺序
（正确推进的轮次会跳过 analyze/diagnose，用全局计数器会错位）。
"""
import re

from app.agent.schemas import Diagnosis, Hint, StepAnalysis
from app.agent.step_matcher import StepMatch
from app.simulation.cases import SimulationCase, SimulationStep


class ScriptedLLM:
    """按用例期望回放结构化输出的假 LLM。"""

    def __init__(self, case: SimulationCase) -> None:
        self.case = case

    # ---- 定位当前步 ----
    def _step_for(self, user_prompt: str) -> SimulationStep:
        """按学生输入定位当前步。

        必须**从后往前**找：Step Matcher / analyze 的提示词里带着
        【近期学习事件】与已完成的作答，历史输入也会出现在提示词中；
        正序查找会命中最早的那一步（表现为整条剧本错位一格）。
        当前输入一定是剧本里最后一个出现在提示词中的，故倒序取首个命中。
        """
        for step in reversed(self.case.steps):
            text = step.input.strip()
            if text and text in user_prompt:
                return step
        # 兜底：措辞微调时不至于整体崩掉
        return self.case.steps[-1]

    # ---- LLMClient 协议 ----
    def chat(self, system: str, user: str) -> str:
        return "[offline] 脚本化 LLM 不生成自由文本"

    def chat_structured(self, system: str, user: str, schema: type):
        step = self._step_for(user)
        expected = step.expected

        if schema is StepMatch:
            return StepMatch(
                matched_nodes=list(expected.nodes),
                attempted_nodes=list(expected.attempted),
                confidence=0.9,
                correct=bool(expected.nodes) and not expected.attempted and not expected.error,
                current_method="脚本化",
                missing_reasoning=(
                    [f"缺少关键推理（脚本化占位 {i + 1}）" for i in range(expected.missing_min)]
                ),
                off_graph=expected.off_graph,
                explanation="offline 脚本化输出",
            )

        if schema is StepAnalysis:
            return StepAnalysis(
                current_step=step.note or "脚本化步骤",
                method="脚本化",
                has_obvious_error=expected.error,
                concepts=[expected.concept] if expected.concept else [],
                is_complete=expected.complete,
                # P23：graphless 下 analyze_step 是缺失推理的唯一来源
                missing_reasoning=list(expected.missing),
            )

        if schema is Diagnosis:
            return Diagnosis(
                issue_type=expected.issue_type if expected.error else "none",
                issue=expected.issue,
                concept=expected.concept,
                severity="medium" if expected.error else "low",
                recommended_action="引导学生回到关键推理" if expected.error else "",
            )

        if schema is Hint:
            # 提示级别写在提示词里，回显出来便于观察逐轮升级
            level = re.search(r"hint_level=(\d+)", user)
            tag = f"L{level.group(1)}" if level else "L?"
            return Hint(hint=f"[offline {tag}] 针对「{step.input[:18]}」的脚本化提示")

        raise AssertionError(f"脚本化 LLM 收到意外的 schema：{schema}")

    def chat_structured_resilient(self, system: str, user: str, schema: type):
        """与真实 provider 行为一致（P20）：永不抛异常，失败时给降级结果。"""
        from app.services.llm.fallback import safe_default
        from app.services.llm.schemas import StructuredOutcome

        try:
            return StructuredOutcome(value=self.chat_structured(system, user, schema))
        except Exception as exc:  # noqa: BLE001
            return StructuredOutcome(
                value=safe_default(schema),
                degraded=True,
                reasons=[f"{type(exc).__name__}: {exc}"],
            )


def install_scripted_llm(case: SimulationCase):
    """把 graph / step_matcher 的 get_llm 换成脚本化实现，返回撤销函数。

    替换函数接受任意参数：P17 起节点会调用 `get_llm("step_matcher")` 这类带 role 的形式。
    """
    from app.agent import graph as graph_module
    from app.agent import step_matcher as step_matcher_module

    scripted = ScriptedLLM(case)
    originals = (
        (graph_module, graph_module.get_llm),
        (step_matcher_module, step_matcher_module.get_llm),
    )
    for module, _ in originals:
        module.get_llm = lambda *args, **kwargs: scripted

    def restore() -> None:
        for module, original in originals:
            module.get_llm = original

    return restore
