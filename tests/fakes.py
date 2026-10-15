"""测试共享夹具：假 LLM 与 canned 输出。

FakeLLM 替换 app.agent.graph / app.agent.step_matcher 中的 get_llm 后，
工作流、Step Matcher 与 API 测试均不发起网络请求。
"""
from app.agent.schemas import Diagnosis, Hint, StepAnalysis
from app.agent.step_matcher import StepMatch
from app.models.candidate_graph import CandidateGraph
from app.services.llm.schemas import PhysicsCheck, ProviderInfo

GOOD_ANALYSIS = StepAnalysis(
    current_step="检验结果",
    method="动能定理",
    has_obvious_error=False,
    concepts=["动能定理", "摩擦力做功"],
    is_complete=True,
)
BAD_ANALYSIS = StepAnalysis(
    current_step="建立能量方程",
    method="机械能守恒",
    has_obvious_error=True,
    concepts=["机械能守恒条件"],
    is_complete=False,
)
ERROR_DIAGNOSIS = Diagnosis(
    issue_type="concept",
    issue="未考虑摩擦力做功",
    concept="机械能守恒条件",
    severity="high",
    recommended_action="引导学生检查斜面是否粗糙",
)
HINT_TEXT = "先想想机械能守恒需要满足什么条件？"
STEP_MATCH_S5 = StepMatch(
    matched_nodes=["S5"],
    confidence=0.9,
    correct=True,
    missing_reasoning=["缺少分离条件 N=0 的判断", "缺少对 Q 的受力分析"],
    explanation="给出了分离位置结论但跳过了推理链",
)
# P14：候选解题图（形状与 graph_generator 提示词的要求一致）
CANDIDATE_GRAPH = CandidateGraph(
    problem_id="phy_fake",
    knowledge_points=["动能定理", "摩擦力做功"],
    nodes=[
        {
            "id": "S1", "name": "识别研究对象", "type": "modeling",
            "description": "物体沿粗糙斜面下滑，受重力、支持力、摩擦力。",
            "knowledge_points": ["受力分析"],
        },
        {
            "id": "S2", "name": "判断功能关系", "type": "key_reasoning",
            "description": "有摩擦力做功，机械能不守恒，改用动能定理。",
            "knowledge_points": ["动能定理"],
            "common_errors": ["E1：误用机械能守恒"],
        },
        {
            "id": "S3", "name": "解出摩擦力做功", "type": "result",
            "description": "W_f = 1/2·m·v² − mgh。",
            "knowledge_points": ["摩擦力做功"],
        },
    ],
    edges=[["S1", "S2"], ["S2", "S3"]],
    valid_paths=[["S1", "S2", "S3"]],
    alternative_methods=["也可用牛顿第二定律结合运动学公式分段求解"],
)


class FakeLLM:
    """按 schema 类型返回 canned 结果的假 LLM。"""

    def __init__(
        self,
        analysis: StepAnalysis = BAD_ANALYSIS,
        diagnosis: Diagnosis = ERROR_DIAGNOSIS,
        hint: Hint | None = None,
        step_match: StepMatch | None = None,
        candidate_graph: CandidateGraph | None = None,
        physics_check: PhysicsCheck | None = None,
        model: str = "fake-model",
        provider: str = "fake",
    ) -> None:
        self.analysis = analysis
        self.diagnosis = diagnosis
        self.hint = hint or Hint(hint=HINT_TEXT)
        self.step_match = step_match or STEP_MATCH_S5
        self.candidate_graph = candidate_graph or CANDIDATE_GRAPH
        self.physics_check = physics_check or PhysicsCheck()
        self.model = model
        self.name = provider
        self.calls: list[type] = []
        self.prompts: list[tuple[str, str]] = []  # (system, user) 记录，供断言

    def describe(self) -> ProviderInfo:
        """与真实 provider 对齐，便于断言生成器记录了模型/厂商来源。"""
        return ProviderInfo(name=self.name, kind="fake", model=self.model)

    def chat(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return "fake-text"

    def chat_structured(self, system: str, user: str, schema: type):
        self.calls.append(schema)
        self.prompts.append((system, user))
        # 每次都返回**深拷贝**：canned 对象是模块级共享的，若直接返回同一个实例，
        # 调用方一旦就地修改（Step Matcher 的后校验曾如此），就会把夹具永久改坏，
        # 后续所有用例都拿到被污染的值。
        if schema is StepAnalysis:
            return self.analysis.model_copy(deep=True)
        if schema is Diagnosis:
            return self.diagnosis.model_copy(deep=True)
        if schema is Hint:
            return self.hint.model_copy(deep=True)
        if schema is StepMatch:
            return self.step_match.model_copy(deep=True)
        if schema is CandidateGraph:
            return self.candidate_graph.model_copy(deep=True)
        if schema is PhysicsCheck:
            return self.physics_check.model_copy(deep=True)
        raise AssertionError(f"意外的 schema: {schema}")

    def chat_structured_resilient(self, system: str, user: str, schema: type):
        """与真实 provider 行为一致（P20）：永不抛异常，失败时给降级结果。

        夹具必须实现这条路径，否则测不出节点的降级行为——
        而"节点降级时不伪造判断"正是 P20 的核心，不能只靠真机验证。
        """
        from app.services.llm.fallback import safe_default
        from app.services.llm.schemas import StructuredOutcome

        try:
            return StructuredOutcome(value=self.chat_structured(system, user, schema))
        except Exception as exc:  # noqa: BLE001 与生产实现保持同样的兜底语义
            return StructuredOutcome(
                value=safe_default(schema),
                degraded=True,
                reasons=[f"{type(exc).__name__}: {exc}"],
            )


def install_fake_llm(monkeypatch, **kwargs) -> FakeLLM:
    """把 graph 与 step_matcher 模块的 get_llm 都替换为返回 FakeLLM 的工厂。

    接受任意参数：P17 起节点会调用 `get_llm("step_matcher")` 这类带 role 的形式。
    """
    from app.agent import graph as graph_module
    from app.agent import step_matcher as step_matcher_module

    fake = FakeLLM(**kwargs)
    monkeypatch.setattr(graph_module, "get_llm", lambda *args, **kw: fake)
    monkeypatch.setattr(step_matcher_module, "get_llm", lambda *args, **kw: fake)
    # **检索重排也调模型**（默认开启，实测 +5.3% p=0.0009）。
    # 不 patch 它，测试就会**打真 API**——既慢又会产生真实 token 用量。
    try:
        from app.rag import rerank as rerank_module

        monkeypatch.setattr(rerank_module, "get_llm", lambda *args, **kw: fake)
    except Exception:  # noqa: BLE001 模块不在时不阻断
        pass
    return fake
