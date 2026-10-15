"""模拟用例的数据结构与加载（P12）。

用例文件放在项目根的 `simulation/`（任务书 P12 指定目录），一类学生一个 JSON：

    spring_good_student.json        正常学生：沿标准路径完成
    spring_missing_reasoning.json   跳步学生：结论对但缺关键推理
    spring_concept_error.json       概念错误学生：在典型节点出错后修正
    spring_stuck_student.json       卡顿学生：反复修改、长时间 idle、多次请求提示
    spring_alternative_path.json    非标准合理解法：走简谐运动法（S6b）分支
    graphless_basic.json            无图题目（P23）：证明 Graphless Mode 不是占位

前五个用例都在 phy_002（有 Graph）上跑，只有 `graphless_basic` 在 phy_001（无 Graph）上跑——
无图路径的判断是**阶段级**的，没有节点可断言，所以它的期望值主要落在
错误判定、缺失推理与提示级别上。

每个 step 描述"模拟学生说什么"以及"期望 Agent 得出什么结论"。期望值同时用于两处：
真实 LLM 模式下作为断言；`--offline` 模式下作为脚本化 LLM 的输出（见 offline.py）。
"""
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMULATION_DIR = PROJECT_ROOT / "simulation"


class StepExpectation(BaseModel):
    """单步期望：既是对真实运行的断言，也是离线模式下的脚本输入。"""

    #: 期望学生本轮**碰到**的节点（对应上 ∪ 尝试过），**不区分对错**。
    #:
    #: ⚠️ 这里原来是"期望**正确完成**的节点"——**注释是错的**，
    #: 实际用例全按"碰到就算"写（例：某一轮 `error=True`、`missing_min=1`
    #: ——学生只给结论未论证——却仍期望 `nodes=[S5]`）。
    #: 断言也曾按 `matched_nodes` 做，比用例意图更严，
    #: 于是出现"教学语义正确但断言失败"。见 docs/排查-节点轨迹三种口径.md。
    #:
    #: 要"正确完成"口径请用 `completed_nodes`；两者都在
    #: `app/agent/node_semantics.py` 里有唯一定义。
    nodes: list[str] = Field(default_factory=list)
    #: 显式口径：期望**正确建立**的节点（= `state.visited_nodes` 口径）
    completed_nodes: list[str] = Field(default_factory=list)
    #: 显式口径：期望**碰到**的节点（与 `nodes` 等价，新用例建议用这个）
    touched_nodes: list[str] = Field(default_factory=list)
    attempted: list[str] = Field(default_factory=list)    # 期望"尝试但未正确完成"的节点
    missing_min: int = 0                                  # 期望指出至少几条缺失推理
    #: P23：graphless 剧本里缺失推理的具体内容（无 Step Matcher，只能脚本给出）
    missing: list[str] = Field(default_factory=list)
    off_graph: bool = False                               # 期望判定为与本题无关
    error: bool = False                                   # 期望诊断出明显错误
    complete: bool = False                                # 期望判定整题完成
    issue_type: str = "none"                              # 期望的错误类型
    issue: str = ""                                       # 期望的错误描述（离线模式用作诊断输出）
    concept: str = ""                                     # 期望的薄弱概念
    hint_level: int | None = None                         # 精确期望（Hint Policy 为确定性规则）
    hint_min_level: int | None = None
    hint_max_level: int | None = None


class SimulationStep(BaseModel):
    """一轮学生输入（对应一次 POST /attempt/analyze）。"""

    input: str
    idle_time: float = 0.0          # 距上次有效操作秒数（驱动卡顿检测）
    rewrite_count: int = 0          # 本轮前的改写次数
    request_hint: bool = False      # 学生主动点击"请求提示"
    repeat: int = 1                 # 同一输入重复提交几次（模拟反复改不动）
    note: str = ""                  # 剧本说明，仅供报告展示
    expected: StepExpectation = Field(default_factory=StepExpectation)


class SimulationCase(BaseModel):
    """一类学生的完整多轮剧本。"""

    case_id: str
    title: str = ""
    profile: str = ""               # 这一类学生的特征描述
    notes: str = ""
    problem_id: str = "phy_002"
    student_id: str = ""            # 留空则由 case_id 推导
    steps: list[SimulationStep] = Field(default_factory=list)
    expected_summary: dict = Field(default_factory=dict)

    @property
    def simulator_student_id(self) -> str:
        return self.student_id or f"sim_{self.case_id}"


def list_cases() -> list[str]:
    """返回全部可用用例 id（按文件名排序，便于 CLI 展示）。"""
    if not SIMULATION_DIR.is_dir():
        return []
    return sorted(path.stem for path in SIMULATION_DIR.glob("*.json"))


def case_path(case_id: str) -> Path:
    return SIMULATION_DIR / f"{case_id}.json"


def load_case(case_id: str) -> SimulationCase:
    """按 id 加载用例；不存在时抛出带可选列表的 ValueError。"""
    path = case_path(case_id)
    if not path.is_file():
        available = "、".join(list_cases()) or "（无）"
        raise ValueError(f"未找到模拟用例 {case_id}（{path}）。可用用例：{available}")
    return SimulationCase.model_validate_json(path.read_text(encoding="utf-8"))
