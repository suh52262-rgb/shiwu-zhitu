"""Graph Validator 测试（P15）。

任务书要求四类检查：结构合法性 / 解题流程完整性 / 物理一致性 / 多路径一致性。

**这个文件最重要的价值是"证明校验器真的会报错"。**
一个永远返回 valid=true 的校验器等于没有校验器——所以每类检查都先用
**故意做坏的图**验证它确实被抓住，再用真实题库的图确认不会误报。
"""
import pytest

from app.models.graph_validation import (
    ACTION_ACCEPT,
    ACTION_REGENERATE,
    ACTION_TEACHER_REVIEW,
    CHECK_ISSUES,
    CHECK_PASS,
    CHECK_SKIPPED,
    ISSUE_FLOW,
    ISSUE_PATH,
    ISSUE_PHYSICS,
    ISSUE_STRUCTURE,
    SEVERITY_ERROR,
)
from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.graph_validator import validate_graph
from app.services.llm.schemas import PhysicsCheck, PhysicsFinding
from app.services.problem_repository import ProblemRepository
from fakes import FakeLLM

# ---------------------------------------------------------------- 构造工具


def node(node_id: str, name: str = "", type: str = "modeling", description: str = "说明") -> GraphNode:
    return GraphNode(id=node_id, name=name or f"步骤 {node_id}", type=type, description=description)


def graph(nodes, edges=None, paths=None) -> ProblemGraph:
    return ProblemGraph(
        problem_id="t",
        nodes=list(nodes),
        edges=[list(e) for e in (edges or [])],
        valid_paths=[list(p) for p in (paths or [])],
    )


def chain(*ids) -> ProblemGraph:
    """一条干净的链：S1→S2→…，含完整路径。"""
    nodes = [node(i, type=t) for i, t in zip(ids, ["modeling", "key_reasoning", "calculation", "result"])]
    edges = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    return graph(nodes, edges, [list(ids)])


@pytest.fixture
def problem():
    class P:
        full_text = "物体从粗糙斜面滑下，求摩擦力做的功。"
        answer = "由动能定理 mgh + W_f = 1/2mv^2"

    return P()


# ---------------------------------------------------------------- 1. 结构合法性

def test_clean_chain_passes_structure(problem):
    report = validate_graph(chain("S1", "S2", "S3", "S4"), problem=problem, llm=FakeLLM())

    assert report.checks["structure"] == CHECK_PASS
    assert report.valid is True
    assert report.recommended_action == ACTION_ACCEPT


def test_duplicate_node_id_is_an_error(problem):
    g = graph([node("S1"), node("S1"), node("S2")], [("S1", "S2")], [["S1", "S2"]])

    report = validate_graph(g, problem=problem, llm=FakeLLM())

    assert report.checks["structure"] == CHECK_ISSUES
    assert report.valid is False
    assert any("重复" in i.description for i in report.by_type(ISSUE_STRUCTURE))


def test_dangling_edge_is_an_error():
    g = graph([node("S1"), node("S2")], [("S1", "S2"), ("S2", "S99")], [["S1", "S2"]])

    report = validate_graph(g, llm=FakeLLM())

    assert report.checks["structure"] == CHECK_ISSUES
    assert any("S99" in i.description for i in report.by_type(ISSUE_STRUCTURE))


def test_self_loop_is_an_error():
    g = graph([node("S1"), node("S2")], [("S1", "S1"), ("S1", "S2")], [["S1", "S2"]])

    report = validate_graph(g, llm=FakeLLM())

    assert any("自环" in i.description for i in report.by_type(ISSUE_STRUCTURE))


def test_cycle_is_detected():
    """S1→S2→S3→S1：解题步骤不该绕回去。"""
    g = graph(
        [node("S1"), node("S2"), node("S3")],
        [("S1", "S2"), ("S2", "S3"), ("S3", "S1")],
    )

    report = validate_graph(g, llm=FakeLLM())

    issues = report.by_type(ISSUE_STRUCTURE)
    assert any("循环" in i.description for i in issues)
    assert report.checks["structure"] == CHECK_ISSUES


def test_missing_terminal_is_an_error():
    """每个节点都有后继 => 学生永远做不完。"""
    g = graph([node("S1"), node("S2")], [("S1", "S2"), ("S2", "S1")])

    report = validate_graph(g, llm=FakeLLM())

    descriptions = " ".join(i.description for i in report.by_type(ISSUE_STRUCTURE))
    assert "终点" in descriptions


def test_isolated_node_is_a_warning():
    g = graph([node("S1"), node("S2"), node("S3")], [("S1", "S2")], [["S1", "S2"]])

    report = validate_graph(g, llm=FakeLLM())

    isolated = [i for i in report.by_type(ISSUE_STRUCTURE) if "孤立" in i.description]
    assert isolated and isolated[0].severity != SEVERITY_ERROR
    # 孤立点不致命，图仍算可用
    assert report.valid is True
    assert report.recommended_action == ACTION_TEACHER_REVIEW


def test_empty_graph_is_an_error():
    report = validate_graph(graph([]), llm=FakeLLM())

    assert report.valid is False
    assert any("没有任何节点" in i.description for i in report.issues)


def test_malformed_edge_does_not_crash():
    """畸形边（长度不对）不能让校验器抛异常——它自己就是来兜异常输入的。"""
    g = ProblemGraph.model_construct(problem_id="t", nodes=[node("S1")], edges=[["S1"], ["S1", "S2", "S3"]])

    report = validate_graph(g, llm=FakeLLM())

    assert any("形状非法" in i.description for i in report.by_type(ISSUE_STRUCTURE))


def test_node_without_description_is_warned():
    g = chain("S1", "S2")
    g.nodes[1].description = ""

    report = validate_graph(g, llm=FakeLLM())

    assert any("没有描述" in i.description for i in report.by_type(ISSUE_STRUCTURE))


# ---------------------------------------------------------------- 2. 流程完整性

def test_only_calculation_nodes_flagged_as_incomplete():
    """全是计算步骤 => 无法判断学生是否理解物理过程。"""
    g = graph(
        [node("S1", type="calculation"), node("S2", type="calculation")],
        [("S1", "S2")],
        [["S1", "S2"]],
    )

    report = validate_graph(g, llm=FakeLLM())

    descriptions = " ".join(i.description for i in report.by_type(ISSUE_FLOW))
    assert "modeling / key_reasoning" in descriptions
    # ⚠️ **原来是 `assert "result" in descriptions`** —— 断言的是旧文案。
    # 判据已从"节点 type 是不是 result"改成"**流程上有没有收尾**"
    # （终点节点有没有体现结果/结论/检验）。理由见 `_check_flow` 里的注释：
    # P14 几乎从不把节点标成 result，旧判据 5/5 全是误报。
    # **这里没放宽，只是换了判据** —— 所以断言新的措辞。
    assert "收尾" in descriptions, descriptions
    assert report.checks["flow"] == CHECK_ISSUES


def test_missing_result_node_is_flagged():
    """**图里没有收尾步骤** → 报。

    注意这个用例的终点是 `S2`（calculation，名字叫 S2），
    **名字和描述里都没有结果/结论/检验**，所以按新判据仍然该报。
    """
    g = graph(
        [node("S1", type="modeling"), node("S2", type="calculation")],
        [("S1", "S2")],
        [["S1", "S2"]],
    )

    report = validate_graph(g, llm=FakeLLM())

    # 旧断言是 `"result" in i.description`（断言文案）。
    # 新判据下措辞变了，但**这条用例该报的性质没变**。
    assert any("收尾" in i.description for i in report.by_type(ISSUE_FLOW))


def test_result_typed_named_terminal_is_not_flagged():
    """**终点只要体现了收尾，就不该报** —— 这正是修掉的那个误报。

    实测：`S10 结果检验与物理合理性判断` 被旧判据误报为"缺少 result 节点"，
    而它 type 是 calculation、名字里有"结果检验"。
    """
    g = graph(
        [node("S1", type="modeling"), node("S2", type="calculation")],
        [("S1", "S2")],
        [["S1", "S2"]],
    )
    g.nodes[-1].name = "结果检验与物理合理性判断"

    report = validate_graph(g, llm=FakeLLM())
    assert not any("收尾" in i.description for i in report.by_type(ISSUE_FLOW)), \
        "终点名字里明明有「结果检验」，不该报缺收尾"


def test_missing_key_reasoning_is_flagged():
    g = graph(
        [node("S1", type="modeling"), node("S2", type="result")],
        [("S1", "S2")],
        [["S1", "S2"]],
    )

    report = validate_graph(g, llm=FakeLLM())

    assert any("key_reasoning" in i.description for i in report.by_type(ISSUE_FLOW))


# ---------------------------------------------------------------- 3. 多路径一致性

def test_missing_valid_paths_is_warned():
    g = graph([node("S1"), node("S2")], [("S1", "S2")], [])

    report = validate_graph(g, llm=FakeLLM())

    assert any("valid_paths" in i.description for i in report.by_type(ISSUE_PATH))


def test_path_through_unknown_node_is_an_error():
    g = graph([node("S1"), node("S2")], [("S1", "S2")], [["S1", "S9"]])

    report = validate_graph(g, llm=FakeLLM())

    issues = report.by_type(ISSUE_PATH)
    assert any(i.severity == SEVERITY_ERROR and "S9" in i.description for i in issues)
    assert report.checks["paths"] == CHECK_ISSUES


def test_path_with_edge_not_in_edges_is_an_error():
    """路径里有 S1→S3，但 edges 里没有这条边 —— 路径不可追踪。"""
    g = graph([node("S1"), node("S2"), node("S3")], [("S1", "S2"), ("S2", "S3")], [["S1", "S3"]])

    report = validate_graph(g, llm=FakeLLM())

    assert any("不存在" in i.description for i in report.by_type(ISSUE_PATH))


def test_multiple_paths_not_converging_is_warned():
    """两条路径终点不同 —— 不同解法应当汇合到同一结论。"""
    g = graph(
        [node("S1"), node("S2"), node("S3"), node("S4")],
        [("S1", "S2"), ("S1", "S3"), ("S2", "S4"), ("S3", "S4"), ("S4", "S9x"[:2])],
        [["S1", "S2", "S4"], ["S1", "S3"]],
    )
    g.edges = [["S1", "S2"], ["S1", "S3"], ["S2", "S4"]]

    report = validate_graph(g, llm=FakeLLM())

    descriptions = " ".join(i.description for i in report.by_type(ISSUE_PATH))
    assert "终点不一致" in descriptions or "起点不一致" in descriptions


def test_forked_paths_that_converge_are_clean():
    """P-Q 弹簧题那种结构：S1→S2 分叉成 S3a/S3b，再汇合到 S4。"""
    g = graph(
        [
            node("S1", type="modeling"),
            node("S2", type="key_reasoning"),
            node("S3a", type="calculation"),
            node("S3b", type="calculation"),
            node("S4", type="result"),
        ],
        [("S1", "S2"), ("S2", "S3a"), ("S2", "S3b"), ("S3a", "S4"), ("S3b", "S4")],
        [["S1", "S2", "S3a", "S4"], ["S1", "S2", "S3b", "S4"]],
    )

    report = validate_graph(g, llm=FakeLLM())

    assert report.by_type(ISSUE_PATH) == []
    assert report.valid is True


# ---------------------------------------------------------------- 4. 物理一致性（LLM）

def test_physics_findings_are_surfaced(problem):
    """任务书举的例子：题目有摩擦力，图里却说机械能守恒。"""
    llm = FakeLLM(
        physics_check=PhysicsCheck(
            consistent=False,
            findings=[
                PhysicsFinding(
                    node="S4",
                    severity="error",
                    description="题目明确有摩擦力，该节点却直接使用机械能守恒",
                )
            ],
        )
    )

    report = validate_graph(chain("S1", "S2", "S3", "S4"), problem=problem, llm=llm)

    physics = report.by_type(ISSUE_PHYSICS)
    assert len(physics) == 1
    assert physics[0].node == "S4"
    assert physics[0].severity == SEVERITY_ERROR
    assert physics[0].source == "llm"
    assert report.checks["physics"] == CHECK_ISSUES
    assert report.valid is False


def test_physics_inconsistency_without_findings_is_still_reported(problem):
    """模型说"不成立"却没给位置时，不能让这个判断被静默吞掉。"""
    llm = FakeLLM(physics_check=PhysicsCheck(consistent=False, findings=[]))

    report = validate_graph(chain("S1", "S2"), problem=problem, llm=llm)

    assert report.checks["physics"] == CHECK_ISSUES
    assert any("未说明具体位置" in i.description for i in report.by_type(ISSUE_PHYSICS))


def test_physics_issues_get_lower_severity_when_warning(problem):
    llm = FakeLLM(
        physics_check=PhysicsCheck(
            consistent=False,
            findings=[PhysicsFinding(node="S2", severity="warning", description="未交代合外力为零")],
        )
    )

    report = validate_graph(chain("S1", "S2"), problem=problem, llm=llm)

    assert report.by_type(ISSUE_PHYSICS)[0].severity != SEVERITY_ERROR
    assert report.recommended_action == ACTION_TEACHER_REVIEW


def test_physical_check_skipped_when_no_problem():
    """没有题面就没法判断物理一致性——必须如实标 skipped，而不是假装查过。"""
    report = validate_graph(chain("S1", "S2"), problem=None, llm=FakeLLM())

    assert report.checks["physics"] == CHECK_SKIPPED
    assert "未检查" in report.summary


def test_llm_failure_degrades_to_rules_only(problem):
    """模型不可用时校验器不能跟着崩——降级为仅规则检查并压低信心。"""

    class Boom(FakeLLM):
        def chat_structured(self, system, user, schema):
            raise RuntimeError("模型不可用")

    report = validate_graph(chain("S1", "S2"), problem=problem, llm=Boom())

    assert report.checks["physics"] == CHECK_SKIPPED
    assert report.checks["structure"] == CHECK_PASS      # 规则检查照常
    assert report.valid is True
    assert report.confidence < 0.8, "最难的一类没查，不该装作很有信心"


def test_use_llm_false_skips_physics_without_calling_model(problem):
    llm = FakeLLM()

    report = validate_graph(chain("S1", "S2"), problem=problem, llm=llm, use_llm=False)

    assert report.checks["physics"] == CHECK_SKIPPED
    assert llm.calls == [], "关掉 LLM 后不该再调用模型"


# ---------------------------------------------------------------- 报告语义

def test_structural_error_recommends_regenerate():
    g = graph([node("S1"), node("S1")], [])

    report = validate_graph(g, llm=FakeLLM())

    assert report.recommended_action == ACTION_REGENERATE


def test_clean_graph_has_full_confidence(problem):
    report = validate_graph(chain("S1", "S2", "S3", "S4"), problem=problem, llm=FakeLLM())

    assert report.confidence == 1.0


def test_confidence_drops_with_issues(problem):
    llm = FakeLLM(
        physics_check=PhysicsCheck(
            consistent=False,
            findings=[PhysicsFinding(node="S2", severity="error", description="规律误用")],
        )
    )

    report = validate_graph(chain("S1", "S2", "S3", "S4"), problem=problem, llm=llm)

    assert report.confidence < 1.0


def test_as_metadata_is_compact(problem):
    """写进 graph.json 的结论要短——metadata 会被反复读写。"""
    report = validate_graph(chain("S1", "S2"), problem=problem, llm=FakeLLM())

    meta = report.as_metadata()

    assert meta["valid"] is True
    assert "issues" not in meta          # 只留摘要，不留完整列表
    assert set(meta) >= {"valid", "confidence", "recommended_action", "checks"}


# ---------------------------------------------------------------- 真实题库（防误报）

@pytest.mark.parametrize("problem_id", ["phy_002", "phy_003"])
def test_real_graphs_pass_rule_checks(problem_id):
    """人工图（phy_002）与自动生成的图（phy_003）都不该被规则检查误报。"""
    repo = ProblemRepository()
    g = repo.get_graph(problem_id)
    if g is None:
        pytest.skip(f"{problem_id} 没有 graph.json")

    report = validate_graph(g, llm=FakeLLM(), use_llm=False)

    assert report.valid is True, [i.description for i in report.errors()]
    assert report.by_type(ISSUE_STRUCTURE) == []
