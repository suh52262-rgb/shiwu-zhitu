"""Auto Problem-Solving Graph Generator 测试（P14）。

任务书 P14 的核心诉求：**导入一道新题时不必人工从零建 Graph**。
因此这里最重要的一条是最后的端到端用例——"新题 → 自动生成候选图 → 保存 →
Agent 直接以 graph 模式跑起来"，全程没有为这道题手写过 Graph。

同时守住任务书的两条边界：
- 候选图**不默认正确**（落盘带 `needs_review=true`）；
- 只做**结构性**归一化保证"Agent 可用"，物理正确性留给 P15 Validator。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent.prompt_registry import PROMPT_ROLES, prompt_version_fields
from app.main import app
from app.models.candidate_graph import CandidateGraph
from app.models.problem_graph import ProblemGraph
from app.services.graph_generator import (
    build_user_prompt,
    generate_and_save,
    generate_candidate_graph,
    normalize_candidate_graph,
)
from app.services.problem_repository import ProblemRepository
from fakes import CANDIDATE_GRAPH, FakeLLM, install_fake_llm

client = TestClient(app)


def _candidate(**overrides) -> CandidateGraph:
    """以样例图为底，按需覆盖字段。"""
    payload = CANDIDATE_GRAPH.model_dump()
    payload.update(overrides)
    return CandidateGraph.model_validate(payload)


@pytest.fixture
def temp_repo(tmp_path, monkeypatch):
    """指向临时题库的仓储，并让默认仓储也跟着走（API 层才会读同一份）。"""
    monkeypatch.setenv("PROBLEMS_DIR", str(tmp_path))
    return ProblemRepository(tmp_path)


def write_problem(root, problem_id: str, **extra) -> None:
    payload = {
        "problem_id": problem_id,
        "title": extra.pop("title", f"题目 {problem_id}"),
        "statement": extra.pop("statement", "题干正文"),
        **extra,
    }
    target = root / problem_id / "problem.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- 输入组装

def test_user_prompt_carries_statement_solution_and_course():
    prompt = build_user_prompt(
        "物体从斜面滑下", "由动能定理 mgh + W_f = 1/2mv²", figures=["fig1.png"], course="大学物理"
    )
    assert "物体从斜面滑下" in prompt
    assert "mgh + W_f" in prompt
    assert "大学物理" in prompt
    assert "fig1.png" in prompt


def test_user_prompt_notes_missing_reference_solution():
    prompt = build_user_prompt("只有题面")
    assert "未提供" in prompt


def test_generate_rejects_empty_statement():
    with pytest.raises(ValueError, match="problem_statement"):
        generate_candidate_graph("   ", llm=FakeLLM())


# ---------------------------------------------------------------- 生成与来源

def test_generate_uses_llm_and_records_provenance():
    llm = FakeLLM(model="demo-model", provider="deepseek")

    candidate = generate_candidate_graph(
        "物体从斜面滑下", "参考答案", problem_id="phy_x", llm=llm
    )

    assert candidate.problem_id == "phy_x"
    assert candidate.generation.source == "auto"
    assert candidate.generation.needs_review is True     # 候选图不默认正确
    assert candidate.generation.model == "demo-model"
    assert candidate.generation.provider == "deepseek"
    assert candidate.generation.generated_at
    assert llm.calls == [CandidateGraph]                 # 确实请求了候选图 schema


def test_generate_sends_the_registered_prompt():
    llm = FakeLLM()
    generate_candidate_graph("题面", llm=llm)

    system, _ = llm.prompts[-1]
    assert "Solution Graph" in system
    assert "edges" in system


# ---------------------------------------------------------------- 归一化

def test_normalize_drops_edges_to_unknown_nodes():
    candidate = _candidate(edges=[["S1", "S2"], ["S2", "S9"], ["S8", "S3"]])

    result = normalize_candidate_graph(candidate)

    assert result.edges == [["S1", "S2"]]
    assert any("不存在节点" in note for note in result.generation.normalizations)


def test_normalize_drops_self_loops_and_duplicate_edges():
    candidate = _candidate(edges=[["S1", "S1"], ["S1", "S2"], ["S1", "S2"]])

    result = normalize_candidate_graph(candidate)

    assert result.edges == [["S1", "S2"]]


def test_normalize_renumbers_invalid_and_duplicate_node_ids():
    candidate = _candidate(
        nodes=[
            {"id": "S1", "name": "第一步"},
            {"id": "S1", "name": "重复的 id"},
            {"id": "有空格 的 id", "name": "非法 id"},
        ],
        edges=[],
        valid_paths=[],
    )

    result = normalize_candidate_graph(candidate)

    ids = [node.id for node in result.nodes]
    assert len(set(ids)) == len(ids), "归一化后 id 必须唯一"
    assert all(node.id.replace("_", "").isalnum() for node in result.nodes)
    assert any("重复" in note or "非法" in note for note in result.generation.normalizations)


def test_normalize_falls_back_unknown_node_type():
    candidate = _candidate(nodes=[{"id": "S1", "name": "x", "type": "不存在的类型"}], edges=[])

    result = normalize_candidate_graph(candidate)

    assert result.nodes[0].type == "modeling"
    assert any("type" in note for note in result.generation.normalizations)


def test_normalize_gives_every_node_a_name():
    candidate = _candidate(nodes=[{"id": "S1", "name": "  "}], edges=[])

    result = normalize_candidate_graph(candidate)

    assert result.nodes[0].name == "步骤 1"


def test_normalize_drops_invalid_paths():
    candidate = _candidate(
        valid_paths=[["S1", "S9"], ["S1"], ["S1", "S2", "S3"]],
    )

    result = normalize_candidate_graph(candidate)

    assert result.valid_paths == [["S1", "S2", "S3"]]


def test_normalize_builds_main_axis_when_no_edges():
    """模型没给边时，必须补出可用的主轴，否则下游"无路可走"。"""
    candidate = _candidate(edges=[], valid_paths=[])

    result = normalize_candidate_graph(candidate)

    assert result.edges == [["S1", "S2"], ["S2", "S3"]]
    assert result.valid_paths == [["S1", "S2", "S3"]]
    assert result.terminal_nodes() == {"S3"}


def test_normalize_builds_longest_chain_when_no_paths():
    candidate = _candidate(valid_paths=[])

    result = normalize_candidate_graph(candidate)

    assert result.valid_paths and result.valid_paths[0][0] == "S1"
    assert result.valid_paths[0][-1] == "S3"


def test_normalize_truncates_oversized_graph():
    from app.services.graph_generator import MAX_NODES

    nodes = [{"id": f"S{i}", "name": f"步骤 {i}"} for i in range(1, MAX_NODES + 5)]
    candidate = _candidate(nodes=nodes, edges=[], valid_paths=[])

    result = normalize_candidate_graph(candidate)

    assert len(result.nodes) == MAX_NODES
    assert any("上限" in note for note in result.generation.normalizations)


def test_normalize_merges_knowledge_points_from_nodes():
    candidate = _candidate(knowledge_points=[])

    result = normalize_candidate_graph(candidate)

    assert result.knowledge_points == ["受力分析", "动能定理", "摩擦力做功"]


def test_normalize_is_idempotent_on_a_clean_graph():
    """干净的图不该被改动，也不该产生噪声提示。"""
    result = normalize_candidate_graph(_candidate())

    assert result.generation.normalizations == []


# ---------------------------------------------------------------- 转成 Agent 用的图

def test_to_problem_graph_carries_provenance_and_alternative_methods():
    candidate = normalize_candidate_graph(_candidate())

    graph = candidate.to_problem_graph()

    assert isinstance(graph, ProblemGraph)
    assert graph.metadata["source"] == "auto"
    assert graph.metadata["needs_review"] is True
    assert graph.metadata["alternative_methods"] == ["也可用牛顿第二定律结合运动学公式分段求解"]
    assert graph.node_ids() == {"S1", "S2", "S3"}
    assert graph.terminal_nodes() == {"S3"}


def test_to_problem_graph_renders_for_prompt():
    """生成的图必须能被 Step Matcher 的提示词渲染器直接消费。"""
    rendered = normalize_candidate_graph(_candidate()).to_problem_graph().render_for_prompt()

    assert "S1" in rendered and "S3" in rendered
    assert "推进关系" in rendered
    assert "完整路径1" in rendered


# ---------------------------------------------------------------- 落盘

def test_generate_and_save_writes_graph_into_repository(temp_repo, tmp_path):
    write_problem(tmp_path, "phy_auto")

    candidate, graph = generate_and_save(
        problem_id="phy_auto",
        problem_statement="物体从粗糙斜面滑下，求摩擦力做功。",
        reference_solution="由动能定理……",
        llm=FakeLLM(),
        repository=temp_repo,
    )

    assert temp_repo.has_graph("phy_auto") is True
    stored = temp_repo.get_graph("phy_auto")
    assert stored.node_ids() == {"S1", "S2", "S3"}
    assert stored.metadata["source"] == "auto"
    assert stored.metadata["needs_review"] is True
    assert graph.problem_id == "phy_auto"
    assert candidate.generation.source == "auto"


# ---------------------------------------------------------------- 提示词注册

def test_graph_generator_prompt_is_registered():
    assert "graph_generator" in PROMPT_ROLES
    fields = prompt_version_fields()
    # v2：要求把一题多解画成图上的分叉、典型错误少而准
    # （P14 对照成员 B 人工图评估后做的改进）
    assert fields["graph_generator_prompt_version"] == "v3"


# ---------------------------------------------------------------- P14 验收：端到端

def test_acceptance_new_problem_generated_graph_runs_in_graph_mode(temp_repo, tmp_path, monkeypatch):
    """P14 验收：新题 → 自动生成候选图 → 保存 → Agent 以 graph 模式跑通。

    全程没有为这道题手写过 Graph：题面由测试给定，图完全来自"生成→归一化→落盘"。
    """
    problem_id = "phy_auto_001"
    write_problem(
        tmp_path,
        problem_id,
        title="粗糙斜面 · 摩擦力做功",
        statement="质量为 m 的物体从高 h 的粗糙斜面顶端由静止滑下，到底端速度为 v，求摩擦力做的功。",
    )

    # 1) 生成候选图并落盘（LLM 用假件，不联网）
    candidate, _ = generate_and_save(
        problem_id=problem_id,
        problem_statement="质量为 m 的物体从高 h 的粗糙斜面顶端由静止滑下，求摩擦力做的功。",
        reference_solution="mgh + W_f = 1/2·m·v²",
        llm=FakeLLM(),
        repository=temp_repo,
    )
    assert temp_repo.has_graph(problem_id) is True

    # 2) 自动候选仍为草稿；作者仓储可读，教学入口须显式人工复核发布。
    assert client.get(f"/problems/{problem_id}/graph").status_code == 404
    assert temp_repo.get_graph(problem_id).metadata["needs_review"] is True
    temp_repo.mark_reviewed(problem_id, reviewed_by="test-human-reviewer")
    temp_repo.publish_graph(problem_id, published_by="test-human-reviewer")
    resp = client.get(f"/problems/{problem_id}/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["nodes"]) == 3
    assert body["metadata"]["source"] == "auto"
    assert body["graph_status"] == "published"

    # 3) Agent 以 graph 模式跑：让假 LLM 命中"生成图里真实存在的节点"
    target_node = candidate.nodes[1].id          # S2「判断功能关系」
    install_fake_llm(monkeypatch, step_match=_match_on(target_node))

    analyzed = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_auto", "problem_id": problem_id, "answer": "有摩擦力做功，应该用动能定理"},
    ).json()

    assert analyzed["graph_mode"] is True, "自动生成的图没有被识别为 graph 模式"
    assert analyzed["step_match"]["matched_nodes"] == [target_node]
    assert analyzed["visited_nodes"] == [target_node]
    assert analyzed["current_node"] == target_node


def _match_on(node_id: str):
    from app.agent.step_matcher import StepMatch

    return StepMatch(matched_nodes=[node_id], confidence=0.9, correct=True)
