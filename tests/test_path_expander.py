"""路径扩展（`expand_alternative_paths`）的测试。

## 这个模块存在的原因
P14 的提示词 v2 里**已经写明**"只把另一种解法写进 `alternative_methods` 是不够的…
不能替代支路节点"，**但实测仍不照做**（跑 `demo_03` 两次都是单路径）。

⇒ 单独一步，只让它干这一件事。

## 测什么
**重点不是"它能不能加支路"，是"它会不会把已经对的部分改坏"。**
所以硬不变量的测试比成功路径的测试更重要。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.models.problem_graph import GraphNode, ProblemGraph  # noqa: E402
from app.services import path_expander as pe  # noqa: E402


# ---------------------------------------------------------------- 夹具

def _demo_like() -> ProblemGraph:
    """仿 demo_03 的形状：线性、有 alternative_methods。"""
    return ProblemGraph(
        problem_id="t_001",
        nodes=[
            GraphNode(id="S1", name="规定正方向", type="key_reasoning",
                      description="d1"),
            GraphNode(id="S2", name="求冲量", type="modeling", description="d2"),
            GraphNode(id="S3", name="求平均冲力", type="result", description="d3"),
        ],
        edges=[["S1", "S2"], ["S2", "S3"]],
        valid_paths=[["S1", "S2", "S3"]],
        metadata={"alternative_methods": ["先用牛顿第二定律求 a，再求 F̄"]},
    )


class _FakeLLM:
    """假 provider：`chat_structured` 返回 `ExpansionOutput`。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_structured(self, system, user, schema):
        self.calls += 1
        return schema.model_validate(self.payload)


def _node(i, name="支路", type_="calculation"):
    return {"id": i, "name": name, "type": type_, "description": "d",
            "knowledge_points": [], "common_errors": []}


# ---------------------------------------------------------------- 成功路径

def test_adds_branch_and_keeps_everything(monkeypatch):
    """正常扩展：加支路、原有节点边路径全保留。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes] + [_node("S2b", "牛顿定律路线")],
        "edges": [list(e) for e in g.edges] + [["S1", "S2b"], ["S2b", "S3"]],
        "valid_paths": [["S1", "S2", "S3"], ["S1", "S2b", "S3"]],
        "rejected_methods": [],
        "notes": "加了牛顿定律支路",
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is True
    assert rep["added_nodes"] == ["S2b"]
    assert rep["added_paths"] == 1
    assert {n.id for n in out.nodes} == {"S1", "S2", "S3", "S2b"}
    assert {tuple(e) for e in g.edges} <= {tuple(e) for e in out.edges}
    assert ["S1", "S2", "S3"] in out.valid_paths


def test_result_is_still_a_draft(monkeypatch):
    """**机器补的支路必须仍标 needs_review，且能被认出来。**"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes] + [_node("S2b")],
        "edges": [list(e) for e in g.edges] + [["S1", "S2b"], ["S2b", "S3"]],
        "valid_paths": [["S1", "S2", "S3"], ["S1", "S2b", "S3"]],
    }
    out, _ = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert out.metadata["needs_review"] is True, "机器改完不能算复核过"
    assert out.metadata["expanded_by"] == "auto"
    assert "机器" in out.metadata["branch_kind"]
    assert "请人工确认" in out.metadata["note"]


# ---------------------------------------------------------------- 硬不变量（重点）

def test_rejects_when_existing_node_removed(monkeypatch):
    """**删了已有节点 → 整次拒绝。** 改坏已经对的部分比不做更糟。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes if n.id != "S2"] + [_node("S2b")],
        "edges": [["S1", "S2b"], ["S2b", "S3"]],
        "valid_paths": [["S1", "S2b", "S3"]],
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is False
    assert "已有节点被删除" in rep["reason"]
    assert out is g, "被拒绝时必须原样返回原图"


def test_rejects_when_existing_edge_removed(monkeypatch):
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes] + [_node("S2b")],
        "edges": [["S1", "S2b"], ["S2b", "S3"]],       # S1→S2、S2→S3 都没了
        "valid_paths": [["S1", "S2b", "S3"]],
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is False
    assert "已有边被删除" in rep["reason"]


def test_rejects_when_original_path_dropped(monkeypatch):
    """把原路径从 `valid_paths` 里挤掉 = 学生走标准解反而认不出来了。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes] + [_node("S2b")],
        "edges": [list(e) for e in g.edges] + [["S1", "S2b"], ["S2b", "S3"]],
        "valid_paths": [["S1", "S2b", "S3"]],          # 原路径没了
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is False
    assert "已有合法路径" in rep["reason"]


def test_rejects_dead_end_branch(monkeypatch):
    """支路必须有出口边汇合回去 —— 死路等于学生走进去了就出不来。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes] + [_node("S2b")],
        "edges": [list(e) for e in g.edges] + [["S1", "S2b"]],   # 没有 S2b→S3
        "valid_paths": [["S1", "S2", "S3"], ["S1", "S2b", "S3"]],
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is False
    assert "不存在的步" in rep["reason"]


def test_rejects_mismatched_entry_or_exit(monkeypatch):
    """每条路径必须同起同终 —— 否则"一条路径"这个概念就散了。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes] + [_node("S2b"), _node("X9")],
        "edges": [list(e) for e in g.edges] + [["S1", "S2b"], ["S2b", "X9"]],
        "valid_paths": [["S1", "S2", "S3"], ["S1", "S2b", "X9"]],
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is False
    assert "终点不一致" in rep["reason"]


def test_rejects_when_nothing_added(monkeypatch):
    """模型原样返回（没加节点）→ 如实报告"没加成"，不是"成功"。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes],
        "edges": [list(e) for e in g.edges],
        "valid_paths": [["S1", "S2", "S3"]],
    }
    out, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["ok"] is False
    assert "没有新增任何节点" in rep["reason"]


# ---------------------------------------------------------------- 健壮性

def test_rejected_methods_are_recorded(monkeypatch):
    """模型判为物理上不成立的解法要留痕 —— 那是有价值的信息。"""
    g = _demo_like()
    payload = {
        "nodes": [n.model_dump() for n in g.nodes],
        "edges": [list(e) for e in g.edges],
        "valid_paths": [["S1", "S2", "S3"]],
        "rejected_methods": [{"method": "能量法", "reason": "碰撞瞬间机械能不守恒"}],
    }
    _, rep = pe.expand_alternative_paths(g, llm=_FakeLLM(payload))
    assert rep["rejected_methods"][0]["method"] == "能量法"


def test_llm_failure_returns_original(monkeypatch):
    """**模型挂了不该让调用方拿不到图。**"""
    class Boom:
        def chat_structured(self, *a, **k):
            raise RuntimeError("网络断了")

    g = _demo_like()
    out, rep = pe.expand_alternative_paths(g, llm=Boom())
    assert rep["ok"] is False and out is g
    assert "模型调用或解析失败" in rep["reason"]


def test_bad_json_returns_original(monkeypatch):
    class Bad:
        def chat_structured(self, *a, **k):
            raise ValueError("输出不可解析")

    g = _demo_like()
    out, rep = pe.expand_alternative_paths(g, llm=Bad())
    assert rep["ok"] is False and out is g
    assert "模型调用或解析失败" in rep["reason"]


def test_no_alternatives_raises():
    """没有替代解法就没有可扩展的东西 —— 这是调用方的错，该抛。"""
    g = _demo_like()
    g.metadata["alternative_methods"] = []
    with pytest.raises(pe.ExpansionError):
        pe.expand_alternative_paths(g, llm=_FakeLLM({}))


def test_prompt_file_exists():
    from pathlib import Path
    p = Path(str(Path(__file__).resolve().parents[1] / 'prompts/graph_expander/v1.md'))
    assert p.is_file(), "提示词文件不存在"
    t = p.read_text(encoding="utf-8")
    for key in ("不许改动已有的节点与边", "不能是死路", "宁缺勿滥"):
        assert key in t, f"提示词缺少关键约束：{key}"
