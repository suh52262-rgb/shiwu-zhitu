"""ProblemRepository 多题管理测试（P16）。

核心是最后一条验收用例：**新增一道题只需要放 problem.json + graph.json，
不改任何 LangGraph 核心代码**——它在临时题库里造一道全新的题，然后驱动真实的
`/attempt/analyze` 跑完一整轮，全程没有为这道题写过一行代码。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agent.step_matcher import StepMatch
from app.models.problem_graph import GraphNode, ProblemGraph
from fakes import install_fake_llm

from app.services.problem_repository import (
    DEFAULT_PROBLEMS_DIR,
    ProblemDataError,
    ProblemRepository,
    problems_dir,
)

client = TestClient(app)


@pytest.fixture
def temp_repo(tmp_path, monkeypatch):
    """指向临时题库目录的仓储，并让默认仓储也跟着走（API 层才会用同一份）。"""
    monkeypatch.setenv("PROBLEMS_DIR", str(tmp_path))
    return ProblemRepository(tmp_path)


def write_problem(root, problem_id: str, **extra) -> None:
    payload = {
        "problem_id": problem_id,
        "title": extra.pop("title", f"题目 {problem_id}"),
        "statement": extra.pop("statement", "题干正文"),
        "knowledge_points": extra.pop("knowledge_points", ["测试知识点"]),
        **extra,
    }
    target = root / problem_id / "problem.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


def write_graph(root, problem_id: str, nodes=None, edges=None) -> None:
    payload = {
        "problem_id": problem_id,
        "nodes": nodes
        or [
            {"id": "S1", "name": "建模", "type": "modeling"},
            {"id": "S2", "name": "结论", "type": "result"},
        ],
        "edges": edges or [["S1", "S2"]],
    }
    target = root / problem_id / "graph.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


# ---------- 目录约定 ----------

def test_default_root_is_data_problems():
    assert DEFAULT_PROBLEMS_DIR.name == "problems"
    assert DEFAULT_PROBLEMS_DIR.parent.name == "data"


def test_problems_dir_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PROBLEMS_DIR", str(tmp_path))
    assert problems_dir() == tmp_path
    assert ProblemRepository().root == tmp_path


def test_explicit_root_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PROBLEMS_DIR", str(tmp_path / "from_env"))
    assert ProblemRepository(tmp_path / "explicit").root == tmp_path / "explicit"


# ---------- 读取真实题库（新布局） ----------

def test_list_problem_ids_reads_new_layout():
    """题库是**数据**，测试不该硬编码它的完整清单。

    项目的核心承诺是"新增一题 = 建目录放 problem.json，零代码改动"；
    原先这里断言 `== ["phy_001", "phy_002"]`，于是每加一道题都要改测试——
    与承诺自相矛盾（P14 用自动生成的图导入 phy_003 时就被绊了一次）。
    改为断言"包含已知题目 + 结果有序 + 格式合法"。
    """
    ids = ProblemRepository().list_problem_ids()

    assert {"phy_001", "phy_002"} <= set(ids), "内置题目应当始终在题库里"
    assert ids == sorted(ids), "题库列表应当有序（便于前端稳定展示）"
    # ⚠️ 这里原来断言 `all(pid.startswith("phy_"))` —— **那不是一个真需求**，
    # 只是当时恰好成立。录入 demo 例题（`demo_02`）时它就成了假约束。
    #
    # **换成真正有意义的性质**：id 必须是安全的目录名。
    # 读取端按 `root / problem_id` 找目录，所以带路径分隔符或 `..` 的 id
    # 会指到题库之外——那才是这条断言该守的东西。
    for pid in ids:
        assert pid and pid == pid.strip(), f"id 有空白的首尾：{pid!r}"
        assert "/" not in pid and "\\" not in pid, f"id 含路径分隔符：{pid!r}"
        assert pid not in (".", "..") and not pid.startswith("."), \
            f"id 会被当成相对路径：{pid!r}"


def test_get_problem_from_new_layout():
    problem = ProblemRepository().get_problem("phy_002")
    assert problem is not None
    assert problem.problem_id == "phy_002"
    assert problem.title
    assert problem.statement
    assert problem.figures == ["figures/phy_002_fig1.png"]


def test_get_problem_unknown_returns_none():
    assert ProblemRepository().get_problem("no_such_problem") is None


def test_get_problem_content_includes_title():
    content = ProblemRepository().get_problem_content("phy_001")
    assert "机械能守恒" in content
    assert ProblemRepository().get_problem_content("no_such") == ""


def test_graph_and_has_graph():
    repo = ProblemRepository()
    assert repo.has_graph("phy_002") is True
    assert repo.has_graph("phy_001") is False        # graphless 题目
    assert repo.get_graph("phy_001") is None
    assert repo.get_graph("phy_002").problem_id == "phy_002"


def test_list_problems_returns_all():
    """同上：断言"内置题目都在且每道都能解析"，而不是穷举题号。"""
    problems = ProblemRepository().list_problems()
    by_id = {p.problem_id: p for p in problems}

    assert {"phy_001", "phy_002"} <= set(by_id)
    for problem in problems:
        assert problem.title, f"{problem.problem_id} 缺少标题"
        assert problem.statement, f"{problem.problem_id} 缺少题面"


# ---------- 损坏数据的行为 ----------

def test_problem_json_with_utf8_bom_is_readable(temp_repo, tmp_path):
    """回归：带 UTF-8 BOM 的 problem.json 必须能读。

    题库文件是给人手写的，而 Windows 记事本 / VS Code / PowerShell 保存 JSON 时
    经常带 BOM。用纯 utf-8 读会抛 "Unexpected UTF-8 BOM"，
    而手工加题的人根本不知道该怎么修——这与"新增一题只需放个文件"的承诺直接冲突。
    """
    target = tmp_path / "bom_problem" / "problem.json"
    target.parent.mkdir(parents=True)
    payload = '{"problem_id":"bom_problem","title":"带 BOM 的题","content":"题面"}'
    target.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))

    problem = temp_repo.get_problem("bom_problem")

    assert problem.title == "带 BOM 的题"
    assert problem.statement == "题面"


def test_graph_json_with_utf8_bom_is_readable(temp_repo, tmp_path):
    target = tmp_path / "bom_graph" / "problem.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"problem_id":"bom_graph","title":"t","content":"c"}', encoding="utf-8")
    graph = tmp_path / "bom_graph" / "graph.json"
    graph.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps(
            {
                "problem_id": "bom_graph",
                "nodes": [{"id": "S1", "name": "第一步"}],
                "edges": [],
                "valid_paths": [],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    loaded = temp_repo.get_graph("bom_graph")

    assert loaded is not None and loaded.node_ids() == {"S1"}


def test_malformed_problem_raises_with_path(temp_repo, tmp_path):
    target = tmp_path / "broken" / "problem.json"
    target.parent.mkdir(parents=True)
    target.write_text("{ 这不是合法 JSON", encoding="utf-8", newline="\n")

    with pytest.raises(ProblemDataError, match="broken"):
        temp_repo.get_problem("broken")


def test_list_problems_skips_malformed_entry(temp_repo, tmp_path):
    """一个坏文件不应让整个题库消失。"""
    write_problem(tmp_path, "good_one")
    (tmp_path / "bad_one").mkdir()
    (tmp_path / "bad_one" / "problem.json").write_text("{坏", encoding="utf-8", newline="\n")

    assert temp_repo.list_problem_ids() == ["bad_one", "good_one"]
    assert [p.problem_id for p in temp_repo.list_problems()] == ["good_one"]


def test_graph_problem_id_mismatch_is_corrected(temp_repo, tmp_path, caplog):
    write_problem(tmp_path, "mine")
    write_graph(tmp_path, "mine")
    (tmp_path / "mine" / "graph.json").write_text(
        json.dumps({"problem_id": "someone_else", "nodes": [], "edges": []}),
        encoding="utf-8",
        newline="\n",
    )

    assert temp_repo.get_graph("mine").problem_id == "mine"


# ---------- 写入（P14 Auto Graph Generator 的出口） ----------

def test_save_graph_roundtrip(temp_repo, tmp_path):
    write_problem(tmp_path, "fresh")
    graph = ProblemGraph(
        problem_id="fresh",
        nodes=[GraphNode(id="S1", name="建模")],
        edges=[],
    )

    path = temp_repo.save_graph("fresh", graph)

    assert path == tmp_path / "fresh" / "graph.json"
    assert temp_repo.has_graph("fresh") is True
    assert temp_repo.get_graph("fresh").nodes[0].name == "建模"


def test_save_graph_creates_directory_and_leaves_no_temp_file(temp_repo, tmp_path):
    temp_repo.save_graph("brand_new", ProblemGraph(problem_id="brand_new", nodes=[], edges=[]))

    assert (tmp_path / "brand_new" / "graph.json").is_file()
    assert list((tmp_path / "brand_new").glob("*.tmp")) == []


def test_save_graph_rejects_mismatched_problem_id(temp_repo):
    with pytest.raises(ProblemDataError, match="不一致"):
        temp_repo.save_graph("target", ProblemGraph(problem_id="other"))


def test_delete_graph_switches_problem_to_graphless(temp_repo, tmp_path):
    write_problem(tmp_path, "with_graph")
    write_graph(tmp_path, "with_graph")
    assert temp_repo.has_graph("with_graph") is True

    assert temp_repo.delete_graph("with_graph") is True
    assert temp_repo.has_graph("with_graph") is False
    assert temp_repo.delete_graph("with_graph") is False   # 再删返回 False


# ---------- 插图路径安全 ----------

def test_figure_path_resolves_existing_figure():
    path = ProblemRepository().figure_path("phy_002", "figures/phy_002_fig1.png")
    assert path is not None and path.is_file()


def test_figure_path_rejects_traversal_and_missing():
    repo = ProblemRepository()
    assert repo.figure_path("phy_002", "../../phy_001/problem.json") is None
    assert repo.figure_path("phy_002", "figures/not_here.png") is None


# ---------- P16 验收：新增题目不需要改代码 ----------

def test_acceptance_adding_a_problem_needs_no_code_change(temp_repo, tmp_path, monkeypatch):
    """P16 验收标准：只加 problem.json + graph.json 即可运行，无需改 LangGraph 核心代码。

    这里造一道全新的题（一道简单的匀加速直线运动题），然后走真实的
    `/attempt/analyze`：Step Matcher 用假 LLM 打桩（离线），其余全部是真代码。
    """
    problem_id = "phy_new_001"
    write_problem(
        tmp_path,
        problem_id,
        title="匀加速直线运动",
        statement="物体由静止开始做匀加速直线运动，加速度 a，求 t 时刻速度。",
        knowledge_points=["匀变速直线运动"],
    )
    write_graph(
        tmp_path,
        problem_id,
        nodes=[
            {"id": "S1", "name": "识别已知量", "type": "modeling"},
            {"id": "S2", "name": "选用速度公式", "type": "key_reasoning"},
            {"id": "S3", "name": "得出速度", "type": "result"},
        ],
        edges=[["S1", "S2"], ["S2", "S3"]],
    )

    # 只替换 LLM（离线）；题目与 Graph 完全来自临时目录，没有任何代码改动。
    # 关键：让假 LLM 返回**这道新题图里真实存在的节点**（S2），
    # 否则 Step Matcher 的后校验会把不存在的 S5 过滤掉，
    # `step_match is not None` 这种断言照样通过，却完全没验证"新题的 Graph 参与了匹配"。
    install_fake_llm(
        monkeypatch,
        step_match=StepMatch(matched_nodes=["S2"], confidence=0.9, correct=True),
    )

    listing = client.get("/problems").json()
    assert problem_id in [item["problem_id"] for item in listing["items"]]

    detail = client.get(f"/problems/{problem_id}").json()
    assert detail["title"] == "匀加速直线运动"
    assert detail["has_graph"] is True
    assert "answer" not in detail        # 参考答案不下发给前端

    graph = client.get(f"/problems/{problem_id}/graph")
    assert graph.status_code == 200
    assert len(graph.json()["nodes"]) == 3

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_new", "problem_id": problem_id, "answer": "先看已知量"},
    ).json()

    assert body["graph_mode"] is True                    # 自动识别为 graph 模式
    assert body["attempt_id"]
    match = body["step_match"]
    assert match["matched_nodes"] == ["S2"], "新题的节点没有被用于匹配"
    assert match["off_graph"] is False                   # S2 在图中，不该被判为无关输入
    assert body["visited_nodes"] == ["S2"]               # 状态确实按新题的图推进了
    assert body["current_node"] == "S2"


def test_acceptance_problem_without_graph_runs_graphless(temp_repo, tmp_path, monkeypatch):
    """只放 problem.json（不给 Graph）也应能跑，自动走 graphless 模式。"""
    write_problem(tmp_path, "phy_new_002", title="只有题目", statement="随便一道题")
    install_fake_llm(monkeypatch)

    assert client.get("/problems/phy_new_002/graph").status_code == 404

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_new2", "problem_id": "phy_new_002", "answer": "我的第一步"},
    ).json()

    assert body["graph_mode"] is False
    assert body["attempt_id"]
