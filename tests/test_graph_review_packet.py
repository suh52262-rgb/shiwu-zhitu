import json

from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.problem_repository import ProblemRepository
from app.services.graph_versions import graph_hash
from scripts.prepare_graph_review import packet, main
import pytest


def test_review_packet_preserves_full_text_and_never_signs_or_publishes(tmp_path):
    repo = ProblemRepository(tmp_path / "problems")
    folder = repo.problem_dir("p")
    folder.mkdir(parents=True)
    (folder / "problem.json").write_text(json.dumps({"problem_id": "p", "statement": "求位移", "answer": "1 m"}), encoding="utf-8")
    long_text = "完整推理与适用条件" * 30 + "不可截断的尾部"
    graph = ProblemGraph(problem_id="p", nodes=[GraphNode(id="S1", name="推导", description=long_text)], valid_paths=[["S1"]])
    repo.save_graph("p", graph)
    content, result = packet(repo, "p")
    assert long_text in content
    assert result["graph_hash"] == graph_hash(graph)
    assert result["reviewer"] is None and result["review_status"] == "pending"
    assert repo.graph_release("p")["graph_status"] == "draft"
    out = tmp_path / "packet"
    assert main(["p", "--root", str(repo.root), "--output", str(out)]) == 0
    (out / "p.md").write_text("teacher notes", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["p", "--root", str(repo.root), "--output", str(out)])
    assert (out / "p.md").read_text(encoding="utf-8") == "teacher notes"
