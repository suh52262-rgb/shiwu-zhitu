"""题目存取 problem_store 测试。"""
from app.services.problem_store import load_problem, load_problem_content


def test_load_demo_problem():
    problem = load_problem("phy_001")
    assert problem is not None
    assert problem.problem_id == "phy_001"
    assert "机械能" in problem.title
    assert problem.knowledge_points


def test_load_missing_problem_returns_none():
    assert load_problem("no_such_problem") is None
    assert load_problem_content("no_such_problem") == ""


def test_load_problem_content_includes_title_and_content():
    text = load_problem_content("phy_001")
    assert text.startswith("斜面下滑")
    assert "粗糙斜面" in text
