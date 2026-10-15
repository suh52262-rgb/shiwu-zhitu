"""P9 评测框架测试：用例文件结构校验 + Hint Policy 离线回归。

评测 case 的期望值是回归基线——修改 Prompt / 模型 / 工作流后
运行 scripts/run_eval.py 与本测试做双重校验。
"""
import importlib.util
from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import json
from pathlib import Path

from app.agent.hint_policy import HintPolicyInput, decide_hint_level

EVAL_DIR = EVALUATION
CASE_FILES = ["step_matching", "diagnosis", "hint", "retrieval", "graph",
              # 合成协议夹具，只测字段与身份；不把它当真实检索质量样本。
              "retrieval_golden", "alternative_path"]


def _load(name: str) -> dict:
    """优先 datasets/，旧的扁平位置兜底（与 run_eval.load_cases 口径一致）。"""
    for base in (EVAL_DIR / "datasets", EVAL_DIR):
        path = base / f"{name}_cases.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"找不到 {name}_cases.json")


def _load_run_eval():
    spec = importlib.util.spec_from_file_location(
        "run_eval", Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVAL_DIR = EVALUATION
    module.DATASETS_DIR = EVALUATION / "datasets"
    return module


def test_all_case_files_exist_and_parse():
    for name in CASE_FILES:
        data = _load(name)
        assert "cases" in data or "rounds" in data, f"{name} 缺少 cases/rounds"


def test_step_matching_cases_reference_existing_graph():
    data = _load("step_matching")
    from app.services.problem_store import load_problem_graph

    graph = load_problem_graph(data["problem_id"])
    assert graph is not None
    valid = graph.node_ids()
    for case in data["cases"]:
        for alt in case["expected"].get("touched_any_of", []):
            assert set(alt) <= valid, f"{case['id']} 引用了不存在的节点"
        for field in ("touched_contains", "attempted_contains"):
            for node in case["expected"].get(field, []):
                assert node in valid, f"{case['id']} 引用了不存在的节点 {node}"


def test_hint_cases_expected_levels_match_policy():
    """Hint Policy 是纯规则：评测集里的期望级别必须与策略实现一致（离线回归）。"""
    for case in _load("hint")["cases"]:
        decision = decide_hint_level(HintPolicyInput(**case["policy_input"]))
        assert decision.level == case["expected"]["level"], (
            f"{case['id']}：策略输出 L{decision.level}，评测集期望 "
            f"L{case['expected']['level']}——策略或用例需要更新"
        )


def test_run_eval_module_registry_complete():
    run_eval = _load_run_eval()
    assert set(run_eval.EVALUATORS) == set(CASE_FILES)
    assert run_eval.load_cases("hint")["metric"] == "hint_quality"
