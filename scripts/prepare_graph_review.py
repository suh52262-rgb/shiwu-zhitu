"""Create an unsigned, full-text teacher review packet without publishing graphs."""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.graph_validator import validate_graph
from app.services.graph_versions import graph_content, graph_hash
from app.services.problem_repository import ProblemRepository, _is_safe_id


def packet(repo, problem_id, baseline_dir=None):
    problem = repo.get_problem(problem_id)
    graph = repo.get_graph(problem_id)
    if problem is None or graph is None:
        raise ValueError("Review requires an existing problem and candidate graph")
    version = graph_hash(graph)
    report = validate_graph(graph, problem=problem, use_llm=False)
    text = [f"# 教师复核材料：{problem_id}", "", f"候选内容哈希：`{version}`", "",
            "状态：待实际复核人确认；本文件由工具生成，不是人工复核或发布记录。",
            "demo_03/04 是已知问题回归材料，不作为盲测样本。", "",
            "复核人：________　日期：________　结论：通过 / 需修改 / 无法判断", "",
            "## 题目与参考答案", "", problem.full_text, "", problem.answer or "（无参考答案）", "",
            "## 逐路径核对", "",
            "请核对目标量是否齐全、每步适用条件、方向/单位、公共汇合是否适用于本路线。",
            "无法确认的内容请保留为未确认，结构通过不能替代物理复核。", ""]
    nodes = graph.node_map()
    for index, path in enumerate(graph.valid_paths, 1):
        text.extend([f"### 路线 {index}：{' → '.join(path)}", ""])
        for node_id in path:
            node = nodes.get(node_id)
            if node is None:
                text.extend([f"- [ ] 缺失节点 {node_id}", ""])
                continue
            text.extend([f"- [ ] `{node.id}` {node.name}", "", node.description, "",
                         "  复核意见：________", ""])
    text.extend(["## 机器辅助检查（未调用物理审校模型）", "", report.summary, ""])
    for issue in report.issues:
        text.extend([f"- {issue.severity} / {issue.type} / {issue.node}：{issue.description}"])
    if baseline_dir:
        before_path = Path(baseline_dir) / f"{problem_id}_before.json"
        if before_path.exists():
            before = json.loads(before_path.read_text(encoding="utf-8"))
            lines = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).splitlines()
            diff = "\n".join(difflib.unified_diff(lines(before), lines(graph.model_dump()), fromfile="before", tofile=version))
            text.extend(["", "## 修订差异", "", "```diff", diff, "```", ""])
    text.extend(["", "## 确认与发布", "",
                 "只有实际复核人确认逐路径内容后才运行 review/publish；操作前再次核对候选哈希。",
                 "若候选内容变化，本材料应重新生成，不能沿用旧签字。", ""])
    return "\n".join(text), {"problem_id": problem_id, "graph_hash": version,
                            "review_status": "pending", "reviewer": None,
                            "physics_verified": False, "checks": report.checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_ids", nargs="+")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args(argv)
    repo = ProblemRepository(args.root)
    targets = []
    for pid in args.problem_ids:
        if not _is_safe_id(pid):
            parser.error("problem_id must be one directory name")
        text, summary = packet(repo, pid, args.baseline_dir)
        targets.extend([(args.output / f"{pid}.md", text),
                        (args.output / f"{pid}.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")])
    if any(path.exists() for path, _ in targets):
        parser.error("Output already exists; choose a new directory to preserve completed review notes")
    args.output.mkdir(parents=True, exist_ok=True)
    for path, content in targets:
        path.write_text(content, encoding="utf-8", newline="\n")
        print(path)
    return 0


if __name__ == "__main__":
    main()
