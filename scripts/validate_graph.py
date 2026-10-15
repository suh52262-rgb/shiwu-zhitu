"""P15 Graph Validator 命令行入口。

用法：
    # 校验某道题的图（真实模型做物理一致性检查）
    .venv/Scripts/python scripts/validate_graph.py phy_003

    # 只跑规则检查（不调模型，快且离线可用）
    .venv/Scripts/python scripts/validate_graph.py phy_003 --rules-only

    # 校验并把结论写进 graph.json 的 metadata.validation
    .venv/Scripts/python scripts/validate_graph.py phy_003 --write

    # 人工确认：清除 needs_review（这是**人**的动作，校验器不会替你做）
    .venv/Scripts/python scripts/validate_graph.py phy_003 --accept --reviewed-by 张三
"""
import argparse
import json
import sys
from pathlib import Path

# 控制台默认 GBK，输出里有 ✓/✗/⚠ 等字符，先重配编码
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.models.graph_validation import (  # noqa: E402
    ACTION_ACCEPT,
    ACTION_REGENERATE,
    ACTION_TEACHER_REVIEW,
    CHECK_SKIPPED,
    SEVERITY_ERROR,
)
from app.services.graph_validator import validate_graph  # noqa: E402
from app.services.problem_repository import get_problem_repository  # noqa: E402

ACTION_LABEL = {
    ACTION_ACCEPT: "可以入库使用",
    ACTION_TEACHER_REVIEW: "建议教师复核",
    ACTION_REGENERATE: "结构有问题，建议重新生成",
}


def _display(path: Path) -> str:
    """尽量显示相对仓库的路径；题库可以配到仓库外（PROBLEMS_DIR），那时显示绝对路径。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_report(report, problem_id: str) -> None:
    print("=" * 68)
    print(f"校验报告 · {problem_id}")
    print("=" * 68)
    verdict = "✓ 通过" if report.valid else "✗ 未通过"
    print(f"结论：{verdict}    信心：{report.confidence}    "
          f"建议：{ACTION_LABEL.get(report.recommended_action, report.recommended_action)}")
    print(f"检查：{'  '.join(f'{k}={v}' for k, v in report.checks.items())}")
    if report.summary:
        print(f"摘要：{report.summary}")

    if report.issues:
        print("-" * 68)
        for issue in report.issues:
            mark = "✗" if issue.severity == SEVERITY_ERROR else "⚠"
            where = f" @{issue.node}" if issue.node else ""
            print(f"  {mark} [{issue.type}]{where} {issue.description}")
            print(f"       （来源：{'模型' if issue.source == 'llm' else '规则'}）")
    else:
        print("-" * 68)
        print("  未发现问题")

    if report.checks.get("physics") == CHECK_SKIPPED:
        print("-" * 68)
        print("  注：物理一致性**未检查**（未提供题面，或模型不可用）。")
        print("      上面通过只代表结构、流程、路径三项没问题。")
    branch_check = report.metadata.get("false_branches", {})
    if branch_check.get("reason"):
        print(f"  注：{branch_check['reason']}。")
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 Problem Graph（P15）")
    parser.add_argument("problem_id", help="题号")
    parser.add_argument("--rules-only", action="store_true", help="只跑规则检查，不调模型")
    parser.add_argument("--write", action="store_true", help="把结论写进 graph.json 的 metadata.validation")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整报告")
    parser.add_argument("--quantity-asked", action="append", dest="quantities_asked",
                        help="明确指定一个目标量（可重复）；省略时读取图 metadata.quantities_asked")
    parser.add_argument("--accept", action="store_true",
                        help="人工确认：清除 needs_review（校验器不会自动做这件事）")
    parser.add_argument("--reviewed-by", default="", help="确认人（配合 --accept）")
    parser.add_argument("--note", default="", help="确认备注（配合 --accept）")
    args = parser.parse_args()

    repo = get_problem_repository()
    graph = repo.get_graph(args.problem_id)
    if graph is None:
        print(f"✗ 题目 {args.problem_id} 没有 graph.json（处于 graphless 模式）")
        return 1

    problem = repo.get_problem(args.problem_id)
    report = validate_graph(graph, problem=problem, use_llm=not args.rules_only,
                            quantities_asked=args.quantities_asked)

    if args.json:
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    else:
        print_report(report, args.problem_id)

    if args.write:
        graph.metadata["validation"] = report.as_metadata()
        repo.save_graph(args.problem_id, graph)
        print(f"✓ 校验结论已写入 {_display(repo.graph_path(args.problem_id))} 的 metadata.validation")

    if args.accept:
        if not report.valid:
            print("\n✗ 校验未通过，拒绝标记为人工确认。请先修图（或重新生成）。")
            return 2
        if report.checks.get("physics") == CHECK_SKIPPED and not args.rules_only:
            print("\n⚠ 提醒：物理一致性未检查，你确认的是'结构上可用'。")
        repo.mark_reviewed(args.problem_id, reviewed_by=args.reviewed_by, note=args.note)
        print(f"✓ 已标记为人工确认（needs_review=false，by {args.reviewed_by or 'unknown'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
