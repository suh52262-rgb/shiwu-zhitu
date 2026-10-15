"""P14 Auto Problem-Solving Graph Generator 命令行入口。

用法：
    # 用题库里已有的题目生成（题面 + 参考答案自动取）
    .venv/Scripts/python scripts/generate_graph.py phy_001

    # 从题目截图直接生成（先用 VL 模型转写题面，再建图）
    .venv/Scripts/python scripts/generate_graph.py phy_004 --image 题目截图.png --save

    # 临时给一段题面（不落盘，只打印）
    .venv/Scripts/python scripts/generate_graph.py --statement "物体从斜面滑下……"

    # 从文件读题面与参考答案，并写入题库
    .venv/Scripts/python scripts/generate_graph.py phy_new_001 \
        --statement-file problem.txt --solution-file solution.txt --save

    # 用样例数据验证链路（不调用真实模型）
    .venv/Scripts/python scripts/generate_graph.py --demo

注意：生成的是**候选图**，落盘时会标记 needs_review=true，
默认不覆盖题库里已有的 graph.json（要覆盖需显式 --force）。
"""
import argparse
import json
import sys
from pathlib import Path

# 控制台默认 GBK，报告里有 ✓/⚠ 等字符，先重配编码（与 run_eval.py 一致）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.candidate_graph import CandidateGraph  # noqa: E402
from app.services.graph_generator import (  # noqa: E402
    describe_node_type,
    generate_candidate_graph,
    normalize_candidate_graph,
)
from app.services.problem_repository import get_problem_repository  # noqa: E402

DEMO_STATEMENT = (
    "质量为 m 的物体从高 h、倾角 θ 的粗糙斜面顶端由静止滑下，到达底端时速度为 v，"
    "重力加速度为 g。求下滑过程中摩擦力做的功。"
)
DEMO_SOLUTION = "由动能定理：mgh + W_f = 1/2·m·v²，所以 W_f = 1/2·m·v² − mgh。"

# --demo 用的样例输出：形状与提示词要求一致，用来验证归一化与落盘链路
DEMO_GRAPH = {
    "knowledge_points": ["动能定理", "摩擦力做功"],
    "nodes": [
        {"id": "S1", "name": "识别研究对象与过程", "type": "modeling",
         "description": "物体从斜面顶端滑到底端，受重力、支持力、摩擦力。",
         "knowledge_points": ["受力分析"]},
        {"id": "S2", "name": "判断可用的功能关系", "type": "key_reasoning",
         "description": "存在摩擦力做功，机械能不守恒，应改用动能定理。",
         "knowledge_points": ["动能定理"], "common_errors": ["E1：误用机械能守恒"]},
        {"id": "S3", "name": "列出动能定理方程", "type": "calculation",
         "description": "mgh + W_f = 1/2·m·v²。", "knowledge_points": ["动能定理"]},
        {"id": "S4", "name": "解出摩擦力做功", "type": "result",
         "description": "W_f = 1/2·m·v² − mgh。", "knowledge_points": ["摩擦力做功"]},
    ],
    "edges": [["S1", "S2"], ["S2", "S3"], ["S3", "S4"]],
    "valid_paths": [["S1", "S2", "S3", "S4"]],
    "alternative_methods": ["也可用牛顿第二定律结合运动学公式分段求解"],
}


def _display(path: Path) -> str:
    """尽量显示相对仓库的路径；题库可以配到仓库外（PROBLEMS_DIR），那时显示绝对路径。

    ⚠️ 这里曾经写成 `return str(_display(path))`——**调用自己**，
    一进这个函数就无限递归。因为只在 `--save` 的**成功路径**上调用，
    平时不触发；是"试卷切题 → 建图"这条新链路第一次走到才暴露出来。
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _print_candidate(candidate: CandidateGraph) -> None:
    print("\n" + "=" * 62)
    print(f"候选解题图 · {candidate.problem_id or '(未指定题号)'}")
    print(f"来源：{candidate.generation.source}   模型：{candidate.generation.model or '—'}"
          f"   需要复核：{'是' if candidate.generation.needs_review else '否'}")
    print("-" * 62)
    print(f"知识点：{'、'.join(candidate.knowledge_points) or '（无）'}")
    print(f"节点（{len(candidate.nodes)}）：")
    for node in candidate.nodes:
        print(f"  {node.id:5} [{describe_node_type(node.type)}] {node.name}")
        if node.description:
            print(f"        {node.description}")
        if node.common_errors:
            print(f"        常见错误：{'；'.join(node.common_errors)}")
    print("推进关系：" + ("，".join(f"{s}→{d}" for s, d in candidate.edges) or "（无）"))
    for i, path in enumerate(candidate.valid_paths, 1):
        print(f"完整路径{i}：" + "→".join(path))
    if candidate.alternative_methods:
        print("其他合理方法：" + "；".join(candidate.alternative_methods))
    if candidate.generation.normalizations:
        print("-" * 62)
        print(f"生成后修正（{len(candidate.generation.normalizations)} 处）：")
        for item in candidate.generation.normalizations:
            print(f"  ⚠ {item}")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="自动生成候选解题图（P14）")
    parser.add_argument("problem_id", nargs="?", help="题库里的题号（也可只配合 --statement 使用）")
    parser.add_argument("--image", help="题目截图：先用 VL 模型转写题面，再建图")
    parser.add_argument("--statement", help="直接给题面")
    parser.add_argument("--statement-file", help="从文件读题面")
    parser.add_argument("--solution", help="直接给参考答案")
    parser.add_argument("--solution-file", help="从文件读参考答案")
    parser.add_argument("--course", default="大学物理")
    parser.add_argument("--save", action="store_true", help="写入题库 data/problems/{id}/graph.json")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有的 graph.json")
    parser.add_argument("--demo", action="store_true",
                        help="用内置样例走完整链路（不调用真实模型，用于验证归一化与落盘）")
    parser.add_argument("--validate", action="store_true",
                        help="生成后立刻跑 P15 校验（结构 / 流程 / 路径 / 物理一致性）")
    parser.add_argument("--quantity-asked", action="append", dest="quantities_asked",
                        help="明确指定一个目标量（可重复），供假分叉检查使用并随图保存")
    parser.add_argument("--expand-paths", action="store_true",
                        help="P15 报 path_inconsistency 时，自动把 alternative_methods "
                             "建成支路节点（机器补的，仍标 needs_review）")
    args = parser.parse_args()

    repo = get_problem_repository()

    # ---------- 取题面 / 参考答案 ----------
    problem_id = args.problem_id or ""
    statement = args.statement or ""
    solution = args.solution or ""

    if args.statement_file:
        statement = Path(args.statement_file).read_text(encoding="utf-8")
    if args.solution_file:
        solution = Path(args.solution_file).read_text(encoding="utf-8")

    # 截图优先：P14 只吃文本，这一步用 VL 模型把图转成题面（见 transcribe_problem.py）
    if args.image:
        from transcribe_problem import TranscribeError, transcribe_problem_image

        print(f"正在转写题目截图（{Path(args.image).name}）……")
        try:
            statement, solution = transcribe_problem_image(args.image)
        except TranscribeError as exc:
            print(f"✗ 截图转写失败：{exc}")
            print("  也可以手工把题面写进 --statement / --statement-file，跳过这一步。")
            return 1
        print(f"✓ 转写完成：题面 {len(statement)} 字，参考答案 {len(solution)} 字")

    if problem_id and not statement:
        problem = repo.get_problem(problem_id)
        if problem is None:
            print(f"✗ 题库里没有题目 {problem_id}（可用 --statement 直接给题面）")
            return 1
        statement = problem.full_text
        solution = solution or problem.answer

    if not statement and not args.demo:
        print("✗ 需要题面：给一个题库里的题号，或用 --statement / --statement-file")
        return 1

    # ---------- 生成 ----------
    if args.demo:
        candidate = normalize_candidate_graph(
            CandidateGraph.model_validate({**DEMO_GRAPH, "problem_id": problem_id or "phy_demo"})
        )
        candidate.generation.model = "demo（未调用模型）"
        candidate.generation.source = "auto"
        print("⚠ --demo：使用内置样例图，不调用真实模型，仅用于验证归一化与落盘链路")
    else:
        print(f"正在生成候选解题图（题面 {len(statement)} 字，参考答案 {len(solution)} 字）……")
        try:
            candidate = generate_candidate_graph(
                problem_statement=statement,
                reference_solution=solution,
                course=args.course,
                problem_id=problem_id,
            )
        except Exception as exc:  # noqa: BLE001 命令行工具：把原因如实打出来即可
            print(f"✗ 生成失败：{type(exc).__name__}: {exc}")
            return 1

    _print_candidate(candidate)

    # ---------- 落盘 ----------
    if not args.save:
        print("\n（未写盘。加 --save 可写入题库 data/problems/<题号>/graph.json）")
        return 0

    if not problem_id:
        print("\n✗ --save 需要题号（第一个位置参数）")
        return 1
    if repo.has_graph(problem_id) and not args.force:
        print(f"\n✗ 题目 {problem_id} 已有 graph.json，未覆盖（要覆盖请加 --force）")
        return 1

    generated_graph = candidate.to_problem_graph()
    if args.quantities_asked is not None:
        generated_graph.metadata["quantities_asked"] = args.quantities_asked
    path = repo.save_graph(problem_id, generated_graph)
    print(f"\n✓ 已写入 {_display(path)}")

    merged = repo.merge_knowledge_points(problem_id, candidate.knowledge_points)
    if merged:
        print(f"✓ 已把整题知识点并入 problem.json（现共 {len(merged)} 个）："
              f"{'、'.join(merged)}")
    else:
        print("  （题目目录不存在或没有知识点，未回写 problem.json）")

    print("  该图标记为 needs_review=true（source=auto）——它是候选图，"
          "深入校验见任务书 P15 Graph Validator。")

    if args.validate:
        # P15：生成完立刻校验一遍。注意校验器**不会**替你清除 needs_review——
        # "机器说没问题"与"人确认过"是两件事。
        from app.services.graph_validator import validate_graph
        from validate_graph import print_report

        print()
        report = validate_graph(repo.get_graph(problem_id), problem=repo.get_problem(problem_id))
        print_report(report, problem_id)

        stored = repo.get_graph(problem_id)
        stored.metadata["validation"] = report.as_metadata()
        repo.save_graph(problem_id, stored)
        print("✓ 校验结论已写入 graph.json 的 metadata.validation")

        # ---- 把"报警"变成"行动指引" ----
        #
        # 实测：P14 能**识别出**替代解法（写进 `alternative_methods`），
        # 但**不把它建模成分叉节点** —— 跑 `demo_03`（成员B 明确写了两条方法）
        # 产出的是 `S1→…→S8` 单路径，替代解法只落在一段说明文字里。
        #
        # P15 的 `path_inconsistency` 会报这条，措辞是
        # "学生走另一条正确路线时会被误判"。**但那只是一条 warning，
        # 混在其它 issue 里很容易被忽略。**
        #
        # 这里把它单独拎出来，并说清**下一步该做什么**。
        alt_issue = [i for i in report.issues
                     if getattr(i, "type", "") == "path_inconsistency"]

        # ---- `--expand-paths`：自动补支路 ----
        # **改图的不是 P15** —— P15 保持只读，由独立的扩展步骤来做，
        # 产物仍是 needs_review 的草稿（机器不自我批准）。
        expanded = None
        if alt_issue and args.expand_paths:
            from app.services.path_expander import (  # noqa: PLC0415
                ExpansionError,
                expand_alternative_paths,
            )

            print()
            print("=" * 62)
            print("  路径扩展（--expand-paths）")
            print("=" * 62)
            cur = repo.get_graph(problem_id)
            try:
                new_graph, exp = expand_alternative_paths(
                    cur, problem_statement=(repo.get_problem(problem_id).statement
                                            if repo.get_problem(problem_id) else ""))
            except ExpansionError as exc:
                new_graph, exp = cur, {"ok": False, "reason": str(exc),
                                       "added_nodes": [], "added_paths": 0}
            if exp["ok"]:
                print(f"  ✓ 新增节点 {exp['added_nodes']}，"
                      f"新增路径 {exp['added_paths']} 条")
                print(f"    新路径：{new_graph.valid_paths}")
                if exp.get("rejected_methods"):
                    print(f"    被拒解法（物理上不成立）：{exp['rejected_methods']}")
                repo.save_graph(problem_id, new_graph)
                expanded = new_graph
                print("  ⚠ **支路是机器补的，未经人复核** —— "
                      "needs_review 仍为 True，请人工确认该解法成立且与主路径等价")

                # 扩展后再校验一次，把结论一并写回
                from app.services.graph_validator import (  # noqa: PLC0415
                    validate_graph as _vg,
                )
                rep2 = _vg(repo.get_graph(problem_id),
                           problem=repo.get_problem(problem_id))
                print_report(rep2, problem_id)
                stored = repo.get_graph(problem_id)
                stored.metadata["validation"] = rep2.as_metadata()
                repo.save_graph(problem_id, stored)
                report = rep2
                alt_issue = [i for i in rep2.issues
                             if getattr(i, "type", "") == "path_inconsistency"]
            else:
                print(f"  ✗ 扩展未成功：{exp['reason']}")
                print("    （原图未改动）")

        if alt_issue:
            print()
            print("=" * 62)
            print("  ⚠ **需要人工补分叉**" if not expanded else "  ⚠ **仍有路径问题**")
            print("=" * 62)
            for i in alt_issue:
                print(f"  {i.description}")
            print()
            print("  P14 识别得出替代方法，但**不会把它建成图上的分叉**——")
            print("  它把替代解法写进了 metadata.alternative_methods，")
            print("  而教学链路吃的是 valid_paths，**用不上那段文字**。")
            print()
            print("  要做的：在 graph.json 里手动加分叉节点（id 用 S6a/S6b 这种")
            print("  带字母后缀的形式），在分叉点后各自建节点、再汇合回公共节点，")
            print("  并把两条路径都写进 valid_paths。")
            print(f"    文件：data/problems/{problem_id}/graph.json")
            print("  补完重跑校验：")
            print(f"    python scripts/validate_graph.py {problem_id}")

        if report.valid and report.checks.get("physics") == "pass":
            print("  人工确认后可清除 needs_review：")
            print(f"    python scripts/validate_graph.py {problem_id} --accept --reviewed-by <你的名字>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
