"""P12 多轮学生学习模拟器运行器。

用法：
    .venv/Scripts/python scripts/run_simulation.py --list
    .venv/Scripts/python scripts/run_simulation.py spring_concept_error
    .venv/Scripts/python scripts/run_simulation.py --all            # 依次跑完 5 类学生
    .venv/Scripts/python scripts/run_simulation.py --all --offline  # 不调 LLM 的离线冒烟

模式说明：
- 默认（真实 LLM）：跑真实模型，`expected` 是**真实断言**，可用于回归与比赛演示；
- `--offline`：用脚本化 LLM 按用例期望回放，验证模拟器自身 / State 演化 /
  Hint Policy / Attempt 落库是否正常。**它不代表模型准确率**，节点断言必然成立。

退出码：全部用例通过为 0，否则为 1（便于接 CI）。
"""
import argparse
import sys
from pathlib import Path

# 控制台默认 GBK，报告里有 ✓/✗ 等字符，先重配编码（与 run_eval.py 一致）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.simulation import (  # noqa: E402
    list_cases,
    load_case,
    print_summary,
    run_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="多轮学生学习模拟器（P12）")
    parser.add_argument("case", nargs="?", help="用例 id，如 spring_concept_error")
    parser.add_argument("--list", action="store_true", help="列出全部用例")
    parser.add_argument("--all", action="store_true", help="依次运行全部用例")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="用脚本化 LLM 离线冒烟（不调真实模型，节点断言必然成立）",
    )
    parser.add_argument("--quiet", action="store_true", help="只打印汇总，不打印逐轮过程")
    args = parser.parse_args()

    if args.list or (not args.case and not args.all):
        cases = list_cases()
        print(f"可用模拟用例（{len(cases)} 个，位于 simulation/）：")
        for case_id in cases:
            case = load_case(case_id)
            print(f"  - {case_id:28} {case.title}（{len(case.steps)} 步）")
        if not args.list:
            print("\n用法：python scripts/run_simulation.py <case_id>  或  --all")
            return 1
        return 0

    targets = list_cases() if args.all else [args.case]
    reports = []
    for case_id in targets:
        try:
            case = load_case(case_id)
        except ValueError as exc:
            print(f"✗ {exc}")
            return 1
        print(f"\n▶ 开始模拟：{case_id}（{case.title}）  模式={'offline' if args.offline else 'real LLM'}")
        report = run_case(case, offline=args.offline, quiet=args.quiet)
        print_summary(report)
        reports.append(report)

    failed = [report for report in reports if not report.ok]
    print(f"\n总计 {len(reports)} 个用例，通过 {len(reports) - len(failed)}，未通过 {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
