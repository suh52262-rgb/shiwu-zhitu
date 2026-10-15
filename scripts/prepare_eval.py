"""Offline label audit and immutable input manifest preparation; no model calls.

Prepare: python scripts/prepare_eval.py --model MODEL --budget-cny 10
Verify before running: python scripts/prepare_eval.py --verify MANIFEST
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.prompt_registry import all_prompt_versions
from app.config import get_settings
from app.evaluation.label_audit import audit_dataset
from app.services.graph_versions import graph_hash
from app.services.problem_store import load_problem_graph

SUITES = ("step_matching", "alternative_path", "graph")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root=ROOT, *, model, budget_cny, limit=99, include_proposed=False,
            budget_approved=False, budget_unlimited=False):
    settings = get_settings()
    files, graphs, coverage = {}, {}, {}
    call_count = 0
    for suite in SUITES:
        path = root / "evaluation" / "datasets" / f"{suite}_cases.json"
        dataset = json.loads(path.read_text(encoding="utf-8"))
        files[str(path.relative_to(root))] = digest(path)
        coverage[suite] = audit_dataset(dataset, load_problem_graph)
        entries = dataset.get("cases") or [dataset]
        for case in entries:
            pid = case.get("problem_id", dataset.get("problem_id"))
            graph = load_problem_graph(pid)
            graphs[pid] = graph_hash(graph) if graph else None
            for filename in ("graph.json", "problem.json"):
                source = root / "data" / "problems" / pid / filename
                if source.exists():
                    files[str(source.relative_to(root))] = digest(source)
        call_count += min(limit, len(dataset.get("rounds", entries)))
    versions = all_prompt_versions()
    for role, version in versions.items():
        path = root / "prompts" / role / f"{version}.md"
        files[str(path.relative_to(root))] = digest(path)
    for directory in (root / "backend" / "app",):
        for path in sorted(directory.rglob("*.py")):
            files[str(path.relative_to(root))] = digest(path)
    for script in ("run_eval.py", "prepare_eval.py"):
        path = root / "scripts" / script
        files[str(path.relative_to(root))] = digest(path)
    return {
        "schema_version": 1, "status": "prepared_not_executed",
        "suites": list(SUITES), "limit": limit, "graph_source": "author_candidate",
        "evaluation_config": {"min_accuracy": 0.7,
                              "label_scope": "proposed_exploratory" if include_proposed else "reviewed_only"},
        "model_config": {"llm_provider": settings.llm_provider, "llm_model": model,
                         "step_matcher_model": model,
                         "llm_temperature": settings.llm_temperature,
                         "llm_timeout": settings.llm_timeout,
                         "llm_max_retries": settings.llm_max_retries},
        "endpoint": {"provider": settings.llm_provider, "base_url": settings.deepseek_base_url
                     if settings.llm_provider == "deepseek" else settings.llm_base_url},
        "prompt_versions": versions, "graphs": graphs, "files": files,
        "coverage": coverage,
        "budget": {"approved": budget_approved, "unlimited": budget_unlimited,
                   "planned_ceiling_cny": None if budget_unlimited else budget_cny,
                   "primary_calls": call_count,
                   "calls_with_sdk_retries_upper_bound": call_count * 2 * (1 + settings.llm_max_retries),
                   "note": "每次结构化生成最多两轮格式重试，再计 SDK 网络重试；费用规划不是运行时费用硬限制"},
        "rerun": [f"python scripts/run_eval.py --suite {s} --limit {limit} --manifest MANIFEST"
                  + (" --include-proposed-labels" if include_proposed else "")
                  for s in SUITES],
    }


def verify(manifest, root=ROOT):
    errors = []
    for name, expected in manifest.get("files", {}).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"invalid_path:{name}")
        elif not path.is_file() or digest(path) != expected:
            errors.append(f"input_drift:{name}")
    if not manifest.get("files"):
        errors.append("missing_file_hashes")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="必须显式指定模型，不能使用动态默认值")
    parser.add_argument("--budget-cny", type=float, default=0)
    parser.add_argument("--limit", type=int, default=99)
    parser.add_argument("--include-proposed-labels", action="store_true")
    parser.add_argument("--budget-approved", action="store_true", help="仅在已有明确用户授权时记录")
    parser.add_argument("--budget-unlimited", action="store_true", help="明确记录用户未设费用上限；仍受 case/retry 限制")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "prepared_manifest.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        errors = verify(json.loads(args.verify.read_text(encoding="utf-8")))
        print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(bool(errors))
    if not args.model or args.limit <= 0 or args.budget_cny < 0:
        parser.error("准备需显式 --model，limit > 0 且 budget >= 0")
    manifest = prepare(model=args.model, budget_cny=args.budget_cny, limit=args.limit,
                       include_proposed=args.include_proposed_labels,
                       budget_approved=args.budget_approved, budget_unlimited=args.budget_unlimited)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({suite: report["counts"] for suite, report in manifest["coverage"].items()},
                     ensure_ascii=False, indent=2))
    print(f"Prepared without external calls: {args.output}")


if __name__ == "__main__":
    main()
