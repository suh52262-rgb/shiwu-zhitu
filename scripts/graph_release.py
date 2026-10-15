"""Human graph review/release workflow (no model or database access).

python scripts/graph_release.py demo_03 status
python scripts/graph_release.py demo_03 diff
python scripts/graph_release.py demo_03 review --reviewer 'Teacher Name' --note 'Checked every route'
python scripts/graph_release.py demo_03 publish --reviewer 'Teacher Name'
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.graph_versions import GraphVersions, graph_content, graph_hash
from app.services.problem_repository import ProblemRepository


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_id")
    parser.add_argument("action", choices=["status", "diff", "review", "publish"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    repo = ProblemRepository(args.root)
    if not args.problem_id or any(s in args.problem_id for s in ("/", "\\", ":")) or args.problem_id.startswith("."):
        parser.error("problem_id must be one directory name")
    graph = repo.get_graph(args.problem_id)
    if graph is None:
        parser.error("No graph candidate exists")
    versions = GraphVersions(repo.problem_dir(args.problem_id))
    try:
        if args.action == "review":
            repo.mark_reviewed(args.problem_id, reviewed_by=args.reviewer, note=args.note)
        elif args.action == "publish":
            repo.publish_graph(args.problem_id, published_by=args.reviewer)
        elif args.action == "diff":
            published = versions.manifest().get("published")
            before = graph_content(versions.load(published)) if published else {}
            lines = lambda value: (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").splitlines(keepends=True)
            print("".join(difflib.unified_diff(lines(before), lines(graph_content(graph)), fromfile=published or "unpublished", tofile=graph_hash(graph))), end="")
        release = repo.graph_release(args.problem_id)
        print(json.dumps({**{key: value for key, value in release.items() if key != "graph"}, "candidate_hash": graph_hash(graph), "manifest": versions.manifest()}, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
