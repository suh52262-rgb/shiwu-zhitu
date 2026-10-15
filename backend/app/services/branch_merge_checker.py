"""Conservative merge review: evidence for reviewers, never a physics proof.

Optional ``metadata.merge_contracts`` is a node-id mapping of
``{"requires": ["token"], "produces": ["token"]}``. Tokens are opaque,
human-supplied facts; edges remain teaching progression, not data dependencies.
Missing annotations yield unknown, never an inferred prerequisite failure.
"""
import re

from app.models.problem_graph import ProblemGraph


def _numeric_outputs(text: str) -> set[str]:
    # Only explicit assignment chains ending in a number count as computed.
    # A symbolic formula (e.g. a=(F-mg)/M) is not a completed numerical result.
    outputs = set()
    for match in re.finditer(r"([A-Za-zΑ-ω][\w̄′₀₁₂]*)\s*=\s*([^。；;\n]+)", text):
        if re.search(r"(?:^|=)\s*[+−-]?\d", match[2]):
            outputs.add(match[1])
    return outputs


def inspect_branch_merges(graph: ProblemGraph) -> dict:
    """Return actual declared routes into each merge and bounded evidence.

    A route must use existing edges and nodes before it can be inspected.
    Invalid paths are handled by the structural validator instead.
    """
    nodes = graph.node_map()
    edges = {(e[0], e[1]) for e in graph.edges
             if isinstance(e, (list, tuple)) and len(e) == 2}
    predecessors = {n: {s for s, d in edges if d == n and s in nodes} for n in nodes}
    paths = [(i, p) for i, p in enumerate(graph.valid_paths, 1)
             if p and len(set(p)) == len(p) and all(n in nodes for n in p)
             and all((s, d) in edges for s, d in zip(p, p[1:]))]
    raw = graph.metadata.get("merge_contracts")
    contracts = raw if isinstance(raw, dict) else {}

    def tags(n, key):
        entry = contracts.get(n, {})
        value = entry.get(key) if isinstance(entry, dict) else None
        if isinstance(value, list) and (value or key == "produces") and all(isinstance(t, str) and t.strip() for t in value):
            return set(value)
        return None

    merges, findings = [], []
    for merge, incoming in predecessors.items():
        if len(incoming) < 2:
            continue
        routes = [(i, p[:p.index(merge)]) for i, p in paths if merge in p and p.index(merge) > 0]
        item = {"node": merge, "incoming": sorted(incoming), "routes": [], "status": "unknown"}
        target = nodes[merge]
        for index, prefix in routes:
            evidence = []
            route = {"path_index": index, "prefix": prefix, "predecessor": prefix[-1],
                     "status": "unknown", "evidence": evidence}
            required = tags(merge, "requires")
            produced = [tags(n, "produces") for n in prefix]
            if required and all(t is not None for t in produced):
                missing = required - set().union(*produced)
                route["status"] = "issues" if missing else "annotated_pass"
                route["required"] = sorted(required)
                route["available"] = sorted(set().union(*produced))
                if missing:
                    evidence.append({"rule": "annotated_prerequisite_missing", "missing": sorted(missing),
                                     "why": "显式标签显示此路线未提供汇合所需前置条件（标签本身需人工核对）"})
            # Restrict textual checks to the branch-specific suffix: common
            # computations before the split are not a merge incompatibility.
            others = [p for _, p in routes if p != prefix]
            common = 0
            while common < len(prefix) and others and all(len(p) > common and p[common] == prefix[common] for p in others):
                common += 1
            branch = prefix[common:]
            branch_text = "\n".join(nodes[n].description for n in branch)
            if others and target.name.startswith(("求", "计算")):
                repeated = _numeric_outputs(branch_text) & _numeric_outputs(target.description)
                if repeated:
                    evidence.append({"rule": "repeated_numeric_calculation", "symbols": sorted(repeated),
                                     "branch_text": branch_text, "merge_text": target.description,
                                     "why": "此支路已给出该量数值，汇合仍要求计算；应允许核对已得结果，避免强制重算"})
            if (others and any(word in target.name + target.description for word in ("取正根", "两个根"))
                    and "振幅" in branch_text and not any(word in branch_text for word in ("二次方程", "取根", "正根"))):
                evidence.append({"rule": "root_selection_after_amplitude", "branch_text": branch_text,
                                 "merge_text": target.description,
                                 "why": "该路线描述用振幅求位移，公共汇合却要求方程取根；未提供此路线需要取根的依据，请复核"})
            if evidence:
                route["status"] = "issues"
                findings.append({"node": merge, "path_index": index, "prefix": prefix, "evidence": evidence})
            route["reason"] = ("仅检查文本风险或显式标签，不证明物理正确性" if route["status"] != "unknown"
                               else "缺少完整前置条件标注，无法确认汇合对本路线适用")
            item["routes"].append(route)
        if any(r["status"] == "issues" for r in item["routes"]):
            item["status"] = "issues"
        elif item["routes"] and incoming <= {r["predecessor"] for r in item["routes"]} and all(r["status"] == "annotated_pass" for r in item["routes"]):
            item["status"] = "annotated_pass"
        merges.append(item)
    return {"merges": merges, "findings": findings,
            "unknown_count": sum(m["status"] == "unknown" for m in merges),
            "reason": "无汇合节点" if not merges else "文本与显式标签辅助复核；unknown 表示未验证，不代表通过"}
