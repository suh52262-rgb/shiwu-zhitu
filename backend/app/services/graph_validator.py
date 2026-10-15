"""Graph Validator（P15，任务书第三阶段 P15 节）。

Auto Graph Generator 由 LLM 生成，所以必须有验证层：

    Candidate Graph → Validator → Validation Report

四类检查按"能不能用规则做"分工（任务书明确允许"规则检查 + LLM 检查"结合）：

| 检查 | 手段 | 为什么 |
|---|---|---|
| 结构合法性 | **纯规则** | 重复 id / 悬空边 / 孤立点 / 有无入口终点 / 环，都是图论事实，不该问 LLM |
| 解题流程完整性 | **纯规则** | 用节点 type 的分布判断是否严重缺步骤，可解释、零成本 |
| 多路径一致性 | **纯规则** | 路径里的节点与边是否存在、是否同起同终，同样是图论事实 |
| 物理一致性 | **LLM** | "有摩擦力却说机械能守恒"需要读懂题面语义，规则做不了 |

另有假分叉启发式检查：显式目标量与直接后继文本匹配，仅提示人工复核。

**降级**（P20 原则）：LLM 不可用 / 没给题面时，物理检查标 `skipped`，
其余三项照常给出，并把置信度上限压低——而不是假装检查过了。
"""
import logging

from app.agent.prompt_registry import get_prompt
from app.models.graph_validation import (
    ACTION_ACCEPT,
    ACTION_REGENERATE,
    ACTION_TEACHER_REVIEW,
    CHECK_ISSUES,
    CHECK_PASS,
    CHECK_SKIPPED,
    ISSUE_FLOW,
    ISSUE_FALSE_BRANCH,
    ISSUE_BRANCH_MERGE,
    ISSUE_PATH,
    ISSUE_PHYSICS,
    ISSUE_STRUCTURE,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    GraphIssue,
    GraphValidationReport,
    new_validation_metadata,
)
from app.models.problem_graph import ProblemGraph
from app.services.branch_checker import find_false_branches
from app.services.branch_merge_checker import inspect_branch_merges
from app.services.llm import LLMClient, get_llm
from app.services.llm.schemas import PhysicsCheck

logger = logging.getLogger(__name__)

#: 单节点图当成"没内容"（题目至少要有建模与结果两步才有教学意义）
MIN_NODES = 2


def validate_graph(
    graph: ProblemGraph,
    problem=None,
    llm: LLMClient | None = None,
    use_llm: bool = True,
    *,
    quantities_asked: list[str] | None = None,
) -> GraphValidationReport:
    """校验一张解题图，返回报告。

    `problem` 用于物理一致性检查（需要题面语义）；不传则该项标 skipped。
    `use_llm=False` 可强制只跑规则检查（离线 / 省成本 / 单测）。
    `quantities_asked` 为显式目标量，优先于图 metadata 中同名字段。
    缺少有效目标量标注时，假分叉启发式检查标 skipped，不推断题面。
    """
    issues: list[GraphIssue] = []
    checks: dict[str, str] = {}

    structure = _check_structure(graph)
    issues += structure
    checks["structure"] = CHECK_ISSUES if structure else CHECK_PASS

    flow = _check_flow(graph)
    issues += flow
    checks["flow"] = CHECK_ISSUES if flow else CHECK_PASS

    paths = _check_paths(graph)
    issues += paths
    checks["paths"] = CHECK_ISSUES if paths else CHECK_PASS

    branch_issues, branch_status, branch_meta = _check_false_branches(graph, quantities_asked)
    issues += branch_issues
    checks["false_branches"] = branch_status

    merge_meta = inspect_branch_merges(graph)
    merge_issues = [_issue(
        ISSUE_BRANCH_MERGE, SEVERITY_WARNING, f["node"],
        f"路径{f['path_index']}（{'→'.join(f['prefix'])}→{f['node']}）汇合疑点："
        + "；".join(e["why"] for e in f["evidence"])) for f in merge_meta["findings"]]
    issues += merge_issues
    checks["branch_merges"] = (CHECK_ISSUES if merge_issues else CHECK_SKIPPED
                               if merge_meta["unknown_count"] else CHECK_PASS)

    physics_issues, physics_status, llm_meta = _check_physics(graph, problem, llm, use_llm)
    issues += physics_issues
    checks["physics"] = physics_status

    errors = [i for i in issues if i.severity == SEVERITY_ERROR]
    warnings = [i for i in issues if i.severity == SEVERITY_WARNING]

    # 结构性错误让图直接不可用；其余交给人工看一眼即可
    if errors:
        action = (
            ACTION_REGENERATE if checks["structure"] == CHECK_ISSUES else ACTION_TEACHER_REVIEW
        )
    elif warnings:
        action = ACTION_TEACHER_REVIEW
    else:
        action = ACTION_ACCEPT

    return GraphValidationReport(
        valid=not errors,
        confidence=_confidence(len(errors), len(warnings), physics_status),
        issues=issues,
        recommended_action=action,
        checks=checks,
        summary=_summarize(graph, checks, errors, warnings, physics_status)
                + (f"；{merge_meta['unknown_count']} 个汇合适用性为 unknown（未获完整前置条件证据）"
                   if merge_meta["unknown_count"] and merge_issues else ""),
        metadata={**new_validation_metadata(**llm_meta), "node_count": len(graph.nodes),
                  "false_branches": branch_meta, "branch_merges": merge_meta},
    )


def validate_and_store(graph: ProblemGraph, problem=None, llm: LLMClient | None = None,
                        *, quantities_asked: list[str] | None = None):
    """校验并把结论写进 `graph.metadata["validation"]`（不改 `needs_review`）。

    `needs_review` 表示"**人工**尚未确认"，校验器通过不等于人看过——
    自动把机器结论冒充成人工确认，正是 P14 那条设计要防的事。
    清除该标记请用 `ProblemRepository.mark_reviewed()`。
    """
    report = validate_graph(graph, problem=problem, llm=llm, quantities_asked=quantities_asked)
    graph.metadata["validation"] = report.as_metadata()
    return report


# ---------------------------------------------------------------- 规则检查


def _check_false_branches(graph: ProblemGraph, quantities_asked):
    source = "argument" if quantities_asked is not None else "graph.metadata.quantities_asked"
    raw = quantities_asked if quantities_asked is not None else graph.metadata.get("quantities_asked")
    meta = {"source": source, "quantities_asked": [], "reason": "", "findings": []}
    if (not isinstance(raw, list) or not raw
            or any(not isinstance(q, str) or not q.strip() for q in raw)):
        meta["reason"] = "缺少有效 quantities_asked 标注（应为非空字符串列表），假分叉检查未执行"
        return [], CHECK_SKIPPED, meta
    asked = list(dict.fromkeys(q.strip() for q in raw))
    meta["quantities_asked"] = asked
    findings = find_false_branches(graph, asked)
    meta["findings"] = findings
    meta["reason"] = "仅按直接后继文本匹配目标量，未检查分支物理等价性或汇合适用性"
    issues = [_issue(ISSUE_FALSE_BRANCH, SEVERITY_WARNING, finding["node"],
                     "疑似假分叉（启发式，需人工复核）：" + finding["why"])
              for finding in findings]
    return issues, CHECK_ISSUES if issues else CHECK_PASS, meta


def _check_structure(graph: ProblemGraph) -> list[GraphIssue]:
    """结构合法性：任务书点名的五项（重复 id / 悬空边 / 孤立点 / 入口终点 / 环）。"""
    issues: list[GraphIssue] = []
    ids = [node.id for node in graph.nodes]

    if not ids:
        return [_issue(ISSUE_STRUCTURE, SEVERITY_ERROR, "", "图中没有任何节点")]

    # 1. 重复 id
    seen: set[str] = set()
    duplicated: list[str] = []
    for node_id in ids:
        if node_id in seen and node_id not in duplicated:
            duplicated.append(node_id)
        seen.add(node_id)
    for node_id in duplicated:
        issues.append(
            _issue(ISSUE_STRUCTURE, SEVERITY_ERROR, node_id, f"节点 id 重复：{node_id}")
        )

    known = set(ids)

    # 2. 空 id / 空名字（Agent 靠 name 与 description 匹配学生作答）
    for node in graph.nodes:
        if not node.id.strip():
            issues.append(_issue(ISSUE_STRUCTURE, SEVERITY_ERROR, "", "存在空 id 的节点"))
        if not node.name.strip():
            issues.append(
                _issue(ISSUE_STRUCTURE, SEVERITY_WARNING, node.id, f"节点 {node.id} 没有名字")
            )
        if not node.description.strip():
            issues.append(
                _issue(
                    ISSUE_STRUCTURE,
                    SEVERITY_WARNING,
                    node.id,
                    f"节点 {node.id} 没有描述——Step Matcher 只能靠名字猜，容易误判",
                )
            )

    # 3. 边引用不存在的节点 / 自环
    valid_edges: list[tuple[str, str]] = []
    for edge in graph.edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            issues.append(
                _issue(ISSUE_STRUCTURE, SEVERITY_ERROR, "", f"边的形状非法：{edge!r}（应为 [起点, 终点]）")
            )
            continue
        src, dst = str(edge[0]), str(edge[1])
        if src not in known or dst not in known:
            issues.append(
                _issue(
                    ISSUE_STRUCTURE,
                    SEVERITY_ERROR,
                    src if src not in known else dst,
                    f"边 {src}→{dst} 引用了不存在的节点",
                )
            )
        elif src == dst:
            issues.append(_issue(ISSUE_STRUCTURE, SEVERITY_ERROR, src, f"自环 {src}→{src}"))
        else:
            valid_edges.append((src, dst))

    # 4. 孤立节点（既无入边也无出边）
    if len(known) > 1:
        touched = {n for pair in valid_edges for n in pair}
        for node_id in sorted(known - touched):
            issues.append(
                _issue(
                    ISSUE_STRUCTURE,
                    SEVERITY_WARNING,
                    node_id,
                    f"节点 {node_id} 是孤立的（没有任何边连接），学生永远走不到它",
                )
            )

    # 5. 入口与终点
    sources = {s for s, _ in valid_edges}
    targets = {d for _, d in valid_edges}
    entries = known - targets
    terminals = known - sources
    if not entries:
        issues.append(
            _issue(ISSUE_STRUCTURE, SEVERITY_ERROR, "", "没有入口节点（每个节点都有前驱，图是环状的）")
        )
    if not terminals:
        issues.append(
            _issue(ISSUE_STRUCTURE, SEVERITY_ERROR, "", "没有终点节点（每个节点都有后继，学生永远做不完）")
        )

    # 6. 环：解题步骤图应当是 DAG
    cycle = _find_cycle(known, valid_edges)
    if cycle:
        issues.append(
            _issue(
                ISSUE_STRUCTURE,
                SEVERITY_ERROR,
                cycle[0],
                "存在循环：" + "→".join([*cycle, cycle[0]]) + "（解题步骤不该绕回去）",
            )
        )

    if len(known) < MIN_NODES:
        issues.append(
            _issue(
                ISSUE_STRUCTURE,
                SEVERITY_WARNING,
                "",
                f"只有 {len(known)} 个节点，撑不起一次解题过程追踪",
            )
        )

    return issues


def _check_flow(graph: ProblemGraph) -> list[GraphIssue]:
    """解题流程完整性。

    任务书明确说"不是每题都必须机械包含全部类型，而是判断是否存在严重缺步骤"，
    所以这里只抓**明显缺失**：没有结果/计算收尾、没有关键推理（全是计算）。
    """
    issues: list[GraphIssue] = []
    types = {node.type for node in graph.nodes}

    if not ({"calculation", "result"} & types):
        issues.append(
            _issue(
                ISSUE_FLOW,
                SEVERITY_WARNING,
                "",
                "全图没有任何 calculation / result 节点：这道题的图停在建模与推理，"
                "学生算完了也没有落点",
            )
        )
    if "result" not in types:
        # ⚠️ **原来只判 `type == "result"`，实测 5/5 全是误报。**
        #
        # 教材例题 5 道试点：P14 生成的图**几乎从不把节点标成 `result`** ——
        # 它给的是 `modeling` / `key_reasoning` / `calculation`。
        # 于是无论图里有没有收尾步骤，这条都报。例如：
        #   `S10 结果检验与物理合理性判断`   ← 明显的收尾，却被判"缺少 result 节点"
        #   `S6 比较两种力的大小并给出物理结论`
        #
        # **100% 误报的检查项会训练人忽略 P15 的输出** ——
        # 那正是"复核人漏掉真问题"的路径。
        #
        # 改判**流程事实**而非**标签**：看终点（出度为 0 的节点）
        # 的名字/描述有没有体现收尾。**收尾是真的，标签只是命名习惯。**
        _tail = ("结果", "结论", "检验", "求得", "最后", "收尾", "答", "总结")
        _out: dict[str, int] = {n.id: 0 for n in graph.nodes}
        for _e in graph.edges:
            # ⚠️ **必须容错**：题库是数据，畸形边（不是二元组）不能让校验崩。
            # 我第一版写 `for _a, _b in graph.edges` —— 直接把
            # `test_malformed_edge_does_not_crash` 打崩了。
            # **校验器的职责是报告问题，不是被问题搞崩。**
            if isinstance(_e, (list, tuple)) and len(_e) == 2:
                _out[_e[0]] = _out.get(_e[0], 0) + 1
        _terms = [n for n in graph.nodes if _out.get(n.id, 0) == 0]
        _has_end = any(any(w in f"{n.name}{n.description}" for w in _tail)
                       for n in _terms)
        if not _has_end:
            issues.append(
                _issue(ISSUE_FLOW, SEVERITY_WARNING,
                       _terms[0].id if len(_terms) == 1 else "",
                       "流程缺收尾：没有任何节点（含终点）体现结果/结论/检验，"
                       "学生算完了也没有落点")
            )
    if not ({"modeling", "key_reasoning"} & types):
        issues.append(
            _issue(
                ISSUE_FLOW,
                SEVERITY_WARNING,
                "",
                "全图没有任何 modeling / key_reasoning 节点：只有计算步骤，"
                "无法判断学生是否理解物理过程",
            )
        )
    if "key_reasoning" not in types:
        issues.append(
            _issue(
                ISSUE_FLOW,
                SEVERITY_WARNING,
                "",
                "缺少 key_reasoning 节点：没有体现关键物理规律的选择与推理",
            )
        )
    return issues


def _check_paths(graph: ProblemGraph) -> list[GraphIssue]:
    """多路径一致性：路径自身合法，且多条路径能合理汇合。"""
    issues: list[GraphIssue] = []
    known = graph.node_ids()
    edge_set: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            edge_set.add((str(edge[0]), str(edge[1])))

    if not graph.valid_paths:
        return [
            _issue(
                ISSUE_PATH,
                SEVERITY_WARNING,
                "",
                "没有给出 valid_paths：无法判断学生走的路线是否完整，"
                "也失去了'路径一致性'这类评测的依据",
            )
        ]

    for index, path in enumerate(graph.valid_paths, 1):
        label = f"路径{index}"
        unknown = [n for n in path if n not in known]
        if unknown:
            issues.append(
                _issue(ISSUE_PATH, SEVERITY_ERROR, unknown[0], f"{label} 含不存在的节点：{unknown}")
            )
            continue
        broken = [
            (path[i], path[i + 1]) for i in range(len(path) - 1) if (path[i], path[i + 1]) not in edge_set
        ]
        for src, dst in broken:
            issues.append(
                _issue(ISSUE_PATH, SEVERITY_ERROR, src, f"{label} 中 {src}→{dst} 这条边在 edges 里不存在")
            )

    if len(graph.valid_paths) > 1:
        starts = {p[0] for p in graph.valid_paths if p}
        ends = {p[-1] for p in graph.valid_paths if p}
        if len(starts) > 1:
            issues.append(
                _issue(
                    ISSUE_PATH,
                    SEVERITY_WARNING,
                    "",
                    f"多条路径的起点不一致（{sorted(starts)}）——同一道题应当从同一起点出发",
                )
            )
        if len(ends) > 1:
            issues.append(
                _issue(
                    ISSUE_PATH,
                    SEVERITY_WARNING,
                    "",
                    f"多条路径的终点不一致（{sorted(ends)}）——不同解法应当汇合到同一结论；"
                    "若确实是一题多问，请拆成不同的图",
                )
            )

    # 声称有替代解法，图上却只有一条路径 —— 替代路径没有建模成节点。
    # 后果：Step Matcher 靠节点判断学生走哪条路，替代解法不在图里时，
    # 学生用另一种正确解法会被误判成"没按标准路径走"。
    # 这条规则来自 P14 对照成员 B 人工图的评估：5 题中 3 题有这个问题。
    alternatives = graph.metadata.get("alternative_methods") or []
    if alternatives and len(graph.valid_paths) <= 1:
        issues.append(
            _issue(
                ISSUE_PATH,
                SEVERITY_WARNING,
                "",
                f"给出了 {len(alternatives)} 条替代解法，但图上只有一条路径——"
                "替代解法没有建模成分叉节点，学生走另一条正确路线时会被误判",
            )
        )
    return issues


# ---------------------------------------------------------------- LLM 检查

def _check_physics(graph, problem, llm, use_llm: bool):
    """物理一致性（LLM）。返回 (issues, status, meta)。"""
    if not use_llm:
        return [], CHECK_SKIPPED, {}
    if problem is None:
        logger.info("未提供题目，跳过物理一致性检查")
        return [], CHECK_SKIPPED, {}
    if not graph.nodes:
        return [], CHECK_SKIPPED, {}

    try:
        client = llm or get_llm("graph_validator")
        user = _physics_prompt(graph, problem)
        # 与 graph.py / step_matcher.py 保持同一调用口径
        result = client.chat_structured(get_prompt("graph_validator"), user, PhysicsCheck)
    except Exception as exc:  # noqa: BLE001 校验器不该因为模型不可用而让调用方失败
        logger.warning("物理一致性检查不可用（%s: %s），本次降级为仅规则检查",
                       type(exc).__name__, exc)
        return [], CHECK_SKIPPED, {}

    issues = [
        _issue(
            ISSUE_PHYSICS,
            SEVERITY_ERROR if finding.severity == SEVERITY_ERROR else SEVERITY_WARNING,
            finding.node,
            finding.description,
            source="llm",
        )
        for finding in result.findings
        if finding.description.strip()
    ]
    meta = {
        "model": getattr(client, "model", "") or "",
        "provider": getattr(client, "name", "") or "",
    }
    if result.notes.strip():
        logger.info("物理一致性检查附注：%s", result.notes)
    # consistent=False 却没有给出任何 finding 时，按"未通过但原因不明"处理，
    # 不要让"检查失败"被静默吞掉
    if not result.consistent and not issues:
        issues.append(
            _issue(ISSUE_PHYSICS, SEVERITY_WARNING, "", "模型判定物理上不成立，但未说明具体位置")
        )
    return issues, CHECK_PASS if result.consistent else CHECK_ISSUES, meta


def _physics_prompt(graph: ProblemGraph, problem) -> str:
    blocks = [f"【题目】\n{(problem.full_text or '').strip()}"]
    answer = (getattr(problem, "answer", "") or "").strip()
    if answer:
        blocks.append(f"【参考答案】\n{answer}")
    blocks.append(f"【待审校的解题步骤图】\n{graph.render_for_prompt()}")
    blocks.append("请只报告物理上确实站不住的地方。")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- 内部工具

def _issue(issue_type: str, severity: str, node: str, description: str, source: str = "rule") -> GraphIssue:
    return GraphIssue(
        type=issue_type, severity=severity, node=node, description=description, source=source
    )



def _find_cycle(known: set[str], edges: list[tuple[str, str]]) -> list[str]:
    """返回一个环上的节点序列（不含重复的收尾），无环返回 []。"""
    successors: dict[str, list[str]] = {}
    for src, dst in edges:
        successors.setdefault(src, []).append(dst)

    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(known, WHITE)
    stack: list[str] = []

    def visit(node: str) -> list[str]:
        color[node] = GREY
        stack.append(node)
        for nxt in successors.get(node, []):
            if color.get(nxt) == GREY:                      # 回到栈上的点 => 有环
                return stack[stack.index(nxt):]
            if color.get(nxt) == WHITE:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        color[node] = BLACK
        return []

    for node in sorted(known):
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return []


def _confidence(error_count: int, warning_count: int, physics_status: str) -> float:
    """对"这张图能直接用于教学"的信心。

    这是**分诊用的启发式，不是概率**：给教师端排序用，
    别拿它当"71% 正确"来读。
    """
    score = 1.0 - 0.30 * error_count - 0.10 * warning_count
    if physics_status == CHECK_SKIPPED:
        # 最难的一类没查，不能装作有信心
        score = min(score, 0.75) - 0.10
    return round(max(0.05, min(1.0, score)), 2)


def _summarize(graph, checks, errors, warnings, physics_status) -> str:
    parts = [f"{len(graph.nodes)} 个节点 / {len(graph.edges)} 条边"]
    parts.append(
        "检查：" + "、".join(f"{name}={status}" for name, status in checks.items())
    )
    parts.append(f"{len(errors)} 个错误、{len(warnings)} 个警告")
    if physics_status == CHECK_SKIPPED:
        parts.append("物理一致性未检查（无题面或模型不可用），结论仅供结构参考")
    if checks.get("false_branches") == CHECK_SKIPPED:
        parts.append("假分叉未检查（缺少有效 quantities_asked 目标量标注）")
    if checks.get("branch_merges") == CHECK_SKIPPED:
        parts.append("汇合适用性尚有 unknown 路线（缺少完整前置条件标注），未验证物理正确性")
    return "；".join(parts)
