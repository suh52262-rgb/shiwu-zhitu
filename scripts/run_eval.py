"""P9 评测运行器（任务书第 13 节 Evaluation Framework）。

五类评测（对应 evaluation/ 下的用例文件）：
- step_matching  Step Matching Accuracy（真实 LLM）
- diagnosis      Diagnosis Accuracy（真实 LLM）
- hint           Hint Quality（策略纯规则 + 真实 LLM 生成）
- retrieval      RAG Recall@K（真实 BGE + Qdrant）
- graph          Graph Path Consistency（真实 LLM，多轮一致性）

用法：
    .venv/Scripts/python scripts/run_eval.py --suite all
    .venv/Scripts/python scripts/run_eval.py --suite retrieval --min-accuracy 0.75
    .venv/Scripts/python scripts/run_eval.py --suite hint --offline   # 只验证策略规则与用例文件

修改 Prompt / 模型 / 工作流后必须重跑本脚本做回归测试。
"""
import argparse
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，输出 ✅/❌ 会抛 UnicodeEncodeError 直接中断评测。
# 必须在任何输出之前重配 stdout/stderr（PYTHONUTF8 不必由调用者记得设置）。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.hint_policy import HintPolicyInput, decide_hint_level  # noqa: E402
from app.agent.state_tracker import apply_step_match  # noqa: E402
from app.agent.step_matcher import match_step  # noqa: E402
from app.models.student import StudentState  # noqa: E402
from app.services.graph_versions import graph_hash  # noqa: E402
from app.services.problem_store import (  # noqa: E402
    load_problem,
    load_problem_content,
    load_problem_graph,
)

EVAL_DIR = ROOT / "evaluation"

# All collectors are reset by main(), so repeated invocations in one process
# cannot silently mix different model/configuration runs.
NODE_RECALL_RESULTS = []
ALTERNATIVE_LABELS = []
STRUCTURE_RESULTS = []
PATH_CASE_RESULTS = []
ALLOW_PROPOSED_LABELS = False


def _record_structure(problem_id, graph):
    """P15 structure status only; no physics/flow score is substituted."""
    version = graph_hash(graph) if graph is not None else None
    graph_id = f"{problem_id}@{version}"
    if any(gid == graph_id for gid, _, _ in STRUCTURE_RESULTS):
        return
    if graph is None:
        PATH_CASE_RESULTS.append({"id": problem_id, "structure": None,
                                  "graph_hash": None, "source": "author_candidate",
                                  "note": "无图，结构指标不适用"})
        return
    from app.services.graph_validator import validate_graph

    report = validate_graph(graph, use_llm=False)
    issues = [i.description for i in report.issues if i.type == "structure"]
    valid = report.checks["structure"] == "pass"
    STRUCTURE_RESULTS.append((graph_id, valid, issues))
    PATH_CASE_RESULTS.append({"id": problem_id, "structure": valid,
                              "graph_hash": version, "source": "author_candidate",
                              "issues": issues, "physics_checked": False})


def _record_path(case, touched, graph, methods=None, accepted=None):
    from app.evaluation.path_metrics import labeled_node_recall
    from app.evaluation.label_audit import label_eligibility
    from app.evaluation.metrics import Rate

    eligible, label_reason = label_eligibility(
        case, graph, allow_proposed=ALLOW_PROPOSED_LABELS)
    rate = (labeled_node_recall(case, touched, graph, methods) if eligible
            else Rate(None, note=f"路线标签未纳入计分：{label_reason}"))
    labels = {**case.get("expected", {}), **case}
    NODE_RECALL_RESULTS.append((case["id"], rate))
    alternative = case.get("is_alternative")
    # The expected correct label must be explicit. Neither note text nor the
    # system's prediction is allowed to manufacture an alternative-path label.
    labeled_alternative = (eligible and alternative is True
                           and case.get("expected", {}).get("correct") is True)
    if labeled_alternative and accepted is not None:
        ALTERNATIVE_LABELS.append((case["id"], accepted))
    PATH_CASE_RESULTS.append({
        "id": case["id"], "touched": sorted(touched),
        "graph_hash": graph_hash(graph) if graph is not None else None,
        "source": "author_candidate",
        "gold_nodes": labels.get("gold_nodes"), "gold_path": labels.get("gold_path"),
        "method": labels.get("method"), "is_alternative": alternative,
        "method_routes": methods or {},
        "label_eligible": eligible, "label_reason": label_reason,
        "annotation": case.get("annotation", {}),
        "node_recall": {"value": rate.value, "numerator": rate.numerator,
                        "denominator": rate.denominator, "note": rate.note,
                        "detail": rate.detail},
        "alternative_accepted": accepted if labeled_alternative else None,
    })


#: P18 三层结构：datasets/ 放用例，reports/ 放报告
DATASETS_DIR = EVAL_DIR / "datasets"
REPORTS_DIR = EVAL_DIR / "reports"


def load_cases(name: str) -> dict:
    """读用例。**优先 datasets/，旧位置兜底**——避免一次性迁移打断既有流程。"""
    for base in (DATASETS_DIR, EVAL_DIR):
        path = base / f"{name}_cases.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"找不到 {name}_cases.json（已找过 {DATASETS_DIR} 与 {EVAL_DIR}）")


def _touched(match) -> set[str]:
    """学生"碰到"的节点。

    **口径来自 `app.agent.node_semantics`，不再自己拼。**
    全项目对"学生到达了哪些节点"有三种口径（完成 / 碰到 / 试了没成），
    各写各的正是数字互相矛盾的根源。见 docs/排查-节点轨迹三种口径.md。
    """
    from app.agent.node_semantics import touched_nodes

    return touched_nodes(match)


# ---------------------------------------------------------------- 各评测器

def eval_step_matching(cases: dict, limit: int) -> tuple[int, int, list[str]]:
    """Step Matching Accuracy：matched/attempted 与期望一致、correct 判定正确。"""
    problem = load_problem(cases["problem_id"])
    graph = load_problem_graph(cases["problem_id"])
    _record_structure(cases["problem_id"], graph)
    passed, total, notes = 0, 0, []
    for case in cases["cases"][:limit]:
        from app.evaluation.label_audit import label_eligibility
        eligible, reason = label_eligibility(case, graph, allow_proposed=ALLOW_PROPOSED_LABELS)
        if not eligible:
            _record_path(case, set(), graph, cases.get("methods"))
            notes.append(f"SKIP {case['id']}: {reason}，未调用模型")
            continue
        total += 1
        exp = case["expected"]
        state = StudentState(visited_nodes=case.get("visited_nodes", []))
        match = match_step(problem, graph, state, case["student_input"])
        touched = _touched(match)
        _record_path(case, touched, graph, cases.get("methods"),
                     accepted=bool(match.correct and not match.off_graph))
        ok = True
        # 只在用例**显式声明**时才校验 correct ——
        # 原写法 `exp.get("correct", match.correct)` 在用例未给该键时是自我比较，恒真，
        # 会让"结论对但缺推理"那类用例（sm_002）少校验一个维度。
        if "correct" in exp:
            ok = match.correct == exp["correct"]
        if "touched_any_of" in exp:
            ok = ok and any(set(alt) <= touched for alt in exp["touched_any_of"])
        if "touched_contains" in exp:
            ok = ok and set(exp["touched_contains"]) <= touched
        if "attempted_contains" in exp:
            ok = ok and set(exp["attempted_contains"]) <= set(match.attempted_nodes)
        if "off_graph" in exp:
            ok = ok and match.off_graph == exp["off_graph"]
        if "missing_reasoning_min" in exp:
            ok = ok and len(match.missing_reasoning) >= exp["missing_reasoning_min"]
        passed += ok
        notes.append(f"{'PASS' if ok else 'FAIL'} {case['id']}: matched={match.matched_nodes} "
                     f"attempted={match.attempted_nodes} correct={match.correct}")
    return passed, total, notes


def eval_diagnosis(cases: dict, limit: int) -> tuple[int, int, list[str]]:
    """Diagnosis Accuracy：错误类型与薄弱知识点归因。"""
    from app.agent.graph import analyze_step, diagnose

    passed, total, notes = 0, 0, []
    problem_content = load_problem_content(cases["problem_id"])
    for case in cases["cases"][:limit]:
        total += 1
        exp = case["expected"]
        state = StudentState(problem_content=problem_content, current_answer=case["student_input"])
        state = state.model_copy(update=analyze_step(state))
        state = state.model_copy(update=diagnose(state))
        ok = True
        if "issue_type" in exp:
            ok = ok and state.issue_type == exp["issue_type"]
        if "issue_type_any_of" in exp:
            ok = ok and state.issue_type in exp["issue_type_any_of"]
        if "concept_any_of" in exp:
            # 子串匹配：LLM 的薄弱点描述常带限定语（如"机械能守恒条件（只有重力或弹力做功）"）
            ok = ok and any(
                keyword in weak
                for keyword in exp["concept_any_of"]
                for weak in state.weak_concepts
            )
        passed += ok
        # **两个口径分开记**：`said` = 系统是否报了错；`real` = 用例标注是否真有错。
        # 假诊断率看的是 said=True 而 real=False 的那些——**冤枉学生**。
        said_error = state.issue_type not in ("", "none", None)
        # **必须显式标注**，缺标注就排除（`None`）。
        # 原写法 `exp.get("issue_type", "none")` 把"没这个键"当成"学生是对的"，
        # 于是真错误用例 dg_003（学生自述代入出错、只是没写 issue_type）
        # 被算成误判——**实测踩过，而且正好违反了本模块自己的原则**：
        # "0 和'没有数据'是两个完全不同的结论"。
        real_error = exp.get("is_error")
        if real_error is None:
            DIAG_LABELS.append((case["id"], said_error, None))
        else:
            DIAG_LABELS.append((case["id"], said_error, bool(real_error)))
        notes.append(f"{'PASS' if ok else 'FAIL'} {case['id']}: type={state.issue_type} "
                     f"weak={state.weak_concepts}")
    return passed, total, notes


def eval_hint(cases: dict, limit: int, offline: bool) -> tuple[int, int, list[str]]:
    """Hint Quality：级别符合策略；L>0 时生成文本不泄露答案、长度受限、包含关键要素。"""
    from app.agent.graph import generate_hint

    passed, total, notes = 0, 0, []
    problem_content = load_problem_content(cases["problem_id"])
    for case in cases["cases"][:limit]:
        total += 1
        exp = case["expected"]
        inp = HintPolicyInput(**case["policy_input"])
        decision = decide_hint_level(inp)
        ok = decision.level == exp["level"]

        if ok and decision.level > 0 and not offline:
            state = StudentState(
                problem_content=problem_content,
                current_answer=case.get("current_answer", case["scenario"]),
                hint_level=decision.level,
                current_node=case.get("current_node", ""),
                missing_reasoning=case.get("missing_reasoning", []),
            )
            hint = generate_hint(state)["hint"]
            text = hint or ""
            forbidden = [f for f in exp.get("forbidden", []) if f in text]
            ok = not forbidden and len(text) <= exp.get("max_len", 400)
            if exp.get("must_include_any"):
                ok = ok and any(k in text for k in exp["must_include_any"])
            note = f"{'PASS' if ok else 'FAIL'} {case['id']}: L{decision.level} len={len(text)}"
            if forbidden:
                note += f" 泄露={forbidden}"
            notes.append(note)
        else:
            notes.append(f"{'PASS' if ok else 'FAIL'} {case['id']}: L{decision.level}"
                         + ("（offline 仅验证策略）" if offline else "（L0 无需生成）"))
        passed += ok
    return passed, total, notes


def eval_retrieval(cases: dict, limit: int) -> tuple[int, int, list[str]]:
    """RAG Recall@K：期望内容是否进入 Top-K。"""
    from app.agent.tools import retrieve_knowledge

    passed, total, notes = 0, 0, []
    for case in cases["cases"][:limit]:
        total += 1
        results = retrieve_knowledge(
            case["query"], knowledge_points=case.get("knowledge_points") or None,
            top_k=case.get("top_k", 3),
        )
        texts = " ".join(r.get("text", "") for r in results)
        gold = case.get("gold_contains_any") or case["expected_contains_any"]
        ok = bool(results) and any(k in texts for k in gold)
        passed += ok
        # **首个相关结果的排名**（1-based）。Hit Rate 只看"进没进 top_k"，
        # 把第 1 位和第 5 位当成一样好；而 Tutor 只把前几条喂给模型，
        # 排名直接决定学生看到什么。所以两个指标都要。
        rank = next(
            (i for i, r in enumerate(results, 1)
             if any(k in (r.get("text") or "") for k in gold)),
            None,
        )
        RETRIEVAL_RANKS.append((case["id"], rank))
        top = results[0] if results else {}
        notes.append(f"{'PASS' if ok else 'FAIL'} {case['id']}: hits={len(results)} "
                     f"top=[{top.get('chapter', '')} p{top.get('page', '?')} {top.get('score', 0)}]")
    return passed, total, notes


def eval_graph(cases: dict, limit: int) -> tuple[int, int, list[str]]:
    """Graph Path Consistency：多轮路径单调增长、不漂移、终态符合预期。"""
    problem = load_problem(cases["problem_id"])
    graph = load_problem_graph(cases["problem_id"])
    _record_structure(cases["problem_id"], graph)
    from app.evaluation.label_audit import label_eligibility
    eligible, reason = label_eligibility(cases, graph, allow_proposed=ALLOW_PROPOSED_LABELS)
    if not eligible:
        _record_path({**cases, "id": cases.get("id", cases["problem_id"] + ":final")},
                     set(), graph, cases.get("methods"))
        return 0, 0, [f"SKIP graph: {reason}，未调用模型"]
    state = StudentState()
    passed, total, notes = 0, 0, []
    prev_visited: list[str] = []
    all_touched: set[str] = set()

    for case in cases["rounds"][:limit]:
        match = match_step(problem, graph, state, case["student_input"])
        all_touched |= _touched(match)
        _record_path(case, _touched(match), graph, cases.get("methods"),
                     accepted=bool(match.correct and not match.off_graph))
        updates = apply_step_match(state, match)
        state = state.model_copy(update=updates)
        visited = state.visited_nodes
        # 已访问节点不消失（不漂移）。
        # 首轮没有前序路径，"空集 ⊆ 任何集合"恒真 —— 那是白送的 PASS 点，
        # 因此首轮**不计入总分**，只标注为 SKIP。
        if prev_visited:
            total += 1
            monotonic = set(prev_visited) <= set(visited)
            passed += monotonic
            notes.append(
                f"{'PASS' if monotonic else 'FAIL'} {case['id']}: path={'→'.join(visited)}"
            )
        else:
            notes.append(f"SKIP {case['id']}: 首轮无前序路径，不做漂移校验")
        prev_visited = visited

    # 终态路径符合预期
    total += 1
    final_ok = any(set(alt) <= set(state.visited_nodes) for alt in cases["expected_contains_any_of"])
    passed += final_ok
    notes.append(f"{'PASS' if final_ok else 'FAIL'} final_path: {'→'.join(state.visited_nodes)}")
    _record_path({**cases, "id": cases.get("id", cases["problem_id"] + ":final")},
                 all_touched, graph, cases.get("methods"))
    return passed, total, notes


def eval_alternative_path(cases: dict, limit: int) -> tuple[int, int, list[str]]:
    """Independent examples: alternative gold identity never comes from output."""
    passed, total, notes = 0, 0, []
    for case in cases["cases"][:limit]:
        problem_id = case.get("problem_id", cases.get("problem_id"))
        problem = load_problem(problem_id)
        graph = load_problem_graph(problem_id)
        _record_structure(problem_id, graph)
        from app.evaluation.label_audit import label_eligibility
        eligible, reason = label_eligibility(case, graph, allow_proposed=ALLOW_PROPOSED_LABELS)
        if not eligible:
            _record_path(case, set(), graph, cases.get("methods"))
            notes.append(f"SKIP {case['id']}: {reason}，未调用模型")
            continue
        state = StudentState(visited_nodes=case.get("visited_nodes", []))
        match = match_step(problem, graph, state, case["student_input"])
        accepted = bool(match.correct and not match.off_graph)
        _record_path(case, _touched(match), graph, cases.get("methods"), accepted)
        expected = case.get("expected", {}).get("correct")
        if not isinstance(expected, bool):
            notes.append(f"SKIP {case['id']}: 缺少 expected.correct 标注")
            continue
        total += 1
        ok = accepted == expected
        passed += ok
        notes.append(f"{'PASS' if ok else 'FAIL'} {case['id']}: accepted={accepted}")
    return passed, total, notes


# ---------------------------------------------------------------- 主流程

#: 额外指标的采集器（评测器填充，主流程汇总）。
#: 用模块级列表而不是改 5 个评测器的返回签名——**改动面小，且不碰既有断言**。
DIAG_LABELS: list[tuple[str, bool, bool]] = []
RETRIEVAL_RANKS: list[tuple[str, int | None]] = []


GOLDEN_RANKS: list[tuple[str, int | None]] = []
GOLDEN_PAGE_RANKS: list[tuple[str, int | None]] = []




def eval_retrieval_golden(cases: dict, limit: int) -> tuple[int, int, list[str]]:
    """**非循环构建的检索评测**：问的是"能否找回出发的那个源段落"。

    gold 是采样时就定下的块 `(doc_id, page, text)`——
    不是"检索返回了什么"。这样才能发现检索真的错了。

    用 `retrieval_golden.json`（由 `scripts/build_retrieval_golden.py` 生成）。
    """
    from app.agent.tools import retrieve_knowledge

    cs = cases["cases"][:limit]
    passed, total, notes = 0, 0, []
    for case in cs:
        total += 1
        gold = case["gold"]
        top_k = case.get("top_k", 5)
        results = retrieve_knowledge(case["query"], knowledge_points=None, top_k=top_k)
        # **严格匹配源段落**：按 (doc_id, page, 文本前 40 字) 比对。
        # 不比整串是因为检索返回的 text 与库里的一致，但为稳妥留一点余量。
        gkey = (gold["doc_id"], gold["page"], (gold["text"] or "")[:40])
        rank = None          # 严格：命中源段落本身
        rank_page = None     # 宽松：命中源段落所在页
        for i, r in enumerate(results, 1):
            rkey = (r.get("source"), r.get("page"), (r.get("text") or "")[:40])
            if rkey == gkey and rank is None:
                rank = i
            if (r.get("source") == gold["doc_id"]
                    and r.get("page") == gold["page"] and rank_page is None):
                rank_page = i
        passed += rank is not None
        GOLDEN_RANKS.append((case["id"], rank))
        GOLDEN_PAGE_RANKS.append((case["id"], rank_page))
        if rank:
            tag = f"rank={rank}"
        elif rank_page:
            # 同页但不是同一块——**多半是"另一块更切题"，不是检索错了**
            tag = f"同页 rank={rank_page}（非同一块）"
        else:
            tag = f"未进 top{top_k}"
        notes.append(f"{'PASS' if rank or rank_page else 'FAIL'} {case['id']}: "
                     f"{tag}  {case['query'][:30]}")
    return passed, total, notes


EVALUATORS = {
    "step_matching": lambda cases, limit, offline=False: eval_step_matching(cases, limit),
    "diagnosis": lambda cases, limit, offline=False: eval_diagnosis(cases, limit),
    "hint": eval_hint,
    "retrieval": lambda cases, limit, offline=False: eval_retrieval(cases, limit),
    "retrieval_golden": lambda cases, limit, offline=False: eval_retrieval_golden(cases, limit),
    "graph": lambda cases, limit, offline=False: eval_graph(cases, limit),
    "alternative_path": lambda cases, limit, offline=False: eval_alternative_path(cases, limit),
}
LLM_FREE_SUITES = {"hint"}  # offline 模式下仍可运行（策略为纯规则）


def main() -> None:
    global ALLOW_PROPOSED_LABELS
    parser = argparse.ArgumentParser(description="视物知途回归评测（P9）")
    parser.add_argument("--suite", default="all",
                        choices=["all", *EVALUATORS], help="评测套件")
    parser.add_argument("--limit", type=int, default=99, help="每个套件最多跑多少 case（控制成本）")
    parser.add_argument("--min-accuracy", type=float, default=0.7, help="低于该准确率则退出码 1")
    parser.add_argument("--offline", action="store_true",
                        help="离线冒烟：不调用 LLM/检索，只验证用例文件与纯规则部分")
    parser.add_argument("--include-proposed-labels", action="store_true",
                        help="显式候选标签试算；报告隔离，不代表人工 gold 基线")
    parser.add_argument("--manifest", type=Path, help="验证冻结输入并应用模型/提示词配置；真实运行需预算 approved=true")
    args = parser.parse_args()
    manifest = None
    if args.manifest:
        import os
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_eval import verify
        from app.config import get_settings
        from app.agent.prompt_registry import clear_cache
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors = verify(manifest, ROOT)
        if errors:
            parser.error("manifest 输入漂移：" + ", ".join(errors))
        if args.suite not in manifest["suites"] or args.limit != manifest["limit"]:
            parser.error("suite/limit 必须与 manifest 一致（按单套件执行）")
        expected_scope = "proposed_exploratory" if args.include_proposed_labels else "reviewed_only"
        if (args.min_accuracy != manifest["evaluation_config"]["min_accuracy"]
                or expected_scope != manifest["evaluation_config"]["label_scope"]):
            parser.error("标签范围及 min_accuracy 必须与 manifest 一致")
        if not args.offline and not manifest.get("budget", {}).get("approved"):
            parser.error("manifest 预算尚未批准；仅允许 --offline")
        if not args.offline and "PENDING" in manifest["model_config"]["llm_model"]:
            parser.error("manifest 模型仍是待选择占位符")
        settings = get_settings()
        endpoint = manifest.get("endpoint", {})
        current_url = (settings.deepseek_base_url if settings.llm_provider == "deepseek"
                       else settings.llm_base_url)
        if endpoint and (endpoint.get("provider") != settings.llm_provider
                         or endpoint.get("base_url") != current_url):
            parser.error("当前 provider/endpoint 与冻结 manifest 不一致")
        for name, value in manifest["model_config"].items():
            setattr(settings, name, value)
        for role, version in manifest["prompt_versions"].items():
            os.environ[f"PROMPT_{role.upper()}_VERSION"] = version
        clear_cache()
    ALLOW_PROPOSED_LABELS = args.include_proposed_labels
    for collector in (NODE_RECALL_RESULTS, ALTERNATIVE_LABELS, STRUCTURE_RESULTS,
                      PATH_CASE_RESULTS, DIAG_LABELS, RETRIEVAL_RANKS,
                      GOLDEN_RANKS, GOLDEN_PAGE_RANKS):
        collector.clear()

    suites = list(EVALUATORS) if args.suite == "all" else [args.suite]
    report: list[str] = []
    overall_ok = True
    label_coverage = {}
    usage_by_suite = {}
    suite_results = {}

    for suite in suites:
        cases = load_cases(suite)
        if suite in {"step_matching", "graph", "alternative_path"}:
            from app.evaluation.label_audit import audit_dataset
            label_coverage[suite] = audit_dataset(cases, load_problem_graph)
        if args.offline and suite not in LLM_FREE_SUITES:
            if suite in {"step_matching", "graph", "alternative_path"}:
                entries = cases.get("cases", [])[:args.limit] or [cases]
                for case in entries:
                    pid = case.get("problem_id", cases.get("problem_id"))
                    _record_structure(pid, load_problem_graph(pid))
                report.append(f"[{suite}] 模型匹配 SKIPPED（offline）；作者候选图结构检查已执行，不代表发布版本质量")
            else:
                report.append(f"[{suite}] SKIPPED（offline 模式跳过，需要真实 LLM/检索）")
            continue
        evaluator = EVALUATORS[suite]
        kwargs = {"offline": args.offline} if suite == "hint" else {}
        from app.services.llm.tracking import track_usage
        with track_usage() as usage:
            try:
                passed, total, notes = evaluator(cases, args.limit, **kwargs)
            except Exception as exc:
                overall_ok = False
                passed, total, notes = 0, 0, [f"ERROR {type(exc).__name__}: {str(exc)[:500]}"]
                suite_results[suite] = {"error": notes[0]}
        usage_by_suite[suite] = usage.as_dict()
        suite_results.setdefault(suite, {"passed": passed, "total": total,
                                        "accuracy": passed / total if total else None})
        if not total:
            report.append(f"[{suite}] n/a（没有已标注、可计分的用例；未执行阈值判断）")
            report.extend(f"    {note}" for note in notes)
            continue
        accuracy = passed / total if total else 0.0
        ok = accuracy >= args.min_accuracy
        overall_ok = overall_ok and ok
        report.append(
            f"[{suite}] {passed}/{total} = {accuracy:.0%} {'✅' if ok else '❌ 低于阈值'}"
        )
        report.extend(f"    {note}" for note in notes)

    # ---- P18：额外指标 ----
    from app.evaluation.metrics import (
        METRIC_SPECS, diagnosis_accuracy, false_diagnosis_rate, hit_rate, mrr,
        alternative_path_acceptance, structure_valid_rate,
    )

    from datetime import datetime, timezone

    extra: dict[str, object] = {}
    from app.evaluation.path_metrics import aggregate_node_recall
    extra["node_recall"] = aggregate_node_recall(NODE_RECALL_RESULTS)
    extra["alternative_path_acceptance"] = alternative_path_acceptance(ALTERNATIVE_LABELS)
    extra["structure_valid_rate"] = structure_valid_rate(STRUCTURE_RESULTS)
    if DIAG_LABELS:
        extra["diagnosis_accuracy"] = diagnosis_accuracy(DIAG_LABELS)
        extra["false_diagnosis_rate"] = false_diagnosis_rate(DIAG_LABELS)
    if RETRIEVAL_RANKS:
        extra["hit_rate"] = hit_rate([(q, r is not None) for q, r in RETRIEVAL_RANKS])
        extra["mrr"] = mrr(RETRIEVAL_RANKS)
    if GOLDEN_RANKS:
        # **与老的 4 条分开报**：混在一起就看不出 60 条上的真实水平。
        # 严格与宽松两个口径都报——只报一个都会误导：
        # 只报严格会低估检索，只报宽松会掩盖真问题。
        extra["golden_hit_strict"] = hit_rate(
            [(q, r is not None) for q, r in GOLDEN_RANKS])
        extra["golden_hit_page"] = hit_rate(
            [(q, r is not None) for q, r in GOLDEN_PAGE_RANKS])
        extra["golden_mrr"] = mrr(GOLDEN_RANKS)

    # **未产出的指标要显式列出来**，而不是静默缺席——
    # 看报告的人分不清"没跑"与"没实现"，那是最容易造成误解的地方。
    not_produced = []
    if not extra.get("diagnosis_accuracy"):
        not_produced.append("diagnosis_accuracy / false_diagnosis_rate（需跑 diagnosis 套件）")
    if not (extra.get("hit_rate") or extra.get("golden_hit_strict")):
        not_produced.append("hit_rate / mrr（需跑 retrieval 或 retrieval_golden 套件）")
    for name in ("node_recall", "alternative_path_acceptance", "structure_valid_rate"):
        if not extra[name].computable:
            not_produced.append(f"{name}（{extra[name].note}）")

    if extra:
        report.append("")
        report.append("── P18 指标（含口径与分母）" + "─" * 30)
        for name, rate in extra.items():
            label, semantics = METRIC_SPECS.get(name, (name, ""))
            report.append(f"  {label:<28}{rate.render()}")
            if semantics:
                report.append(f"      口径：{semantics}")
            report.extend(f"      {d}" for d in rate.detail[:6])

    if not_produced:
        report.append("")
        report.append("── 本次未产出的指标" + "─" * 36)
        report.extend(f"  · {x}" for x in not_produced)

    print("\n".join(report))

    # ---- 落盘：**结果不能只 print**。没有报告就没有跨版本的对比。----
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "graph_source": "author_candidate",
        "label_scope": "proposed_exploratory" if ALLOW_PROPOSED_LABELS else "reviewed_only",
        "manifest": {"path": str(args.manifest), "status": manifest["status"]} if manifest else None,
        "graph_scope_note": "图评测使用 load_problem_graph 读取的作者候选图，不代表已发布版本质量",
        "generated_at": stamp,
        "offline": args.offline,
        "limit": args.limit,
        "min_accuracy": args.min_accuracy,
        "overall_ok": overall_ok,
        "suites": suites,
        "not_produced": not_produced,
        "path_cases": PATH_CASE_RESULTS,
        "label_coverage": label_coverage,
        "suite_results": suite_results,
        "token_usage": usage_by_suite,
        "usage_note": "归集 provider 记录的调用及 token；网络失败/SDK 内部尝试可能无 usage，非账单金额",
        "console": report,
        "metrics": {
            name: {
                "value": rate.value,
                "numerator": rate.numerator if rate.value is not None else None,
                "denominator": rate.denominator if rate.value is not None else None,
                "note": rate.note,
                "detail": rate.detail,
                "semantics": METRIC_SPECS.get(name, ("", ""))[1],
            }
            for name, rate in extra.items()
        },
    }
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        mode = "offline" if args.offline else "full"
        is_full = (args.suite == "all")
        # **只有全量运行才配更新 `latest-full.json`。**
        # 单套件运行写它会把基线覆盖成残缺报告——实测跑一次
        # `--suite retrieval_golden` 之后，基线里只剩 3 个指标，
        # 看的人会以为诊断/检索/提示的指标不存在。
        stable = (f"latest-{mode}.json" if is_full
                  else f"latest-suite-{args.suite}.json")
        if ALLOW_PROPOSED_LABELS:
            stable = f"candidate-{stable}"
        payload["suite_scope"] = ("all" if is_full else args.suite)
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        out = REPORTS_DIR / f"{stamp}-{'candidate-' if ALLOW_PROPOSED_LABELS else ''}{args.suite}-{mode}.json"
        out.write_text(body, encoding="utf-8")
        (REPORTS_DIR / stable).write_text(body, encoding="utf-8")
        print(f"\n报告已落盘：{out.relative_to(ROOT)}"
              + (f"（基线另存 {stable}）" if is_full
                 else f"（单套件运行，写 {stable}，**不动全量基线**）"))
    except Exception as exc:  # noqa: BLE001 落盘失败不该让评测本身判失败
        print(f"\n⚠ 报告落盘失败：{type(exc).__name__}: {exc}")

    print(f"\nEVAL {'OK' if overall_ok else 'FAILED'}"
          + ("（offline 冒烟）" if args.offline else ""))
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
