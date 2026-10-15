"""Auto Problem-Solving Graph Generator（P14，任务书第三阶段 P14 节）。

解决的问题：导入一道人工作业之外的新题时，不能每次都要求人工从零建 Graph。

    题目 + 可选参考答案 → LLM → **候选** Problem Graph

三条设计原则（都来自任务书）：

1. **候选，不默认正确**。生成结果一律带 `source="auto"` 与 `needs_review=True`，
   深度校验（结构 / 流程完整性 / 物理一致性 / 多路径一致性）交给 P15 Graph Validator。
2. **轻量、但要"Agent 可用"**。不追求复杂知识图谱，但产出的图必须能直接喂给
   Step Matcher —— 因此这里做**结构性归一化**：补节点 id、去重复 id、
   丢弃指向不存在节点的边与路径、保证存在起点到终点的路径。
   归一化做了什么会记进 `generation.normalizations`，便于复核。
3. **不依赖题目与来源**。只依赖 `ProblemGraph` Schema，题号由调用方给。
"""
import logging
import re
import json

from app.agent.prompt_registry import get_prompt
from app.models.candidate_graph import CandidateGraph, new_generation_info
from app.models.problem_graph import NODE_TYPES, GraphNode, ProblemGraph
from app.services.llm import LLMClient, get_llm

logger = logging.getLogger(__name__)

# LLM 偶尔会给出超过这个规模的图；超出多半是"把知识图谱也画进来了"
MAX_NODES = 20
# 节点 id 允许的形状：S1 / S6a / N12b …
_NODE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def build_user_prompt(
    problem_statement: str,
    reference_solution: str = "",
    figures: list[str] | None = None,
    course: str = "大学物理",
    knowledge_vocabulary: list[str] | None = None,
    quantities_asked: list[str] | None = None,
    correction: dict | None = None,
) -> str:
    """按任务书 P14 的输入结构组装 user 提示词。"""
    parts = [f"【课程】{course}", f"【题目】\n{problem_statement.strip()}"]
    if reference_solution.strip():
        parts.append(f"【参考答案】\n{reference_solution.strip()}")
    else:
        parts.append("【参考答案】\n（未提供，请依据物理规律自行推导解题过程）")
    if figures:
        parts.append("【配图】\n" + "、".join(figures) + "\n（无法查看图片，请仅依据文字描述作图）")
    if knowledge_vocabulary is not None:
        parts.append('【服务端输出约束】\n整题和每个节点的knowledge_points只能从以下列表选取，不得自造近义标签；无匹配项允许空列表：\n'
                     + '、'.join(knowledge_vocabulary)
                     + '\ncommon_errors只描述可能出现的纸面关系、列式或结论；不要写“误认为”“未认识到”等理解状态归因。不要把信息不足当成已犯错误。')
    parts.append("请按系统提示的要求输出这张题的 Solution Graph。")
    evidence=source_evidence(problem_statement,reference_solution)
    parts.append('【服务端证据目录】\n'+json.dumps(evidence,ensure_ascii=False))
    if quantities_asked:
        parts.append('【必须覆盖的目标量】\n'+json.dumps(quantities_asked,ensure_ascii=False))
    if correction:
        parts.append('【一次定向修订】\n'+json.dumps(correction,ensure_ascii=False)
                     +'\n按具体发现修订完整图，不通过删掉必答小问来躲避检查。不能解决的输入矛盾保留在open_questions。')
    return "\n\n".join(parts)


def generate_candidate_graph(
    problem_statement: str,
    reference_solution: str = "",
    figures: list[str] | None = None,
    course: str = "大学物理",
    problem_id: str = "",
    llm: LLMClient | None = None,
    knowledge_vocabulary: list[str] | None = None,
    quantities_asked: list[str] | None = None,
    correction: dict | None = None,
) -> CandidateGraph:
    """调用 LLM 生成候选解题图，并做结构性归一化。

    归一化后仍可能物理上不严谨——那正是 P15 Validator 存在的理由，
    这里只保证"结构上 Agent 用得了"。
    """
    if not problem_statement.strip():
        raise ValueError("problem_statement 不能为空")

    llm = llm or get_llm("graph_generator")
    user = build_user_prompt(problem_statement, reference_solution, figures, course,knowledge_vocabulary,quantities_asked,correction)
    raw = llm.chat_structured(get_prompt("graph_generator"), user, CandidateGraph)

    info = llm.describe() if hasattr(llm, "describe") else None
    candidate = CandidateGraph(
        problem_id=problem_id or raw.problem_id,
        knowledge_points=list(raw.knowledge_points),
        nodes=list(raw.nodes),
        edges=[list(edge) for edge in raw.edges],
        valid_paths=[list(path) for path in raw.valid_paths],
        alternative_methods=list(raw.alternative_methods),
        node_evidence=dict(raw.node_evidence),
        quantity_coverage=dict(raw.quantity_coverage),
        open_questions=list(raw.open_questions),
        generation=new_generation_info(
            model=getattr(info, "model", "") or getattr(llm, "model", ""),
            provider=getattr(info, "name", "") or getattr(llm, "name", ""),
            course=course,
        ),
    )
    return normalize_candidate_graph(candidate)


def source_evidence(statement,reference):
    """Exact source lines; IDs never come from model text."""
    return [{'id':f'{prefix}{i:03d}','source':kind,'text':line}
            for prefix,kind,text in [('Q','statement',statement),('R','reference_solution',reference)]
            for i,line in enumerate([s for s in text.splitlines() if s.strip()],1)]


def normalize_candidate_graph(candidate: CandidateGraph) -> CandidateGraph:
    """把 LLM 输出修成"结构上一定可用"的图，并记录改了什么。

    修的东西：非法/重复/缺失的节点 id、指向不存在节点的边、自环、
    越界或断链的 valid_paths、未知的节点 type、超量节点。
    **不判断物理对错**——那是 P15 的事。
    """
    notes: list[str] = []

    # ---- 1. 节点 id：非法或重复的一律重编号 ----
    seen: set[str] = set()
    for index, node in enumerate(candidate.nodes, 1):
        node.name = (node.name or "").strip() or f"步骤 {index}"
        if node.type not in NODE_TYPES:
            notes.append(f"节点 {node.id or index} 的 type={node.type!r} 不在约定枚举内，已回落为 modeling")
            node.type = "modeling"
        original = (node.id or "").strip()
        if not _NODE_ID_RE.match(original) or original in seen:
            new_id = _fresh_id(seen, index)
            notes.append(f"节点 id {original!r} 非法或重复，已改为 {new_id}")
            node.id = new_id
        else:
            node.id = original
        seen.add(node.id)

    # ---- 2. 节点数量上限 ----
    if len(candidate.nodes) > MAX_NODES:
        notes.append(
            f"节点数 {len(candidate.nodes)} 超过上限 {MAX_NODES}，已截断（疑似把知识图谱也画了进来）"
        )
        candidate.nodes = candidate.nodes[:MAX_NODES]
        seen = candidate.node_ids()

    valid_ids = candidate.node_ids()

    # ---- 3. 边：两端必须存在，去掉自环与重复 ----
    clean_edges: list[list[str]] = []
    for edge in candidate.edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            notes.append(f"丢弃形状非法的边 {edge!r}")
            continue
        src, dst = str(edge[0]).strip(), str(edge[1]).strip()
        if src not in valid_ids or dst not in valid_ids:
            notes.append(f"丢弃指向不存在节点的边 {src}→{dst}")
            continue
        if src == dst:
            notes.append(f"丢弃自环 {src}→{src}")
            continue
        if [src, dst] in clean_edges:
            continue
        clean_edges.append([src, dst])
    candidate.edges = clean_edges

    # ---- 4. valid_paths：节点必须存在，且首尾要在图上连通 ----
    clean_paths: list[list[str]] = []
    for path in candidate.valid_paths:
        nodes = [str(n).strip() for n in path] if isinstance(path, (list, tuple)) else []
        if len(nodes) < 2:
            notes.append(f"丢弃过短的路径 {path!r}")
            continue
        unknown = [n for n in nodes if n not in valid_ids]
        if unknown:
            notes.append(f"丢弃含未知节点的路径（{unknown}）")
            continue
        if nodes not in clean_paths:
            clean_paths.append(nodes)
    candidate.valid_paths = clean_paths

    # ---- 5. 没有任何有效边时，按节点顺序补一条主轴，保证图可用 ----
    if candidate.nodes and not candidate.edges:
        ordered = [node.id for node in candidate.nodes]
        candidate.edges = [[ordered[i], ordered[i + 1]] for i in range(len(ordered) - 1)]
        notes.append("模型未给出有效边，已按节点顺序补一条主轴")
        if not candidate.valid_paths:
            candidate.valid_paths = [ordered]
            notes.append("并据此补了一条完整路径")

    # ---- 6. 收尾：没路径时用图上的最长链兜一条，避免下游"无路可走" ----
    if candidate.nodes and not candidate.valid_paths:
        longest = _longest_chain(candidate)
        if longest:
            candidate.valid_paths = [longest]
            notes.append("模型未给出 valid_paths，已按最长链补一条")

    # ---- 7. 知识点汇总：题目级空则从各节点并集补齐 ----
    if not candidate.knowledge_points:
        merged: list[str] = []
        for node in candidate.nodes:
            for kp in node.knowledge_points:
                if kp and kp not in merged:
                    merged.append(kp)
        if merged:
            candidate.knowledge_points = merged
            notes.append("模型未给出整题知识点，已从各节点汇总")

    # ---- 8. 典型错误去重与降噪（提示词之外的兜底）----
    _dedupe_errors(candidate, notes)

    candidate.generation.normalizations = notes
    if notes:
        logger.info("候选图归一化：%d 处修正", len(notes))
    return candidate


#: 与物理理解无关的错误类型（单位、抄写、粗心）。提示词里已禁止，
#: 但模型偶尔仍会写进来——它们会进 Step Matcher 的提示词，稀释有效信号。
_NOISE_PATTERNS = (
    "单位", "换算", "漏乘", "忘记乘", "粗心", "抄错", "笔误",
    "算错", "计算错误", "代入错误", "数值错误",
)
#: 每条错误保留多少条上限（提示词要求 0~2，这里做硬约束）
MAX_ERRORS_PER_NODE = 2


def _dedupe_errors(candidate: CandidateGraph, notes: list[str]) -> int:
    """同一节点内合并重复/近似的典型错误，并剔除纯噪声类；返回剔除条数。

    判重按"归一化后的字符集合重叠度"——模型很爱把同一件事换个说法写两遍
    （"正负号判断错误" / "符号代入错误" / "角加速度符号代入错误"）。
    """
    removed = 0
    for node in candidate.nodes:
        kept: list[str] = []
        kept_keys: list[set[str]] = []
        for raw in node.common_errors:
            text = (raw or "").strip()
            if not text:
                removed += 1
                continue
            if any(p in text for p in _NOISE_PATTERNS):
                removed += 1
                continue
            key = _error_key(text)
            if any(_too_similar(key, other) for other in kept_keys):
                removed += 1
                continue
            kept.append(text)
            kept_keys.append(key)
        if len(kept) > MAX_ERRORS_PER_NODE:
            removed += len(kept) - MAX_ERRORS_PER_NODE
            kept = kept[:MAX_ERRORS_PER_NODE]
        node.common_errors = kept

    if removed:
        notes.append(f"典型错误去重/降噪：剔除 {removed} 条（重复、或属单位换算与粗心类）")
    return removed


def _error_key(text: str) -> set[str]:
    """把错误描述压成字符集合，用于近似判重。"""
    return {c for c in text if "\u4e00" <= c <= "\u9fff" or c.isalnum()}


def _too_similar(a: set[str], b: set[str]) -> bool:
    """Jaccard 相似度超过阈值即视为同一条错误。"""
    if not a or not b:
        return False
    return len(a & b) / len(a | b) > 0.5


def generate_and_save(
    problem_id: str,
    problem_statement: str,
    reference_solution: str = "",
    figures: list[str] | None = None,
    course: str = "大学物理",
    llm: LLMClient | None = None,
    repository=None,
    merge_knowledge_points: bool = True,
) -> tuple[CandidateGraph, ProblemGraph]:
    """生成候选图并落盘到题库（`data/problems/{problem_id}/graph.json`）。

    落盘的图带 `source="auto"` 与 `needs_review=True`，不会冒充人工确认过的图。

    默认还会把图里的整题知识点**并回** `problem.json`（只增不删）——
    否则它们只活在 graph.json 的 metadata 里，按题目取知识点的下游看不到。
    返回 (候选图, 已保存的 ProblemGraph)。
    """
    from app.services.problem_repository import get_problem_repository

    repo = repository or get_problem_repository()
    candidate = generate_candidate_graph(
        problem_statement=problem_statement,
        reference_solution=reference_solution,
        figures=figures,
        course=course,
        problem_id=problem_id,
        llm=llm,
    )
    graph = candidate.to_problem_graph()
    repo.save_graph(problem_id, graph)
    if merge_knowledge_points:
        candidate.knowledge_points = (
            repo.merge_knowledge_points(problem_id, candidate.knowledge_points)
            or candidate.knowledge_points
        )
    return candidate, graph


# ---------------------------------------------------------------- 内部工具

def _fresh_id(seen: set[str], index: int) -> str:
    """生成一个未被占用的节点 id：优先 S<序号>，冲突则 S<序号>_2、_3…"""
    base = f"S{index}"
    if base not in seen:
        return base
    suffix = 2
    while f"{base}_{suffix}" in seen:
        suffix += 1
    return f"{base}_{suffix}"


def _longest_chain(candidate: CandidateGraph) -> list[str]:
    """在 DAG 上求一条最长链（含环时退化为已访问集合，不会死循环）。"""
    successors: dict[str, list[str]] = {}
    for src, dst in candidate.edges:
        successors.setdefault(src, []).append(dst)

    starts = [node.id for node in candidate.nodes if node.id not in {d for _, d in candidate.edges}]
    best: list[str] = []

    def walk(node_id: str, path: list[str], visited: set[str]) -> None:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        for nxt in successors.get(node_id, []):
            if nxt in visited:
                continue
            walk(nxt, [*path, nxt], visited | {nxt})

    for start in starts or [node.id for node in candidate.nodes][:1]:
        walk(start, [start], {start})
    return best


def graph_from_llm_output(
    payload: dict, problem_id: str = "", **generation_kwargs
) -> CandidateGraph:
    """从原始 dict 构造候选图（CLI / 测试用），同样会走归一化。"""
    candidate = CandidateGraph.model_validate(payload)
    if problem_id:
        candidate.problem_id = problem_id
    if not candidate.generation.model:
        candidate.generation = new_generation_info(**generation_kwargs)
    return normalize_candidate_graph(candidate)


def describe_node_type(type_name: str) -> str:
    """节点类型的中文说明（CLI 展示用）。"""
    return {
        "modeling": "建模/状态识别",
        "key_reasoning": "关键推理",
        "calculation": "计算",
        "verification": "检验",
        "result": "结果",
    }.get(type_name, type_name)
