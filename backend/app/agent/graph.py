"""LangGraph 状态驱动教学工作流（P6 条件路由版，任务书第 9 节）。

        New Learning Event（API 层传入 current_answer / request_hint 等）
                ↓
          step_matching（Step Matcher + State Tracker；graphless 时走近似路径）
                ↓ 是否正常推进？
        Yes ────┼──────────────────────→ hint_policy
        No      ↓                            ↓ 是否应该干预？（Hint Policy L0-L5）
          analyze_step                  Yes ↙        ↘ No
                ↓                  generate_hint     update_state
            diagnose（无错误时短路）
                ↓ 是否需要外部知识？
        Yes → retrieve_knowledge（RAG 未实现时降级为空）
        No ──────────────────────────↗
                ↓
          update_state → END

设计要点：
- Hint Policy（P5）决策在图内 hint_policy 节点完成，级别由策略规则决定，LLM 只生成话术
- 正确推进的轮次跳过 analyze/diagnose（省两次 LLM 调用），直接进入干预判断
- Problem / ProblemGraph 通过 configurable 注入；不注入时图退化为 graphless 近似路径，
  保证无 Graph 数据也可离线测试与运行（P8 Graphless Mode）
- 持久化仍由 API 层完成，图保持无副作用
"""
import logging
import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent import state as agent_state
from app.agent.hint_policy import (
    HintPolicyInput,
    decide_hint_level,
)
from app.agent.prompt_registry import get_prompt, resolve_version
from app.agent.state_tracker import apply_step_match
from app.agent.step_matcher import match_step_resilient
from app.agent.schemas import Diagnosis, Hint, StepAnalysis
from app.agent.tools import retrieve_knowledge as retrieve_knowledge_tool
from app.services.concept_normalizer import snap_concept
from app.services.subquestion_context import (resolve_subquestion, context_block as subquestion_block, clarification,
                                             render_missing_fields)
from app.services.llm import get_llm
from app.services.stuck_detector import build_graph_signals
# ⚠️ 复用**复盘那道闸门同一份**判据（`unsupported_claims`）——两边各写一份就会漂开。
#    `review_builder` 不 import 本模块 ⇒ 无循环。
from app.services.review_builder import unfounded_statement_quotes, unsupported_claims

logger = logging.getLogger(__name__)

#: **本地兜底提示**（不调用模型）：面向**行动**，**不断言学生用了什么方法**。
#:
#: 为什么是这句（用户给的契约口径）：
#: ```
#: · hint 要的是"下一步做什么 / 该说明什么依据"，不是"你其实是怎么做的"
#: · 一旦模型写了「说明串联关系掌握得很好」这类**对不可见过程的断言**，
#:   整句就不可信了 —— 而 hint 是学生**当场**照着做的东西，代价比复盘更大
#: ⇒ 检出即替换，且**如实记进 degraded_reasons**（不冒充模型输出）
#: ```
CONSERVATIVE_HINT = (
    "请补写这一步的依据：说明你用到了哪条规律或关系式、代入了哪些已知量，"
    "并检查单位与量纲是否一致。")


def _deps(config: RunnableConfig | None) -> dict[str, Any]:
    """从 configurable 取运行时依赖（Problem / ProblemGraph）。"""
    return (config or {}).get("configurable", {}) or {}


def _problem_block(state: agent_state.TutorState) -> str:
    return f"【题目】\n{state.problem_content or '（未提供题干）'}" + subquestion_block(state.subquestion_context)


#: 兼容旧每命中预算；当前总预算为三倍，即默认1200字符。
RAG_TUTOR_BUDGET = 400


def _render_rag_material(item: dict, budget: int = RAG_TUTOR_BUDGET) -> str:
    """按**实际进入预算的片段**渲染教材材料，**每片标自己的页码**（核查第 1 项）。

    ## 🔴 旧写法错在哪（两条都是被查出来的）
    ```
    body = item["context"] or item["text"]
    f"- {body[:400]}（{item['source']} 第{item['page']}页）"
    ```
    ```
    ① **整段只标一个页码**：`context` 可能含**下一页**的块（实测 4 个跨页窗口），
       却统一标成命中页 ⇒ **来源归属是错的**。
    ② **`[:400]` 之后的内容仍被算作依据**：截掉了却还标着"这是教材依据"，
       而模型根本没看到那部分 ⇒ 引用会指到它没收到的东西上。
    ```
    ⇒ 逐片走 `context_blocks[*].context_start/context_end`（服务端算好的区间），
      **只把真正进预算的片段标出来**；超出的片段**不出现、也不标注**。
    """
    body = str(item.get("context") or item.get("text") or "")
    source = item.get("source", "未知来源")
    spans = list(item.get("context_blocks") or [])
    if not spans:
        # 没有片段目录（旧数据或其他来源）⇒ 退回单页，但**如实说明只给了前 N 字**，
        # 不能把截掉的部分也说成依据。
        head = body[:budget]
        suffix = "（**仅前 %d 字**）" % budget if len(body) > budget else ""
        return f"- {head}{suffix}（{source} 第{item.get('page', '?')}页）"

    delivered, seen_pages = [], []
    for s in spans:
        start, end = int(s.get("context_start", 0)), int(s.get("context_end", 0))
        if start >= budget:
            continue                                  # 完全在窗口之外 ⇒ 不算依据
        cut = min(end, budget)
        text = body[start:cut]
        if not text.strip():
            continue
        page = s.get("page", item.get("page", "?"))
        if page not in seen_pages:
            seen_pages.append(page)
        truncated = "，**该页此处被截断**" if cut < end else ""
        delivered.append(f"「{text}」（{source} 第{page}页{truncated}）")
    if not delivered:
        return f"- （本轮未取到可用片段）（{source}）"
    # ⚠️ 页码列**如实给出全部涉及的页**（不再统一成命中页）。
    pages = "、".join(f"第{p}页" for p in seen_pages)
    return "- " + " ".join(delivered) + f"｜共涉及 {pages}"
#:
#: ⚠️ **链式等式要取最后的值**：`a=6/2=3 m/s²` 里 `a` 写出来的是 **3 m/s²**，不是 6。
#:    第一版只匹配到紧邻的那个数，实测抽成了 `a = 6`（错）。
#: ⚠️ 名字里**必须允许下标与上标**（`U₁` `P₂` `v²`）——
#:    第一版没有，实测把「U₁=4 V，P₂=16 W」整段漏掉了（而那正是缺陷案例里的主角）。
_ASSIGNED = re.compile(r"([A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff₀-₉²³]{0,6})"
                       r"\s*[=＝]\s*([^,，。；\n]{1,40})")
#: 从右侧表达式里取**最后一个**「数值 + 可选单位」。
_VALUE = re.compile(r"(-?\d+(?:\.\d+)?(?:\s*[×x]\s*10\s*\^?-?\d+)?"
                    r"(?:\s*(?:m/s²|m/s\^2|m/s|kg·m/s|kg|J|W|N|A|V|Ω|s|m))?)")


def _event_field(event, name: str) -> str:
    """`timeline` 里的元素**可能是 dict，也可能是 `TimelineEvent` 模型**。

    ⚠️ **第一版只当 dict 处理，实测 56 项测试全红** —— 生产路径传的是
    `TimelineEvent`（pydantic 模型，**没有 `.get`**）⇒ 不只是测试问题，
    生产里也会直接抛 `AttributeError`。**两种都要吃。**
    """
    raw = event.get(name) if isinstance(event, dict) else getattr(event, name, None)
    return str(raw or "")


def _prior_submissions(timeline: list | None) -> tuple[list[str], list[str]]:
    """从**历史提交**里抽出「已经写出的量」与**此前提交的原文**（服务端确定性抽取）。

    ## 🔴 为什么要有这个（2026-10-10 实测缺陷）
    ```
    原先 `generate_hint` **只拿到 `current_answer`（本轮）** ——
      ⇒ 模型**看不见前几轮学生写了什么**，
      ⇒ 于是会让他去求一个**他第 1 轮就已经写出来的量**。
    实测（v14 归档，两评审都扣分）：
      Q028 第 2 轮 hint「式中的 a 目前还是一个待求量」—— 而**第 1 轮已写 a=3 m/s²**
      Q049 hint 问「算损失还缺碰前的什么量」   —— 而**初动能 18 J 第 1 轮已写**
      Q044 第 2 条 hint 重问已在第 1 轮解决的回路电阻合成
    ```
    ⚠️ **性质与 ③族不同**：③族是"说了看不见的"；这个是"**没看见已经写出来的**"。
    后果更直接：**提示会让学生去求一个他已经求出来的量**。

    ⚠️ 数据本来就在手上：`timeline` 里的 `submit` 事件（`content = 该轮作答`）
    早就传进了图，**只是没给模型看** ⇒ 不需要新增字段。
    """
    submits = [_event_field(e, "content") for e in (timeline or [])
               if _event_field(e, "event_type") == "submit" and _event_field(e, "content").strip()]
    # ⚠️ **最后一条是本轮**（调用方在 invoke 前刚追加），要从"此前"里排除。
    prior = submits[:-1] if submits else []
    seen, facts = set(), []
    for text in prior:
        for m in _ASSIGNED.finditer(text):
            name, rhs = m.group(1), m.group(2)
            values = _VALUE.findall(rhs)
            if not values:
                continue
            # **取最后一个** —— 链式等式（`a=6/2=3`）里写出来的是最后那个值。
            value = re.sub(r"\s+", "", values[-1])
            key = f"{name}={value}"
            if key not in seen:
                seen.add(key)
                facts.append(f"{name} = {value}")
    return facts, prior


# ---------------------------------------------------------------- 节点

def step_matching(state: agent_state.TutorState, config: RunnableConfig | None = None) -> dict:
    """Step Matching 调度：按是否有 Problem Graph 显式分流（P8，任务书第 11 节）。"""
    deps = _deps(config)
    problem = deps.get("problem")
    problem_graph = deps.get("problem_graph")
    context = resolve_subquestion(problem, state.current_answer)
    scoped = state.model_copy(update={'subquestion_context': context})
    if context.get('status') == 'conflict':
        return {'subquestion_context': context, 'graph_mode': problem_graph is not None,
                'round_correct': False, 'round_off_graph': False, 'has_obvious_error': None,
                'diagnosed_issue': '', 'issue_type': 'none', 'missing_reasoning': [], 'is_complete': False}
    if context.get('verified_partial'):
        message, _ = render_missing_fields([item['key'] for item in context['result_fields']], context)
        return {'subquestion_context': context, 'graph_mode': problem_graph is not None,
                'round_correct': True, 'round_off_graph': False, 'has_obvious_error': False,
                'diagnosed_issue': '', 'issue_type': 'none', 'missing_reasoning': [message] if message else [],
                'current_node': (context.get('nodes') or [state.current_node])[-1],
                'is_complete': False}
    if context.get('verified_error'):
        return {'subquestion_context': context, 'graph_mode': problem_graph is not None,
                'round_correct': False, 'round_off_graph': False,
                'current_node': (context.get('nodes') or [state.current_node])[-1],
                'is_complete': False}
    if problem is not None and problem_graph is not None:
        if context.get('target'):
            allowed = set(context.get('nodes', [])) | set(problem.metadata.get('shared_nodes', []))
            problem_graph = problem_graph.model_copy(update={
                'nodes': [node for node in problem_graph.nodes if node.id in allowed]})
        updates = graph_mode_matching(scoped, problem, problem_graph)
    else:
        updates = graphless_mode_matching(scoped)
    return {**updates, 'subquestion_context': context}


def graph_mode_matching(
    state: agent_state.TutorState, problem, problem_graph
) -> dict:
    """Graph Mode（任务书第 11 节）：Problem + Graph + Timeline → 精确步骤追踪。"""
    outcome = match_step_resilient(
        problem=problem,
        graph=problem_graph,
        state=state,
        student_input=state.current_answer,
        timeline=state.timeline,
    )
    if outcome.degraded:
        # P20：匹配降级 → **不写学生状态**。
        # 空匹配的语义是"学生什么都没做对"：若采用它，visited_nodes 不推进、
        # consecutive_failures 累加，系统会开始指责一个可能完全正确的学生。
        # 因此这里只声明模式与降级原因，剩下的交给 Hint Policy（降级 → L0 不提示）。
        return {
            "graph_mode": True,
            "round_off_graph": False,
            "degraded_reasons": [
                *state.degraded_reasons,
                f"step_matching 降级：{'；'.join(outcome.reasons) or '模型未返回可用的匹配结果'}",
            ],
        }
    match = outcome.match
    if state.subquestion_context.get('result_fields'):
        missing, _ = render_missing_fields(
            [item['key'] for item in state.subquestion_context['result_fields'] if not item['present']],
            state.subquestion_context)
        # For authored result fields, absence is generated from input spans, not
        # from a model's free-text account of what the student supposedly omitted.
        match = match.model_copy(update={'missing_reasoning': [missing] if missing else []})
    updates = apply_step_match(state, match)
    updates.update(
        {
            "graph_mode": True,
            "round_correct": match.correct,
            "round_off_graph": match.off_graph,
            "last_step_match": match.model_dump(),
        }
    )
    return updates


def graphless_mode_matching(state: agent_state.TutorState) -> dict:
    """Graphless Mode（任务书第 11 节）：无 Graph 时的近似路径。

    本节点**不判断本轮对错**——没有 Graph 就没有匹配结果可依据。
    它只把 `round_correct` 置 False，强制路由走 `analyze_step`；
    真实的对错与连续失败计数由 analyze_step 依据本轮分析结果写入。

    历史坑：这里曾写 `round_correct = state.has_obvious_error is False`。
    `has_obvious_error` 是**上一轮**的分析结论，而且 API 并未把它传进图内
    （恒为 None），于是 `None is False` **恒为 False** →
    学生答对也一直算失败：提示级别 L1→L2→L3 只升不降、连续失败永不归零、
    trace 里 `diagnosis.issue_type='none'` 与"连续未推进 N 轮"自相矛盾。
    """
    return {
        "graph_mode": False,
        "round_correct": False,   # 占位：真实值由 analyze_step 在本轮写入
        "round_off_graph": False,
    }


def analyze_step(state: agent_state.TutorState) -> dict:
    """分析学生当前作答：步骤 / 方法 / 是否有明显错误 / 涉及知识点 / 是否完成。

    graphless 模式下同时负责判定"本轮是否正确推进"与连续失败计数——
    这两项在图模式由 Step Matcher 决定，无 Graph 时只能依据本节点的分析结果。
    """
    if state.subquestion_context.get('verified_error'):
        return {'has_obvious_error': True, 'is_complete': False, 'current_step': '核对水平力方向'}
    llm = get_llm("analyze_step")
    user = (
        f"{_problem_block(state)}\n\n"
        f"【学生当前作答】\n{state.current_answer or '（空）'}"
    )
    outcome = llm.chat_structured_resilient(get_prompt("analyze_step"), user, StepAnalysis)
    if outcome.value is None or outcome.degraded:
        # P20：分析降级 → **不写任何判断**。
        # 关键是不要给 has_obvious_error / is_complete 填默认值：
        # 填 False 会判"学生本轮推进正确"→ 不提示 → 学生卡住却得不到帮助；
        # 填 True 会凭空指责学生。两者都是编造，正确行为是如实记录降级。
        return {"degraded_reasons": [*state.degraded_reasons, *_analyze_reasons(outcome)]}

    analysis = outcome.value
    updates = {
        "current_step": analysis.current_step,
        "current_method": analysis.method,
        "has_obvious_error": analysis.has_obvious_error,
        "concepts": analysis.concepts,
        "is_complete": analysis.is_complete,
        "recommended_next_step": analysis.recommended_next_step,
    }
    if not state.graph_mode:
        round_correct = not analysis.has_obvious_error
        updates["round_correct"] = round_correct
        updates["consecutive_failures"] = (
            0 if round_correct else state.consecutive_failures + 1
        )
        # P23：无 Graph 时没有 Step Matcher 提供节点级的缺失推理，
        # 只能由本节点给出——否则 graphless 下这个字段永远是空的，
        # "结论对但跳过推理"的学生拿不到任何反馈。
        # 有 Graph 时**不要**写：那会覆盖掉 Step Matcher 更精确的结论
        # （step_matching 在本节点之前运行）。
        updates["missing_reasoning"] = list(analysis.missing_reasoning)
    return updates


def _analyze_reasons(outcome) -> list[str]:
    detail = "；".join(outcome.reasons) or "模型未返回可用的分析结果"
    return [f"analyze_step 降级：{detail}"]


def _knowledge_vocabulary(deps: dict[str, Any]) -> list[str]:
    """这道题的知识点词表 —— `concept` 必须归位到它。

    **优先用 Problem Graph 的**（节点上也标了知识点），没有图就用 Problem 的。
    两者都没有则返回空表 —— 调用方据此不做加工。

    ⚠️ 词表要**两边都取**：`phy_001` 那种无图题只有 Problem 上的知识点，
    而 `demo_*` 的图节点上也标了，合起来才完整。
    """
    out: list[str] = []
    graph = deps.get("problem_graph")
    for n in getattr(graph, "nodes", None) or []:
        out += [k for k in (getattr(n, "knowledge_points", None) or []) if k]
    problem = deps.get("problem")
    out += [k for k in (getattr(problem, "knowledge_points", None) or []) if k]
    seen: list[str] = []
    for k in out:
        if k not in seen:
            seen.append(k)
    return seen


def diagnose(state: agent_state.TutorState,
             config: RunnableConfig | None = None) -> dict:
    """错误归因。无明显错误时直接短路，不消耗 LLM 调用。

    ⚠️ **`config` 是为拿这道题的 knowledge_points 而加的** ——
    `concept` 要归位到那个词表，否则画像与推荐对不上（见 `snap_concept`）。
    """
    if state.subquestion_context.get('verified_error'):
        return {'diagnosed_issue': state.subquestion_context['verified_error'], 'issue_type': 'verification',
                'severity': 'low', 'recommended_action': '根据T=0条件核对水平力方向。'}
    if not state.has_obvious_error:
        return {
            "issue_type": "none",
            "diagnosed_issue": "",
            "severity": "low",
            "recommended_action": "",
        }
    llm = get_llm("diagnosis")
    # **把本题知识点喂进提示词**（`diagnosis/v2.md` 要求 `concept` 从中照抄）。
    # 这是"请求对齐"；真正的"保证对齐"是下面的 `snap_concept`。
    _vocab = _knowledge_vocabulary(_deps(config))
    _vocab_block = (
        "【本题知识点】（`concept` 必须**逐字照抄**其中一项；没有合适的就留空）\n"
        + "、".join(_vocab) + "\n\n" if _vocab else ""
    )
    user = (
        f"{_problem_block(state)}\n\n"
        f"{_vocab_block}"
        f"【学生当前作答】\n{state.current_answer}\n\n"
        f"【步骤分析】当前步骤：{state.current_step}；方法：{state.current_method or '未知'}；"
        f"涉及知识点：{'、'.join(state.concepts) or '未知'}"
    )
    diag_outcome = llm.chat_structured_resilient(get_prompt("diagnosis"), user, Diagnosis)
    if diag_outcome.value is None or diag_outcome.degraded:
        # 归因失败时不编造错误类型：宁可"本轮没有诊断结论"，
        # 也不要凭空给出一个 issue_type 让下游照着它检索与提示。
        return {
            "issue_type": "none",
            "diagnosed_issue": "",
            "severity": "low",
            "recommended_action": "",
            "degraded_reasons": [
                *state.degraded_reasons,
                f"diagnose 降级：{'；'.join(diag_outcome.reasons) or '模型未返回可用的归因结果'}",
            ],
        }
    diag = diag_outcome.value
    diagnosis_adjustment = None
    if resolve_version('diagnosis') == 'v5':
        from app.services.hint_citations import FREE_QUOTES
        plain = FREE_QUOTES.sub(lambda match: match.group(0)[1:-1], diag.issue)
        if plain != diag.issue:
            diagnosis_adjustment = {'field':'issue','reason':'显示为模型概括，不声称逐字引用；语义仍需检查',
                                    'original':diag.issue,'displayed':plain,'semantic_validated':False}
            diag = diag.model_copy(update={'issue':plain})
    if diag.finding_kind == 'uncertain':
        return {'diagnosed_issue': '', 'issue_type': 'none', 'has_obvious_error': None,
                'is_complete': False, 'recommended_action': '请核对当前书写与方向约定，信息不足时不判错。'}
    if diag.finding_kind == 'missing_information' and state.subquestion_context.get('result_fields'):
        message, rejected = render_missing_fields(diag.missing_fields, state.subquestion_context)
        return {'diagnosed_issue': message, 'issue_type': 'verification' if message else 'none',
                'severity': 'low', 'recommended_action': message, 'has_obvious_error': False,
                'round_correct': True,
                'missing_reasoning': [message] if message else [],
                'safety_substitutions': [*state.safety_substitutions, *([
                    {'field': 'issue', 'reason': '已写出或未经定义的遗漏项被拒绝',
                     'rejected_fields': rejected}] if rejected else [])]}
    weak_concepts = list(state.weak_concepts)
    recorded = any(e.get("concept") == diag.concept and e.get("issue") == diag.issue
                   and e.get("student_text") == state.current_answer
                   and e.get("round") == sum(event.event_type == "submit" for event in state.timeline)
                   for e in state.error_evidence)

    # ---- ⚠️ **必须把 `concept` 归位到题库的知识点** ----
    #
    # 实测的问题：提示词只举例 `（如"机械能守恒条件"）`，**没有封闭词表**，
    # 模型把例子当前缀自己加限定，同一个知识点被记成三条：
    #   "机械能守恒条件（只有重力或弹力做功）"
    #   "机械能守恒条件（除重力外是否有其他力做功）"
    #   "机械能守恒条件（是否只有重力/弹力做功）"
    #
    # ⇒ 画像越滚越长；而**推荐的判据是「画像知识点 ∩ 题库知识点」**，
    #   两边措辞对不上就直接漏掉。教师端的"反复出错知识点"统计也失真。
    #
    # `diagnosis/v2.md` 里已经**要求**它照抄，但**提示词约束不住输出**
    # （P14 那条"硬性要求一"就没管住）。**代码这层兜底才是保证。**
    _concept = snap_concept(diag.concept, _vocab)   # `_vocab` 在上面构造 user 时已算好
    if (_concept and diag.finding_kind == "observed_error"
            and diag.issue and state.current_answer.strip() and not state.degraded_reasons and _concept not in weak_concepts):
        weak_concepts.append(_concept)
    return {
        "issue_type": diag.issue_type,
        "diagnosed_issue": diag.issue,
        'diagnosis_text_kind': 'model_summary_not_verbatim' if resolve_version('diagnosis')=='v5' else 'legacy',
        'safety_substitutions': [*state.safety_substitutions, *([diagnosis_adjustment] if diagnosis_adjustment else [])],
        "severity": diag.severity,
        "recommended_action": diag.recommended_action,
        "weak_concepts": weak_concepts,
        "error_evidence": ([*state.error_evidence, {
            "concept": _concept, "issue": diag.issue,
            "student_text": state.current_answer,
            "round": sum(e.event_type == "submit" for e in state.timeline),
            "node": state.current_node, "judgement_source": "model_diagnosis",
            "semantic_validated": False,
        }] if _concept and diag.finding_kind == "observed_error" and diag.issue
             and state.current_answer.strip() and not state.degraded_reasons and not recorded
             else state.error_evidence),
    }


#: **需要教材依据**的 issue 类型（用户 2026-10-10 口径第 1 条：明确概念错误、规律适用条件问题）。
_RAG_NEEDED_ISSUES = frozenset({"concept", "modeling", "method"})
#: **不需要检索**的：纯算术核对、纯核对（口径第 5 条 —— **先澄清或核对**）。
_RAG_SKIP_ISSUES = frozenset({"calculation", "verification"})


def needs_textbook_evidence(state: agent_state.TutorState) -> bool:
    """**本轮要不要检索教材** —— 用户 2026-10-10 定的五条口径，逐条落成判据。

    ```
    场景                                 是否检索   本条判据
    明确概念错误、规律适用条件问题          **是**    issue_type ∈ {concept, modeling, method}
    学生主动请求提示                       **是**    state.request_hint（**含当前答案正确**）
    缺少关键推理，且本轮需要生成提示        **是**    missing_reasoning ∧ hint_level > 0（**含图模式**）
    正确推进、无需提示                     **否**    以上都不成立
    仅 OCR 歧义 / 输入缺失 / 纯算术核对     **否**    issue_type ∈ {calculation, verification}
                                                   ⇒ **先澄清或核对**，不检索
    ```

    ⚠️ 原实现是 `has_obvious_error or diagnosed_issue or weak_concepts`：
    ```
    漏：**主动求助**（无图正确但点了「请求提示」⇒ 检索为零；核查报告实证过）
        **缺关键推理**（图模式正确但缺推理 ⇒ 同样检索为零）
    滥：`diagnosed_issue` 非空就检索 ⇒ **OCR 歧义 / 输入缺失 / 纯算术核对也去查教材**
    ```
    ⚠️ `method`（方法问题）**不在用户表格任何一行** —— 我按"它不属于第 5 条的排除项"**放进"要"**，
       并在此写明是为了可复核；若产品认为方法问题不需要教材，改这一行即可。
    """
    issue = (state.issue_type or "").strip()
    if issue in _RAG_SKIP_ISSUES:
        return False                                   # 第 5 条：先澄清或核对
    if issue in _RAG_NEEDED_ISSUES:
        return True                                    # 第 1 条
    if state.request_hint:
        return True                                    # 第 2 条（含答对时主动求助）
    if state.missing_reasoning and int(state.hint_level or 0) > 0:
        return True                                    # 第 3 条（含图模式）
    # 第 4 条：正确推进、无需提示 ⇒ 不检索。
    # ⚠️ 这里**不再用 `diagnosed_issue` 兜底** —— 那正是把 OCR 歧义/纯核对也拖去检索的原因。
    return False


def retrieve_knowledge(state: agent_state.TutorState) -> dict:
    """按需调用 RAG Tool（P7）：只有诊断出错误 / 薄弱点时才会被条件路由进来。

    RAG 是增强层而非硬前提（P20 错误恢复）：检索器未实现、向量库不可用、
    Embedding 模型加载失败等任何异常都只降级为"本轮无教材依据"，
    绝不允许击穿整条工作流——Tutor 仍可基于 Problem / Graph / StudentState 生成提示。

    ## ⚠️ **必须区分状态，不能把故障说成"教材里没有"**（2026-10-10 核查第 2 项）

    ```
    原实现有**两条路都返回空列表**：
        异常分支   return {"rag_query": query, "retrieved_context": []}
        正常无命中 return {"retrieved_context": [], ...}
    ⇒ 下游与页面**分不出**"检索失败了"和"教材里确实没有" ——
      前者是**系统故障**（该重试/该报错），后者是**知识库的正常结论**。
    ```
    ⇒ 现在返回结构化的 `rag_status`（**与审计状态是两套**，别混）：
    ```
    not_triggered    本轮没走检索（没有错误/薄弱点）—— **不是**"查了没有"
    no_hit           查了，知识库里没有相关结果
    failed           检索过程出错（含向量库不可用、embedding 失败）
    ok               查到结果；`rerank` 字段说明重排是否真的生效
    ```
    并带**分阶段耗时**（`timings`）：embedding / 检索 / 重排 / 总耗时。
    """
    if not needs_textbook_evidence(state):
        # ⚠️ **未触发**要显式记 —— 空列表本身说明不了"是没查还是没查到"。
        return {"rag_status": {"state": "not_triggered", "timings": {}}}
    # Correct/help rounds often have a generic step name such as "检查结果".
    # Keep the actual question and writing; no gold block IDs or model rewrite.
    query = state.diagnosed_issue or '\n'.join(filter(None, [
        state.subquestion_context.get('statement') or state.problem_content,
        state.current_answer, state.current_step]))
    _rk: dict = {}
    try:
        results = retrieve_knowledge_tool(
            query=query,
            knowledge_points=state.weak_concepts or None,
            top_k=3,
            status=_rk,
        )
    except NotImplementedError:
        logger.info("RAG 尚未实现，本次跳过知识检索")
        return {"rag_status": {"state": "not_triggered", "reason": "rag_not_implemented",
                               "timings": {}}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG 不可用，本轮降级为无教材上下文（%s: %s）", type(exc).__name__, exc)
        # 🔴 **这是失败，不是"教材里没有"** —— 结构化上报，供页面与 Trace 区分。
        return {"rag_query": query, "retrieved_context": [],
                "rag_status": {"state": "failed", "reason": f"{type(exc).__name__}: {exc}",
                               "timings": {}}}
    # rag_query 记入状态，供 P9 Tracing 定位"是检索错了还是生成错了"
    return {"retrieved_context": results, "rag_query": query,
            "rag_status": {"state": "ok" if results else "no_hit",
                           "returned": len(results),
                           # ⚠️ 重排状态**由重排自己报**（含"明确判无相关"这一支）。
                           #    没有它，空结果分不出
                           #    「向量检索没命中」与「重排判断候选里没有相关材料」。
                           "rerank": _rk.get("rerank_status", "unknown"),
                           "rerank_reason": _rk.get("rerank_reason", ""),
                           "timings": {**_rag_timings(results), **_rk.get('timings', {})}}}


def delivered_rag_sources(retrieved_context: list[dict] | None,
                          budget: int = RAG_TUTOR_BUDGET, *, selected_policy: str | None = None) -> list[dict]:
    """Return the exact catalogue used by Tutor; default total budget is 1200.

    Explicit legacy selection preserves historical 3 x 400 audits. Sources and
    quoted text come from this single plan; this is not semantic validation.
    """
    from app.services.rag_materials import fragments
    return fragments(retrieved_context or [], per_hit_budget=budget, selected_policy=selected_policy)['sources']


def _rerank_state(results: list[dict]) -> str:
    """重排到底跑没跑成功 —— 从工具返回里读，**读不到就 unknown，不猜**。

    ⚠️ 核查报告指出：现有代码注释里"额外约 0.5 秒"是历史测量，不能用；
    而且**逻辑检索成功不代表重排成功**（有超时后按设计回退原序的情况）。
    """
    for r in results or []:
        mark = r.get("rerank") if isinstance(r, dict) else None
        if mark in ("ok", "applied", "success"):
            return "ok"
        if mark in ("fallback", "timeout", "failed"):
            return str(mark)
        if isinstance(r, dict) and r.get('rerank') in {'disabled','no_relevant','unparseable'}:
            return r['rerank']
    return "unknown"


def _rag_timings(results: list[dict]) -> dict:
    """分阶段耗时：工具报了什么就记什么，**没报的不编**。"""
    keys = ("embedding_setup_sec", "embed_sec", "retrieve_sec", "search_sec", "context_sec", "rerank_sec", "total_sec")
    out: dict = {}
    for r in results or []:
        if not isinstance(r, dict):
            continue
        for k in keys:
            if isinstance(r.get(k), (int, float)) and k not in out:
                out[k] = round(float(r[k]), 3)
    return out


def hint_policy(state: agent_state.TutorState, config: RunnableConfig | None = None) -> dict:
    """Hint Policy（P5）：在图内显式决策本轮提示级别（L0-L5）。"""
    if state.subquestion_context.get('status') == 'conflict':
        return {'hint_level': max(1, state.hint_level),
                'hint_reasons': ['小问标号与条件冲突，先核对归属，不判作答错误或增加薄弱点']}
    deps = _deps(config)
    graph_signals = build_graph_signals(deps.get("problem_graph"), state)
    same_hint_before = bool(
        state.current_node
        and state.last_hint_node == state.current_node
        and state.hint_level > 0
    )
    # ---- P21：反向逻辑的触发条件 ----
    # "刚进入新节点" = 本轮匹配到了节点、它就是当前节点、且**还没在本节失败过**。
    # 用 `last_step_match`（本轮匹配结果）而不是别的新字段——它已经在状态里。
    _last = getattr(state, "last_step_match", None) or {}
    matched_now = list((_last.get("matched_nodes") if isinstance(_last, dict)
                        else getattr(_last, "matched_nodes", None)) or [])
    just_entered_node = bool(
        matched_now
        and state.current_node
        and state.current_node in matched_now
        and not same_hint_before
        and state.consecutive_failures == 0
    )
    # "刚取得进展"：本轮推进正确，或当前没有连续失败
    recent_progress = bool(state.round_correct or state.consecutive_failures == 0)
    decision = decide_hint_level(
        HintPolicyInput(
            request_hint=state.request_hint,
            correct=state.round_correct,
            off_graph=state.round_off_graph,
            stuck_score=state.stuck_score,
            consecutive_failures=state.consecutive_failures,
            current_hint_level=state.hint_level,
            is_key_node=graph_signals.is_key_node,
            same_hint_before=same_hint_before,
            missing_reasoning=list(state.missing_reasoning),
            # P20：本轮任何环节降级 → 一律不提示（理由见 HintPolicyInput）
            degraded=bool(state.degraded_reasons),
            # ---- P21：干预控制器需要的上下文 ----
            # `current_node` 与 `hint_history` 一起表达"这个节点上已提示过几次"，
            # 比单个 `same_hint_before` 布尔细。
            current_node=state.current_node,
            hint_history=list(getattr(state, "hint_history", None) or []),
            # 累计错误数用 `node_errors` 求和：没有单独的 `mistakes` 字段，
            # 而 `node_errors` 就是"每个节点错了几次"的权威来源。
            error_count=sum((state.node_errors or {}).values()),
            idle_time=float(getattr(state, "idle_time", 0.0) or 0.0),
            recent_progress=recent_progress,
            just_entered_node=just_entered_node,
        )
    )
    # same_hint_before 是**策略入参**，不是本轮产物：update_state 之后
    # last_hint_node 会被改写成本轮节点，调用方再重算就永远得不到当时的值。
    # 因此把它随结果一起返回，供 Trace 如实记录。
    updates = {
        "hint_level": decision.level,
        "hint_reasons": decision.reasons,
        "hint_same_before": same_hint_before,
        # P21：把本轮决策记进 `hint_history`，供**下一轮**回答
        # "这个节点上已经提示过几次"。**只记真正下发的**（level>0）——
        # 把 withhold 也记上会让"提示过几次"虚高，反而更容易误升级别。
        "hint_history": (
            list(getattr(state, "hint_history", None) or [])
            + ([{"node": state.current_node, "level": decision.level}]
               if decision.level > 0 else [])
        ),
        # P21：动作与抑制理由一并回传，供 Trace 与前端区分
        # "没事发生"与"本来能给、故意不给"。
        "hint_action": decision.action,
        "hint_withheld_reason": decision.withheld_reason,
    }

    # 求助是操作事实，不是知识薄弱或认知卡顿证据。
    if state.request_hint:
        updates["help_events"] = [*state.help_events, {
            "node": state.current_node, "level": decision.level,
            "round": sum(e.event_type == "submit" for e in state.timeline),
            "timestamp": state.timeline[-1].timestamp if state.timeline else 0.0,
        }]
    # 图模式下若还没拿到人类可读的步骤名，用命中节点的 name 补上。
    # 为什么需要：答对轮会跳过 analyze_step，于是 current_step 保持模型默认的 ""，
    # 响应里就成了"current_node=S3 但 current_step 为空"——前端只能显示一个编号。
    # 节点名本来就是给学生看的（如"解出同时跳下后车的速度"），没有理由空着。
    problem_graph = deps.get("problem_graph")
    if problem_graph is not None and not state.current_step and state.current_node:
        node = problem_graph.node_map().get(state.current_node)
        if node is not None and node.name:
            updates["current_step"] = node.name
    return updates


def generate_hint(state: agent_state.TutorState) -> dict:
    """按 Hint Policy 确定的级别生成分级提示（L0 已被条件路由拦下，双保险短路）。"""
    if state.subquestion_context.get('status') == 'conflict':
        return {'hint': clarification(state.subquestion_context), 'hint_level': state.hint_level}
    if state.hint_level <= 0:
        return {"hint": "", "hint_level": 0}
    if (state.hint_level == 1 and state.request_hint and state.is_complete and state.round_correct
            and not state.has_obvious_error and not state.missing_reasoning
            and not state.diagnosed_issue and not state.degraded_reasons):
        # This is an optional reinforcement response, not a new grading verdict.
        # No model call and no invented missing step. Higher levels still use Tutor.
        return {'hint': '本轮分析已标记作答为完成。可选巩固：挑一个已写出的公式，说明它适用的条件。',
                'hint_optional': True,
                'hint_level': state.hint_level,
                'hint_reasons': [*state.hint_reasons, '完整正确作答的L1求助：本地可选巩固提示（非模型文本）']}
    llm = get_llm("tutor")
    materials_in_prompt = False
    context_block = ""
    if state.retrieved_context:
        from app.services.rag_materials import fragments, render
        plan = fragments(state.retrieved_context)
        materials_in_prompt = bool(plan['sources'])
        context_block = "\n\n【教材参考（可引用其思路，不要照抄）】\n" + render(plan)
    node_block = ""
    if state.current_node:
        node_block = f"学生当前所在节点：{state.current_node}；"
    if state.missing_reasoning:
        node_block += "缺失的关键推理：" + "；".join(state.missing_reasoning)
    if node_block:
        node_block = f"\n\n【步骤定位】{node_block}"
    # ⚠️ **把"他已经写出来的东西"交给它**（见 `_prior_submissions` 的 docstring）：
    #    不给这段，模型就会让他去求一个他第 1 轮就已经写出来的量。
    _facts, _prior = _prior_submissions(state.timeline)
    prior_block = ""
    if _facts or _prior:
        prior_block = "\n\n【学生此前已经写出的内容（**服务端从他前几轮的提交里抽取的**，"
        prior_block += "不是你的推断）】\n"
        if _facts:
            prior_block += "已写出的量：" + "、".join(_facts) + "\n"
        for i, text in enumerate(_prior, 1):
            prior_block += f"· 第{i}轮：{text[:200]}\n"
        prior_block += ("⚠️ **不许让他去求上面已经写出来的量**；他已经写出来的那一步，"
                        "就在它的基础上往下走（要提到它时按他的原话，别改写成你的说法）。")
    user = (
        f"{_problem_block(state)}\n\n"
        f"【学生当前作答】\n{state.current_answer}\n\n"
        f"【当前诊断】错误类型：{state.issue_type or 'none'}；"
        f"问题：{state.diagnosed_issue or '无明显错误'}\n\n"
        f"【本次提示级别】hint_level={state.hint_level}"
        f"{node_block}"
        f"{prior_block}"
        f"{context_block}"
    )
    from app.services.hint_contract import render_hint_contract
    user += render_hint_contract(state)
    citation_pool = None
    if resolve_version('tutor') == 'v5':
        from app.services.hint_citations import pool_for_hint, catalogue
        citation_pool = pool_for_hint(state, _prior)
        user += catalogue(citation_pool)
    outcome = llm.chat_structured_resilient(get_prompt("tutor"), user, Hint)
    if outcome.value is None or outcome.degraded:
        # 提示文本没生成出来 → 返回空提示而不是编一句。
        # hint_level 保持不变，让调用方知道"策略本来是要提示的，只是没生成出来"。
        return {
            "hint": "",
            'rag_materials_in_prompt': materials_in_prompt,
            "hint_level": state.hint_level,
            "degraded_reasons": [
                *state.degraded_reasons,
                f"generate_hint 降级：{'；'.join(outcome.reasons) or '模型未返回提示文本'}",
            ],
        }
    hint_text = outcome.value.hint or ""
    original_hint = hint_text
    hint_evidence, citation_issues = [], []
    if citation_pool is not None:
        from app.services.hint_citations import render_hint
        hint_text, hint_evidence, citation_issues = render_hint(hint_text, citation_pool)
    optional = bool(state.is_complete and state.round_correct and not state.has_obvious_error
                    and not state.missing_reasoning and not state.diagnosed_issue
                    and not state.degraded_reasons)
    # ⚠️ **hint 的独立校验 + 本地兜底**（2026-10-10，用户口径）：
    #
    # 复盘那道闸门**管不到 hint** —— 它是教学阶段的主产物。但"拦不住"≠"不设防"：
    # ```
    # · 不能拦：拦了学生连提示都没有了（照片路径尤其如此，提示就是主产物）
    # · 不能重试：**默认不增加模型重试**（多一次付费 + 延迟，且失败还是没提示）
    # ⇒ 换成本地保守提示（**不调用模型**），识别与批改结果**照常返回**。
    # ```
    # ⚠️ 校验用的是与复盘同一份 `unsupported_claims`，所以**它漏的它也漏**
    #    （"把路径当学生结果的来源"那一族没有词法锚点）。
    #    ⇒ 兜底只能挡住"检得出的"，检不出的要靠**提示词减少自由生成的归因性表述**。
    _defects = unsupported_claims(hint_text)
    # ⚠️ **同一道兜底也管"误引题面"**（2026-10-10 实测后加）：
    #    归属给题面的引文**必须逐字在题面里**（契约 §2.7③）。
    #    为什么这一路能上闸门而学生材料那一路不能：**题面是服务端给的**（ground truth），
    #    且归属词让"该去哪个来源里找"**无歧义** —— 实测三批 **3/3 精确、零误报**；
    #    而"归属给学生的引文"实测召回 1~2/3、精度约一半 ⇒ 那一路只走提示词。
    _misquotes = unfounded_statement_quotes(hint_text, state.problem_content or "")
    _misquotes += ['hint_quote_contract:' + issue['kind'] for issue in citation_issues]
    if _defects or _misquotes:
        # ⚠️ **日志必须把两支都打出来**（2026-10-10 实测踩到）：
        #    原实现只打 `_defects`，于是**误引题面单独触发时**日志显示 `[]` ——
        #    看起来像"无缘无故替换"，下次排障会被它误导。
        logger.warning("hint 含无依据断言/误引题面，换成本地保守提示：断言=%s 误引=%s",
                       _defects, _misquotes)
        # 🔴 **记进 `safety_substitutions`，不是 `degraded_reasons`**（用户 2026-10-10 裁决）：
        # ```
        # 原来的写法把它记进 degraded_reasons ⇒ reliable=False ⇒ finished=False
        #   ⇒ 运行器判 "Teaching degraded; not a successful complete case" ⇒ **整条案例判失败**
        #   ⇒ 按可评性口径该案例**退出评分** —— 一个 hint 的安全替换，
        #     把整条案例（含那份可能完全合格的复盘）**挤出了评分范围**。
        # ⇒ 两种状态必须分开：**判断降级** vs **文本安全替换**。
        # ```
        _sub = {
            "field": "hint",
            "detector": "unsupported_claims+unfounded_statement_quotes",
            "reasons": [*_defects, *_misquotes],
            # 审计用：**原模型提示**（学生看不到；学生只看到替换后的保守提示）
            "original": original_hint,
            "prompt_version": resolve_version("tutor"),
        }
        # ⚠️ **"判断没问题"必须有独立依据**（用户口径）：
        #    若替换的触发原因**同时暴露了批改里的无依据判断**（同一判据命中 `diagnosed_issue`），
        #    那就**不能只当提示替换** ⇒ 真实降级照记，两类都留。
        _judgment_defects = unsupported_claims(state.diagnosed_issue or "")
        _out: dict = {
            "hint": ('当前分析已标记作答为完成；以下为可选巩固：' + CONSERVATIVE_HINT) if optional else CONSERVATIVE_HINT,
            'hint_optional': optional,
            'rag_materials_in_prompt': materials_in_prompt,
            "hint_level": state.hint_level,
            "safety_substitutions": [*state.safety_substitutions, _sub],
        }
        if _judgment_defects:
            logger.warning("替换触发原因同时暴露批改越界，按真实降级记录：%s", _judgment_defects)
            _out["degraded_reasons"] = [
                *state.degraded_reasons,
                f"批改含无依据判断（{_judgment_defects}）；同轮提示已由本地兜底替换",
            ]
        return _out
    return {"hint": ('当前分析已标记作答为完成；以下为可选巩固：' + hint_text) if optional else hint_text,
            'hint_evidence': hint_evidence,
            'hint_optional': optional, "hint_level": state.hint_level,
            'rag_materials_in_prompt': materials_in_prompt}


def update_state(state: agent_state.TutorState, config: RunnableConfig | None = None) -> dict:
    """汇总本轮结果写回 StudentState。

    finished 判定：
    - 模型明确判完整时，仍须本轮无明显错误且未降级；
    - graph 模式的终端补判还须已完成一条合法完整路径，且没有缺失推理。
      单独命中末节点不能替代前置步骤或尚未完成的另一个小问。
      替代路径只需完成其中一条；没有 valid_paths 时不猜测分支的 AND/OR 语义。
    同时维护 last_hint_node（重复提示检测）。
    """
    data = state.model_dump()
    if state.subquestion_context.get('status') == 'conflict':
        return {**data, 'finished': False, 'is_complete': False}
    reliable = not state.has_obvious_error and not state.degraded_reasons
    finished = bool(state.is_complete and reliable)
    if state.graph_mode and state.missing_reasoning:
        finished = False
    if (not finished and reliable and state.graph_mode and state.round_correct
            and state.current_node and not state.missing_reasoning):
        problem_graph = _deps(config).get("problem_graph")
        if problem_graph is not None and state.current_node in problem_graph.terminal_nodes():
            visited = set(state.visited_nodes)
            known = problem_graph.node_ids()
            completed_path = any(
                path and path[-1] == state.current_node
                and set(path).issubset(known) and set(path).issubset(visited)
                for path in problem_graph.valid_paths
            )
            if completed_path:
                finished = True
                data["is_complete"] = True
    data["finished"] = finished
    if state.hint_level > 0 and data.get("hint"):
        data["last_hint_node"] = state.current_node
    return data


# ---------------------------------------------------------------- 路由

def route_after_matching(state: agent_state.TutorState) -> str:
    """是否正常推进？正确推进 → 直接进入干预判断；否则先分析/诊断。"""
    return "hint_policy" if state.round_correct or state.subquestion_context.get('status') == 'conflict' else "analyze_step"


def route_after_diagnose(state: agent_state.TutorState) -> str:
    """先决定本轮提示级别，再按统一规则检索；只决策一次。"""
    return "hint_policy"


def route_after_policy(state: agent_state.TutorState) -> str:
    """是否应该干预？L0 = 继续观察不提示；L1-L5 → Tutor 生成提示。"""
    if needs_textbook_evidence(state):
        return 'retrieve_knowledge'
    return "generate_hint" if state.hint_level > 0 else "update_state"


def prepare_round(state: agent_state.TutorState) -> dict:
    """Do not advertise previous-round references as this round's input."""
    return {'retrieved_context': [], 'rag_query': '', 'rag_materials_in_prompt': False,
            'hint_optional': False,
            'hint_evidence': [],
            'diagnosis_text_kind': 'not_generated',
            'rag_status': {'state':'not_triggered','timings':{}}}


def route_after_retrieval(state: agent_state.TutorState) -> str:
    return 'generate_hint' if state.hint_level > 0 else 'update_state'


def build_graph():
    """编译状态驱动教学工作流。独立封装便于测试与后续替换节点。"""
    graph = StateGraph(agent_state.TutorState)
    from app.services.llm.tracking import timed_node
    graph.add_node('prepare_round', timed_node('prepare_round', prepare_round))
    graph.add_node("step_matching", timed_node('step_matching', step_matching))
    graph.add_node("analyze_step", timed_node('analyze_step', analyze_step))
    graph.add_node("diagnose", timed_node('diagnose', diagnose))
    graph.add_node("retrieve_knowledge", timed_node('retrieve_knowledge', retrieve_knowledge))
    graph.add_node("hint_policy", timed_node('hint_policy', hint_policy))
    graph.add_node("generate_hint", timed_node('generate_hint', generate_hint))
    graph.add_node("update_state", timed_node('update_state', update_state))

    graph.add_edge(START, 'prepare_round')
    graph.add_edge('prepare_round', 'step_matching')
    graph.add_conditional_edges(
        "step_matching", route_after_matching, ["analyze_step", "hint_policy"]
    )
    graph.add_edge("analyze_step", "diagnose")
    graph.add_conditional_edges(
        "diagnose", route_after_diagnose, ["hint_policy"]
    )
    graph.add_conditional_edges('retrieve_knowledge', route_after_retrieval, ['generate_hint','update_state'])
    graph.add_conditional_edges(
        "hint_policy", route_after_policy, ["retrieve_knowledge", "generate_hint", "update_state"]
    )
    graph.add_edge("generate_hint", "update_state")
    graph.add_edge("update_state", END)
    return graph.compile()


# 模块级单例：API 层直接 import 使用
tutor_graph = build_graph()
