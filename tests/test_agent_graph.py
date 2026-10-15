"""LangGraph 教学工作流测试：用假 LLM 替换真实调用，不访问网络。"""
import pytest

from app.agent.graph import tutor_graph
from app.agent.schemas import Diagnosis, Hint, StepAnalysis
from app.agent.step_matcher import StepMatch
from fakes import BAD_ANALYSIS, GOOD_ANALYSIS, STEP_MATCH_S5, install_fake_llm

BASE_INPUT = {
    "student_id": "stu_001",
    "problem_id": "phy_001",
    "problem_content": "物体从粗糙斜面顶端由静止滑下……",
    "current_answer": "mgh = 1/2mv^2，所以机械能守恒",
    "hint_level": 1,
}


def test_erroneous_answer_flow(monkeypatch):
    install_fake_llm(monkeypatch)  # 默认 BAD_ANALYSIS + concept 错误
    result = tutor_graph.invoke(BASE_INPUT)

    assert result["current_step"] == "建立能量方程"
    assert result["has_obvious_error"] is True
    assert result["issue_type"] == "concept"
    assert result["diagnosed_issue"] == "未考虑摩擦力做功"
    assert "机械能守恒条件" in result["weak_concepts"]
    assert result["hint"]  # 生成了提示
    assert result["finished"] is False


def test_diagnose_skipped_and_finished_when_no_error(monkeypatch):
    fake = install_fake_llm(monkeypatch, analysis=GOOD_ANALYSIS)
    result = tutor_graph.invoke({**BASE_INPUT, "current_answer": "W_f = 1/2mv^2 - mgh"})

    assert result["issue_type"] == "none"
    assert result["diagnosed_issue"] == ""
    assert result["finished"] is True
    # 无错误时 diagnose 短路：LLM 只被调用 analyze_step 一次（Diagnosis 未被调用）
    assert fake.calls == [StepAnalysis]
    # 答对且做完整题不该再提示。
    # 历史缺陷：graphless 的 round_correct 恒为 False（`None is False`），
    # 于是答对也被当成"连续未推进"，一路 L1→L2→L3 升级并反复下发提示。
    assert result["round_correct"] is True
    assert result["consecutive_failures"] == 0
    assert result["hint_level"] == 0
    assert result["hint"] == ""


def test_correct_round_exposes_current_step_from_graph(monkeypatch):
    """回归：graph 模式的答对轮也必须给出人类可读的步骤名。

    答对轮会跳过 analyze_step，`current_step` 若不从图上取就保持默认的 ""，
    响应于是变成"current_node=S5 但 current_step 为空"——前端只能显示一个节点编号，
    完全看不出学生做到了哪一步。

    这是在实跑 P14 自动生成的图（phy_003，N 人跳车题）时观察到的真实现象：
    学生正确答出第(1)问，响应里 current_node=S3 而 current_step=""。
    """
    install_fake_llm(monkeypatch)          # 默认假 LLM：正确匹配 S5
    result = tutor_graph.invoke(BASE_INPUT, config=_graph_config())

    assert result["round_correct"] is True
    assert result["current_node"] == "S5"

    from app.services.problem_store import load_problem_graph

    node_name = load_problem_graph("phy_002").node_map()["S5"].name
    assert node_name, "phy_002 的 S5 应当有名字"
    assert result["current_step"] == node_name


def test_missing_reasoning_on_correct_step_still_hints(monkeypatch):
    """回归：结论对但跳过推理（任务书 Case B / E2）必须给提示。

    历史缺陷：Hint Policy 没有 missing_reasoning 输入，correct=True 直接返回 L0，
    再叠加 `route_after_matching` 走 hint_policy、`hint_level<=0` 不生成提示，
    这类学生**拿不到任何反馈**——跳步被静默放过。
    而"结论正确 + 缺关键推理"正是任务书第 5 节列明必须支持的情形。
    """
    install_fake_llm(monkeypatch, step_match=STEP_MATCH_S5)   # correct=True 且带缺失推理
    result = tutor_graph.invoke(BASE_INPUT, config=_graph_config())

    assert result["round_correct"] is True          # 结论确实是对的
    assert result["missing_reasoning"]              # 但推理链缺失
    assert result["hint_level"] >= 1, "缺推理却完全不提示"
    assert result["hint"]
    assert any("推理" in reason for reason in result["hint_reasons"])


def test_weak_concepts_accumulate_across_calls(monkeypatch):
    install_fake_llm(monkeypatch, analysis=BAD_ANALYSIS)
    first = tutor_graph.invoke(BASE_INPUT)
    assert first["weak_concepts"] == ["机械能守恒条件"]

    diagnosis = Diagnosis(
        issue_type="calculation",
        issue="代入时漏了负号",
        concept="动能定理",
        severity="medium",
        recommended_action="检查功的正负",
    )
    install_fake_llm(
        monkeypatch,
        analysis=BAD_ANALYSIS,
        diagnosis=diagnosis,
    )
    state = {**BASE_INPUT, "weak_concepts": first["weak_concepts"]}
    second = tutor_graph.invoke(state)
    assert second["weak_concepts"] == ["机械能守恒条件", "动能定理"]


def test_retrieve_not_implemented_is_graceful(monkeypatch):
    install_fake_llm(monkeypatch)

    def _raise(query, top_k=5):
        raise NotImplementedError("Phase 4")

    monkeypatch.setattr("app.agent.tools.retrieve", _raise)
    result = tutor_graph.invoke(BASE_INPUT)
    assert result["retrieved_context"] == []  # 降级为空，不阻塞工作流


def test_retrieve_arbitrary_failure_is_graceful(monkeypatch):
    """P20：向量库不可用 / Embedding 加载失败等任意异常都不得击穿工作流。

    回归背景：此前 retrieve_knowledge 节点只捕获 NotImplementedError，
    Qdrant 或本地向量库抛 PermissionError 时异常会裸冒泡出图，整条链中断。
    """
    install_fake_llm(monkeypatch)

    def _raise(query, top_k=5):
        raise PermissionError("[Errno 13] Permission denied: data/qdrant_local/.lock")

    monkeypatch.setattr("app.agent.tools.retrieve", _raise)
    result = tutor_graph.invoke(BASE_INPUT)

    assert result["retrieved_context"] == []   # 本轮无教材上下文
    assert result["rag_query"]                 # 仍记录查询，便于 Trace 定位
    assert result["hint"]                      # Tutor 仍按 Problem/State 生成了提示


def test_retrieve_results_flow_into_context(monkeypatch):
    """P7：RAG Tool 调用时，薄弱知识点作为 knowledge_points 参与检索。"""
    install_fake_llm(monkeypatch)

    def _fake_retrieve(query, top_k=5):
        assert "未考虑摩擦力做功" in query
        assert "机械能守恒条件" in query  # knowledge_points 拼入检索语句
        return [{"text": "机械能守恒的条件是只有重力做功", "source": "教材.pdf", "page": 85}]

    monkeypatch.setattr("app.agent.tools.retrieve", _fake_retrieve)
    result = tutor_graph.invoke(BASE_INPUT)
    assert result["retrieved_context"][0]["page"] == 85


# ---- P6：条件路由（graph 模式通过 configurable 注入题目与 Graph） ----

def _graph_config():
    from app.services.problem_store import load_problem, load_problem_graph

    return {
        "configurable": {
            "problem": load_problem("phy_002"),
            "problem_graph": load_problem_graph("phy_002"),
        }
    }


def test_correct_graph_round_skips_analyze_diagnose_and_hint(monkeypatch):
    """P6：正确推进 → 跳过 analyze/diagnose，L0 不提示——整轮只调 Step Matcher。"""
    from app.agent.step_matcher import StepMatch

    fake = install_fake_llm(
        monkeypatch,
        step_match=StepMatch(matched_nodes=["S7"], confidence=0.95, correct=True),
    )
    result = tutor_graph.invoke(
        {**BASE_INPUT, "current_answer": "v = d√(k/(m1+m2))", "graph_mode": True},
        config=_graph_config(),
    )

    assert fake.calls == [StepMatch]  # 只有一次 Step Matcher 调用
    assert result["hint"] == ""
    assert result["hint_level"] == 0
    assert result["current_node"] == "S7"
    assert result["visited_nodes"] == ["S7"]


def test_graph_error_round_runs_full_chain(monkeypatch):
    """P6：错误作答 → StepMatch → analyze → diagnose → RAG → hint 全链路。"""
    fake = install_fake_llm(
        monkeypatch,
        step_match=StepMatch(
            matched_nodes=[],
            attempted_nodes=["S10"],
            correct=False,
            missing_reasoning=["分离后振子只剩 P"],
        ),
    )
    result = tutor_graph.invoke(
        {**BASE_INPUT, "current_answer": "分离后振幅还是 d", "graph_mode": True},
        config=_graph_config(),
    )

    from app.agent.step_matcher import StepMatch as StepMatchSchema

    assert fake.calls == [StepMatchSchema, StepAnalysis, Diagnosis, Hint]
    assert result["current_node"] == "S10"  # 错误作答定位到尝试节点
    assert result["visited_nodes"] == []    # 未完成不计入路径
    assert result["missing_reasoning"] == ["分离后振子只剩 P"]
    assert result["finished"] is False


def test_terminal_node_match_finishes_problem(monkeypatch):
    """正确完成一条合法路径，并推进到终端节点（S11），才视为整题完成。"""
    install_fake_llm(
        monkeypatch,
        step_match=StepMatch(
            matched_nodes=["S10", "S11"], confidence=0.95, correct=True
        ),
    )
    from app.services.problem_store import load_problem_graph
    path = next(path for path in load_problem_graph("phy_002").valid_paths if path[-1] == "S11")
    result = tutor_graph.invoke(
        {
            **BASE_INPUT,
            "current_answer": "A = d√(m1/(m1+m2))，E = m1kd²/(2(m1+m2))",
            "graph_mode": True,
            "visited_nodes": path[:-2],
        },
        config=_graph_config(),
    )
    assert result["finished"] is True
    assert result["is_complete"] is True


def test_off_graph_round_gives_no_hint(monkeypatch):
    """P6：与题目无关的输入 → L0 继续观察，不提示、不推进、不累计失败。"""
    install_fake_llm(
        monkeypatch,
        step_match=StepMatch(matched_nodes=[], confidence=0.1, correct=False, off_graph=True),
    )
    result = tutor_graph.invoke(
        {**BASE_INPUT, "current_answer": "老师，这道题什么时候交？", "graph_mode": True},
        config=_graph_config(),
    )
    assert result["hint"] == ""
    assert result["round_off_graph"] is True
    assert result["consecutive_failures"] == 0  # 无关输入不计入失败（State Tracker 设计）


def test_hint_level_0_skips_tutor_via_conditional_route(monkeypatch):
    """P5/P6：L0 → hint_policy 条件路由直接跳过 generate_hint，Hint 不被调用。"""
    fake = install_fake_llm(
        monkeypatch,
        step_match=StepMatch(matched_nodes=["S7"], confidence=0.95, correct=True),
    )
    result = tutor_graph.invoke(
        {**BASE_INPUT, "current_answer": "v = d√(k/(m1+m2))", "graph_mode": True},
        config=_graph_config(),
    )
    assert result["hint"] == ""
    assert result["hint_level"] == 0
    assert Hint not in fake.calls


def test_hint_prompt_carries_level_and_missing_reasoning(monkeypatch):
    """P5/P6：generate_hint 的提示词携带级别、节点与缺失推理。"""
    fake = install_fake_llm(
        monkeypatch,
        step_match=StepMatch(
            matched_nodes=[],
            attempted_nodes=["S10"],
            correct=False,
            missing_reasoning=["缺少分离条件 N=0 的判断"],
        ),
    )
    tutor_graph.invoke(
        {
            **BASE_INPUT,
            "current_answer": "振幅还是 d",
            "graph_mode": True,
            "consecutive_failures": 3,
        },
        config=_graph_config(),
    )
    system, user = fake.prompts[-1]
    assert "Hint Policy" in system
    assert "S10" in user
    assert "N=0" in user


# ================================================================ hint 的独立校验 + 本地兜底
#
# ⚠️ **为什么 hint 要单独设防**（2026-10-10，用户口径）：
#   复盘那道闸门**管不到 hint** —— 它是教学阶段的主产物。但"拦不住"≠"不设防"：
#   ```
#   · 不能拦：拦了学生连提示都没有了（照片路径尤其如此，提示就是主产物）
#   · 不能重试：**默认不增加模型重试**（多一次付费 + 延迟，失败还是没提示）
#   ⇒ 换成本地保守提示（**不调用模型**），识别与批改结果照常返回，
#     并**如实记进 degraded_reasons** —— 不冒充模型输出。
#   ```
# ⚠️ 校验用与复盘**同一份** `unsupported_claims`，所以**它漏的它也漏**
#    （"把路径当学生结果的来源"那一族没有词法锚点）。

@pytest.mark.parametrize("why,kw,want", [
    # ---- 用户 2026-10-10 定的五条口径，逐条钉住 ----
    ("明确概念错误", {"issue_type": "concept", "has_obvious_error": True}, True),
    ("规律适用条件（modeling）", {"issue_type": "modeling"}, True),
    ("学生主动请求提示（**答对也要**）", {"issue_type": "none", "request_hint": True}, True),
    ("缺关键推理且本轮要生成提示（**含图模式**）",
     {"issue_type": "none", "missing_reasoning": ["缺少 N=0 判断"], "hint_level": 2}, True),
    # ---- 不该检索的 ----
    ("缺推理但本轮不生成提示", {"issue_type": "none", "missing_reasoning": ["x"],
                                "hint_level": 0}, False),
    ("正确推进、无需提示", {"issue_type": "none"}, False),
    ("**纯算术核对** ⇒ 先核对，不检索",
     {"issue_type": "calculation", "diagnosed_issue": "4/4=1 这一步算错了"}, False),
    ("**纯核对（verification）** ⇒ 先核对，不检索",
     {"issue_type": "verification", "diagnosed_issue": "代入后再核一遍"}, False),
])
def test_textbook_retrieval_routing_follows_the_product_spec(why, kw, want):
    """⭐ **求助路由的验收**（用户 2026-10-10 定的口径）。

    ```
    原实现 `has_obvious_error or diagnosed_issue or weak_concepts`：
      漏：**主动求助**（无图正确点「请求提示」⇒ 检索为零，核查报告实证过）
          **缺关键推理**（图模式正确但缺推理 ⇒ 同样为零）
      滥：`diagnosed_issue` 非空就查 ⇒ **OCR 歧义 / 输入缺失 / 纯算术核对也去查教材**
    ⇒ 现在逐条落成 `needs_textbook_evidence()`，**不再用 diagnosed_issue 兜底**
      （那正是把纯核对也拖去检索的原因）。
    ```
    """
    from app.agent.graph import needs_textbook_evidence
    from app.agent.state import TutorState

    assert needs_textbook_evidence(
        TutorState(**{"problem_content": "题", **kw})) is want, why


def test_rag_status_distinguishes_failure_from_no_hit():
    """⭐ **不能把故障显示成"教材里没有"**（2026-10-10 RAG 核查第 2 项）。

    ```
    原实现**两条路都返回空列表**：
        异常分支   {"rag_query": q, "retrieved_context": []}
        正常无命中 {"retrieved_context": [], ...}
    ⇒ 下游与页面**分不出**"检索失败了"和"教材里确实没有" ——
      前者是**系统故障**（该重试/该报错），后者是**知识库的正常结论**。
    ```
    """
    from app.agent import graph as G
    from app.agent.state import TutorState

    def _state(**kw):
        # ⚠️ `issue_type` 必须给 —— 新口径下**是否检索由 `needs_textbook_evidence` 决定**
        #    （概念错误/规律适用 ⇒ 要），而不再是"diagnosed_issue 非空就查"。
        return TutorState(**{"has_obvious_error": True, "issue_type": "concept",
                             "diagnosed_issue": "未考虑摩擦力做功",
                             "problem_content": "题", **kw})

    # ① 检索抛异常 ⇒ **failed**，不是 no_hit
    monkeypatch = None
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "retrieve_knowledge_tool",
                   lambda **_k: (_ for _ in ()).throw(RuntimeError("vector store down")))
        got = G.retrieve_knowledge(_state())
    assert got["rag_status"]["state"] == "failed", got
    assert "vector store down" in got["rag_status"]["reason"]
    assert got["retrieved_context"] == []

    # ② 查到但空 ⇒ **no_hit**（这与 failed 是**两回事**）
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "retrieve_knowledge_tool", lambda **_k: [])
        got2 = G.retrieve_knowledge(_state())
    assert got2["rag_status"]["state"] == "no_hit", got2

    # ③ 没走检索 ⇒ **not_triggered**（同样不是"查了没有"）
    got3 = G.retrieve_knowledge(TutorState(problem_content="题"))
    assert got3["rag_status"]["state"] == "not_triggered", got3

    # ④ 有结果 ⇒ ok，并把"实际返回几条"记下来
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "retrieve_knowledge_tool",
                   lambda **_k: [{"text": "摩擦力", "page": 47, "source": "d"}])
        got4 = G.retrieve_knowledge(_state())
    assert got4["rag_status"]["state"] == "ok"
    assert got4["rag_status"]["returned"] == 1


def test_rag_status_does_not_claim_rerank_succeeded_without_evidence():
    """重排**读不到就写 unknown**，不默认成"成功了"。

    ⚠️ 核查报告指出：逻辑检索成功**不代表**重排成功（有超时后按设计回退原序的情况）。
    """
    from app.agent.graph import _rag_timings, _rerank_state

    assert _rerank_state([{"text": "x"}]) == "unknown"          # 没报 ⇒ unknown
    assert _rerank_state([{"text": "x", "rerank": "fallback"}]) == "fallback"
    assert _rerank_state([{"text": "x", "rerank_sec": 0.4}]) == "unknown"  # time alone does not prove success
    # 耗时分阶段：报了什么记什么，**没报的不编**
    t = _rag_timings([{"embed_sec": 0.01, "rerank_sec": 1.5}])
    assert t == {"embed_sec": 0.01, "rerank_sec": 1.5}
    assert _rag_timings([{"text": "x"}]) == {}


def test_the_hint_prompt_carries_what_the_student_already_wrote():
    """🔴 **"没看见已经写出来的量"**（2026-10-10 实测缺陷，两评审都扣分）。

    ```
    Q028 第 2 轮 hint 称「式中的 a 目前还是一个待求量」—— 而**第 1 轮已写 a=3 m/s²**
    Q049 hint 问「算损失还缺碰前的什么量」   —— 而**初动能 18 J 第 1 轮已写**
    ⇒ 根因：`generate_hint` **只拿到 `current_answer`（本轮）**，
      前几轮学生写了什么**根本看不见**（`timeline` 里明明有 `submit` 事件，只是没给它）。
    ```
    ⚠️ 性质与 ③族不同：③族是"说了看不见的"，这个是"**没看见已经写出来的**"——
    后果更直接：**提示会让他去求一个他已经求出来的量**。
    """
    from app.agent.graph import _prior_submissions

    facts, prior = _prior_submissions([
        {"event_type": "submit", "content": "先算 a：合力 10-4=6 N，a=6/2=3 m/s²"},
        {"event_type": "hint_request", "content": "学生主动请求提示"},
        {"event_type": "submit", "content": "现在求速率"},          # ← 本轮，要排除
    ])
    # ① 链式等式取**最后**的值（`a=6/2=3` ⇒ 3，不是 6）
    assert "a = 3m/s²" in facts, f"链式等式取错了：{facts}"
    # ② 本轮**不算**"此前"
    assert prior == ["先算 a：合力 10-4=6 N，a=6/2=3 m/s²"]
    # ③ 下标量名（`U₁` `P₂`）与符号量都要抽到 —— 它们正是缺陷案例里的主角
    facts2, _ = _prior_submissions([
        {"event_type": "submit", "content": "I=2 A，U₁=4 V，P₂=16 W。"},
        {"event_type": "submit", "content": "x"}])
    assert {"I = 2A", "U₁ = 4V", "P₂ = 16W"} <= set(facts2), f"下标量漏了：{facts2}"
    facts3, _ = _prior_submissions([{"event_type": "submit", "content": "W_G=-mgh=-25 J"},
                                    {"event_type": "submit", "content": "x"}])
    assert facts3 == ["W_G = -25J"], f"符号量抽错：{facts3}"


def test_the_hint_prompt_actually_receives_that_block(monkeypatch):
    """上面那段**必须真的进提示词** —— 抽出来却没用上等于没做。"""
    from app.models.student import TimelineEvent

    fake = install_fake_llm(monkeypatch)
    tutor_graph.invoke({**BASE_INPUT,
                        "timeline": [
                            TimelineEvent(timestamp=1.0, event_type="submit",
                                          content="a=6/2=3 m/s²"),
                            # ⚠️ 最后一条是**本轮**（调用方在 invoke 前刚追加）⇒ 不算"此前"。
                            TimelineEvent(timestamp=2.0, event_type="submit",
                                          content="求速率")]})
    system, user = fake.prompts[-1]
    assert "此前已经写出的内容" in user, "没把历史提交交给模型"
    assert "a = 3m/s²" in user, "抽出的量没进提示词"
    assert "不许让他去求上面已经写出来的量" in user, "没写那条禁令"


def test_a_hint_with_an_unfounded_claim_is_replaced_locally(monkeypatch):
    """**验收①**：单独触发提示替换 ⇒ **完成状态不变**，且**不算降级**。

    🔴 为什么这条最要紧：旧实现把替换记进 `degraded_reasons` ⇒ `reliable=False`
    ⇒ `finished=False` ⇒ 运行器判 "not a successful complete case" ⇒ **整条案例判失败**
    ⇒ 按可评性口径**退出评分** —— 一个 hint 的安全替换把整条案例（含那份可能完全合格的复盘）
    **挤出了评分范围**。所以"判断降级"与"文本安全替换"必须是两种状态。
    """
    from app.agent.graph import CONSERVATIVE_HINT

    install_fake_llm(monkeypatch, hint=Hint(
        hint="思路正确，结果也一致，说明串联关系掌握得很好。最后核对一下单位与量纲。"))
    result = tutor_graph.invoke(BASE_INPUT)

    assert result["hint"] == CONSERVATIVE_HINT, "越界提示没有被本地兜底换掉"
    # 🔴 关键断言：**不许写 degraded_reasons**
    assert result["degraded_reasons"] == [], "安全替换被记成了降级 ⇒ 会翻转完成状态、挤出评分"
    subs = result["safety_substitutions"]
    assert subs, "安全替换没被记下来 —— 那就成了偷偷替换"
    assert subs[-1]["field"] == "hint"
    assert subs[-1]["detector"] == "unsupported_claims+unfounded_statement_quotes"
    assert subs[-1]["reasons"], "没记触发原因"
    assert "掌握" in subs[-1]["original"], "审计里没留**原模型提示**"
    assert subs[-1]["prompt_version"], "审计里没留**提示词版本**"
    # 兜底**只换提示**：级别与批改结果照常
    assert result["hint_level"] == 1
    assert result["issue_type"] == "concept"


def test_the_substitution_does_not_change_the_completion_state(monkeypatch):
    """**验收①后半**：替换前后状态必须一致（它只由诊断与降级决定）。

    ⚠️ 夹具必须走**会生成提示**的那条路（`BASE_INPUT` = 有错 + hint_level=1）；
    答对且做完整题时 graph **根本不会生成提示**，拿它对比是测不到东西的。
    """
    install_fake_llm(monkeypatch, hint=Hint(hint="思路正确，说明你已经掌握了。"))
    replaced = tutor_graph.invoke(BASE_INPUT)
    install_fake_llm(monkeypatch, hint=Hint(hint="最后核对一下单位与量纲。"))
    clean = tutor_graph.invoke(BASE_INPUT)

    assert replaced["safety_substitutions"], "夹具没触发替换"
    assert not clean["safety_substitutions"]
    assert replaced["finished"] == clean["finished"], "替换改变了完成状态"
    assert replaced["degraded_reasons"] == clean["degraded_reasons"] == []


def test_a_real_judgment_degradation_survives_the_substitution(monkeypatch):
    """**验收②**：替换与**真实判断降级**同时触发 ⇒ 仍判降级，**不能恢复成功**。

    ⚠️ "判断没问题"必须有**独立依据**：若替换的触发原因**同时暴露了批改里的无依据判断**
    （同一判据命中 `diagnosed_issue`），就**不能只归类为提示替换**。
    """
    from app.agent.graph import CONSERVATIVE_HINT

    install_fake_llm(
        monkeypatch,
        diagnosis=Diagnosis(issue_type="concept", issue="说明学生当时没有犹豫，一步就到位了",
                            concept="机械能守恒条件", severity="low",
                            recommended_action="引导学生检查斜面是否粗糙"),
        hint=Hint(hint="思路正确，说明串联关系掌握得很好。"))

    result = tutor_graph.invoke(BASE_INPUT)

    assert result["hint"] == CONSERVATIVE_HINT
    assert result["safety_substitutions"], "替换本身仍要如实记录"
    reasons = " ".join(result["degraded_reasons"])
    assert result["degraded_reasons"], "批改里也有越界 ⇒ **必须**按真实降级记录"
    assert "批改" in reasons and "替换" in reasons, f"降级理由没说清两者同时发生：{reasons}"


def test_a_clean_hint_passes_through_untouched(monkeypatch):
    """**兜底不许把正常提示也换掉** —— 它只在检出越界时动手。"""
    from app.agent.graph import CONSERVATIVE_HINT

    good = "先写出这一步用到的那条关系式，再把已知量代进去，最后核对单位。"
    install_fake_llm(monkeypatch, hint=Hint(hint=good))
    result = tutor_graph.invoke(BASE_INPUT)

    assert result["hint"] == good
    assert result["hint"] != CONSERVATIVE_HINT
    assert not result["safety_substitutions"]
    assert not result["degraded_reasons"]


def test_the_fallback_hint_itself_is_not_a_violation():
    """兜底文本自己不能被判据判违规（否则每次兜底都在制造新的越界）。"""
    from app.agent.graph import CONSERVATIVE_HINT
    from app.services.review_builder import unsupported_claims

    assert unsupported_claims(CONSERVATIVE_HINT) == []
    # 它应当是**面向行动**的：给"该做什么"，不给"你是怎么做的"
    assert "依据" in CONSERVATIVE_HINT
