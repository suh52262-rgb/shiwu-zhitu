"""教学干预控制器（P21）。

提示强度不由 LLM 自由决定：本模块用显式、可解释的规则计算本轮**是否干预**
以及**干预多强**，Tutor（generate_hint 节点）只负责按给定级别生成自然语言。

## P21 相对旧版改了什么

旧版把"是否干预"和"给几级"耦合在一个函数里——`level == 0` 既表示
"没事发生"也表示"故意不给"，调用方分不清。而且它**只会加压**：
失败越多级别越高，缺一条"现在不该打断学生"的反向逻辑。

三处改动：
1. **`action` 字段**：把"是否干预"提升为一等公民，且区分
   `observe`（没事发生）与 `withhold`（**本来能给，故意不给**）
2. **输入补全**：`current_node` / `error_count` / `hint_history` /
   `idle_time` / `recent_progress` —— 旧版这五项一个都没有
3. **反向逻辑**：刚进入新关键节点、且只错一次时**抑制提示**，
   把思考的空间留给学生

## 等级定义（任务书第 8 节）
- L0 不提示，继续观察
- L1 方向提示（方向性提问，不点出具体对象与公式）
- L2 概念提示（指出被忽略的对象/条件/概念，不给公式）
- L3 方法提示（给出可行方法路径，不写具体方程）
- L4 关键步骤提示（给出关键推理/方程，但不完成整题）
- L5 完整讲解（仅在学生主动请求、或多轮提示仍无法推进时提供）
"""
from dataclasses import dataclass, field

# 卡顿分达到该值时，错误作答额外升一级（提前加压）
STUCK_HIGH = 0.75
# 连续失败达到该轮数后，学生主动请求即给完整讲解（L5）
FAILURES_FOR_FULL_EXPLANATION = 4
# 关键节点上连续失败达到该轮数后，额外升一级（关键节点卡住代价高）
KEY_NODE_FAILS = 3
# 无学生主动请求时，自动提示的最高级别（L5 永不自动下发）
AUTO_MAX_LEVEL = 4

#: 刚进入新节点后，允许学生自己尝试的失败轮数。
#: 低于这个轮数**抑制提示**——学生刚到新步骤，一次失误就介入会剥夺他
#: 自己纠错的机会，而且会让他形成"一卡就等提示"的依赖。
GRACE_FAILS_ON_NEW_NODE = 1

# Deprecated compatibility constant; hint levels never determine profile weakness.
WEAK_HINT_THRESHOLD = 2

#: 空转多久才算"学生可能真卡住了"（秒）。
#: 只在**结合失败**时用于判断；单纯发呆不提示（学生可能在想）。
IDLE_considered_stuck = 45.0


# ---------------------------------------------------------------- 动作

#: 干预动作。`observe` 与 `withhold` 的区别是这个模块的核心：
#: 前者是"没事发生"，后者是"**本来能给，故意不给**"。
ACTION_OBSERVE = "observe"      # 不干预：没有需要介入的信号
ACTION_WITHHOLD = "withhold"    # 不干预：**主动抑制**，有明确理由
ACTION_NUDGE = "nudge"          # L1–L2 轻度介入
ACTION_GUIDE = "guide"          # L3–L4 方法/关键步骤
ACTION_EXPLAIN = "explain"      # L5 完整讲解

ACTION_LABELS = {
    ACTION_OBSERVE: "继续观察",
    ACTION_WITHHOLD: "暂不提示",
    ACTION_NUDGE: "轻度引导",
    ACTION_GUIDE: "方法引导",
    ACTION_EXPLAIN: "完整讲解",
}


def action_for_level(level: int) -> str:
    """级别 → 动作。**只在"已决定干预"之后调用**——
    `level=0` 时无法区分 observe 与 withhold，那种区分由决策逻辑给出。"""
    if level <= 0:
        return ACTION_OBSERVE
    if level <= 2:
        return ACTION_NUDGE
    if level <= 4:
        return ACTION_GUIDE
    return ACTION_EXPLAIN


@dataclass
class HintPolicyInput:
    """决策输入（由 API 层从 StepMatch / 卡顿检测 / 会话状态汇总）。"""

    request_hint: bool = False        # 学生主动点击"请求提示"
    correct: bool = False             # 本轮是否正确推进
    off_graph: bool = False           # 本轮输入与题目无关
    stuck_score: float = 0.0
    consecutive_failures: int = 0     # 含本轮的连续未推进轮数
    current_hint_level: int = 0       # 上一轮下发的提示级别
    is_key_node: bool = False         # 当前节点是否关键节点
    same_hint_before: bool = False    # 同一节点上已给过同级提示
    #: 本轮 Step Matcher 指出的缺失推理。"结论对但跳过推理"必须提示，
    #: 否则任务书 Case B / E2 这类学生会**完全得不到任何反馈**（P12 审计发现）
    missing_reasoning: list[str] = field(default_factory=list)
    #: P20：本轮分析是否降级（模型不可用 / 输出解析失败）。
    #: 降级时必须**不提示**——见 decide_intervention 的说明。
    degraded: bool = False

    # ---------------- P21 新增 ----------------
    #: 当前所在节点（用于"同一节点上的提示次数"与报告）
    current_node: str = ""
    #: 该题累计错误数（与 `consecutive_failures` 不同：后者会因一次成功清零）。
    #: 用来区分"一直磕磕绊绊"与"刚刚卡了一下"。
    error_count: int = 0
    #: 已下发的提示记录（按时间顺序，每项含 `node` / `level`）。
    #: 用于"这个节点上已经提示过几次"——比 `same_hint_before` 一个布尔更细。
    hint_history: list[dict] = field(default_factory=list)
    #: 距上次有效操作秒数。**单纯发呆不触发提示**（学生可能在想），
    #: 只在结合失败时用于判断是否真卡住。
    idle_time: float = 0.0
    #: 上一轮是否刚取得进展（刚完成一个节点）。与 `just_entered_node` 配合
    #: 表达"学生正在推进中"。
    recent_progress: bool = False
    #: **本轮刚进入一个新节点**。这是反向逻辑的触发条件。
    just_entered_node: bool = False


@dataclass
class HintDecision:
    level: int = 0
    reasons: list[str] = field(default_factory=list)
    #: 干预动作，取值见 `ACTION_*`。**这是 P21 加的一等公民字段**：
    #: 调用方不必再从 `level == 0` 反推"到底是不该提示还是故意不提示"。
    action: str = ACTION_OBSERVE
    #: 抑制时的明确理由（`action == withhold` 时必须有）
    withheld_reason: str = ""

    @property
    def should_intervene(self) -> bool:
        return self.action in (ACTION_NUDGE, ACTION_GUIDE, ACTION_EXPLAIN)

    @property
    def is_withheld(self) -> bool:
        """**本来能给，故意不给**——与"没事发生"不同。"""
        return self.action == ACTION_WITHHOLD


def _hints_on_node(inp: HintPolicyInput) -> int:
    """该节点上已下发的提示条数。

    比 `same_hint_before`（一个布尔）细：知道"已经提示过 2 次"
    和"提示过 1 次"对是否升级的判断不一样。
    """
    if not inp.current_node:
        return 0
    return sum(1 for h in inp.hint_history
               if isinstance(h, dict) and h.get("node") == inp.current_node)


def _hint_for_request(inp: HintPolicyInput) -> HintDecision:
    """学生**主动请求提示**时给哪一级（P21 反向逻辑：永远不抑制）。

    ## 为什么单独抽出来
    这段原来只写在"判错"那条分支里（现在的 L189 附近），
    而"**本轮推进正确**"分支在它前面就短路掉了：

        if inp.correct:
            ...
            if inp.request_hint:
                return HintDecision(1, ["本轮推进正确，按请求给巩固性方向说明"])

    **后果：答案被判对之后，反复点「请求提示」永远停在 L1。**
    而且会自我延续 —— 返回 L1 就存成 `hint_level=1`，
    下一轮 `current_hint_level` 还是 1，又被那条短路返回 L1。

    **实测就是用户报的现象**："无论点多少次都只给 L1"。

    这直接违反 L187 那条注释的承诺：*主动请求是明确的求助信号，
    压制它等于让学生白问*。**两处必须用同一套升级逻辑。**

    ## 判对时也升级 —— 产品决定
    原来"判对 + 请求"固定给 L1 巩固性说明（`test_correct_answer_with_request_
    gives_reinforcement` 钉着）。**产品方决定改成也逐级升级**，理由是：
    **做对的学生点"请求提示"，意思往往是"我想多懂一点"** ——
    可能是"为什么是这个方向"、"还有没有别的解法"，而不是"我卡住了"。
    固定的 L1 巩固语太浅，点了没变化。

    所以对判对的情形，理由换一套措辞（**深入解释**而非**求助**），
    但**阶梯共用** —— 不该有两套升级规则。
    """
    if (inp.current_hint_level >= 4
            or inp.consecutive_failures >= FAILURES_FOR_FULL_EXPLANATION):
        return HintDecision(
            5,
            ["学生主动请求", f"已提示至 L{inp.current_hint_level} 或连续失败 "
             f"{inp.consecutive_failures} 轮，多轮提示仍未推进 → L5 完整讲解"],
            action=ACTION_EXPLAIN,
        )
    level = min(max(inp.current_hint_level + 1, 2), 4)
    if inp.correct:
        # 判对：措辞是"想更深入"，不是"卡住了"
        return HintDecision(
            level,
            [f"本轮已判对，学生请求更深入的解释 → 从 L{inp.current_hint_level} "
             f"升至 L{level}"],
            action=action_for_level(level),
        )
    return HintDecision(level,
                        [f"学生主动请求提示 → 从 L{inp.current_hint_level} 升至 L{level}"],
                        action=action_for_level(level))



def decide_intervention(inp: HintPolicyInput) -> HintDecision:
    """决定**是否干预**以及**干预多强**。规则顺序即优先级，全部可解释。

    分两段：先处理"明确不干预"的情形（含主动抑制），再算级别。
    """
    reasons: list[str] = []

    # ================= 第一段：明确不干预 =================

    # ---- P20：本轮分析降级 → 一律不提示 ----
    # 为什么不是"给个保守的提示"：降级意味着我们**不知道**学生这一步对不对。
    # 按错处理会冤枉学生（凭空指责），按对处理会漏掉真正卡住的人；两者都是编造。
    if inp.degraded:
        return HintDecision(0, ["本轮分析降级（模型或解析不可用），不提示以免给出错误引导"],
                            action=ACTION_WITHHOLD,
                            withheld_reason="本轮分析不可用")

    # ---- 本轮推进正确：通常不提示，但"结论对而推理缺"必须指出 ----
    if inp.correct:
        if inp.request_hint:
            # Explicit help also escalates when the answer is correct but reasoning
            # is missing; the automatic L1 reminder must not override this request.
            return _hint_for_request(inp)
        if inp.missing_reasoning:
            return HintDecision(
                1,
                ["本轮结论正确但缺少关键推理（"
                 + "；".join(inp.missing_reasoning[:3])
                 + "）→ L1 要求补齐推理链"],
                action=ACTION_NUDGE,
            )
        return HintDecision(0, ["本轮推进正确，L0 继续观察，不提示"],
                            action=ACTION_OBSERVE)

    # ---- 无关输入：不消耗提示，继续观察 ----
    if inp.off_graph:
        return HintDecision(0, ["输入与本题无关，L0 继续观察"], action=ACTION_OBSERVE)

    # ---- P21 反向逻辑：学生主动请求，永远不抑制 ----
    # （主动请求是明确的求助信号，压制它等于让学生白问）
    if inp.request_hint:
        return _hint_for_request(inp)

    # ---- P21 反向逻辑：刚进入新节点，给他自己试的机会 ----
    #
    # **这是旧版完全没有的一条。** 旧版只会加压：失败 1 轮就给 L1。
    # 但学生刚走到新步骤时，一次失误就介入会剥夺他自我纠错的机会，
    # 还会让他形成"一卡就等提示"的依赖。
    #
    # 注意只压制**轻度过失**：连着错超过宽限轮数，还是按常规加压——
    # 抑制不是放任。
    if (inp.just_entered_node
            and not inp.same_hint_before
            and inp.consecutive_failures <= GRACE_FAILS_ON_NEW_NODE
            and _hints_on_node(inp) == 0):
        # 但"真卡住"要区分开：空转很久 + 已经错过，就别再晾着他
        if not (inp.idle_time >= IDLE_considered_stuck and inp.error_count >= 2):
            return HintDecision(
                0,
                [f"刚进入节点 {inp.current_node or '（未知）'}，"
                 f"本轮首次失误 → 暂不提示，把自我纠错的机会留给学生"],
                action=ACTION_WITHHOLD,
                withheld_reason="刚进入新节点，首次失误不介入",
            )

    # ================= 第二段：决定级别 =================
    level = min(max(inp.consecutive_failures, 1), AUTO_MAX_LEVEL)
    reasons.append(f"连续未推进 {inp.consecutive_failures} 轮 → L{level}")

    if inp.stuck_score >= STUCK_HIGH:
        level += 1
        reasons.append(f"卡顿分 {inp.stuck_score:.2f} ≥ {STUCK_HIGH} → 额外升一级")
    if inp.is_key_node and inp.consecutive_failures >= KEY_NODE_FAILS:
        level += 1
        reasons.append(f"卡在关键节点且已失败 {inp.consecutive_failures} 轮 → 额外升一级")
    if inp.same_hint_before:
        level += 1
        reasons.append("同一节点上已给过同级提示 → 升级避免重复")

    # P21：同一节点上提示过多次 → 升级（`same_hint_before` 只说"给过同级的"，
    # 这里管"给过好几条不同级别的"）
    on_node = _hints_on_node(inp)
    if on_node >= 2 and not inp.same_hint_before:
        level += 1
        reasons.append(f"该节点上已提示 {on_node} 次仍未推进 → 升级")

    level = min(level, AUTO_MAX_LEVEL)
    return HintDecision(level, reasons, action=action_for_level(level))


def decide_hint_level(inp: HintPolicyInput) -> HintDecision:
    """兼容旧名。**新代码请用 `decide_intervention`** ——
    这个名字只说了"给几级"，而本模块现在还要回答"该不该给"。"""
    return decide_intervention(inp)
