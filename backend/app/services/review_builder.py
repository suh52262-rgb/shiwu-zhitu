"""离线复盘：把一整段作答轨迹变成**一份复盘**，而不是一串逐轮提示。

## 与实时提示的区别（这是本模块存在的全部理由）
```
实时  L1「先想一想：静止时绳中张力是否为零？」        ← **提问**，逼学生自己想
离线  「这里你把张力当成了零，但静止时它必须平衡 m₂ 的重力」← **陈述**，直接纠正
```
视频是**录完再上传**的：学生录的时候看不到提示，等他看到时题早做完了。
**六级递进在那儿没有作用对象** —— 它是在对一个不在场的人做渐进引导。
⇒ **诊断复用，话重说。**

## 为什么要"归并卡点"
实测 9~15 轮的逐轮记录，往往只指向**一两个概念错误**。
逐轮列出来学生要从十几行里自己找规律；**归并成 1~3 处**才有用。

## ⚠️ 必须把**学生实际写了什么**喂进去
第一版我猜字段名猜错了（取了 `current_answer`，实际是 `student_input`），
输入里学生作答是空的 —— **而产出"看着依然通顺"**，只是卡点描述变成
"或者…或者…"（在猜）。修好之后卡点从"猜三个可能"变成"指出凑对"。
> **"输出看着像样"不能证明"输入是对的"。**
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class StickingPoint(BaseModel):
    where: str = Field(description="哪一问/哪一步，学生看得懂的说法")
    what: str = Field(description="他做错的具体是什么（陈述句，不要提问）")
    why: str = Field(description="只核对物理关系和适用条件；不得解释学生的认识、选择或未展示过程")
    downstream: str = Field(default="", description="兼容字段；新版填空，不推断未展示的后续过程")
    # ---- v3：让"引用"变成真的 ----
    #
    # ## 为什么加这一个字段就够
    # v3 的结构（`evidence` / `findings` / `validation`）已经在了，
    # 但 `findings` 是**服务端从 sticking_points 映射**来的，
    # 所以 `evidence_ids` **只能是空的** —— 而空引用的"校验"等于没校验。
    #
    # ⇒ 不重写报告结构（那会动 schema、动缓存、动前端），
    #   **只给卡点加一个"你指的是哪几条证据"** —— 纯增量、有默认值、旧报告照样读。
    #
    # **模型只能引用服务端给的 ID**（目录在提示词里给出）。
    # 填错/编造的 ID 由 `build_validation` 的 `dangling_reference` 抓出来。
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="这条判断依据的证据 ID（**只能取自服务端给的目录**）")


class AnswerCheck(BaseModel):
    """**结论与推导的自洽性** —— `review/v2` 新增，专治「凑对」。

    起因是一次真实的误判：学生写 `F = 49 N`（**对的**），
    而识别文本把 `49` 读成 `4`，复盘据此说成"没有物理意义的结果" ——
    **把"做对了"说成了"错得离谱"。**

    ⇒ 复盘不能只转述结论，要**自己用学生的方程推一遍再比对**。
    **这个检查不需要知道正确答案，只需要内部一致性。**
    """

    student_answer: str = Field(
        description="**逐字引用**学生最终写的结论，不要改写、不要替他算")
    derived_from_work: str = Field(description="你自己用他的方程推出来的结果")
    consistent: str = Field(description="三选一：一致 / 不一致 / 无法判断")
    note: str = Field(default="",
                      description="不一致时说明差在哪；一致时说明推导成立")
    ocr_caution: str = Field(
        default="",
        description="识别可疑之处（同一处前后读法不同、出现 ?、数值不合理）。"
                    "**宁可说看不准，也不要拿可能读错的数下结论。**")


class ReviewReport(BaseModel):
    _raw_generation: dict | None = PrivateAttr(default=None)
    render_policy: str = 'legacy'
    canonicalized_fields: list[str] = Field(default_factory=list)
    timeline: str = Field(description="他做了什么，两三句")
    sticking_points: list[StickingPoint] = Field(description="0~3处有证据的错误或待核查点；没有明确证据时为空")
    key_concept: str = Field(description="最该补的一个知识点，逐字取自列表")
    next_step: str = Field(description="下一步做什么")
    encouragement: str = Field(description="一句实在的话，不要空话")
    # ⚠️ **给默认值** —— v1 的提示词不产出这个字段，
    # 而登记表允许用环境变量切回 v1 做 A/B。
    # **没有默认值的话，切回 v1 会直接解析失败。**
    answer_check: AnswerCheck | None = Field(
        default=None, description="结论与推导是否自洽（v2 起）")
    # ---- v3 三个结构：**服务端填，模型不产出** ----
    #
    # ⚠️ **光有默认值不够，必须配合出口摘除**（见 `api/video.py` 的
    # `_validated_review_payload`）。我第一版只加字段就接线，**打挂 21 条缓存测试** ——
    # `model_dump()` 会把它们加给**每一条**响应，包括缓存里的旧报告，
    # 而契约写着「**历史 v1/v2 不伪装成带证据报告**」。
    review_schema_version: str = Field(default="", description="报告结构版本")
    evidence: list[dict] = Field(default_factory=list,
                                 description="服务端生成的证据目录（v3 起）")
    findings: list[dict] = Field(default_factory=list,
                                 description="问题或建议，引用证据 ID（v3 起）")
    validation: dict = Field(default_factory=dict,
                             description="结构与引用校验结果（v3 起）")



# ================================================================ 服务端统计
#
# ## 为什么要有这一段（契约硬约束④）
# > **轮次、重复次数、时间范围由服务端统计，不由模型编造。**
#
# 起因是一次真实的计数错（审计 docs/review_packets/2026-10-09/）：
# `ocr_caution` 说某数值出现在「四轮片段」，实际是**五轮**（第 4、5、6、7、8 轮）。
#
# **这不是"模型不小心"，是任务本身不该给它** ——
# 让模型从自由文本里数"第几轮出现了几次"它做得不稳，
# **而服务端手里有全部轮次，数得准。** ⇒ 把"数数"从模型手里拿走，改成喂给它。

#: 抓数值 token：整数/小数，可带科学计数法（粗口径，与 OCR 抖动无关）
#: 抓数值 token —— **必须连符号一起抓**。
#  ⚠️ 第一版不抓符号，于是 `-19.6` 与 `+19.6` 都被归成 `19.6`，
#  **方向信息在统计里丢了** —— 而复盘恰恰要判方向。
_NUM_RE = re.compile(
    r"[-−+]?\s*\d+(?:\.\d+)?(?:\s*[×xX*]\s*10\s*\^?\s*[-−]?\s*\d+)?")


def _norm_num(s: str) -> str:
    """归一化数值**写法** —— 只改写法，**不改数值、不丢符号**。

    ⚠️ `^` 必须留着。第一版把它删了，`1.98×10^4` 显示成 `1.98×104` ——
    **那是在改数值，不是在归一化写法。**
    """
    out = s.replace(" ", "").replace("x", "×").replace("X", "×").replace("*", "×")
    return out.replace("−", "-")


def _looks_like_a_quantity(tok: str) -> bool:
    """**单字符/双位整数一律丢掉。**

    ## 为什么（实测出来的）
    冒烟时从 `(1) I=1.98*10^4 N·s (2) F=I/Δt=1.65e5` 里抓出了 `1`、`2`、`5` ——
    **那是题号编号和指数残片，不是物理量。**
    把它们列进「数值出现情况」**会误导模型**，而这份统计**全部的意义就是可信**。

    ## 代价（如实写在这里）
    **会漏掉 `m₂ = 2 kg` 这类真量。** 但：
    · 漏了 ⇒ 模型本来也不知道，它只会说"无法从材料确认"（**口径允许**）
    · 混进噪声 ⇒ 模型会**拿编号当物理量去引用**
    ⇒ **宁可漏，也不要让它引错。**
    """
    body = tok.lstrip("-+")          # 符号不算长度
    if "." in body or "×10" in body:
        return True
    return body.isdigit() and len(body) >= 3


# ================================================================ v3 证据结构（壳）
#
# ## 本次只做「壳」，为什么
# 完整 v3 要改提示词、schema、缓存身份 —— 那是独立的一块。
# **但前端和后端不该互相等**，所以先把三个结构定下来并接进响应：
# ```
# evidence      **服务端生成**（ID / 来源 / 修订号 / 轮次 / 时刻 / 原文）
# findings      从现有 sticking_points + answer_check **映射**，**不开新提示词**
# validation    结构与引用检查 + 上一版只能进 logger 的那两条
# ```
# ⇒ 前端可以照 `findings.category` 先做显示；
#   而 `validation` 让「模型自己写了次数」这件事**第一次能被看见**。
#
# ⚠️ **没做的**：模型侧仍输出 `sticking_points`（老结构），
#    `findings` 是服务端映射来的 —— **不是模型按证据 ID 引用生成的**。
#    ⇒ 所以「引用校验」现在只能校**服务端自己填的引用**，
#      **还不是真正的"模型不许编造引用"**。那条要等模型侧改到 v3。

REVIEW_SCHEMA_VERSION = "v3"


def review_schema_version(prompt_version: str) -> str:
    """报告版本 = **结构版本 + 实际跑过的提示词版本**（契约 §2.3）。

    ## 为什么必须绑在一起
    默认提示词是 **v2**，而 **v2 不要求模型引用证据 ID** ⇒ 那些报告的
    `evidence_ids` 全是空的。它们**结构上确实是 v3**（三个字段都在），
    但**不满足 v3 提示词想要的东西**。

    旧版无论跑哪版提示词都标一个 `"v3"` —— 读的人会以为
    「这份报告有真正的引用校验」，**而那正是本轮要堵的洞**
    （`valid=true` 不能保证每项指控都有依据）。

    ⇒ 版本串里带上提示词版本，读的人一眼能看出这份报告是在什么条件下生成的。
      `REVIEW_SCHEMA_VERSION` 仍是**结构**版本，判定"是不是 v3 结构"用 `startswith`。
    """
    return f"{REVIEW_SCHEMA_VERSION}+prompt-{prompt_version or 'unknown'}"

#: finding 的三类（契约固定）
CATEGORY_OBSERVED = "observed_error"
CATEGORY_UNCERTAIN = "uncertain"
CATEGORY_SUGGESTION = "suggestion"

#: 「这条判断指不回原文」的**固定说明后缀**。
#:
#: ⚠️ **追加处（`build_findings`）与计数处（`count_downgraded_claims`）共用它** ——
#: 两处各写一遍字符串，改一处就会静默失配（计数永远为 0，而没人看得出来）。
DOWNGRADE_NOTE = "（**这条找不到对应的原文引用**，需人工核对）"

#: 学生材料的**总预算**（契约 §2.3 硬约束⑨）。
#:
#: ⚠️ 旧版**没有总预算**：轨迹每轮截 300 字，而证据目录发**完整原文** ⇒
#: 同一份材料进了两次，**整体输入反而更长（净增，不是净减）**。
#: 现在两者由**同一个** `build_material()` 产出，只进一次。
REVIEW_MATERIAL_BUDGET = 12000

#: 单轮至少给这么多字符 —— 否则"新轮优先"会把最老的一轮压成零字，
#: 而那一轮可能正是学生写步骤的地方。
_MIN_ROUND_CHARS = 200


def build_material(rounds: list[dict], budget: int = REVIEW_MATERIAL_BUDGET,
                   statement: str = "") -> dict:
    """把轨迹整理成**唯一一份**受预算约束的材料（契约 §2.3⑨）。

    ## 三条规则（照契约原文）
    ```
    ① 按**完整行**组织 —— 不在行中间切断（切断会把半句话交给模型当证据）
    ② 优先保留**最新作答** —— 从最后一轮往前分配，不是从第一轮往后
    ③ 未发送部分**明确记录为省略** —— 省略是"没发给他"，
      **不等于"学生没写"**，提示词里也要写明这条
    ```

    ## 为什么要有这个函数
    原来「轨迹正文」和「证据目录」是**两次独立构造**：
    正文每轮截 300 字，证据目录发全文。后果有两个，
    **第二个更严重**：
    ```
    · 预算形同虚设，整体输入净增
    · **模型看到的正文和它被允许引用的原文不是同一份**
      ⇒ 它可能引用一段自己没看过的文本（或反过来，正文里有的原文它引不到）
    ```
    ⇒ 现在两者共用这里的输出，**引用的原文必然是它看过的**。

    返回 `items[*]` 同时带 `text`（可能在**行边界**截断）与省略计数。
    """
    usable = []
    for i, r in enumerate(rounds or [], 1):
        n = r.get("round")
        cur = n if isinstance(n, int) else i
        raw = _txt(r.get("answer") or r.get("student_input"))
        if not raw.strip():
            continue                      # 空轮**不产出材料**，也不造假
        rev = r.get("revision") or r.get("source_revision")
        sid = r.get("source_id")
        usable.append({
            "id": f"E{cur}.1",
            "source": "student_answer",
            "source_id": str(sid) if sid else None,
            "revision": rev if isinstance(rev, int) else None,
            "round": cur,
            "ts": r.get("ts") if isinstance(r.get("ts"), (int, float)) else None,
            "text": raw,
        })

    # ---- 从最新一轮往前分配（契约②）----
    remaining = max(0, int(budget))
    kept: list[dict] = []
    omitted_rounds: list[int] = []
    for item in reversed(usable):
        if remaining < _MIN_ROUND_CHARS:
            omitted_rounds.append(item["round"])
            continue
        text, dropped_lines, dropped_chars, mid_line_cut = _fit_lines(item["text"], remaining)
        entry = {**item, "text": text,
                 "omitted_lines": dropped_lines, "omitted_chars": dropped_chars,
                 "mid_line_cut": mid_line_cut}
        kept.append(entry)
        remaining -= len(text)
    kept.reverse()
    omitted_rounds.reverse()

    # ⚠️ **题面单独成池**（契约 §2.8③：来源分池、不许互相冒充）。
    #    ID 用 `S1`（statement），与学生作答的 `E<轮>.1` 不同前缀 ——
    #    让"这段字是题面还是学生写的"在**记号层面**就分得开。
    #    ⚠️ **参考答案（reference_solution）绝不入池** —— 它是评测专用。
    stmt_text = _txt(statement)
    if stmt_text.strip():
        kept.insert(0, {"id": "S1", "source": "statement", "source_id": None,
                        "revision": None, "round": None, "ts": None, "text": stmt_text,
                        "omitted_lines": 0, "omitted_chars": 0, "mid_line_cut": False})

    return {
        "items": kept,
        "omitted_rounds": omitted_rounds,
        "budget": int(budget),
        "chars": sum(len(e["text"]) for e in kept),
        # ⚠️ `mid_line_cut` 也要算进来 —— 只切了一行中间同样是"截断了"。
        # 旧实现漏掉它，于是 13000 字的单行输入报 `truncated=False`。
        "truncated": (bool(omitted_rounds)
                      or any(e["omitted_lines"] or e.get("mid_line_cut") for e in kept)),
    }


def material_evidence(material: dict) -> list[dict]:
    """从**同一份材料**派生证据目录 —— 唯一的派生入口。

    ⚠️ 证据目录与提示词正文**必须是同一份文本**（见 `build_review_user`）。
    所以这里不接受 `rounds`，只接受已经定稿的 `material` ——
    **签名上就堵死"各建一次"那条路。**
    """
    return [{k: it[k] for k in ("id", "source", "source_id", "revision",
                                "round", "ts", "text")}
            for it in material["items"]]


def _fit_lines(text: str, limit: int) -> tuple[str, int, int]:
    """按**完整行**取前若干行，直到装不下为止。

    ⚠️ **尽量不切断行**：半行交给模型，等于让它把"没写完的半句"当成证据。
    装不下的整行如实计数，调用方会写进提示词。

    ⚠️ 但**当第一行本身就超预算时只能硬切** —— 否则整条作答一个字都发不出去，
    那比"发一段截断的"更糟。硬切时**必须把那截尾巴也算进省略**：
    实测过一个 13000 字的单行输入，旧实现只切了 12000 字却报
    `omitted_chars=0`、`truncated=False` —— **截断了却说自己没截**，
    比不截更危险（下线会据此以为材料是完整的）。
    ⇒ 返回第 4 个值 `mid_line_cut`，把"切在了行中间"这件事**显式说出来**。
    """
    if len(text) <= limit:
        return text, 0, 0, False
    lines = text.splitlines() or [text]
    kept: list[str] = []
    used = 0
    cut_tail = ""
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if kept and used + cost > limit:
            break
        if not kept and cost > limit:      # 第一行本身就超预算 ⇒ 只能硬切
            kept.append(line[:limit])
            cut_tail = line[limit:]        # **被切掉的尾巴要记账**
            used = limit
            break
        kept.append(line)
        used += cost
    body = "\n".join(kept)
    dropped = lines[len(kept):]
    dropped_chars = sum(len(line) + 1 for line in dropped) + len(cut_tail)
    dropped_lines = len(dropped) + (1 if cut_tail else 0)
    return body, dropped_lines, dropped_chars, bool(cut_tail)


def build_evidence_catalog(rounds: list[dict],
                           budget: int = REVIEW_MATERIAL_BUDGET) -> list[dict]:
    """**服务端生成**证据目录 —— 模型只能引用它，不能改写原文。

    · `id` 形如 `E5.1`（轮 5 的第 1 条），**由服务端稳定生成**
    · `text` **逐字取自轨迹**，服务端填 —— 这正是契约那条
      「引用原文由服务端填充，模型不能自行改写」的落点
    · `revision` 若轨迹带就更上（更正的版本号），没有则 `None`（**如实为空，不编**）
    · `source_id` **是实际来源标识**（契约 §2.3「来源及修订号」）——
      没有它就验不了「跨来源引用」和「更正前引用」，
      因为 `source` 那个字段是**种类**（恒为 `student_answer`），不是来源。

    ⚠️ 材料**经过预算约束**（见 `build_material`）：目录里的 `text`
    就是模型实际看到的那一份，两者不会再漂开。
    """
    return material_evidence(build_material(rounds, budget))


def format_evidence_catalog(evidence: list[dict]) -> str:
    """把证据目录渲染成**给模型看的**清单（**带 ID**）。

    ⚠️ 原文在这里是**服务端填的逐字文本** ——
    模型引用 ID 即可，**不得改写原文**（契约②）。
    """
    if not evidence:
        return "（本题没有可用证据）"
    #: 来源标签：**让模型看得见"这段字是谁写的"**（契约 §2.8③ 来源分池）。
    _label = {"statement": "题面", "student_answer": "学生作答"}
    out = []
    for e in evidence:
        rev = f"，修订 {e['revision']}" if e.get("revision") is not None else ""
        ts = f"，{e['ts']}s" if e.get("ts") is not None else ""
        src = f"，来源 {e['source_id']}" if e.get("source_id") else ""
        who = _label.get(str(e.get("source")), str(e.get("source")))
        rnd = f"第{e['round']}轮" if e.get("round") is not None else ""
        out.append(f"[{e['id']}]（{who}）{rnd}{ts}{rev}{src}：{e['text']}")
    out.append("⚠️ **`evidence_ids` 只能填上面出现过的 ID**；"
               "没有证据支持的判断，**宁可留空** —— 不要编 ID，也不要改写上面的原文。")
    out.append(
        "⚠️ **要引用原文给人看，必须用引用记号 `⟦证据ID:你想引的那段字⟧`**"
        "—— **把那段字照抄进去**（服务端会去核对并截取原文，**不需要你数位置**）。\n"
        "   ✅ 例：`你写的 ⟦E1.1:合力F-f=10-4=6 N⟧ 这一步是对的`\n"
        "   ✅ 连空格都不用一模一样（服务端只差空白也算通过），但**字不能错**"
        "——抄错一个符号整份报告会被拒收。\n"
        "   ❌ **不许自己敲引号里的字**：「你写的「合力 6 N」」「题目说「移动3 m 时的速率」」——\n"
        "      凡「你写的/题目说/题干给的/原文是」+ 引号，**一律视为非法引用**。\n"
        "   ℹ️ 普通术语/自标仍可用引号（「机械能守恒」「取右为正」），**只要不带归属框架**。\n"
        "   ℹ️ 题面与学生作答是**两个池子**：`S…` 是题面，`E…` 是学生写的，**不许互相冒充**。")
    return "\n".join(out)


def _cite_for_text(evidence: list[dict], text: str) -> list[str]:
    """给一段"服务端自己填的原文"找它对应的证据 ID。

    先精确匹配，再退化到包含匹配（模型可能只引了结论那一行）。
    **找不到就返回空** —— 不猜、不指一条无关的证据。
    """
    needle = (text or "").strip()
    if not needle:
        return []
    for e in evidence:
        if (e.get("text") or "").strip() == needle:
            return [e["id"]]
    for e in evidence:
        if needle and needle in (e.get("text") or ""):
            return [e["id"]]
    return []


#: 「**没有可疑处**」的说法 —— 模型常在 `ocr_caution` 里写这类话。
#:
#: ⚠️ 只匹配**开头**（去掉标点后），因为要判的是"这一句到底在报可疑，还是在报没有"。
#: 这是一个**词表启发式**，不是理解：它只挡最常见的几种说法。
#: 真正的修法在提示词里（**没有可疑处就留空**），这里是防御性兜底 ——
#: 漏挡的代价 = 多一条噪声待核查项；误挡的代价 = 少一条真实提醒。
#: ⇒ **只在开头命中时才当作"没有问题"**，尽量不误挡。
#: ⚠️ **调用方必须先摘掉服务端自己渲染的片段**（`strip_server_renders`）。
#: 直接拿**填充后**的文本调本函数，会把服务端的产物判成"模型自写次数"（见 `_SERVER_RENDER_RE`）。
_NO_SUSPICION_PREFIXES = ("未发现", "未见", "没有发现", "未察觉", "无可疑",
                          "识别清晰", "无明显可疑", "未看到可疑", "没发现",
                          # 2026-10-09 独立复核补：模型实际会写这些说法，旧词表挡不住
                          "没有可疑", "识别正常", "无异常", "没有问题", "暂无可疑",
                          "尚未发现", "未检测到", "识别尚可", "读法一致", "未见异常")

#: **转折词**：一旦出现，就说明这句话后面还有内容 —— 那就**不能**因为开头是
#: "未发现…" 就把整条丢掉。
#:
#: ⚠️ 复核报的误挡（实测）：
#: ```
#: "未发现明显问题，但第4轮 1.98 与第8轮 198 不一致，需核查"   → 整条被吞
#: "识别清晰，但第7轮的 1.6 与第9轮的 1.49 无法确认"           → 整条被吞
#: ```
#: 这两条恰恰是最该留给学生自查的 OCR 歧义提醒。
#: ⇒ 只在**整句为纯否定且无转折**时才当作"没有可疑"。
#: ⚠️ 只列**转折连词** —— 不能把「，」「；」也算进来，
#: 否则归档里那条真正的"没有可疑"（「未发现同一数字前后读法矛盾；2、18…」）又会被当成可疑。
_ADVERSATIVE = ("但", "不过", "然而", "可是", "却", "只是")


# ================================================================ 全文归因（无依据的过程/认知断言）
#
# ## 材料是**判据的真值**，不是我的猜测
# 判据来自 T4 开发集里**两名评审独立标出、并经复核确认**的 10 条原文（逐字引用）：
# ```
# Q007「不是靠…直觉猜的」            Q008「他后面凡是…都会…翻车」
# Q023「串联总电阻相加这一步没有犹豫」  Q026「不是凑出来的」
# Q034「不是套结论套出来的」          Q036「说明结果不是碰巧写出来的数字」
# Q053「不是碰出来的数」+「没有出现…的痕迹」
# Q056「不是随手写的近似值」          Q066「说明串联电流相同这条主线你抓住了」
# Q071「说明…mgh…这条线你用对了」
# ```
# ## 为什么用**句式**而不是关键词
# 关键词会误杀：`用对了` 出现在「公式用对了」里是**对可见书写的正当评价**，
# 只有 `这条线你用对了`（**方法类宾语 + 未见过程**）才是越界。
# ⇒ 每条判据都**钉住宾语或否定框架**，宁窄不宽。
#
# ## ⚠️ 有一族它**够不到**，如实写明
# Q026 / Q071 还包含「把**未展示的**方法叙述成学生实际走过的路线」——
# 那是**语义**问题（缺证据），不是措辞问题，**句式扫描器原理上抓不到**。
# ⇒ 它只覆盖 10 条里的措辞部分；剩下那部分只能靠提示词与人。
_UNSUPPORTED_PATTERNS = (
    # A 否定式断言：不是猜/凑/碰巧/蒙/随手/套出来的
    #
    # ⚠️ 窗口 {0,40}：Q007 的真实原文是
    #    「不是靠「下落所以重力做正功」的直觉猜的」——`不是` 与 `猜` 之间隔了 20 多字。
    #    第一版写 {0,14}，**实测漏掉了它**。
    #
    # ⚠️ 引号后视 `(?<![「『"'])`：Q001/Q021 的原文是
    #    「所以这**不是「凑对」**，而是「前提错→结论跟着错」」——
    #    模型在**否认**自己凑，是好话。宾语被引号括起来 ⇒ 那是在**给一类错误命名**，
    #    不是在对学生下断言。第一版把它判成违规，**实测误报两条**。
    re.compile(r"不是[^。；！？]{0,40}(?<![「『\"'])(?:猜|凑|碰巧|碰出来|蒙|随手|套(?:结论|出来)|侥幸)"),
    # B 心理/过程状态：没有犹豫 / 纠结；**以及"没有发生某事"整族**
    #
    # ⚠️ **2026-10-09 契约 ⑦-3 裁决后扩的这一族**：
    #    裁决规则第③条明确 —— **「没有发生某事」一律违规**
    #    （没有犹豫 / 改口 / 换向 / 换值 / 回退 / 改写 / 反复），
    #    理由：**静态快照无法证明过程事件不存在**。
    #
    #    裁决的依据锚在已确认真值上：上一批 Q023「串联总电阻相加这一步没有犹豫」。
    #    本批**两位评审都漏掉**的 4 条同族句（Q011「没有中途换过数值」、
    #    Q033「没有出现回退或改写的痕迹」「没有中途换值或换公式」、Q058「一直没换过」）
    #    正因为"两评审都判 false ⇒ 不在分歧清单 ⇒ 裁决不到"而漏网 ——
    #    **这一族是"只裁决分歧"那个系统性盲区的直接受益者，所以必须由规则兜住。**
    #
    #    ⚠️ 曾经删过 `没有出现…痕迹`（当时两批判定相反、形态有歧义）。
    #    **现在歧义已被 ⑦-3 解决**，所以按裁决重新加回来 —— 这不是反复，
    #    是"判据未定就不当闸门、定了再当闸门"。
    re.compile(r"(?:没有|未见|不曾|毫无|一直没|从来没有|从未)"
               r"[^。；！？]{0,10}(?:犹豫|纠结|停顿|迟疑|换过|换值|换公式|换向|改口|回退|改写|反复|变更|改变"
               # ⚠️ 变体实测补的：「没有**默认**取正根」「公式关系没有**用错**」——
               #    同一族的另两种动词。来源：第三轮评审 A′（Q070）与 B′（Q021）。
               # ⚠️ 2026-10-10 再补 `中断`/`修改`（第五轮共识项 Q016/Q030 就是这两个词）
               #    与 `改动`（第六轮评审 A 的 Q038「两条结论都给出后没有再作改动」）。
               #    实测（五批 360 份归档）：**净增 6 处命中，全部同族、0 误报**，
               #    其中 2 处正是 v7 的真值；另 4 处是前几批**评审漏掉**的同族句。
               #    同时测了 4 条合法反例（"若中途被打断…"建议句、"材料里有涂改痕迹"正面描述、
               #    "他在第 2 轮修改了合力"可见变化、"只在「损失为」处中断"材料描述）
               #    ⇒ **一条都没误杀**（它们都没有 `没有/未见…` 这个否定前缀）。
               #    `改动/更换/改换` 另测：v7 净增 0、v8 净增 1（正是 Q038）⇒ 也是 0 误报。
               # ⚠️ 2026-10-10 第六轮裁决再补两处（**它建议的先量后加，我照做了**）：
               #    · pattern A 加 `侥幸` —— v8 Q010「说明结果不是侥幸来的」
               #    · pattern B 加 `再?回到前面的量`（**窄化**，不写成裸的 `回到`）——
               #      v8 Q033「中间没有再回到前面的量」
               #    六批归档实测：**各自只命中目标那一条、零净增误报**。
               # ⚠️ 裁决者还纠正了我一处表述：我说"闸门不区分射程"**不准确** ——
               #    它区分的不是「材料里」这个字面，而是**否定词后面跟的是不是过程动词**；
               #    只是覆盖面靠词表。⇒ **不要**加"必须带『材料里』才算/才不算"的射程规则。
               r"|默认|用错|中断|修改|改动|更换|改换|再?回到前面的量)"),
    # ⚠️ 变体实测补的：这一族不只有「没有出现X」，还有「**没有留下**X」——
    #    实测「整段没有留下推导痕迹」原版漏掉（只认 `没有出现`）。
    re.compile(r"没有(?:出现|留下)[^。；！？]{0,20}(?:痕迹|迹象)"),
    # C 由结果断言**方法/掌握** —— 宾语必须是方法类，否则「公式用对了」会被误杀
    #
    # ⚠️ 单独的「线」也要算（Q071「说明…这条**线**你用对了」）——
    #    但**必须排除可见的线**（辅助线/切线/实线…）：评价"他辅助线画对了"是
    #    **对可见书写的正当评价**，不是过程归因。
    #    用单字符后视排除；Python 的后视要求定宽，所以只能逐字列出来。
    re.compile(r"(?:主线|思路|方法|路线|套路|精髓|(?<![辅助切实虚横竖斜曲直法垂])线)"
               r"[^。；！？]{0,6}(?:抓住了|用对了|掌握)"),
    re.compile(r"(?:抓住了|用对了|掌握了)[^。；！？]{0,4}"
               r"(?:主线|思路|方法|路线|(?<![辅助切实虚横竖斜曲直法垂])线)"),
    re.compile(r"(?:说明|可见|表明|证明)[^。；！？]{0,14}(?:掌握|抓住|学会)"),
    # D 对未来的绝对预测
    re.compile(r"(?:凡是|以后|后面|接下来)[^。；！？]{0,20}(?:都会|必然|一定|肯定)"),
    # E **第二人称的方法/认知归属** —— "从正确结论反推"那一族的**词法代理**
    #
    # ⚠️ 我先前断言"这一族没有词法代理"（实测 0/7）。**那个结论只对当时那批形态成立。**
    #    换一批评审标记重看，它们有**共同措辞**：
    #      「这一环**你用的**是匀变速规律」「**你是按**同一套串联规律写下来的」
    #      「说明**你注意到了**「粘在一起」…」「所以 16 W 支持**你用的是**乘号」
    #    对照**不算违规**的那些，分界线很清楚：
    #      ✅「**你写的**是 W_G=-mgh=-25 J」「无法确认**你写的**是 16 W 还是别的」← 可见书写
    #      ❌「**你用的**…」「**你抓住了**…」← 推出来的
    #
    # 实测（142 份归档逐条人工判过）：
    #   上一批 6 处命中 **6/6 都是这一族**（其中 Q066/Q071 是**已确认真值**）⇒ 0 误报
    #   v4 批 1 处（"这两点你用对了"）同族；v5 批 B′ 标出的 5 条**全部命中**
    #   ⚠️ 唯一误报是「**不能据此判断**你用的是哪条路径」——靠 `_UNSUPPORTED_GUARDS` 挡住 ✓
    re.compile(r"你(?:用的|用的是|用了|用对|选的是|选对|注意到了?|抓住|掌握|使用的是|采用的是)"),
    # 「你按…方法类名词」——**必须带方法名词**，否则「你按一条线做完」这种
    # 对**可见结构**的描述会被误伤（实测确认过这一条）。
    re.compile(r"你(?:是)?按[^。；！？]{0,12}(?:规律|公式|关系|方法|思路|路线|主线|定理|定律)"),
)

#: 命中处**前面**出现这些词 ⇒ 是"不能说明他掌握"这类**否定/存疑**，**不算违规**。
#: ⚠️ 没有这道闸，`说明…掌握` 会把「**不能**说明他掌握了」判成违规 —— 那是反的。
_UNSUPPORTED_GUARDS = ("不能", "不足以", "无法", "难以", "并不能", "不应", "别", "不要")


def unsupported_claims(text: str) -> list[str]:
    """找出**无依据的过程/认知断言**（判据见 `_UNSUPPORTED_PATTERNS`）。

    ⚠️ **只管措辞，不管语义。** 见上：把"未展示的方法说成学生走过的路线"
    这一类**抓不到**，别把它当"归因问题已堵住"。
    """
    text = text or ""
    hits: list[str] = []
    for pattern in _UNSUPPORTED_PATTERNS:
        for m in pattern.finditer(text):
            window = text[max(0, m.start() - 6):m.start()]
            if any(g in window for g in _UNSUPPORTED_GUARDS):
                continue                      # 是否定/存疑的表述，不是违规
            # ⚠️ **对比式否认守卫**（2026-10-09 实测补）：
            #    「**不是凑对，而是一致地建立在同一个前提上**」——
            #    这是模型在**说明这个错误的性质**（系统性而非蒙对），
            #    与已确认的违规「不是凑出来的」（**表扬**学生没蒙）**不是一回事**。
            #    判据：命中之后紧跟着「而是」⇒ 是在对比，跳过。
            #    实测（四批 142+72 份）：已确认的 12 处「不是…」违规**没有一处**带这个结构 ⇒ 守卫安全。
            if "而是" in text[m.end():m.end() + 8]:
                continue
            hits.append(m.group(0).strip())
    return hits


#: Balanced quotation forms. ASCII word/Greek primes remain mathematical marks.
_QUOTED_SPAN = re.compile(
    r'「([^」]{1,12000})」|『([^』]{1,12000})』|“([^”]{1,12000})”|‘([^’]{1,12000})’'
    r'|(?<![A-Za-z0-9_\u0370-\u03ff])"([^"\n]{1,12000})"(?![A-Za-z0-9_\u0370-\u03ff])'
    r"|(?<![A-Za-z0-9_\u0370-\u03ff])'([^'\n]{1,12000})'(?![A-Za-z0-9_\u0370-\u03ff])")


def _quote_fragment(match):
    return next(value for value in match.groups() if value is not None)
#: **归属给题面**的框架词。必须紧挨在引号**前面**（`{0,14}` 窗口内）。
_STATEMENT_ATTRIBUTION = re.compile(
    r"(?:题干|题面|题目(?:说|给出|给的|中)|题目条件|已知条件|条件里|题中说)")


def unfounded_statement_quotes(text: str, statement: str) -> list[str]:
    """找出**归属给题面、但题面里并不逐字存在**的引文（契约 §2.7③）。

    ## 为什么只查这一个来源
    ```
    引文归属的完整判据是"引号内必须是**所称来源**里逐字存在的连续子串"。
    但**只有题面这一路可以上闸门**——因为**题面是服务端给的**（ground truth），
    且归属词（「题干给的」「题目说」）让"该去哪个来源里找"**无歧义**。
    实测（v7/v8/v9 三批）：归属给题面的引文一共 3 个，**3 个都不逐字** ⇒ 精确率 3/3、**零误报**。
    ```
    ⚠️ **学生材料那一路不能照搬**（实测过两次）：
       全量引号 → v7 里 85/298 不在材料里，**几乎全是正当术语/转述**；
       只查"归属给学生的"引文 → 召回约 1~2/3、精度约一半 ⇒ 会拒掉每批 5~9 份。
    ⇒ 学生材料那一路**只走提示词**（`prompts/review/v3.md`「引号是承诺」），不做闸门。

    ⚠️ 判据本身**只看"能不能在题面里逐字找到"**，不看意思是否相近 ——
       把「且」改成「、」也算引用不实（那正是本判据抓到的真实形态）。
    """
    text, statement = text or "", statement or ""
    if not text or not statement:
        return []
    flat = re.sub(r"\s+", "", statement)
    hits: list[str] = []
    for m in _QUOTED_SPAN.finditer(text):
        before = text[max(0, m.start() - 14):m.start()]
        if not _STATEMENT_ATTRIBUTION.search(before):
            continue                      # 不是归属给题面的引文，本判据不管
        frag = re.sub(r"\s+", "", _quote_fragment(m))
        if frag and frag not in flat:
            hits.append(m.group(0).strip())
    return hits


# ================================================================ 契约 §2.8：引文结构化

#: 引用记号：`⟦证据ID:…⟧`。`…` 有**两种**写法（契约 §2.8①）：
#:
#: ```
#: ① **文本式（推荐，提示词教这个）**：`⟦E1.1:合力F-f=10-4=6 N⟧`
#:    —— 模型**抄它想引的那一段**，**服务端自己定位**并截取。
#: ② 数值式：`⟦E1.1:18-27⟧`（0 基码点区间，`-` 后省略 = 到末尾）
#:    —— 留给程序化调用与测试；**不教给模型**。
#: ```
#:
#: 🔴 **为什么必须有①**（2026-10-10 实测教训）：
#: ```
#: v11 首次上线只教了②。结果模型**认真地在用新格式、但偏移算错了**：
#:   ⟦E1.1:18-27⟧ ⇒ 越界（该证据只有 25 字符）  ⟦E1.1:4-27⟧ ⇒ 同样越界
#: ⇒ 校验器正确拒收（设计意图），但**那是接口设计的错，不是模型不服从** ——
#:   要求 LLM 稳定数到"第 18 个码点"是不可执行的。
#: ⇒ 把定位这件事从模型手里拿走：**它只需要抄对那段字**，定位由服务端做。
#: ```
#: ⚠️ **重复片段**：同一段字在证据里出现多次时，**取第一处**（确定性）。
#:    需要指定第二处时用数值式。这条规则写进契约，避免"取哪一处"变成隐式行为。
CITATION_TOKEN = re.compile(r"⟦([A-Za-z0-9_.\-]+):?([^⟧]*)⟧")
#: 数值式判据：`18-27` / `18-` / `18`
_NUMERIC_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)?)?\s*$")
#: 去掉所有空白后的形式（"只差空白"是**唯一**允许的等价，契约 §2.7③）。
_squash = lambda s: re.sub(r"\s+", "", str(s or ""))

#: **归属框架**：出现它 + 引号 = 模型在"声称这句话来自某处" ⇒ 必须走引用记号。
#: 判据是**有没有归属框架**，不是"有没有引号"（普通术语引号必须放行）。
_ATTRIBUTION_FRAME = re.compile(
    r"(?:你写(?:的|出|的是)?|学生写(?:的|的是)?|他写(?:的|的是)?|原文(?:是|写的是)?"
    r"|材料中写(?:有|的)?|题干(?:给的|说的是)?|题面(?:上|写的)?|题目(?:说|给的|给的是)"
    r"|条件里写(?:的)?|用到了|写的是)")

_EXPLICIT_QUOTE_ATTRIBUTION = re.compile(
    r"(?P<statement>(?:题干|题面|题目|题设|原题)(?:中|上)?(?:明确|直接)?"
    r"(?:明说|说|写|给出|给定|提供|标注|描述|记载|给的|写的|的条件))"
    r"(?:[^。；！？「『“‘\"']{0,8})$"
    r"|(?P<student>(?:你|学生|他)(?:明确)?写(?:的|出|的是)?|你的(?:作答|书写)(?:中)?(?:写|是))"
    r"(?:[^。；！？「『“‘\"']{0,8})$"
    r"|(?P<material>原文(?:是|写的是|写有|写着|注明|提到))"
    r"(?:[^。；！？「『“‘\"']{0,8})$")


def _quote_source(before):
    match = _EXPLICIT_QUOTE_ATTRIBUTION.search(before)
    if match is None:
        return None
    return 'statement' if match.group('statement') else ('student_answer' if match.group('student') else None)

#: 允许出现在证据池里的来源。**`reference_solution` 不在列** ——
#: 它是**评测专用**，绝不许进教学/复盘模型的输入（契约 §2.8③）。
ALLOWED_EVIDENCE_SOURCES = frozenset({"statement", "student_answer"})


def build_server_timeline(material: dict, stats: dict | None = None) -> str:
    """**服务端生成 `timeline`**（契约 §2.9）。模型不再自由叙述学生的实际解题过程。

    ## 只写三样，多一样都不写
    ```
    ① **提交顺序** —— 由服务端的轮次/时间确定（模型无从知道，也无权编）
    ② **可见原文** —— 逐字取自已投递的材料（可能按预算截断，**截断如实说**）
    ③ **修订状态** —— 该轮有没有改过、当前修订号
    ```
    ## 🔴 为什么**不**做"材料完备度判断"（用户 2026-10-10 纠正）
    ```
    我原设计：先判"材料有没有写全路径各步"，判得出才允许叙述顺序。
    用户否掉了：那等于**再引入一个不可靠的判据**去决定"能不能断言学生怎么做" ——
    而这个判断本身没有 ground truth（"完整路径"是主观的，且路径可能有多种）。
    ⇒ 改成：**任何情况下都不叙述计算顺序**，只呈现"纸面有什么"。
      不再需要那个判断，也就没有它判错的余地。
    ```
    ## 推导链住哪
    ```
    **独立重建区**（`answer_check.derived_from_work`）—— 那里才是"系统算的"，
    且必须标「核对时重建」。`timeline` 里**只描述纸面**。
    ```
    """
    items = list((material or {}).get("items") or [])
    if not items:
        return "材料没有可用的提交记录（未展示作答内容）。"
    # The evidence catalogue also contains the statement and may split one
    # submission into several evidence items. Neither creates a student round.
    visible_rounds = {
        e.get('round') if isinstance(e.get('round'), int) else ('unlabelled', index)
        for index, e in enumerate(items) if e.get('source') == 'student_answer'
    }
    parts = [f"材料展示 {len(visible_rounds)} 轮学生提交，按**提交顺序**如下（仅列纸面可见内容）："]
    for e in items:
        who = "题面" if str(e.get("source")) == "statement" else f"第{e.get('round')}轮"
        rev = e.get("revision")
        rev_note = f"，当前修订 {rev}" if isinstance(rev, int) else ""
        cut = ""
        if e.get("omitted_lines") or e.get("omitted_chars"):
            cut = f"，**该轮有 {e.get('omitted_chars') or 0} 字未纳入材料**"
        body = str(e.get("text") or "").strip()
        parts.append(f"· {who}{rev_note}：{body or '（空白）'}{cut}")
    # ⚠️ **常设免责**：措辞说的是**"本段不推断"**，不是"材料没写" ——
    #    后者会是**无条件的事实断言**（材料真写了步骤时它就假了），
    #    而判断"材料写没写全"正是用户否掉的那个不可靠判断。
    #    ⇒ 只声称**本段自身的性质**，这样它在任何材料上都成立。
    parts.append("⚠️ 以上仅为**提交顺序与纸面可见内容**；**本段不推断实际计算顺序**，"
                 "也不表示材料是否展示了完整推导。系统自己的复算见「结论核对」栏。")
    return "\n".join(parts)


def citation_pool_status(pool: list[dict]) -> list[str]:
    """池子本身合规吗？（来源必须在白名单里）"""
    bad = [str(e.get("source")) for e in (pool or [])
           if str(e.get("source")) not in ALLOWED_EVIDENCE_SOURCES]
    return sorted(set(bad))


def parse_citations(text: str) -> list[dict]:
    """把记号解析成 `{raw, evidence_id, start|None, end|None, quote|None}`。

    数值式 ⇒ 填 `start/end`；文本式 ⇒ 填 `quote`（**定位留给服务端**，见 `_locate`）。
    """
    out = []
    for m in CITATION_TOKEN.finditer(text or ""):
        eid, body = m.group(1), m.group(2)
        num = _NUMERIC_RANGE.match(body)
        if num:
            out.append({"raw": m.group(0), "evidence_id": eid,
                        "start": int(num.group(1)),
                        "end": int(num.group(2)) if num.group(2) else None,
                        "quote": None})
        else:
            out.append({"raw": m.group(0), "evidence_id": eid,
                        "start": None, "end": None, "quote": body})
    return out


def _locate(body: str, quote: str) -> tuple[int, int] | None:
    """在证据文本里定位 `quote`；**逐字优先，退一步只差空白**。

    返回 `(start, end)` 或 `None`（定位不到 ⇒ 非法引用）。
    ⚠️ 允许"只差空白"是沿用已定案的唯一等价（契约 §2.7③）——
       **渲染出来的仍是原文的字符**（不是模型那份），所以字面保证不被削弱。
    ⚠️ 重复片段**取第一处**（契约 §2.8①）。
    """
    if not quote:
        return None
    i = body.find(quote)
    if i >= 0:
        return i, i + len(quote)
    # 只差空白：把两边都压掉空白再找，再把位置**映射回原文**。
    idx = [pos for pos, ch in enumerate(body) if not ch.isspace()]
    flat = "".join(body[p] for p in idx)
    want = _squash(quote)
    j = flat.find(want)
    if j < 0:
        return None
    return idx[j], idx[j + len(want) - 1] + 1


def citation_defects(text: str, pool: list[dict],
                     source_revisions: dict | None = None) -> list[dict]:
    """校验**引用记号**（契约 §2.8①）。**任何一条不过 ⇒ 该报告不得投递/缓存。**

    逐项：证据 ID 存在 → 来源在白名单 → 范围合法 → （`revision` 由调用方比对，见下）。
    ⚠️ **Unicode 偏移规则**：`start/end` 是 **0 基码点索引**，坐标系是
    **服务端已投递的那份 `evidence.text`**（可能已按预算截断）——不是 UTF-16 单元，
    也不是未截断的原文。
    """
    index = {str(e.get("id")): e for e in (pool or [])}
    defects: list[dict] = []
    for c in parse_citations(text):
        ev = index.get(c["evidence_id"])
        if ev is None:
            defects.append({**c, "kind": "unknown_evidence",
                            "detail": f"证据 {c['evidence_id']} 不在此次授权的证据池里"})
            continue
        src = str(ev.get("source"))
        if src not in ALLOWED_EVIDENCE_SOURCES:
            defects.append({**c, "kind": "source_not_authorized",
                            "detail": f"来源 {src} 未授权进教学/复盘输入"})
            continue
        # ⚠️ **过期修订 ⇒ 拒**（契约 §2.8①）：证据取自某一修订，
        #    而服务端当前修订已经变了 ⇒ 这段字**可能已经被学生改过**，引用不再成立。
        sid = str(ev.get("source_id")) if ev.get("source_id") else None
        if sid and isinstance(source_revisions, dict) and sid in source_revisions:
            now = source_revisions[sid]
            if isinstance(now, int) and ev.get("revision") != now:
                defects.append({**c, "kind": "stale_revision",
                                "detail": f"证据修订 {ev.get('revision')} 已过期"
                                          f"（当前 {now}）"})
                continue
        body = str(ev.get("text") or "")
        if c["quote"] is None:
            end = len(body) if c["end"] is None else c["end"]
            if not (0 <= c["start"] < end <= len(body)):
                defects.append({**c, "kind": "range_out_of_bounds",
                                "detail": f"范围 [{c['start']},{end}) 超出该证据长度 {len(body)}"})
        elif _locate(body, c["quote"]) is None:
            defects.append({**c, "kind": "quote_not_in_evidence",
                            "detail": f"所引文字在证据 {c['evidence_id']} 里找不到"
                                      f"（逐字与只差空白都不匹配）"})
    return defects


def render_citations(text: str, pool: list[dict]) -> tuple[str, list[dict]]:
    """把**合法**记号替换成原文；返回 `(渲染后文本, 非法清单)`。

    ⚠️ 非法记号**原样保留**在文本里（不静默丢弃）——调用方必须先看非法清单，
    非空就**不得投递**（契约 §2.8①）。保留是为了让"哪里错了"可见，不是让它流出去。
    """
    index = {str(e.get("id")): e for e in (pool or [])}
    defects = citation_defects(text, pool)
    bad_raw = {d["raw"] for d in defects}

    def _sub(m: re.Match) -> str:
        if m.group(0) in bad_raw:
            return m.group(0)
        ev = index.get(m.group(1))
        if ev is None:
            return m.group(0)
        body = str(ev.get("text") or "")
        parsed = parse_citations(m.group(0))[0]
        if parsed["quote"] is None:
            start = parsed["start"]
            end = len(body) if parsed["end"] is None else parsed["end"]
        else:
            located = _locate(body, parsed["quote"])
            if located is None:
                return m.group(0)
            start, end = located
        return body[start:end]

    return CITATION_TOKEN.sub(_sub, text or ""), defects


def unattributed_quotes(text: str, pool: list[dict] | None = None, *, strict_sources=False) -> list[str]:
    """模型**自己写的**"归属框架 + 引号"里，**那段字在任何授权来源里都找不到**的。

    🔴 **为什么加了 `pool` 这个参数**（2026-10-10 第十轮实测）：
    ```
    原实现只判"有没有归属框架 + 引号" ⇒ **fail-closed 之后误杀术语**：
      v12 有 4/7 条失败是这个判据，而被拒的是「取右为正」「粘在一起」「损失为0」
      —— 它们是术语/题面词组，**引文是准确的、无害的**。
    我 v8 就量过这个判据的精度只有约一半，**却仍把它设成阻断** —— 那是方法错误：
    量出精度不够之后，**必须据此改设计**，不是照旧上线。
    ⇒ 判据改成**可判定的那一步**：不是"猜它是不是引文"，
      而是"**这段字在源里找不找得到**"。
    ⇒ 「粘在一起」在题面里 ⇒ 放行（引文准确）；"他没写过的句子"找不到 ⇒ 拒（字面不实）。
    ```
    ⚠️ `pool is None` ⇒ **回退到旧行为**（只看归属框架）—— 只为历史兼容与单测，
       生产路径**必须**传池子。
    """
    # New quote styles require an explicit nearby attribution: terminology and
    # hypothetical OCR repairs are not claims of verbatim source text. Explicit
    # statement/student attribution restricts lookup to that source, even when
    # the fragment exists elsewhere. Legacy missing sources remain unknown;
    # current physical_check_only_v1 reports fail closed on missing sources.
    hits: list[str] = []
    for m in _QUOTED_SPAN.finditer(text or ""):
        before = (text or "")[max(0, m.start() - 14):m.start()]
        explicit = _EXPLICIT_QUOTE_ATTRIBUTION.search(before)
        if explicit and re.search(r"(?:若|如果|假如|假设)[^。；！？]{0,12}$", before):
            continue
        if not explicit and (m.group(0)[0] not in '「『' or len(_quote_fragment(m)) < 2):
            continue
        if not explicit and not _ATTRIBUTION_FRAME.search(before):
            continue
        if pool is not None:
            frag = _squash(_quote_fragment(m))
            if not frag:
                continue
            claimed_source = _quote_source(before)
            if claimed_source and not strict_sources and not any(e.get('source') == claimed_source for e in pool):
                # Legacy reports may omit the question catalogue. Unknown source
                # cannot be judged by comparing it against a different source.
                continue
            found = any(frag in _squash(e.get("text"))
                        for e in pool if str(e.get("source")) in ALLOWED_EVIDENCE_SOURCES
                        and (claimed_source is None or e.get('source') == claimed_source))
            if found:
                continue                      # 引文准确 ⇒ 放行（术语/题面词组都在这一类）
        hits.append(m.group(0).strip())
    return hits


#: **次序/依赖**标记（契约 §2.9）。出现它 = 句子在说"先做了什么、再由什么得到什么"。
_ORDER_MARKERS = re.compile(
    r"先[^。；！？]{0,24}(?:再|然后|随后|接着|而后|继而)"
    r"|(?:由|从)[^。；！？]{0,16}(?:再|求出|算出|求得|得到)"
    r"|据此|从而|进而|再?由[^。；！？]{0,12}(?:求|算|得)"
    r"|(?:最后|最终)[^。；！？]{0,12}(?:求|算|得)")
#: **材料里有没有中间步骤** —— 由服务端判（"材料完备度"）。判据：等号附近出现运算。
#: ```
#: 「1×6=3v」      ⇒ `×` 与 `=` 相邻 ⇒ **有步骤**
#: 「I=2 A，U₁=4 V」⇒ 等号右边是纯量值 ⇒ **只有结论**
#: ```
_HAS_STEPS = re.compile(
    r"[=＝][^,，。；\n]{0,24}[+\-×÷*/^√²]|[+\-×÷*/^√²][^,，。；\n]{0,24}[=＝]"
    r"|代入|联立|解得|用.{0,8}公式")


#: **明确拒绝推断**的窗口 —— 命中就**不判**（那正是我们要鼓励的写法）。
_REFUSAL_GUARDS = ("无法", "不能", "难以", "没法", "看不出", "未展示", "没有留下",
                   "不可确认", "无从", "不足以")
#: ⚠️ **`next_step` 按设计就是面向行动的建议**（「先写…再…」＝ 让他这么做），
#:    **不是断言他做过** ⇒ 该字段**整类排除**（v14 实测：不排除会误杀 Q002/Q029）。
_ACTION_FIELDS = ("next_step",)


def evidence_has_intermediate_steps(pool: list[dict] | None) -> bool:
    """证据池里**有没有中间步骤**（对比"只有结论值"）。—— 这是**材料完备度**，不是措辞。"""
    return any(_HAS_STEPS.search(str(e.get("text") or ""))
               for e in (pool or []) if str(e.get("source")) in ALLOWED_EVIDENCE_SOURCES)


def path_claim_defects(text: str, pool: list[dict] | None,
                       where: str = "") -> list[str]:
    """**路径/次序断言**（契约 §2.9）：看**材料完备度**，不看措辞。

    ```
    材料里**已写出该路径各步** ⇒ 同一句式**合规**（那是在复述纸面）
    材料**只有结论值、中间式没有留下** ⇒ 同一句式**越界**（路径不由结果唯一确定）
    ```
    ⚠️ **为什么逐步引用不够**（用户纠正）：三条结果分别能引用，
    **不等于**"先算电流、再据此算电压"成立 —— 前者证"字在纸上"，后者说"数**出自**那条路径"。

    ## ⚠️ 两道守卫（**都是量出来的，不是想出来的**）
    ```
    v14 首版实测（v13 归档 68 份）：命中 8 处，**至少 4 处误报** ——
      Q002/Q029 `next_step`「先写…再…」＝**建议**（让他这么做），不是断言他做过
      Q064「**无法**从材料确认他是怎么算出」＝**明确拒绝推断**，被模式误抓
    ⇒ 加守卫：① 拒绝推断的窗口内不判 ② `next_step` 整类排除（它按设计就是面向行动的）
    ```
    """
    if pool is None:
        return []                                  # 没池子 ⇒ 不判（旧报告不误报）
    if where.startswith(_ACTION_FIELDS):
        return []                                  # 建议句不判（见上）
    if evidence_has_intermediate_steps(pool):
        return []                                  # 材料写了步骤 ⇒ 复述纸面是合规的
    out: list[str] = []
    for m in _ORDER_MARKERS.finditer(text or ""):
        window = (text or "")[max(0, m.start() - 10):m.start()]
        if any(g in window for g in _REFUSAL_GUARDS):
            continue                               # 明确拒绝推断
        out.append(m.group(0).strip())
    return out


def _is_a_real_suspicion(caution: str | None) -> bool:
    """这条 `ocr_caution` 是在**报可疑**，还是在**报"没有可疑"**？

    ## 为什么要有这个判断
    旧版只要字段非空就产出一条 `uncertain` 待核查项 ⇒
    模型写「未发现同一数字前后读法矛盾…未见可疑识别点」也照收，
    **凭空给学生多一条"需要核查"**（T4 评测里每份报告都有这一条）。

    ## ⚠️ 两个方向都放过（2026-10-09 独立复核实测）
    ```
    漏挡（噪声照旧）  "没有可疑之处，识别清晰" / "识别正常" / "无异常" / "暂无可疑"
                      ⇒ 旧词表里是 `无可疑`，模型写的是 `没有可疑`，前缀不匹配
    误挡（真提醒被吞）"未发现明显问题，但第4轮 1.98 与第8轮 198 不一致，需核查"
                      ⇒ 只看开头就整条丢弃
    ```
    ⇒ 现在：**开头命中"没有可疑"的说法** 且 **全句没有转折词**，才算"没有可疑"。
    """
    text = (caution or "").strip().lstrip("⚠️：: 　")
    if not text:
        return False
    if any(mark in text for mark in _ADVERSATIVE):
        # 有转折 ⇒ 后面还有话，不能因为开头是"未发现…"就丢掉
        return True
    return not text.startswith(_NO_SUSPICION_PREFIXES)


def build_findings(report, stats: dict, evidence: list[dict] | None = None) -> list[dict]:
    """把现有报告**映射**成 `findings`（**不开新提示词**）。

    映射规则：
    ```
    sticking_points[*]  → observed_error（**有引用时**），否则 uncertain
    answer_check.ocr_caution  → uncertain（识别可疑，需核查）
    answer_check.note（判"不一致"时）→ observed_error（**挂上原文证据**）
    next_step / encouragement → suggestion
    ```
    **无明确问题时允许空列表**（契约⑥）—— 这里如实：模型没给卡点就是空。

    ## ⚠️ 一条硬规则：**「观察到的问题」必须有引用**
    契约把三类的含义写死了：`observed_error` = 「**有具体原文支持**的错误判断」。
    ⇒ 一条**没有任何引用**的判断，按定义就**不是** observed_error。
    旧版允许它挂着"错误"的类别、引用却是空的，于是
    `valid=true` **不能保证每条指控都有依据** —— 那正是本轮要堵的洞。

    判据不是"模型有没有填 `evidence_ids`"（它会填错、也会忘），
    而是**填的 ID 在不在服务端给出的目录里**：
    ```
    填了且存在           → observed_error（引用可用）
    填了但不存在         → observed_error（保留，**由 dangling_reference 拦下并拒绝缓存**）
    没填 / 目录里找不到   → **降级为 uncertain**，并如实计数
    ```
    降级**不丢内容**：那条判断照样显示给学生，只是归到「需要核查」——
    这正是它该在的位置（"我看到了这个，但指不回具体原文"）。
    """
    out: list[dict] = []
    known = {e["id"] for e in (evidence or [])}
    downgraded = 0

    def _claim(where, what, why, downstream, ids, fallback_what=""):
        nonlocal downgraded
        usable = [i for i in (ids or []) if i]
        if usable:
            return {"category": CATEGORY_OBSERVED, "where": where, "what": what,
                    "why": why, "downstream": downstream, "evidence_ids": usable}
        downgraded += 1
        note = (what or fallback_what or "").strip()
        return {"category": CATEGORY_UNCERTAIN, "where": where,
                "what": (note + DOWNGRADE_NOTE).strip(),
                "why": why, "downstream": downstream, "evidence_ids": []}

    for sp in getattr(report, "sticking_points", None) or []:
        ids = list(getattr(sp, "evidence_ids", None) or [])
        # ⚠️ 引用了**不存在的 ID** 时**不降级** —— 那是另一种缺陷
        # （"结论指不回去"），要留给 `dangling_reference` 拦下并拒绝缓存。
        # 降级会把它洗成"没引用"，反而把那个缺陷藏起来。
        if ids and not any(i in known for i in ids):
            out.append({"category": CATEGORY_OBSERVED, "where": sp.where, "what": sp.what,
                        "why": sp.why, "downstream": sp.downstream, "evidence_ids": ids})
            continue
        out.append(_claim(sp.where, sp.what, sp.why, sp.downstream, ids))
    ac = getattr(report, "answer_check", None)
    if ac is not None:
        if _is_a_real_suspicion(ac.ocr_caution):
            out.append({"category": CATEGORY_UNCERTAIN, "where": "识别",
                        "what": ac.ocr_caution, "why": "", "downstream": "",
                        "evidence_ids": []})
        if "不一致" in (ac.consistent or ""):
            # 「结论核对」的原文**由服务端逐字填过**（`student_answer`），
            # 所以这条**本来就有据可依** —— 只是旧版没把引用挂上去。
            out.append(_claim("结论核对", ac.note or "", ac.derived_from_work or "", "",
                              _cite_for_text(evidence or [], ac.student_answer or ""),
                              fallback_what="结论与推导对不上"))
    for field, where in (("next_step", "下一步"), ("encouragement", "鼓励")):
        v = (getattr(report, field, "") or "").strip()
        if v:
            out.append({"category": CATEGORY_SUGGESTION, "where": where,
                        "what": v, "why": "", "downstream": "",
                        "evidence_ids": []})
    if downgraded:
        logger.info("复盘有 %d 条判断因缺少可用引用，降级为「需要核查」", downgraded)
    return out


def count_downgraded_claims(findings: list[dict]) -> int:
    """有多少条判断因为**指不回原文**而没能成为「观察到的问题」。

    判据用 `DOWNGRADE_NOTE` 这个常量 —— **追加处与计数处共用它**，不会漂开。
    """
    return sum(1 for f in (findings or [])
               if isinstance(f, dict) and DOWNGRADE_NOTE.strip("（）") in (f.get("what") or ""))


def reference_defects(evidence: list[dict], findings: list[dict],
                      source_revisions: dict | None = None) -> list[dict]:
    """**重算**引用类缺陷 —— 只依赖 `evidence` + `findings`（+ 当前修订号）。

    ## 为什么是"重算"而不是"读 `validation.issues`"
    缓存里存着上一版代码写的 `validation`。**信任它，就等于信任写它时的代码** ——
    而缺陷恰恰可能正是那时的代码漏掉的。
    ⇒ 生成路径与缓存路径**共用这一个函数**，判据只有一份。

    ## 四类
    ```
    dangling_reference  引用的 ID 不在服务端给出的目录里（结论指不回去）
    unknown_source      证据的来源不在本次材料的来源集合里（跨来源引用）
    stale_reference     证据的修订号与本次材料记录的修订号不一致（更正前引用）
    unsupported_claim   「观察到的问题」没有任何可用引用（错误指控无依据）
    ```
    后三条要 `source_revisions` 才判得了；**不给就不判**（如实跳过，不猜）。
    """
    ids = {e.get("id") for e in (evidence or []) if isinstance(e, dict)}
    issues: list[dict] = []

    for f in findings or []:
        if not isinstance(f, dict):
            continue
        refs = [r for r in (f.get("evidence_ids") or []) if r]
        for eid in refs:
            if eid not in ids:
                issues.append({"kind": "dangling_reference",
                               "detail": f"引用了不存在的证据 {eid}"})
        # ⚠️ **只在"一个引用都没填"时才算无依据**。
        # 填了但指向不存在的东西，已经由上面的 `dangling_reference` 更精确地报过了 ——
        # 两条都报等于把同一个毛病数两遍，还会让"该修哪一处"变得含糊。
        if f.get("category") == CATEGORY_OBSERVED and not refs:
            issues.append({"kind": "unsupported_claim",
                           "detail": "「观察到的问题」没有任何引用："
                                     + (f.get("what") or f.get("where") or "（无描述）")})

    if source_revisions is not None:
        # `source_revisions` **就是本次材料的来源集合**（来源 → 当前修订号）。
        # 证据的来源不在里面 = 跨来源引用；修订对不上 = 更正前的引用。
        current = {str(k): v for k, v in (source_revisions or {}).items()}
        for e in evidence or []:
            if not isinstance(e, dict):
                continue
            sid = e.get("source_id")
            if not sid:
                continue          # 没有来源标识 ⇒ 判不了，**跳过而不是猜**
            if sid not in current:
                issues.append({"kind": "unknown_source",
                               "detail": f"证据 {e.get('id')} 的来源 {sid} "
                                         "不在本次材料里（跨来源引用）"})
            elif e.get("revision") != current[sid]:
                issues.append({
                    "kind": "stale_reference",
                    "detail": f"证据 {e.get('id')} 的修订 {e.get('revision')} "
                              f"与材料当前修订 {current[sid]} 不一致（更正前的引用）"})
    return issues


def build_validation(evidence: list[dict], findings: list[dict], *,
                     missed: list[str] | None = None,
                     stray: list[str] | None = None,
                     stats: dict | None = None,
                     source_revisions: dict | None = None,
                     prompt_version: str = "") -> dict:
    """结构与引用检查。**只代表结构与引用，不代表物理语义正确。**

    ⚠️ 契约⑧明确：`validation` **不得显示成「物理语义已验证」** ——
    所以返回里带 `scope` 字段，把这句话写进数据本身，
    **免得前端拿到一个 `valid: true` 就当成"物理没问题"。**
    """
    # 引用类缺陷**重算**，不读任何传进来的 validation（见 `reference_defects`）。
    issues: list[dict] = reference_defects(evidence, findings, source_revisions)

    if missed:
        issues.append({"kind": "unresolved_placeholder",
                       "detail": "未命中的统计占位符：" + "、".join(missed)})
    if stray:
        # **这一条就是上一版只能进 logger 的那条** —— 现在它能被看见了
        issues.append({"kind": "model_written_count",
                       "detail": "报告里出现模型自写的次数表述（应由服务端填充）："
                                 + "、".join(stray)})

    return {
        "valid": not issues,
        "scope": "结构与引用校验；**不代表物理语义已验证**",
        "validator_version": REVIEW_SCHEMA_VERSION,
        "prompt_version": prompt_version or "",
        # 降级的条数**单独记**，不进 `issues` —— 它不是缺陷，是**如实标注**：
        # 这些判断照样给学生看，只是归到了「需要核查」。
        # 放进 issues 会让每一份（默认 v2 提示词下的）报告都 `valid=false`，
        # 而那是**常态不是异常**，只会让 `valid` 失去意义。
        "downgraded_claims": count_downgraded_claims(findings),
        "issues": issues,
        "evidence_count": len(evidence),
        "value_count": len((stats or {}).get("value_rounds") or {}),
    }
def compute_trajectory_stats(rounds: list[dict]) -> dict:
    """**服务端数一遍**，供复盘直接引用（契约硬约束④）。

    ## ⚠️ 两条被复核指出、已修的缺陷
    ```
    ① **同轮重复被算成多轮**：第 7 轮写两次 `19.6` → 旧版给 [7, 7]，渲染成「共 2 轮」
       ⇒ 现分列 `rounds`（**去重后的轮号**）与 `occurrences`（**总出现次数**）
    ② **正负号丢失**：`-19.6` 与 `+19.6` 被归成一个
       ⇒ 现连符号一起抓、连符号一起归一化
    ```
    ## 轮号口径
    **一律用材料里真实的 `round`，并与轨迹正文用同一套编号** ——
    旧版统计用真实 `round`、正文按顺序重编，**两套号互相矛盾**，
    **比原来那个计数错更糟。** ⇒ 现在两处都用真实 `round`。
    """
    rounds = rounds or []
    idx: list[int] = []
    #: 值 → {"rounds": [去重轮号], "occurrences": 总次数}
    agg: dict[str, dict] = {}
    empty: list[int] = []
    tss: list[float] = []

    for i, r in enumerate(rounds, 1):
        n = r.get("round")
        cur = n if isinstance(n, int) else i
        idx.append(cur)

        txt = _txt(r.get("answer") or r.get("student_input"))
        if not txt.strip():
            empty.append(cur)
        for m in _NUM_RE.findall(txt):
            tok = _norm_num(m)
            if not _looks_like_a_quantity(tok):
                continue
            slot = agg.setdefault(tok, {"rounds": [], "occurrences": 0})
            slot["occurrences"] += 1
            if cur not in slot["rounds"]:
                slot["rounds"].append(cur)

        ts = r.get("ts")
        if isinstance(ts, (int, float)):
            tss.append(float(ts))

    value_rounds = {k: v for k, v in sorted(
        agg.items(), key=lambda kv: (-kv[1]["occurrences"], kv[0]))}
    stats: dict = {
        "count_basis": "服务端统计，模型不得自行计数或改写",
        "round_count": len(rounds),
        "round_indices": idx,
        "value_rounds": value_rounds,
        "empty_rounds": empty,
        "time_range": None,
    }
    if tss:
        stats["time_range"] = {
            "start_sec": round(min(tss), 3),
            "end_sec": round(max(tss), 3),
            "span_sec": round(max(tss) - min(tss), 3),
        }
    return stats
def format_stats_block(stats: dict) -> str:
    """渲染成给模型看的块 —— **明说"不许自己数"**。"""
    out = [
        "【服务端统计】（**以下数字由服务端数出，必须逐字引用，不要自己数、不要改写**）",
        "总轮次：" + str(stats["round_count"]),
        "轮次编号：" + str(stats["round_indices"]),
    ]
    tr = stats.get("time_range")
    out.append("时间范围：" + (
        str(tr["start_sec"]) + "s – " + str(tr["end_sec"])
        + "s（跨度 " + str(tr["span_sec"]) + "s）"
        if tr else "（本材料无时间戳，**不得编造时刻**）"))
    if stats["empty_rounds"]:
        out.append("无识别内容的轮次：" + str(stats["empty_rounds"]))

    vr = stats.get("value_rounds") or {}
    if vr:
        out.append("数值出现情况（**要引用出现轮次时，一律以此为准**）：")
        for v, slot in list(vr.items())[:40]:
            rs = slot["rounds"]
            occ = slot["occurrences"]
            out.append("  " + v + "  → 第 " + "、".join(str(x) for x in rs)
                       + " 轮（出现 " + str(occ) + " 次，涉及 "
                       + str(len(rs)) + " 轮）")
    else:
        out.append("数值出现情况：（这段材料里没有可统计的数值）")
    out.append("⚠️ 上面没列出的量，**不要声称它出现了几次或在哪几轮**；"
               "说不准就写「无法从材料确认」。")
    return "\n".join(out)


def _txt(v: Any) -> str:
    """轨迹里的字段可能是字符串、也可能是嵌套 dict。**两种都要能吃。**"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("hint", "answer", "text", "content", "statement", "step"):
            if v.get(k):
                return str(v[k])
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def build_review_user(problem_statement: str,
                      knowledge_points: Iterable[str],
                      rounds: list[dict],
                      material: dict | None = None) -> str:
    """组给 `review` 提示词的输入。

    **把学生实际写的东西放进去** —— 那是判"他是真会还是凑对"的唯一依据。

    ## ⚠️ 材料**只进一次**
    旧版把同一份材料进了两次：轨迹正文每轮截 300 字，证据目录发**完整原文**。
    后果不只是"预算形同虚设"，更要紧的是 ——
    **模型看到的正文，和它被允许引用的原文，不是同一份**。
    ⇒ 现在证据目录与正文都由**同一个 `material`** 渲染（见 `build_material`），
      它引用的原文必然是它看过的。
    """
    material = material if material is not None else build_material(rounds)
    items = material["items"]
    # ⚠️ 证据目录**从同一份 material 派生** —— 不再各建一次。
    # 各建一次时，只要两边的构造参数有一点点差别（比如以后谁改了一边的预算），
    # 模型看到的正文和它能引用的原文就会**静默漂开**。
    evidence = [{k: it[k] for k in ("id", "source", "source_id", "revision",
                                    "round", "ts", "text")} for it in items]

    lines = []
    for it in items:
        omitted = ""
        if it.get("mid_line_cut"):
            # ⚠️ **截在行中间**要单独说：这段文字看起来是完整的句子，但它是半句。
            # 不说的话，模型会把"没写完"当成"学生写到这就停了"。
            omitted = (f"【⚠️ 本轮**被截断在行中间**：上面是原文的前一段，"
                       f"后面还有约 {it['omitted_chars']} 字没发出 —— "
                       "**截断不代表学生只写到这，也不得据此说推理不完整**】")
        elif it.get("omitted_lines"):
            omitted = (f"【本轮后 {it['omitted_lines']} 行（约 {it['omitted_chars']} 字）"
                       "因预算未展示 —— **未展示不代表学生未写**】")
        lines.append(
            f"第{it['round']}轮  视频时刻={it['ts'] if it['ts'] is not None else '未知'}秒\n"
            f"    识别/更正文本：{it['text']}{omitted}"
        )
    if material["omitted_rounds"]:
        lines.append(
            "【以下轮次因预算**整轮未展示**：" +
            "、".join(f"第{n}轮" for n in material["omitted_rounds"]) +
            " —— **未展示不代表这些轮没有内容，也不得据此推断学生漏了推理**】")

    kps = "、".join(sorted({k for k in knowledge_points if k}))
    return ("【证据边界】以下仅含稀疏OCR快照；系统提示、诊断和分数已排除，非停顿标注。"
            "轮数、分数、重复文本均不能证明停笔时长或认知卡住；不得断言学生不会、猜测或凑对。"
            "只指出原文支持的错误；未展示的推理不得当作错误推理。大小加明确物理方向与一致带符号表达同样合法，"
            "不能强制只有一种选轴或符号写法。没有明确错误时sticking_points可为空。\n\n"
            # 契约⑨：省略必须如实说，且**不得据此判断遗漏推理**
            "【材料预算】学生材料按**完整行**组织，优先保留最新作答，"
            f"总预算 {material['budget']} 字符（本次实际 {material['chars']} 字符）。"
            "被省略的部分已逐处标明 —— **省略是「没发给你」，不等于「学生没写」，"
            "更不得据此说学生跳过了推理**。\n\n"
            f"【题目】\n{problem_statement}\n\n"
            f"【本题知识点】（`key_concept` 必须逐字取自这里）\n{kps}\n\n"
            # ⭐ 契约硬约束④：轮次/次数/范围由**服务端**统计，不由模型编造
            f"{format_stats_block(compute_trajectory_stats(rounds))}\n\n"
            f"【证据目录】（**引用时只能用这里的 ID**）\n"
            + format_evidence_catalog(evidence) + "\n\n"
            f"【可用识别轨迹片段】（{len(items)} 轮）\n" + "\n".join(lines))


# ================================================================ 服务端填充（硬约束④后半）
#
# ## 为什么只"喂正确输入"不够
# 上一版我把统计喂给了模型，**但报告正文里的次数仍由模型自由写** ——
# 它还是可以写「四轮」。**那个原始缺陷并没有被堵死，只是被提示了。**
#
# ⇒ 把数字**从模型手里拿走**：它只能写占位符，**真实轮次由服务端填**。

#: 模型要引用某个数值的出现情况时，写这个占位符
PLACEHOLDER_RE = re.compile(r"\[\[\s*value\s*:\s*([^\]]+?)\s*\]\]")

#: 兜底：模型可能不听话，直接写「N 轮」「N 次」。**这些要报出来。**
#:
#: ## ⚠️ 但必须区分「计数」和「定位」—— 这是真机跑出来的
#: 第一版判据太宽，把合法内容也报了。真机跑 v3 时剩下的 3 处是：
#: ```
#: 「**第7轮**出现 a = 1.6，**第9轮**起改为 a = 1.49」
#: 「**第14轮** F = 49 N 中的 49 识别清晰」
#: ```
#: **那不是"出现了几次"，那是"在某一轮里"** —— 而**定位正是可追溯性要的东西**，
#: 不该被报成问题。
#:
#: ⇒ 判据据此收紧：
#: ```
#: 计数（要报）  「第4、5、6、7、8轮」/「共五轮」/「出现在 5 轮里」  ← 多轮或带量词
#: 定位（不报）  「第7轮写的是…」                                  ← **单轮，且带「第」**
#: ```
#: ## 收紧判据**顺带修掉了原来的误报**
#: ```
#: 原误报  「两轮车里」→ 抓 `两轮`（那不是计数）
#:         ⇒ **没有了** —— 新判据要求「逗号列表 / 量词 / 次」，裸的「N 轮」不再报
#: ```
#: ## 仍知的洞（如实留着）
#: ```
#: 漏报  单轮计数写成「出现了 5 轮」以外的花样（如「五轮里都有」）可能漏
#: ```
#: **因为只报不改**：误报的代价 = 一条日志，漏报的代价 = 少一条提醒。
#: **真正的堵法是 `[[value:X]]` 占位符，不是这个检测。**
_STRAY_CN = "一二三四五六七八九十两"
#: 多轮/带量词的计数。「第4、5、6轮」这种**逗号列表**是关键形态 ——
#: **那次真实计数错（五轮写成四轮）就是它。**
#:
#: ⚠️ **2026-10-09 修 P0（独立复核发现，实测复现）**：
#: 旧版最后一条是**裸的** `[0-9一二三四五六七八九十两]{1,4}\s*次` ——
#: 于是中文里最普通的那个副词被当成计数：
#: ```
#: 「整份材料只有一次快照」        → 命中 `一次`
#: 「本轮学生一次性给出了完整解答」  → 命中 `一次`
#: 「y 是 x 的一次函数」           → 命中 `一次`
#: ```
#: **归档 72 份里 42 份（58%）的唯一命中就是「一次」**（复核独立重放实测）。
#: 在**只报不改**时那只是日志噪声；**改成 fail-closed 阻断之后，它变成误杀**：
#: 一份内容正确的报告被告知"统计字段不合法"，而它无从改起。
#: ⇒ 现在要求 `N 次` 必须带**计数上下文**（共/一共/出现/重复…）。
#: **代价是漏报**（裸的「改了 3 次」抓不到）—— 方向刻意选"少拦不误杀"。
STRAY_COUNT_RE = re.compile(
    rf"(?:[0-9{_STRAY_CN}]{{1,4}}\s*[、,，]\s*){{1,}}[0-9{_STRAY_CN}]{{1,4}}\s*轮"   # 第4、5、6轮
    rf"|共\s*[0-9{_STRAY_CN}]{{1,4}}\s*(?:轮|次)"                                    # 共五轮 / 共 3 次
    rf"|出现在\s*[0-9{_STRAY_CN}]{{1,4}}\s*轮"                                       # 出现在 5 轮
    # N 次：**必须带计数上下文**，否则「一次」这种普通副词会被误杀（见上）。
    # 允许上下文与数字之间隔几个字（「一共改了 3 次」），但仍要求数字**紧贴**「次」。
    rf"|(?:一共|合计|总计|重复|出现了?)[^。；！？]{{0,6}}[0-9{_STRAY_CN}]{{1,4}}\s*次"
)

#: **服务端自己渲染**的统计片段 —— 扫之前必须先摘掉。
#:
#: ⚠️ **不摘掉就会自伤**（2026-10-09 修 P0，实测复现）：
#: ```
#: render_value_occurrence(...) = '第 4、8 轮（出现 2 次，涉及 2 轮）'
#: find_stray_counts(它)        = ['4、8 轮', '2 次']      ← 服务端自己的产物
#: ⇒ 每一次**占位符成功填充**都必然产出一条 model_written_count
#: ⇒ 模型越听话用 `[[value:X]]`，越必然被闸门打回（而它无从改起 ⇒ 重试死循环）
#: ```
#: `tests/test_trajectory_stats.py` 有一条测试把**渲染格式与这个正则**绑在一起：
#: 改了 `render_value_occurrence` 的格式而没改这里，测试会红。
_SERVER_RENDER_RE = re.compile(r"第\s*[0-9、]+\s*轮（出现\s*\d+\s*次，涉及\s*\d+\s*轮）")


def strip_server_renders(text: str) -> str:
    """摘掉服务端自己渲染的统计片段，剩下的才是**模型写的**（见 `_SERVER_RENDER_RE`）。"""
    return _SERVER_RENDER_RE.sub("", text or "")


def render_value_occurrence(stats: dict, value: str) -> str:
    """**服务端**把这个数值的出现情况渲染成一句话。

    找不到该数值时**如实说找不到**，不编。
    """
    key = _norm_num(value)
    slot = (stats.get("value_rounds") or {}).get(key)
    if slot is None:
        # 试一次宽松匹配（模型可能少写/多写符号）
        for k, v in (stats.get("value_rounds") or {}).items():
            if k.lstrip("+-") == key.lstrip("+-"):
                slot, key = v, k
                break
    if slot is None:
        return f"（服务端统计里没有 {value}，**不得据此推断次数**）"
    rs = slot["rounds"]
    return ("第 " + "、".join(str(x) for x in rs) + " 轮"
            + f"（出现 {slot['occurrences']} 次，涉及 {len(rs)} 轮）")


def fill_stats_placeholders(text: str, stats: dict) -> tuple[str, list[str]]:
    """把 `[[value:X]]` 换成**服务端渲染的**真实轮次。

    返回 `(填好的文本, 未命中的占位符列表)` ——
    **未命中要报出来**，不能静默留一个占位符给学生看。
    """
    missed: list[str] = []

    def _sub(m):
        v = m.group(1)
        rendered = render_value_occurrence(stats, v)
        if rendered.startswith("（服务端统计里没有"):
            missed.append(v)
        return rendered

    return PLACEHOLDER_RE.sub(_sub, text or ""), missed


def find_stray_counts(text: str) -> list[str]:
    """**兜底**：找出模型自己写的「N 轮 / N 次」。

    ⚠️ **这里只是"找出来报"，不是"改掉"。**
    因为**改自由文本有改错的风险**（它可能不是在说统计，而是在说别的）。

    ## 判据的两条边界（真机跑出来的）
    ```
    计数（要报）  「第4、5、6、7、8轮」/「共五轮」/「出现在 5 轮里」/「改了 3 次」
    定位（不报）  「第7轮写的是…」「第14轮 F = 49 N」  ← **单轮 + 带「第」= 可追溯性要的东西**
    ```
    ## ⚠️ 2026-10-09 更正：**本文档此前把两个已修的洞写成了"未修"**
    ```
    旧文本写着  漏报「第4、5、6、7、8轮」只抓到 `8轮` / 误报「两轮车里」抓到 `两轮`
    实测（scripts 探针，13 条样本）：**两个都已修**
        「两轮车里坐着两个人」      → 不报  ✅
        「第4、5、6、7、8轮都有」    → 报 `4、5、6、7、8轮`  ✅
        「第7轮 / 第14轮 / 第9轮」   → 不报  ✅（定位不误伤）
    ⇒ 当时照着旧注释去"修检测器"，会去修一个**已经修好的东西**。
      **文档里的"已知缺陷"必须能复现，否则它自己就是缺陷。**
    ```
    ## 仍知的洞（**实测确认存在**，如实留着）
    ```
    漏报  **裸量词 + 轮** 的形态一律抓不到，例如
          「五轮里都有」「这五轮都稳」「前四轮都对」
    范围  = 「逗号列表 / 共N轮 / 出现在N轮 / N次」之外的一切写法
    ```
    ⚠️ **这个洞的方向是"少拦"，不是"误杀"。** 在本函数**只报不改**时，
    两个方向的代价都只是一条日志；但一旦它被用作**阻断判据**，方向就重要了：
    **漏报 = 一份坏报告放行（等同现状）；误报 = 一份好报告被打掉、还要重试。**
    所以判据**刻意偏保守**：宁可漏，不误杀。要收紧请先补样本、别直接加宽正则。

    ⚠️ **因此它只是局部兜底，不是第 3 条修复的完成标志。**
    真正的堵法是让模型**没有机会写计数**（占位符 + 服务端填充），
    这个检测只兜住"它还是不听话"的一部分形态。
    """
    return [m.group(0).strip() for m in STRAY_COUNT_RE.finditer(text or "")]


#: **模型自由书写**的文本字段 —— 服务端填充的与学生原文的**不在内**。
#:
#: ⚠️ 这份范围是"只报不改"类检查的判据基础，**写错会误杀**：
#: ```
#: 排除 answer_check.student_answer  逐字引用学生 ⇒ 里面出现「两轮车」是**学生写的**
#: 排除 evidence[*].text            服务端逐字取自轨迹 ⇒ 同上
#: ```
#: 把它们算进来，等于**拿学生自己的话去判模型违规**。
_MODEL_TEXT_FIELDS = ("timeline", "key_concept", "next_step", "encouragement")
_SP_FIELDS = ("where", "what", "why", "downstream")
_AC_FIELDS = ("derived_from_work", "consistent", "note", "ocr_caution")


def report_text_fields(payload: dict) -> list[tuple[str, str]]:
    """列出 `(字段路径, 文本)`，只含**模型自由书写**的字段（见 `_MODEL_TEXT_FIELDS`）。"""
    out: list[tuple[str, str]] = []
    if not isinstance(payload, dict):
        return out
    for key in _MODEL_TEXT_FIELDS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            out.append((key, value))
    for i, item in enumerate(payload.get("sticking_points") or []):
        if isinstance(item, dict):
            for f in _SP_FIELDS:
                v = item.get(f)
                if isinstance(v, str) and v.strip():
                    out.append((f"sticking_points[{i}].{f}", v))
    ac = payload.get("answer_check")
    if isinstance(ac, dict):
        for f in _AC_FIELDS:
            v = ac.get(f)
            if isinstance(v, str) and v.strip():
                out.append((f"answer_check.{f}", v))
    for i, item in enumerate(payload.get("findings") or []):
        if isinstance(item, dict):
            for f in _SP_FIELDS:
                v = item.get(f)
                if isinstance(v, str) and v.strip():
                    out.append((f"findings[{i}].{f}", v))
    return out


def structural_defects(payload: dict, source_revisions: dict | None = None,
                       evidence_pool: list[dict] | None = None) -> list[dict]:
    """**重算**这份报告的全部结构缺陷 —— 生成路径与缓存路径**共用这一个判据**。

    ⚠️ **不读 `payload["validation"]`。** 缓存里存着上一版代码写的 `validation`；
    信它，就等于信写它时的代码 —— 而缺陷恰恰可能是那时的代码漏掉的。

    ⚠️ **这是 fail-closed 的落点**：调用方以「返回非空」为阻断条件，
    而**不是**按"已知种类白名单"判断 —— 白名单会让**将来任何新种类静默不阻断**
    （契约 ⑦ 2026-10-09 澄清；那个形态正是缺陷长期存在的原因）。

    `evidence_pool`：**本次授权的证据池**（契约 §2.8）。传了才判引用类缺陷；
    **不传 = 不判**（历史报告缺新结构，不因此算违规 —— 契约 §2.8 验收第 1 条）。
    """
    if not isinstance(payload, dict):
        return [{"kind": "malformed", "detail": "报告不是对象"}]

    issues: list[dict] = []
    evidence, findings = payload.get("evidence"), payload.get("findings")
    if isinstance(evidence, list) and isinstance(findings, list):
        # v1/v2 没有这两个结构 ⇒ 无从判引用（契约：历史报告保留读取能力）
        issues.extend(reference_defects(evidence, findings, source_revisions))

    stray: list[tuple[str, str]] = []
    missed: list[tuple[str, str]] = []
    claims: list[tuple[str, str]] = []
    for where, text in report_text_fields(payload):
        # ⚠️ **先摘掉服务端自己渲染的统计片段** —— 否则每一次成功的占位符填充
        # 都会被判成"模型自写次数"（见 `_SERVER_RENDER_RE`，实测复现过）。
        own = strip_server_renders(text)
        stray.extend((where, s) for s in find_stray_counts(own))
        missed.extend((where, m.group(0)) for m in PLACEHOLDER_RE.finditer(own))
        claims.extend((where, c) for c in unsupported_claims(text))
    if stray:
        issues.append({"kind": "model_written_count",
                       "detail": "报告里出现模型自写的次数表述（应由服务端填充）："
                                 + "、".join(_dedupe_hits(stray))})
    if missed:
        issues.append({"kind": "unresolved_placeholder",
                       "detail": "未命中的统计占位符：" + "、".join(_dedupe_hits(missed))})
    if claims:
        issues.append({"kind": "unsupported_process_claim",
                       "detail": "无依据的过程/认知断言（原文里没有依据支持）："
                                 + "、".join(_dedupe_hits(claims))})
    # ---- 契约 §2.8：引文只由结构化引用生成 ----
    # ⚠️ 这两条**只在传了证据池时**才判（`evidence_pool is None` = 老调用方/历史报告）
    #    —— 历史报告保留读取能力，**不因缺新结构就判违规**（契约 §2.8 验收第 1 条）。
    if evidence_pool is not None:
        pool_bad = citation_pool_status(evidence_pool)
        if pool_bad:
            issues.append({"kind": "unauthorized_evidence_source",
                           "detail": "证据池里有未授权来源（教学/复盘输入不得含评测专用来源）："
                                     + "、".join(pool_bad)})
        cite_bad, quote_bad = [], []
        order_bad = []
        for where, text in report_text_fields(payload):
            cite_bad.extend((where, f"{d['kind']}：{d['raw']}（{d['detail']}）")
                            for d in citation_defects(text, evidence_pool, source_revisions))
            # Verbatim server catalogues may contain quotations inside the
            # student's own text; those are not model source attributions.
            server_timeline = where == 'timeline' and text == build_server_timeline({'items': evidence_pool})
            if not server_timeline:
                quote_bad.extend((where, q) for q in unattributed_quotes(text, evidence_pool,
                    strict_sources=payload.get('render_policy') == 'physical_check_only_v1'))
            # 契约 §2.9：**路径/次序断言**（材料只有结论值时不许叙述他的计算顺序）。
            # ⚠️ 覆盖**全部字段** —— 防的就是"路径断言从 timeline 搬去别处"。
            order_bad.extend((where, o) for o in path_claim_defects(text, evidence_pool, where))
        if cite_bad:
            issues.append({"kind": "invalid_citation",
                           "detail": "非法引用（不得投递）："
                                     + "、".join(_dedupe_hits(cite_bad))})
        if quote_bad:
            issues.append({"kind": "unattributed_quote",
                           "detail": "模型自写的归属引文（该走引用记号）："
                                     + "、".join(_dedupe_hits(quote_bad))})
        if order_bad:
            # 🔴 **暂不阻断**（2026-10-10 实测）：跨 8 批归档命中 **29 处**，而其中
            #    很大一部分落在 `answer_check.derived_from_work` ——
            #    那**正是契约 §2.9 指定的"系统重建区域"**（推导本来就该写在那里）。
            #    ⇒ 精度不够格当阻断。按项目纪律（也是 v12 的教训）：
            #      **量出精度不够就不许设成阻断**；先只记日志，收窄后再上。
            logger.warning("路径/次序断言（暂只记录，未阻断）：%s",
                           _dedupe_hits(order_bad))
    return issues


def _dedupe_hits(hits: list[tuple[str, str]]) -> list[str]:
    """同一段文字只报一次（保留第一次出现的位置）。

    ⚠️ **为什么需要**：`report_text_fields()` 把 `findings[]` 也算进来了，
    而 `findings` 是 `build_findings()` **从 `sticking_points` / `answer_check` 复制**出来的
    —— 同一句话会在两个字段里各命中一次。
    **判定不受影响**（非空即阻断），但明细会重复，读的人会以为"问题有两处"。
    ⇒ 保留扫描范围（将来若有**不是**派生出来的 finding 仍然覆盖到），只把明细去重。
    """
    first: dict[str, str] = {}
    for where, hit in hits:
        first.setdefault(hit, where)
    return [f"{where}：{hit}" for hit, where in first.items()]


def classify_defects(issues: list[dict]) -> str:
    """把结构缺陷**分类**，好让失败信息说清是哪一类（契约 ⑦ / 用户第 4 条）。

    ⚠️ 只分**能由规则判定的**几类。**物理质量不在其列** ——
    规则判不了物理对错，把它混进来就是冒充。
    """
    kinds = {i.get("kind") for i in issues}
    reference = {"dangling_reference", "unknown_source", "stale_reference",
                 "unsupported_claim", "malformed"}
    stats = {"model_written_count", "unresolved_placeholder"}
    attribution = {"unsupported_process_claim"}
    parts = []
    if kinds & reference:
        parts.append("引用不合法")
    if kinds & stats:
        parts.append("统计字段不合法")
    if kinds & attribution:
        parts.append("无依据的过程/认知断言")
    if not parts:
        return "结构不合法（未分类）"
    return "、".join(parts)



def review_trajectory(problem, rounds: list[dict], *,
                      knowledge_points: Optional[Iterable[str]] = None,
                      llm=None,
                      source_revisions: Optional[dict] = None) -> Optional[ReviewReport]:
    """跑一次复盘。**失败返回 None，绝不抛** ——

    复盘是锦上添花：**没有它，逐轮轨迹照样能看**。
    为一份复盘把整条视频链路打挂，不划算。

    `source_revisions` 由调用方给（视频那条路有），用于判「跨来源 / 更正前引用」；
    **不给就跳过那两条**，不猜。
    """
    if not rounds:
        return None
    try:
        from app.agent.prompt_registry import get_prompt, resolve_version
        from app.services.llm import get_llm

        # ⚠️ **实际跑的是哪一版提示词，要记下来** —— 见 `review_schema_version()`。
        prompt_version = resolve_version("review")
        # ⚠️ **题面必须与正文用的是同一份**（契约 §2.8③）：正文在 `build_review_user` 里
        #    取 `problem.statement`，材料池也必须取**同一个表达式** ——
        #    两处各写一遍，将来谁改了一处就会静默漂开（这个函数上面的注释记过同类事故）。
        _statement = getattr(problem, "statement", "") or getattr(problem, "content", "")
        material = build_material(rounds, statement=_statement)
        kps = list(knowledge_points or [])
        if not kps:
            kps = list(getattr(problem, "knowledge_points", None) or [])
        user = build_review_user(
            _statement,
            kps, rounds, material)
        out = (llm or get_llm("review")).chat_structured_resilient(
            get_prompt("review"), user, ReviewReport)
        if out.value is None or getattr(out, "degraded", False):
            # Exhausted resilient calls may return a schema-safe default value.
            # Its existence is not evidence that a model produced a report.
            logger.warning("复盘失败（不影响轨迹）：%s", getattr(out, "reasons", []))
            return None
        out.value._raw_generation = out.value.model_dump()
        out.value.render_policy = 'legacy'
        out.value.canonicalized_fields = []
        if prompt_version == 'v4':
            out.value.render_policy = 'physical_check_only_v1'
            for index, point in enumerate(out.value.sticking_points):
                # The old field requests an unobservable causal continuation.
                # Preserve its model value privately; display no inference of it.
                point.downstream = '实际后续过程与认知原因未由材料确认。'
                out.value.canonicalized_fields.append(f'sticking_points[{index}].downstream')
        if not isinstance(out.value.timeline, str) or not out.value.timeline.strip():
            logger.warning("复盘失败（不影响轨迹）：模型返回空白轨迹总结")
            return None
        if out.value.key_concept not in kps or not out.value.sticking_points:
            out.value.key_concept = ""
        if out.value.answer_check is not None:
            # Quote the actual latest text, never the model's reconstructed prose.
            actual = next((_txt(r.get("answer") or r.get("student_input"))
                           for r in reversed(rounds) if r.get("answer") or r.get("student_input")), "")
            out.value.answer_check.student_answer = actual
        # ---- 服务端填充（硬约束④后半）----
        #
        # 只"喂正确输入"不够 —— 报告正文里的次数仍由模型自由写。
        # ⇒ 把数字**从模型手里拿走**：它写 `[[value:X]]`，服务端填真实轮次。
        # 并**兜底揪出**它自己写的「N 轮 / N 次」（含中文数字 ——
        # 那次真实计数错就是「四轮」）。
        _stats = compute_trajectory_stats(rounds)
        _missed: list[str] = []
        _stray: list[str] = []

        def _fill(text: str) -> str:
            filled, missed = fill_stats_placeholders(text, _stats)
            _missed.extend(missed)
            # ⚠️ **扫模型原文，不扫填充后的文本**（2026-10-09 修 P0）。
            # 旧版扫的是 `filled`，而 `render_value_occurrence` 的产物
            # （'第 4、8 轮（出现 2 次，涉及 2 轮）'）**自身就带计数形态** ⇒
            # 每一次成功填充都必然记一条"模型自写次数"，
            # **模型越听话用占位符，越必然被打回**。实测复现过。
            _stray.extend(find_stray_counts(text))
            return filled

        out.value.timeline = _fill(out.value.timeline)
        out.value.next_step = _fill(out.value.next_step)
        out.value.encouragement = _fill(out.value.encouragement)
        for _sp in out.value.sticking_points:
            for _attr in ("where", "what", "why", "downstream"):
                _cur = getattr(_sp, _attr, "") or ""
                setattr(_sp, _attr, _fill(_cur))
        if out.value.answer_check is not None:
            for _attr in ("note", "ocr_caution", "derived_from_work"):
                _cur = getattr(out.value.answer_check, _attr, "") or ""
                setattr(out.value.answer_check, _attr, _fill(_cur))

        # ⚠️ **只报不改。**
        # 改自由文本有改错的风险（它可能不是在说统计，而是在说别的）——
        # 而"悄悄替换"比"留着让人看见"更危险。
        # ⇒ 记日志；**v3 的 `validation` 要把它带给前端**。
        if _missed:
            logger.warning("复盘里有未命中的统计占位符：%s", _missed)
        if _stray:
            logger.warning(
                "复盘里出现模型自写的次数表述（**应由服务端填充**）：%s", _stray)
        # ---- v3 三个结构：**服务端填** ----
        out.value.review_schema_version = review_schema_version(prompt_version)
        out.value.evidence = material_evidence(material)
        # ⚠️ **渲染引用记号**（契约 §2.8①）：把 `⟦E1.1:5-19⟧` 换成**服务端截取的原文**。
        #    **必须在 `evidence` 填好之后**（池子是这一刻的证据目录）；
        #    非法记号**原样留着** —— 它随后会被结构闸门判 `invalid_citation` 而**不得投递**，
        #    在这里悄悄删掉反而会让缺陷变得不可见。
        _cite_bad: list[dict] = []
        for _attr in ("timeline", "next_step", "encouragement"):
            _rendered, _bad = render_citations(getattr(out.value, _attr) or "",
                                               out.value.evidence)
            setattr(out.value, _attr, _rendered)
            _cite_bad.extend(_bad)
        for _sp in out.value.sticking_points:
            for _attr in ("where", "what", "why", "downstream"):
                _rendered, _bad = render_citations(getattr(_sp, _attr) or "",
                                                   out.value.evidence)
                setattr(_sp, _attr, _rendered)
                _cite_bad.extend(_bad)
        if out.value.answer_check is not None:
            for _attr in ("note", "ocr_caution", "derived_from_work"):
                _rendered, _bad = render_citations(
                    getattr(out.value.answer_check, _attr) or "", out.value.evidence)
                setattr(out.value.answer_check, _attr, _rendered)
                _cite_bad.extend(_bad)
        if _cite_bad:
            logger.warning("复盘里有非法引用（**不得投递**，交由结构闸门阻断）：%s", _cite_bad)
        # ⚠️ **`timeline` 由服务端生成**（契约 §2.9）：模型写的那份**整段替换**。
        #    理由见 `build_server_timeline` 的 docstring —— 模型无权叙述学生的实际解题过程，
        #    而"先算哪个再算哪个"正是 v13 唯一卡点 Q066 的形态。
        #    ⚠️ **放在引用渲染之后**：服务端这段本来就不含记号，替换掉模型那份即可。
        out.value.timeline = build_server_timeline(material, _stats)
        out.value.findings = build_findings(out.value, _stats, out.value.evidence)
        # ⚠️ **`_missed` / `_stray` 到这里才真正被看见** ——
        # 上一版它们只进 logger，前端与人审都看不到。
        out.value.validation = build_validation(
            out.value.evidence, out.value.findings,
            missed=_missed, stray=_stray, stats=_stats,
            source_revisions=source_revisions, prompt_version=prompt_version)
        return out.value
    except Exception:  # noqa: BLE001
        logger.exception("复盘异常（不影响轨迹）")
        return None
