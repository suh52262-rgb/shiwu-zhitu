"""教师端的**纯逻辑**：知识点分类 + 聚合 + 学情诊断文本生成。

## 为什么单独一个模块

把"数据 → 该说什么"从渲染里拆出来，**可单测**。
本模块**不导入 streamlit、不连数据库、不发网络请求** —— 只做纯计算。
调用方负责取数、负责画图。

## 学科分类为什么放在前端（而不是后端）

后端 `/teacher/knowledge-points/stats` 返回的是**扁平的**知识点列表
（`{knowledge_point, student_count, mention_count}`），**没有学科字段**。
而"按力学 / 热学 / 电磁学…分组"是**纯展示需求**。

按「让前端适应后端」的原则，这一层映射就放在前端：
  · 后端零改动，也不新增表 / 字段
  · 分类规则集中在 `DISCIPLINE_RULES` 一处，要调整只改这里
  · 代价：**规则是关键词映射，不是权威分类**。知识点的权威归属应由
    领域专家裁定（`data/knowledge_vocabulary.json` 的注释也写了
    "词表本身需要领域判断，不能由机器定"）。
    所以界面会注明"分类基于关键词规则"，不宣称它是权威结论。

## 映射的优先级

规则**按顺序匹配，先具体后宽泛**：
`"电磁感应"` 必须命中「电磁学」而不是被力学里的单字 `"力"` 抢走 ——
所以学科规则里把量子 / 光学 / 电磁 / 热学排在力学前面，
力学放最后（它含"力""运动"这类宽泛词，兜底能力最强）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

#: 未命中任何学科规则时的归类。
#: 不硬塞进"力学" —— 那会把分类错误伪装成分类成功。
UNCLASSIFIED = "未分类"

#: 学科大类规则：**(学科名, 关键词元组)**，顺序敏感（先具体后宽泛）。
DISCIPLINE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "量子物理",
        ("量子", "光子", "光电效应", "波函数", "薛定谔", "能级", "德布罗意",
         "康普顿", "不确定关系", "黑体辐射", "放射性", "衰变", "核反应",
         "氢原子", "玻尔", "波粒二象", "跃迁", "基态", "激发态"),
    ),
    (
        "光学",
        ("折射", "反射", "干涉", "衍射", "偏振", "透镜", "焦距", "光栅",
         "双缝", "波长", "光谱", "相干", "成像", "全反射", "色散", "光程",
         "薄透镜", "放大率"),
    ),
    (
        "电磁学",
        ("电场", "磁场", "电荷", "电流", "电容", "电感", "电阻", "电势",
         "电动势", "电磁感应", "磁通", "洛伦兹", "安培", "法拉第", "楞次",
         "库仑", "高斯定理", "欧姆", "基尔霍夫", "自感", "互感", "极化",
         "电通量", "磁矩", "位移电流"),
    ),
    (
        "热学",
        ("温度", "热", "熵", "内能", "焓", "卡诺", "绝热", "等温", "等容",
         "等压", "理想气体", "分子", "热力学", "麦克斯韦速率", "玻尔兹曼",
         "自由能", "相变", "熔化", "汽化", "热容量", "热机"),
    ),
    (
        "力学",
        ("力", "运动", "速度", "加速度", "动量", "能量", "功", "弹簧",
         "碰撞", "简谐", "振动", "转动", "角", "牛顿", "刚体", "质心",
         "平衡", "摩擦", "张力", "惯性", "位移", "轨迹", "参考系", "胡克",
         "力矩", "约束", "矢量", "做功", "势能", "动能", "守恒", "单位换算",
         "一元二次", "分离条件", "机械能", "连接体", "滑轮", "绳", "冲量",
         "变质量", "冲力", "反冲",
         # 以下几条不含"力 / 运动"这类词，靠上面匹配不到，需显式列出
         # （实测核对题库 47 个知识点时发现漏了它们）
         "隔离法", "整体法", "正方向", "系统边界", "力的合成"),
    ),
)

#: 力学内部的知识领域细分：**(子领域, 关键词元组)**，同样顺序敏感。
#:
#: ## 为什么需要这一层
#: 当前题库 8 道题**全部是力学题**（见 `data/problems/`），
#: 所以"按学科"分只能得到一个扇区 —— 饼图没有信息量。
#: 再细分一层，教师才能真正看出"力学里具体弱在哪一块"。
#: 这一层刻意做得**比学科规则更贴近教学口径**（讲评时就是这么分块的）。
MECH_SUBFIELD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("刚体转动", ("刚体", "定轴转动", "转动惯量", "力矩", "转动定律", "质心",
                  "角动量", "进动", "陀螺", "平行轴")),
    ("振动与波动", ("简谐", "振动", "平衡位置", "胡克", "弹性势能", "振幅",
                    "周期", "频率", "相位", "波动", "波速", "驻波", "阻尼")),
    ("动量与碰撞", ("动量", "冲量", "碰撞", "变质量", "相对速度", "参考系",
                    "反冲", "平均冲力", "冲击")),
    ("功与能量", ("机械能", "动能定理", "势能", "做功", "保守力", "系统边界",
                  "功和能", "能量守恒")),
    ("牛顿力学与受力", ("牛顿", "受力", "隔离", "整体", "张力", "内力", "外力",
                        "平衡条件", "摩擦", "连接体", "绳", "滑轮", "接触力",
                        "分离条件", "约束", "惯性", "临界", "力的合成")),
    ("运动学", ("转速", "角速度", "角加速度", "切向", "法向", "匀变速",
                "加速度", "速度", "位移", "轨迹", "运动阶段", "匀加速",
                "自由落体", "抛体")),
    ("通用与数学", ("单位换算", "一元二次", "矢量性", "正方向", "量纲", "取舍")),
)

#: 力学以外的学科，子领域退化为"该学科本身"（数据量通常不足以再分）。
DEFAULT_SUBFIELD = "综合"


def _match(text: str, rules: Sequence[tuple[str, tuple[str, ...]]]) -> str | None:
    """按规则顺序返回第一个命中的分类名；都没命中返回 None。"""
    if not text:
        return None
    for name, keywords in rules:
        for kw in keywords:
            if kw in text:
                return name
    return None


def classify_discipline(concept: str) -> str:
    """知识点 → 学科大类。未命中返回 `UNCLASSIFIED`。"""
    return _match(concept or "", DISCIPLINE_RULES) or UNCLASSIFIED


def classify_subfield(concept: str, discipline: str | None = None) -> str:
    """知识点 → 学科内的知识领域。

    力学走二级细分（`MECH_SUBFIELD_RULES`）；
    其它学科数据量通常不足以再分，统一返回 `DEFAULT_SUBFIELD`。
    """
    disc = discipline or classify_discipline(concept)
    if disc == "力学":
        return _match(concept or "", MECH_SUBFIELD_RULES) or DEFAULT_SUBFIELD
    return DEFAULT_SUBFIELD


def classify(concept: str) -> tuple[str, str]:
    """一次拿到 `(学科, 知识领域)`，避免调用方重复分类。"""
    disc = classify_discipline(concept)
    return disc, classify_subfield(concept, disc)


# ---------------------------------------------------------------- 聚合


def normalize_rows(raw: Iterable[dict]) -> list[dict]:
    """把后端 / SQL 的扁平行补上分类字段，并为缺失字段兜底。

    输入行形如 `{"knowledge_point": ..., "student_count": ..., "mention_count": ...}`。
    容错点（都踩得到）：
      · 字段名缺失 / 为 None → 归零，不抛异常（教师端不该因为一个坏行整页崩）
      · 计数为 None（SQL 的 SUM 在空集上是 NULL）→ 0
    """
    out: list[dict] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        concept = str(r.get("knowledge_point") or r.get("concept") or "").strip()
        if not concept:
            continue
        disc, sub = classify(concept)
        out.append(
            {
                "concept": concept,
                "mentions": int(r.get("mention_count") or r.get("mentions") or 0),
                "students": int(r.get("student_count") or r.get("students") or 0),
                "discipline": disc,
                "subfield": sub,
            }
        )
    return out


def _aggregate(rows: Sequence[dict], key: str) -> list[dict]:
    """按 `rows` 里的某个字段（`discipline` 或 `subfield`）汇总。

    返回 `[{name, mentions, students, concepts}]`，按提及次数降序。

    ⚠️ `students` 是**各知识点涉及学生数的最大值**，
    **不是**该模块的真实去重人数。真实并集需要再查一次库，这里拿不到。

    因此这个字段**只标注为 `students_upper_hint`**，不得当作人数使用：
    审阅实测过按它算覆盖面会**少算**
    （A 生只涉及动量、B 生只涉及电容，两个知识点各 1 人 →
     取 max 得 1，真实并集是 2）。

    需要精确覆盖人数时，用 `teacher_data.covered_students()` ——
    那是独立的 `COUNT(DISTINCT student_id)`。本函数不再输出会被误用成
    "人数"的字段名。

    内部共用一个实现：学科与知识领域只是换了分组字段，
    各写一遍迟早出现"两处口径不一致"。
    """
    buckets: dict[str, dict] = {}
    for r in rows:
        name = r.get(key) or UNCLASSIFIED
        b = buckets.setdefault(
            name, {"name": name, "mentions": 0, "students_upper_hint": 0, "concepts": []}
        )
        b["mentions"] += int(r.get("mentions") or 0)
        b["students_upper_hint"] = max(
            b["students_upper_hint"], int(r.get("students") or 0)
        )
        b["concepts"].append(r.get("concept", ""))
    return sorted(buckets.values(), key=lambda x: (-x["mentions"], x["name"]))


def discipline_totals(rows: Sequence[dict]) -> list[dict]:
    """按学科汇总。返回 `[{name, mentions, students_upper_hint, concepts}]`，提及次数降序。

    注意 `students_upper_hint` **不是**人数上限的精确值，只是各知识点
    "涉及学生数"的最大值。展示时不要写成"N 名学生的 N"。
    """
    return _aggregate(rows, "discipline")


def subfield_totals(rows: Sequence[dict]) -> list[dict]:
    """按知识领域汇总（同一学科内使用）。

    ⚠️ 上一版这里写的是 `[{"discipline": r["subfield"], **r}]`：
    字典展开时 `**r` 在后，`r["discipline"]` 会把刚设置的值**覆盖掉**，
    于是领域分组全部塌回学科名 —— 力学内部的细分饼图只有一个扇区。
    现在两个函数共用 `_aggregate`，不会再出现这类"换个字段名就写错"的问题。
    """
    return _aggregate(rows, "subfield")


def concept_rows(rows: Sequence[dict], *, discipline: str | None = None,
                 subfield: str | None = None) -> list[dict]:
    """按知识点明细，可按学科 / 知识领域过滤，按提及次数降序。"""
    out = [
        r for r in rows
        if (discipline is None or r["discipline"] == discipline)
        and (subfield is None or r["subfield"] == subfield)
    ]
    return sorted(out, key=lambda x: (-x["mentions"], -x["students"], x["concept"]))


# ---------------------------------------------------------------- 诊断

#: 诊断条目的严重度。用于前端选图标/配色，不用于排序。
LEVEL_ALERT = "alert"      # 需要立即关注
LEVEL_WATCH = "watch"      # 值得留意
LEVEL_INFO = "info"        # 陈述事实
LEVEL_GOOD = "good"        # 正面信号

#: `weak_rows` 的**时间口径**。审阅第二批 #6 要求
#: 「个人画像是跨时累计时，也不得与当前日期范围作答次数拼成同一段诊断」。
#:
#:   · `WEAK_SCOPE_RANGE`    —— 来自 `attempts.weak_concepts`，**按筛选范围过滤**
#:   · `WEAK_SCOPE_LIFETIME` —— 来自 `student_profiles.weak_concepts`，**跨时累计**
#:
#: 调用方必须如实告知用哪一种；混用会让"范围作答数 / 累计弱项数"拼进同一句，
#: 读者会据此做出错误推断（"这几次作答产生了这些弱项"）。
WEAK_SCOPE_RANGE = "range"
WEAK_SCOPE_LIFETIME = "lifetime"


def scope_fingerprint(*, start=None, end=None, student_ids=None,
                      attempt=None, problem_id=None, extra: str = "") -> str:
    """算出一个**范围指纹**（短串），标识"这批数字是哪个范围算出来的"。

    ## 为什么需要它

    审阅要求「**返回同一个 aggregation/snapshot 标识**」。
    原因是这批统计分别来自多次查询：知识点、提示、错题、人数……
    它们必须属于**同一个范围**才有可比性。

    只要筛选条件不同（哪怕只是漏传了一个 `end`），两次查询就会落在
    不同范围上，而界面上的数字**看起来仍然合理** —— 这类口径漂移极其难发现。

    印上同一个指纹后，一眼就能看出"这两个数字是不是同一批"。
    指纹只用于**标识一致性**，不含任何个人信息（口令/姓名都不进）。
    """
    import hashlib

    parts = [
        f"s={start or '-'}",
        f"e={end or '-'}",
        f"a={attempt or '-'}",
        f"p={problem_id or '-'}",
    ]
    ids = sorted(str(x) for x in (student_ids or []))
    #: 学生集合可能很长，用**长度 + 内容摘要**代表：
    #: 长度不同或内容不同 → 指纹不同，足以发现口径漂移。
    blob = ",".join(ids)
    parts.append(f"n={len(ids)}")
    #: ⚠️ 用 hashlib 而不是内置 `hash()` —— 后者受 PYTHONHASHSEED 影响，
    #: **每次进程启动结果都不同**，那就没法用它判断"两次是否同一范围"了。
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]
    parts.append(f"h={digest}")
    if extra:
        parts.append(f"x={extra}")
    return "/".join(parts)


def _stamp_scope(items: list[dict], scope_id: str, scope_label: str) -> list[dict]:
    """给一批诊断卡片盖上**范围指纹**，然后原样返回。

    审阅要求「返回同一个 aggregation/snapshot 标识」：这批结论由多次查询拼成，
    必须能证明它们出自同一个范围。指纹一致 ⇒ 同源；
    若某处显示别的指纹 ⇒ 口径已漂移（这类问题平时极难发现）。

    抽成函数是为了**所有出口都盖章** —— 早返回（无数据分支）最容易漏，
    而恰恰是那些分支最容易被误读成"真的没有数据"。
    """
    if scope_id:
        for item in items:
            item["scope_id"] = scope_id
            item.setdefault("scope_label", scope_label)
    return items


def build_diagnosis(
    *,
    weak_rows: Sequence[dict],
    hint_stats: dict | None = None,
    stuck_nodes: Sequence[dict] | None = None,
    active_students: int = 0,
    covered_students: int | None = None,
    attempt_count: int = 0,
    scope_label: str = "全部数据",
    scope_id: str = "",
    weak_scope: str = WEAK_SCOPE_RANGE,
) -> list[dict]:
    """生成**学情诊断**条目。

    每条是 `{"level", "title", "text"}`。全部结论都从传入的数据算出来，
    **不调模型、不编造**：数字是多少就写多少，样本不足就如实说"样本不足"。

    ## 措辞与口径（审阅后收紧）

    这里的措辞刻意保守，且**每条都注明信号性质**：

      · 弱项画像是**只增不减**的并集 → 一律称「历史提及」，
        不称"当前薄弱"、不称"反复出错"
      · 覆盖人数用调用方传入的 **`covered_students`（精确去重）**，
        不用"各知识点学生数的最大值"（那会少算）
      · 提示的 `effective` 是**系统启发式判定**（依据节点/状态变化），
        不是教学因果效果 → 只报三态计数，**不评价"提示整体有效"**，
        也不给 0.4/0.7 这类没有依据的效果阈值
      · 提示下发频次**不能**当作"学生卡住次数" → 分开列
        「提示下发较集中」与「错误记录较集中」
      · 样本量小时明说，不给出"这个班很差"这类过度推断

    `covered_students`：范围内**至少被标记过一个薄弱知识点**的学生数
    （精确值）。传 None 时退化为不展示覆盖面结论 —— 宁可不说，
    也不能用会少算的数字说。

    `weak_scope`：`weak_rows` 的**时间口径**，默认 `WEAK_SCOPE_RANGE`。
      · `range`    —— 与 `attempt_count` **同口径**（都按筛选范围过滤），
        可以放在同一句里陈述
      · `lifetime` —— **跨时累计**（学生画像并集），**不可**与
        `attempt_count` 混算；此时"总体规模"会拆成两句分别标明口径

    ⚠️ 传错这个参数的后果：把"范围内作答 N 次"与"历史累计 M 个弱项"
    拼进一句，读者会误解为"N 次作答产生了 M 个弱项"。**即使两个数字各自都对，
    这句话也是错的。** 审阅第二批 #6 明确禁止这种拼接。
    """
    out: list[dict] = []
    rows = list(weak_rows or [])

    if not rows:
        out.append({
            "level": LEVEL_INFO,
            "title": "暂无薄弱知识点数据",
            "text": f"分析范围「{scope_label}」内没有诊断出薄弱知识点。"
                    "可能是该范围内没有作答记录，或学生未暴露问题。",
        })
        return _stamp_scope(out, scope_id, scope_label)

    total_mentions = sum(r["mentions"] for r in rows)
    discs = discipline_totals(rows)

    # ---- ① 总体规模 ----
    #
    # ⚠️ **这里必须按 `weak_scope` 分开表述，不能把两种时间口径拼成一句。**
    #
    # 审阅原文（第二批 #6）：
    #   「个人画像是跨时累计时，**也不得与当前日期范围作答次数拼成同一段诊断**」
    #
    # 为什么：`weak_rows` 有两种来源 ——
    #   · 主视图 / 班级诊断 → `attempts.weak_concepts`，**按范围过滤**，与 attempt_count 同口径
    #   · 个体诊断         → `student_profiles.weak_concepts`，**跨时累计**（并集只增不减）
    # 后者若与「范围内作答次数」写在同一句，读者会以为"这 N 次作答产生了 M 个弱项"，
    # 而实际上 M 可能是学生历史上任何时间累积的。**两个不同时间范围的数字放一起，
    # 本身就是错误陈述** —— 即使两个数字各自都对。
    if weak_scope == WEAK_SCOPE_LIFETIME:
        #: **拆成两条独立卡片** —— 不是"一句话里分两句"，而是彻底不放在同一段诊断里。
        #: 审阅要求「不得与当前日期范围作答次数**拼成同一段诊断**」。
        #: 放在同一张卡里（哪怕分句）读者仍会把两个数字联系起来：
        #: "这 1 次作答诊断出了那 3 个弱项"。分开成卡，各自只有一个口径。
        out.append({
            "level": LEVEL_INFO,
            "title": "本次筛选范围的规模",
            "text": f"范围「{scope_label}」内：{active_students} 名学生、"
                    f"{attempt_count} 次作答。\n\n"
                    "（本卡片只反映**当前筛选范围**。）",
        })
        out.append({
            "level": LEVEL_INFO,
            "title": "历史累计画像（不受筛选影响）",
            "text": f"该范围覆盖的学生，其**画像**累计有 {len(rows)} 个薄弱知识点、"
                    f"{total_mentions} 次历史提及。\n\n"
                    "⚠️ 画像按并集累积、**只增不减**，**不随上方时间/班级筛选变化**。"
                    "它是**跨时间累计**结果，与上面那张卡片的作答次数"
                    "**不是同一个时间口径，不能相互推算**"
                    "（不能说「这几次作答产生了这些弱项」）。",
        })
    else:
        scale = f"分析范围「{scope_label}」：{active_students} 名学生、{attempt_count} 次作答，"
        scale += f"历史画像提及 {len(rows)} 个知识点、{total_mentions} 次历史提及。"
        out.append({"level": LEVEL_INFO, "title": "总体规模", "text": scale})

    # ---- ② 集中度：历史提及最集中的模块 ----
    if len(discs) == 1:
        only = discs[0]
        out.append({
            "level": LEVEL_WATCH if only["mentions"] >= 3 else LEVEL_INFO,
            "title": f"历史提及全部集中在「{only['name']}」",
            "text": f"该模块下有 {len(only['concepts'])} 个知识点有历史提及、"
                    f"共 {only['mentions']} 次历史提及。其余学科在本范围内没有记录。",
        })
    else:
        top = discs[0]
        share = top["mentions"] / total_mentions if total_mentions else 0
        out.append({
            "level": LEVEL_WATCH if share >= 0.5 else LEVEL_INFO,
            "title": f"历史提及分布在 {len(discs)} 个模块，以「{top['name']}」为主",
            "text": f"「{top['name']}」占 {share:.0%}（{top['mentions']}/{total_mentions} 次提及），"
                    f"其次是「{discs[1]['name']}」（{discs[1]['mentions']} 次）。",
        })

    # ---- ③ 历史提及最多的单个知识点 ----
    worst = max(rows, key=lambda r: (r["mentions"], r["students"]))
    wshare = worst["mentions"] / total_mentions if total_mentions else 0
    out.append({
        "level": LEVEL_ALERT if wshare >= 0.3 or worst["students"] >= 3 else LEVEL_WATCH,
        "title": f"历史提及最多的知识点：{worst['concept']}",
        "text": f"累计提及 {worst['mentions']} 次、涉及 {worst['students']} 名学生"
                f"（占全部提及的 {wshare:.0%}），归属「{worst['discipline']} · {worst['subfield']}」。"
                "⚠️ 这是**历史画像的累计提及**，不是已确认的当前薄弱点；"
                "讲评时可优先覆盖，但建议结合原始作答核对。",
    })

    # ---- ④ 覆盖面：多少学生被标记过（用精确去重值）----
    if active_students > 0 and covered_students is not None:
        covered = int(covered_students)
        if covered > active_students:
            #: ⚠️ **不用 min() 钳制掩盖矛盾。**
            #: 审阅明确要求「数据矛盾报异常，不用 min 钳制掩盖不同查询范围」。
            #: 分子（覆盖人数）大于分母（范围内学生数）**只可能**是两个数字
            #: 来自不同的查询范围 / 不一致的筛选条件。
            #: 钳制会让比率看起来正常，把"口径已经不一致"这件事藏起来 ——
            #: 那比显示一个荒谬的 150% 更危险（后者至少逼人去看）。
            #: 所以：**报出矛盾并停止输出覆盖率**，而不是修成一个好看的数字。
            out.append({
                "level": LEVEL_ALERT,
                "title": "数据不一致：覆盖面分子大于分母，已停止输出覆盖率",
                "text": f"覆盖人数 {covered} 名学生 **大于** 范围内的学生总数 "
                        f"{active_students} 名。这只能说明分子与分母来自"
                        "**不同的查询范围或不同的筛选条件**。"
                        "为避免掩盖口径问题，此处**不输出覆盖率** ——"
                        "请先核对取数口径（时间范围 / 班级 / 学生集合是否一致）。",
            })
        else:
            ratio = covered / active_students
            lvl = (LEVEL_ALERT if ratio >= 0.5
                   else (LEVEL_WATCH if ratio >= 0.3 else LEVEL_GOOD))
            out.append({
                "level": lvl,
                "title": "薄弱点的学生覆盖面",
                "text": f"{active_students} 名学生中有 {covered} 名（{ratio:.0%}）"
                        "至少被标记过一个薄弱知识点（按学生去重的精确计数）。"
                        + ("覆盖较广，可考虑集体讲评。" if ratio >= 0.5
                           else "相对集中在部分学生。"),
            })

    # ---- ⑤ 提示判定分布（**不评价效果**）----
    if hint_stats:
        total_h = int(hint_stats.get("total") or 0)
        eff = int(hint_stats.get("effective") or 0)
        ineff = int(hint_stats.get("ineffective") or 0)
        undec = int(hint_stats.get("undecided") or 0)
        decided = eff + ineff
        if total_h == 0:
            out.append({
                "level": LEVEL_INFO,
                "title": "提示判定",
                "text": "本范围内没有下发提示记录。",
            })
        else:
            #: ⚠️ 措辞刻意不写"提示整体有效/偏弱"。
            #: `effective` 是系统按节点/状态变化做的**启发式判定**
            #: （含"离开原节点"这类情形），不是教学因果效果；
            #: 0.4/0.7 那种阈值也没有效果研究依据。所以只陈述分布。
            out.append({
                "level": LEVEL_INFO,
                "title": f"提示判定分布（可判定 {decided} / 共 {total_h}）",
                "text": f"有效 {eff} 条、无效 {ineff} 条、"
                        f"**未判定 {undec} 条**（未判定不等于无效，单列不计入分母）。"
                        + (f"可判定部分中判为有效 {eff / decided:.0%}。"
                           if decided else "尚无可判定样本，不能评价。")
                        + "此处的「有效」指**系统按节点/状态变化作出的判定**，"
                          "不是教学效果的验证，请结合具体提示原文判断。",
            })

    # ---- ⑥ 提示下发 & 错误记录：**两种信号分开列** ----
    if stuck_nodes:
        nodes = list(stuck_nodes)[:3]
        if nodes:
            head = nodes[0]
            hint_parts = "、".join(
                f"{n.get('node')}（{int(n.get('from_hint') or 0)} 次）"
                for n in nodes if int(n.get("from_hint") or 0) > 0)
            err_parts = "、".join(
                f"{n.get('node')}（{int(n.get('from_mistake') or 0)} 次）"
                for n in nodes if int(n.get("from_mistake") or 0) > 0)
            #: ⚠️ 原文案是"在某节点下发提示 = 学生在那里卡住"。
            #: 审阅指出这**越界**：主动求助、连续求助、缺推导
            #: 都不能证明认知卡顿。现在只陈述"记录集中在哪"。
            body = f"提示下发较集中的节点：{hint_parts or '—'}。" if hint_parts else ""
            if err_parts:
                body += f"错误记录较集中的节点：{err_parts}。"
            body += ("提示下发频次反映**求助/被提示的分布**，"
                     "不表示学生在该处认知卡住；错误记录是过程证据，两者口径不同，已分开列出。")
            out.append({
                "level": LEVEL_WATCH
                if int(head.get("from_mistake") or 0) >= 3 else LEVEL_INFO,
                "title": f"记录较集中的节点：{head.get('node')}",
                "text": body,
            })

    # ---- ⑦ 样本量提示（避免过度推断）----
    if attempt_count and attempt_count < 5:
        out.append({
            "level": LEVEL_INFO,
            "title": "样本量偏小",
            "text": f"本范围仅有 {attempt_count} 次作答，上述比例的统计波动会很大，"
                    "建议仅作为方向性参考，不要据此下强结论。",
        })

    return _stamp_scope(out, scope_id, scope_label)


# ---------------------------------------------------------------- 错题诊断


def wrong_problem_concentration(wrong_rows: Sequence[dict]) -> dict:
    """把错题明细按**题号**汇总。

    返回 `{"by_problem": [{problem_id, students, records, mistake_count, types}],
            "total_records": int, "total_mistakes": int}`。

    `by_problem` 按"涉及学生数"降序 —— 教学上"多少人都错在这道题"比
    "总错了几次"更有意义（一个人反复错会把次数刷高）。
    """
    agg: dict[str, dict] = {}
    for r in wrong_rows or []:
        pid = r.get("problem_id") or "（未知）"
        b = agg.setdefault(pid, {
            "problem_id": pid, "students": set(), "records": 0,
            "mistake_count": 0, "types": set(),
        })
        b["students"].add(r.get("student_id"))
        b["records"] += 1
        b["mistake_count"] += int(r.get("mistake_count") or 0)
        for t in str(r.get("issue_types") or "").split("、"):
            if t.strip():
                b["types"].add(t.strip())
    rows = [
        {
            "problem_id": b["problem_id"],
            "student_count": len(b["students"]),
            "records": b["records"],
            "mistake_count": b["mistake_count"],
            "types": "、".join(sorted(b["types"])),
        }
        for b in agg.values()
    ]
    rows.sort(key=lambda x: (-x["student_count"], -x["mistake_count"], x["problem_id"]))
    return {
        "by_problem": rows,
        "total_records": sum(r["records"] for r in rows),
        "total_mistakes": sum(r["mistake_count"] for r in rows),
    }


def build_wrong_diagnosis(
    wrong_rows: Sequence[dict],
    *,
    active_students: int = 0,
    scope_label: str = "全部数据",
    scope_id: str = "",
) -> list[dict]:
    """**错题诊断** —— 围绕"哪道题、多少人、错在什么性质"给结论。

    与 `build_diagnosis` 分开：那个讲"知识点薄弱"，这个讲"题目层面的问题"。
    两者是不同视角 —— 一道题可能涉及多个知识点，一个知识点也可能横跨多题。
    """
    out: list[dict] = []
    if not wrong_rows:
        out.append({
            "level": LEVEL_INFO,
            "title": "未取得模型错误记录",
            "text": f"分析范围「{scope_label}」内未取得模型错误记录。"
                    "这不证明学生没有做错；请同时核查作答覆盖、识别与复核状态。",
        })
        return _stamp_scope(out, scope_id, scope_label)

    conc = wrong_problem_concentration(wrong_rows)
    top = conc["by_problem"][0]
    n_stu = len({r.get("student_id") for r in wrong_rows})

    out.append({
        "level": LEVEL_INFO,
        "title": "错题总体情况",
        "text": f"范围内有 {n_stu} 名学生曾被系统记录错误，共 {conc['total_records']} 条模型历史记录、"
                f"{conc['total_mistakes']} 个错误点，涉及 {len(conc['by_problem'])} 道题。",
    })

    # ---- 最集中的题目 ----
    share = top["student_count"] / n_stu if n_stu else 0
    out.append({
        "level": LEVEL_ALERT if share >= 0.5 or top["student_count"] >= 3 else LEVEL_WATCH,
        "title": f"错得最集中的题目：{top['problem_id']}",
        "text": f"该题有 {top['student_count']} 名学生出错（占出错学生的 {share:.0%}）、"
                f"累计 {top['mistake_count']} 个错误点"
                + (f"，错误性质：{top['types']}。" if top["types"] else "。")
                + ("属于共性错题，建议课堂集中讲评。" if share >= 0.5
                   else "目前偏个别现象，可个别辅导。"),
    })

    # ---- 题目分布：共性问题 vs 分散 ----
    if len(conc["by_problem"]) == 1:
        out.append({
            "level": LEVEL_WATCH,
            "title": "错题高度集中在一道题上",
            "text": f"所有错题都出自 {top['problem_id']}。"
                    "说明问题很聚焦，讲评这一道题就能覆盖大部分薄弱环节。",
        })
    else:
        multi = [r for r in conc["by_problem"] if r["student_count"] >= 2]
        if multi:
            names = "、".join(r["problem_id"] for r in multi[:4])
            out.append({
                "level": LEVEL_WATCH,
                "title": f"有 {len(multi)} 道题是多人出错",
                "text": f"涉及 {names}，每题有多名学生的模型错误记录。"
                        "建议核查原作答、计算步骤和识别质量，不能仅凭分布判断知识掌握原因。",
            })
        else:
            out.append({
                "level": LEVEL_GOOD,
                "title": "错题较分散",
                "text": f"{len(conc['by_problem'])} 道题都有错，但每道题都只有 1 名学生出错 —— "
                        "更像个体的个性化问题，而非班级共性。",
            })

    # ---- 错误性质分布 ----
    type_count: dict[str, int] = {}
    for r in wrong_rows:
        for t in str(r.get("issue_types") or "").split("、"):
            if t.strip():
                type_count[t.strip()] = type_count.get(t.strip(), 0) + 1
    if type_count:
        ordered = sorted(type_count.items(), key=lambda x: (-x[1], x[0]))
        desc = "、".join(f"{k}（{v} 次）" for k, v in ordered)
        head = ordered[0]
        total_t = sum(type_count.values())
        out.append({
            "level": LEVEL_WATCH if head[1] / total_t >= 0.6 else LEVEL_INFO,
            "title": f"错误性质以「{head[0]}」为主",
            "text": f"全部错误点的性质分布：{desc}。"
                    + ("以单一性质为主，说明这是**同类问题**，可用一套方法纠正。"
                       if head[1] / total_t >= 0.6
                       else "性质较为分散，建议按类型分别设计练习。"),
        })

    # ---- 出错节点 ----
    node_count: dict[str, int] = {}
    for r in wrong_rows:
        for n in str(r.get("nodes") or "").split("、"):
            if n.strip():
                node_count[n.strip()] = node_count.get(n.strip(), 0) + 1
    if node_count:
        ordered = sorted(node_count.items(), key=lambda x: (-x[1], x[0]))
        desc = "、".join(f"{k}（{v} 次）" for k, v in ordered[:4])
        out.append({
            "level": LEVEL_INFO,
            "title": f"出错环节集中在 {ordered[0][0]}",
            "text": f"出错节点分布：{desc}。"
                    "节点是该题解题路径上的环节坐标，表示系统曾定位到的流程节点，不证明学生实际推导顺序或认知卡顿。",
        })

    if len(conc["by_problem"]) and active_students and n_stu / active_students >= 0.5:
        out.append({
            "level": LEVEL_ALERT,
            "title": "错题覆盖面广",
            "text": f"{active_students} 名学生中有 {n_stu} 名（{n_stu/active_students:.0%}）"
                    "有模型历史错误记录；建议核查后安排针对性讲评，不据此直接判断群体能力。",
        })

    return _stamp_scope(out, scope_id, scope_label)


# ---------------------------------------------------------------- 班级

#: 班级映射文件。**放在前端**（而非数据库）的原因见 `load_classes` 的文档。
CLASSES_FILE = Path(__file__).resolve().parent / "classes.json"

#: 不在任何班级里的学生归入这个虚拟班级。
UNCLASSED = "未分班"

#: "全部班级"这个虚拟选项。
ALL_CLASSES = "全部班级"


def load_classes(path: Path | str | None = None) -> tuple[dict[str, list[str]], str]:
    """读取班级映射。返回 `(classes, error)`，`error` 为空表示读取正常。

    `path` 为 None 时用 `CLASSES_FILE`（默认与 `teacher_logic.py` 同目录）。
    配置层（`teacher_config`）可以覆盖它，从而实现"班级文件放在任意位置"。

    ## 为什么班级放在前端配置而不是数据库

    `students` 表**只有 `id` / `display_name` / `created_at`**，没有班级字段。
    加字段属于改后端 schema，因此班级改由前端配置定义。

    ⚠️ 如实说明其代价：**这是前端映射，不是权威归属**。
    接入真实教务数据后，班级应当来自学生表或教务系统，
    届时把 `students_of_class()` 换成查库即可，界面无需改动。

    ## 容错

    文件不存在 / JSON 坏 / 结构不对 → 返回空映射 + 原因，
    **不抛异常**：一个配置文件坏掉不该让整页崩掉。
    空映射时界面会把所有学生归入「未分班」，仍然可用。

    ## 重复成员会被拒绝（审阅要求）

    同一学生出现在多个班级时**不接受**，而是返回空映射 + 明确错误。
    理由：`class_of` 是"第一个命中就返回"，那么归属就由**字典顺序**决定 ——
    JSON 里把两个班换个前后位置，学生的班级就变了。
    这种"看起来正常、实则随机"的归属会直接污染班级对比，宁可不显示。
    """
    f = Path(path) if path else CLASSES_FILE
    try:
        if not f.is_file():
            return {}, f"未找到班级配置：{f}"
        data = json.loads(f.read_text(encoding="utf-8"))
        raw = data.get("classes")
        if not isinstance(raw, dict):
            return {}, "班级配置缺少 classes 字段（应为对象）"
        out: dict[str, list[str]] = {}
        owner: dict[str, str] = {}
        dup: list[str] = []
        for name, members in raw.items():
            if not isinstance(members, list):
                continue
            clean = [str(m) for m in members if str(m).strip()]
            out[str(name)] = clean
            for m in clean:
                if m in owner and owner[m] != str(name):
                    dup.append(f"{m}（{owner[m]} / {name}）")
                else:
                    owner[m] = str(name)
        if not out:
            return {}, "班级配置里没有任何班级"
        if dup:
            #: 返回空映射而不是"取第一个" —— 见上面 docstring。
            return {}, ("同一学生出现在多个班级，归属无法确定，请先修正：" + "、".join(dup[:5])
                        + ("…" if len(dup) > 5 else ""))
        return out, ""
    except Exception as exc:  # noqa: BLE001 配置坏了不该让页面崩
        return {}, f"{type(exc).__name__}: {exc}"


def class_of(student_id: str, classes: dict[str, list[str]]) -> str:
    """反查学生所属班级。不在任何班级里 → `UNCLASSED`。"""
    for name, members in classes.items():
        if student_id in members:
            return name
    return UNCLASSED


def students_of_class(classes: dict[str, list[str]], name: str,
                      roster_ids: Sequence[str] | None = None) -> list[str]:
    """某班级的成员。

    `ALL_CLASSES` 返回全体；`UNCLASSED` 返回"不在任何班级里的学生"
    —— 后者需要 `roster_ids` 才能算出来（配置里没有他们的名字）。
    传了 `roster_ids` 时结果会与之取交集，避免把已不存在的学生算进来。
    """
    known = set(roster_ids) if roster_ids is not None else None
    if name == ALL_CLASSES:
        return list(roster_ids) if known is not None else [
            m for ms in classes.values() for m in ms
        ]
    if name == UNCLASSED:
        if known is None:
            return []
        assigned = {m for ms in classes.values() for m in ms}
        return [s for s in roster_ids if s not in assigned]
    members = list(classes.get(name, []))
    if known is not None:
        members = [m for m in members if m in known]
    return members


def class_summaries(
    roster: Sequence[dict],
    classes: dict[str, list[str]],
    *,
    active_ids: Sequence[str] | None = None,
) -> list[dict]:
    """各班级的规模与**记录分布**（供班级概览与班级选择器）。

    `roster` 每项需含 `student_id` / `attempt_count` / `problem_attempt_count` /
    `weak_count` / `hint_count`。

    ## ⚠️ 这里统计的是"记录分布"，不是"班级能力"

    审阅指出原实现的问题：把"弱项次数 / 班级人数"当人均弱项来排名，
    并称"相对更弱"。但人均化**没有控制**：题目、题目难度、作答次数、
    活跃覆盖、曝光时间。**积极作答的班反而会累积更多历史信号**，
    于是"更努力"可能被读成"更差"。

    所以本函数只给**事实性计数**，不做能力评价；
    排序仅按"有记录的学生数"，**并列不强行分先后**。

    `active_ids`：该范围内**有过作答**的学生。给了就额外返回
    `active_count` 与 `coverage`（活跃/在册），让"覆盖面不足"可见 ——
    否则一个只有 1 人作答的班会被当成"这个班很干净"。
    """
    active = set(active_ids) if active_ids is not None else None
    buckets: dict[str, list[dict]] = {}
    for s in roster or []:
        buckets.setdefault(class_of(s.get("student_id", ""), classes), []).append(s)
    if not classes and not buckets:
        return []

    out = []
    for name, members in buckets.items():
        n = len(members)
        att = sum(int(m.get("attempt_count") or 0) for m in members)
        bad = sum(int(m.get("problem_attempt_count") or 0) for m in members)
        weak = sum(int(m.get("weak_count") or 0) for m in members)
        hinted = sum(int(m.get("hint_count") or 0) for m in members)
        act = None
        if active is not None:
            act = sum(1 for m in members if m.get("student_id") in active)
        out.append({
            "name": name,
            "student_count": n,
            "active_count": act,
            "attempt_count": att,
            "problem_attempt_count": bad,
            "weak_count": weak,
            "hint_count": hinted,
            #: 保留人均值只为**看记录密度**，字段名已标明是记录口径而非能力。
            #: 界面上不得据此排名并称"更弱"。
            "weak_per_student_record": round(weak / n, 2) if n else 0.0,
            #: 出错率：有过程错误记录的作答占比（详见 _has_error_records 的口径）
            "problem_rate": round(bad / att, 3) if att else None,
            #: 活跃覆盖率：有作答的学生 / 在册学生。低覆盖时任何比例都不可比。
            "coverage": round(act / n, 3) if (act is not None and n) else None,
        })
    #: 按"在册人数"降序，**不按问题指标排序** —— 排序本身就会被读成评价。
    #: 并列时按班级名，只是为了让输出稳定（不是分优劣）。
    out.sort(key=lambda x: (-x["student_count"], x["name"]))
    return out


def compare_classes(
    summaries: Sequence[dict],
    *,
    common_problems: int | None = None,
    common_time_window: str = "",
    scope_label: str = "全部数据",
) -> list[dict]:
    """班级横向**记录分布**对照。少于 2 个班时不做对照（没有参照物）。

    ## 为什么不评价"哪个班更弱"

    审阅指出：人均弱项排名并称"相对更弱"**不具备可比性** ——
    没有控制题目、题目难度、作答次数、活跃覆盖和曝光时间；
    积极作答的班可能只是因为做得多而累积了更多历史信号。
    相同值也会被挑出第一名/最后一名。这在教学中会误导资源投放。

    因此这里**只陈述分布事实**，并明确写出不可比的原因。

    ## 条件性对照：需要「共同题目 + 共同时间 + 覆盖足够」三个前提

    审阅原文（第二批 #7）：
      「只有**共同题目、共同时间及足够覆盖**时展示条件性对照」

    三个前提**缺一不可**，所以本函数要求调用方**显式提供**这两项证据：

      · `common_problems`     —— 各班共同作答的题目数（>0 才谈得上同题比较）
      · `common_time_window`  —— 共同时间窗口的人可读描述（如 `2026-10-01~10-14`）
                                 **非空字符串才表示"时间窗口一致"**

    ⚠️ 为什么时间窗口也要显式传：各班若在不同时间段作答，
    题目难度、教学进度、学生状态都不同 —— 那时比"出错率"等于比两件不同的事。
    本函数**无法自己判断**两个班的时间窗口是否一致（那需要跨班时间交集分析），
    所以**要求调用方给出证据**，而不是默认"大概差不多"。
    这也是为什么两个参数都不给时**不显示**条件性对照。

    并列时不点"最弱/最优"。
    """
    rows = [s for s in summaries if s.get("student_count")]
    if len(rows) < 2:
        return []
    out: list[dict] = []

    def _desc(r) -> str:
        bits = [f"{r['name']}：{r['attempt_count']} 次作答"]
        if r.get("active_count") is not None:
            bits.append(f"活跃 {r['active_count']}/{r['student_count']}")
        bits.append(f"历史提及 {r['weak_count']} 次")
        if r.get("problem_rate") is not None:
            bits.append(f"出错率 {r['problem_rate']:.0%}")
        return "、".join(bits)

    out.append({
        "level": LEVEL_INFO,
        "title": f"各班级记录分布（{len(rows)} 个班）",
        "text": "；".join(_desc(r) for r in rows)
                + "。⚠️ 这是**记录量**的分布，**不是能力或水平对比** ——"
                  "各班作答次数、活跃覆盖、题目难度都不同，"
                  "一个班记录多很可能只是因为它做得多。",
    })

    #: 覆盖不足时明确提示不可比（否则读者会自行脑补"哪个班更差"）。
    low = [r for r in rows if r.get("coverage") is not None and r["coverage"] < 0.5]
    if low:
        out.append({
            "level": LEVEL_WATCH,
            "title": "部分班级活跃覆盖不足，暂不宜横向比较",
            "text": "、".join(f"{r['name']}（{r['coverage']:.0%}）" for r in low)
                    + " 的学生在本范围内没有作答记录。"
                      "覆盖不足时，任何人均或比率都不可比。"
                      "建议先确认这些班是否未参与，再决定比较口径。",
        })

    #: 仅在「共同题目 + 共同时间 + 覆盖足够」**三者齐备**时才给条件性对照。
    #: 缺任何一项都不显示 —— 审阅要求三个前提缺一不可。
    comparable = [r for r in rows
                  if (r.get("coverage") is None or r["coverage"] >= 0.5)]
    #: 缺哪个前提就如实说清，而不是默默不显示（否则读者会以为"没有差异"）。
    missing: list[str] = []
    if not common_problems:
        missing.append("**共同题目数**（未提供，无法确认各班做过同一批题）")
    if not common_time_window:
        missing.append("**共同时间窗口**（未提供，各班可能在不同时间段作答）")
    if len(comparable) < 2:
        missing.append("**足够的活跃覆盖**（有班的覆盖 < 50%）")

    if not missing:
        rates = [r for r in comparable if r.get("problem_rate") is not None]
        if len(rates) >= 2:
            hi = max(rates, key=lambda x: x["problem_rate"])
            lo = min(rates, key=lambda x: x["problem_rate"])
            gap = hi["problem_rate"] - lo["problem_rate"]
            if gap >= 0.05:
                out.append({
                    "level": LEVEL_WATCH if gap >= 0.2 else LEVEL_INFO,
                    "title": "条件性对照：出错率差异（仅在共同题目、共同时间、覆盖达标时有效）",
                    "text": f"限定在 {common_problems} 道**共同题目**、"
                            f"时间窗口 {common_time_window}、"
                            f"且覆盖达标的班级内：{hi['name']} 出错率 "
                            f"{hi['problem_rate']:.0%}，{lo['name']} 为 "
                            f"{lo['problem_rate']:.0%}，相差 {gap:.0%}。"
                            "⚠️ 该对照**仅**在已控制题目、时间与覆盖的前提下成立；"
                            "不同题目数量或不同时间段的班级之间不可直接比较。",
                })
    else:
        #: **不显示 ≠ 没有差异。** 必须说清是"前提不足所以不做对照"，
        #: 否则读者会把"没输出"理解成"两班水平相当"。
        out.append({
            "level": LEVEL_INFO,
            "title": "未做条件性对照（前提不足）",
            "text": "两班之间若要比「出错率」，需要**同时**满足三个前提。"
                    "当前缺少：" + "；".join(missing) + "。\n\n"
                    "⚠️ 因此**不给出**任何班际出错率对比 ——"
                    "这不代表两个班水平相当，只说明**前提不成立、尚不可比**。"
                    "如需对照，请先限定到共同题目与共同时间窗口，"
                    "并确保各班活跃覆盖达标。",
        })
    return out
