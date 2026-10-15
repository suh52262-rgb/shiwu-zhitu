"""视物知途 · 教师端（班级学情分析）。

## 定位：独立教师处理台；历史读取与显式核对写入分开

| | 学生端 | 教师端（本包） |
|---|---|---|
| 回答的问题 | 这个学生**这一步**对不对 | 这批学生**整体**弱在哪 |
| 数据单位 | 单次作答、逐轮 trace | 跨学生 / 跨题 / 跨班的**聚合** |
| 是否调 LLM | 是（诊断、提示） | **否**，只读已落库的结果 |
| 是否产生数据 | 是 | 只通过后端显式保存本地核对决定 |

本页已接入主项目；启动入口和数据读取边界见项目README。
界面风格与学生端保持一致（同一套原生 Streamlit 组件：
`st.sidebar` 分组 / `st.header` 分段 / `st.caption` 说明 / `st.divider` 分区 /
`st.metric` 指标 / `st.dataframe` 表格 / `st.success|warning|error|info` 状态），
共享样式由 workspace_style.py / workspace.css 维护，不改变原生控件语义。

## 为什么是独立入口，而不是学生端的多页面

学生端有 **10+ 个测试文件**在保护它的行为。Streamlit 的 `frontend/pages/`
多页面机制会**给学生端侧边栏注入导航**，从而改变学生端 UI ——
有触发既有测试失败的风险。独立入口零侵入，且两端可**并排演示**。

## 与后端的边界

历史统计使用只读数据库通道，当前错误和材料通过项目只读教师接口核对。本页需要的"时间段 / 最近 14 天 / 单次提交 / 学生集合 /
错题题号 / 按题目和类型聚合"——这些后端都没有对应接口；
因此由 `teacher_data.py` 以**只读**方式查询已有表
（`attempts.weak_concepts` / `.mistakes` 自带 `start_time` 与 `problem_id`），
分类、班级映射与诊断文本由 `teacher_logic.py` 在前端计算。

## 四种分析视角（本页视图）

    ① 知识点分布   —— 「弱在哪个知识点」，两级饼图可下钻
    ② 平行检索     —— 换检索键：按知识点 / 按题目 / 按错误类型（三者平级）
    ③ 错题分析     —— 「哪道题、多少人、错在什么性质、记录集中在哪个节点」
    ④ 班级与个体   —— 班级分类与对比、班级诊断、学生下钻

## 启动

    streamlit run frontend/teacher_app.py --server.port 8502
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_config import describe, load_config  # noqa: E402
#: ⚠️ **必须把用到的模式常量都导入。**
#: 原实现只导入 `MODE_DB`，却在下面 `elif _mode == MODE_API` 里引用 `MODE_API` ——
#: 正常连库时不会走到那个分支所以看不出来，
#: 一旦数据库/后端不可用就抛 `NameError: name 'MODE_API' is not defined`，
#: 而 **README 承诺的恰恰是"故障时能启动降级"**。审阅已用离线 AppTest 复现。
#: 教训：故障分支最需要被测，却也最容易漏 —— 所以这里显式导入全部三种模式。
from teacher_data import (  # noqa: E402
    DBERR_CONNECT,
    DBERR_QUERY,
    DBERR_SCHEMA,
    MODE_API,
    MODE_DB,
    MODE_DOWN,
    REVIEW_STORE_DETECTED,
    REVIEW_STORE_INTEGRATED,
    REVIEW_STORE_UNAVAILABLE,
    TeacherData,
)
from teacher_logic import (  # noqa: E402
    ALL_CLASSES,
    LEVEL_ALERT,
    LEVEL_GOOD,
    LEVEL_INFO,
    LEVEL_WATCH,
    UNCLASSED,
    build_diagnosis,
    build_wrong_diagnosis,
    class_summaries,
    compare_classes,
    concept_rows,
    discipline_totals,
    normalize_rows,
    scope_fingerprint,
    students_of_class,
    subfield_totals,
    wrong_problem_concentration,
)
#: `weak_scope` 常量单独导入（用于个体诊断，标明弱项是跨时累计口径）
from teacher_logic import WEAK_SCOPE_LIFETIME  # noqa: E402
#: 统一只读聚合契约的指纹 —— 界面上回显，回答「这个数字按哪一版口径算的」。
#: 审阅第三批要求「**冻结统一只读聚合契约**」，指纹就是它的版本标识。
from teacher_contract import contract_fingerprint  # noqa: E402

#: 连接配置：环境变量 > teacher_config.json > 项目内自动探测。
#: 因此本页面可以独立部署、接入任意的后端与数据库。
CFG = load_config()

st.set_page_config(page_title="视物知途 · 教师端", page_icon="📊", layout="wide")
from workspace_style import apply_workspace_style, page_intro
apply_workspace_style('teacher')

VIEW_TASKS = "待办与证据"
VIEW_DIST = "① 知识点分布"
VIEW_PARALLEL = "② 平行检索"
VIEW_WRONG = "③ 错题分析"
VIEW_CLASS = "④ 班级与个体"

SCOPE_ALL = "全部数据"
SCOPE_14D = "最近 14 天"
SCOPE_RANGE = "自定义时间段"
SCOPE_ONE = "单次提交"

#: 平行检索的三个检索键。三者共用同一套"饼图 + 排行 + 明细"呈现，
#: 只是换了聚合维度 —— 这就是"平行"的含义：同一批数据、平级的三个入口。
DIM_KNOWLEDGE = "知识点"
DIM_PROBLEM = "题目"
DIM_ISSUE = "错误类型"


# ============================ 数据源 ============================

@st.cache_resource(show_spinner=False)
def get_data(cache_key: str) -> TeacherData:
    """数据源单例。

    `cache_resource` 避免每次交互重跑都重新探测数据库；
    参数用 `cache_key`（配置指纹）而不是 URL —— 配置一变缓存就失效，
    否则改了 `teacher_config.json` 还得手动清缓存才生效。
    真正的配置对象在函数内部读取，保证拿到的是最新值。
    """
    return TeacherData(load_config())


def config_key(cfg) -> str:
    """配置指纹：任一连接参数变化都会让 `get_data` 缓存失效。"""
    return f"{cfg.backend_url}|{cfg.database_url}|{cfg.classes_file}"


@st.cache_data(ttl=300, show_spinner=False)
def env_consistency_cached(_config_key: str, backend_url: str, database_url: str) -> dict:
    """环境一致性交叉校验（缓存 5 分钟）。

    ⚠️ 参数用 `_config_key` / 连接串是为了**缓存失效**：
    任一连到别处时都应重新核对，而不是沿用上一次的结论。

    ⚠️ `_config_key` 前缀下划线 → Streamlit 不对它做哈希，
    所以真正的缓存键是 `backend_url` + `database_url` 两个字符串
    （而它们恰好完整刻画了"连到哪"）。函数体内部重新构造 `TeacherData`，
    避免把不可哈希的资源对象塞进 `cache_data`。
    """
    from teacher_data import TeacherData
    from teacher_config import load_config
    fresh = TeacherData(load_config())
    return fresh.environment_consistency()


@st.cache_data(ttl=300, show_spinner=False)
def account_readonly_cached(_config_key: str, database_url: str) -> dict:
    """库侧只读授权证明（缓存 5 分钟）。见 `TeacherData.account_readonly_status`。"""
    from teacher_data import TeacherData
    from teacher_config import load_config
    fresh = TeacherData(load_config())
    return fresh.account_readonly_status()


DATA = get_data(config_key(CFG))
#: 每次刷新开始时**清空累计的查询失败**（审阅 T1）——
#: `_sql` 的失败只增不减（成功不清除），必须在本次渲染开始时归零，
#: 否则上一次刷新的旧失败会一直残留。
DATA.reset_query_errors()
#: 班级归属：**数据库优先、classes.json 兜底**（可插拔）。
#: 审阅：「本地 classes.json 只能作为演示映射」—— 接入真实教务数据时
#: 只需让库里出现班级来源（或用 TEACHER_CLASS_SOURCE 指过去），无需改代码。
CLASSES, CLASS_ERR, CLASS_SOURCE = DATA.class_source(CFG.classes_file)


# ============================ 小工具 ============================

def picked(event, param: str, field: str):
    """从 `st.altair_chart(on_select=...)` 的返回值里取出被点中的字段值。

    结构是 `{"selection": {"<param>": [{"<field>": "...", ...}]}}`，
    未选中时是空列表。**任何异常都返回 None** —— 图表交互失败不该让整页崩，
    用户仍可用旁边的选择器操作。
    """
    try:
        sel = getattr(event, "selection", None)
        if sel is None and isinstance(event, dict):
            sel = event.get("selection")
        pts = (sel or {}).get(param) or []
        if isinstance(pts, list) and pts:
            v = pts[0].get(field)
            return v if v else None
    except Exception:  # noqa: BLE001 交互解析失败即视为未选中
        pass
    return None


def pie_chart(df: pd.DataFrame, *, category: str, value: str, param: str) -> alt.Chart:
    """环形图 + 扇区外标签；`add_params` 挂点选参数供 `on_select` 回传。"""
    selector = alt.selection_point(fields=[category], name=param, empty=False)
    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{value}:Q", stack=True, sort="descending"),
        color=alt.Color(f"{category}:N", title=category, scale=alt.Scale(range=['#2458B5','#087F86','#697BC1','#B17632','#54677E','#A35564'])),
        tooltip=[
            alt.Tooltip(f"{category}:N", title=category),
            alt.Tooltip(f"{value}:Q", title="数值"),
        ],
    )
    arc = base.mark_arc(innerRadius=62, stroke="white", strokeWidth=1)
    labels = base.mark_text(radius=118, fontSize=12).encode(text=alt.Text(f"{category}:N"))
    return (arc + labels).add_params(selector)


def hbar(df: pd.DataFrame, *, category: str, value: str, title: str) -> alt.Chart:
    """横向条形图（排行用，比饼图更易比较量级）。"""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{category}:N", sort="-x", title=None),
            x=alt.X(f"{value}:Q", title=title),
            tooltip=[
                alt.Tooltip(f"{category}:N", title="项目"),
                alt.Tooltip(f"{value}:Q", title=value),
            ],
        )
    )


def pct(value) -> str:
    """`None` 表示**没有可判定的样本**，不是 0%。"""
    return f"{value:.0%}" if isinstance(value, (int, float)) else "—"


#: 诊断条目的符号。不用 emoji —— 与学生端的图标使用口径保持一致（少而克制）。
_LEVEL_MARK = {
    LEVEL_ALERT: "▲ 重点关注",
    LEVEL_WATCH: "△ 值得留意",
    LEVEL_INFO: "· 情况说明",
    LEVEL_GOOD: "✓ 正面信号",
}


def render_diagnosis(items: list[dict], *, heading: str) -> None:
    """渲染诊断卡片。用 `st.container(border=True)`，零自定义样式。

    顺便把**范围指纹**印出来（审阅要求「返回同一个 aggregation/snapshot 标识」）。
    这批结论由多次查询拼成，指纹一致才说明它们同源；
    若你看到不同的小节显示不同指纹，就说明口径漂移了。
    """
    st.subheader(heading)
    if not items:
        st.caption("暂无诊断结论。")
        return
    #: 同一批的指纹必然相同，取第一条即可。
    _sid = items[0].get("scope_id") or ""
    if _sid:
        st.caption(f"范围指纹：`{_sid}` —— 标识当前筛选条件，不证明同一数据库快照"
                   "（范围不同则指纹不同）。")
    for it in items:
        mark = _LEVEL_MARK.get(it.get("level", LEVEL_INFO), "·")
        with st.container(border=True):
            st.markdown(f"**{mark}｜{it.get('title', '')}**")
            st.caption(it.get("text", ""))


def data_hint() -> None:
    """无数据时的引导。

    ⚠️ **不要在这里给"生成演示数据"的命令** ——
    本包是**独立的教师端包**，不含后端、不含模拟器脚本
    （那是部署包 `scripts/run_simulation.py` 的职责）。
    写一条在本包内根本跑不通的命令，只会把人引到死路。

    正确做法：说明数据从哪来，以及去哪个包找模拟器。
    """
    st.info("当前范围内没有可分析的数据。")
    st.markdown(
        "教师端历史统计只读；本地核对决定通过后端显式保存。\n\n"
        "**要看到内容，先让学生端产生数据：**\n"
        "1. 打开学生端，至少提交几步作答；或\n"
        "2. 在**部署包**里跑离线模拟器（走真实作答链路、不需要 API 密钥）：\n"
        "   `scripts/run_simulation.py --all --offline`\n\n"
        "数据落库后，教师端**刷新页面**即可看到。"
    )


def render_query_failures() -> None:
    """汇总本次刷新里**累计**的查询失败（审阅 T1）。

    `_sql` 的失败是**只增不减**的（成功不清除），所以这里能同时看到
    "知识点分布查询失败" + "错题明细查询成功"这样的**混合状态**，
    而不是被最后一条成功查询抹掉失败证据。每个失败的统计区域都会
    **明确警告**，否则教师会把"查询失败显示成的暂无数据"当成"真的没有记录"。
    """
    errs = DATA.query_errors
    if not errs:
        return
    _named = {k: v for k, v in errs.items() if k != "_default"}
    _unnamed = errs.get("_default")
    _lines = [f"· **{k}**：{v}" for k, v in _named.items()]
    if _unnamed:
        _lines.append(f"· **（未标注区域）**：{_unnamed}")
    st.warning(
        "⚠️ **本次加载存在查询失败** —— 下列统计区域的结果可能不全，"
        "失败区域会显示成「暂无数据」，**切勿当成「没有记录」**：\n\n"
        + "\n".join(_lines)
    )


def render_evidence(attempt_id: str) -> None:
    """渲染**单次作答的原始证据**（#8 下钻）。

    这是"统计 → 学生 → attempt → 当前版本证据"这条链的末端。
    审阅指出原实现只有聚合后的 issue/types/node，教师**无法核对**
    "这个数字是怎么来的"。这里把库里已有的原始材料摊开：

      · 状态投影（当前节点 / 已访问节点 / 弱项 / 缺失推理）
      · 过程错误记录（节点 / 性质 / 描述）
      · 每一次提示的**原文** + **三态判定** + **判定依据**
        （`effective_evidence` 是后端写入的可复核文本）
      · 逐轮分析轨迹 + 从轨迹还原的**状态修订序列**
      · **逐轮诊断与提示**（按轮次并列，不是「当前/过期复盘」）

    ⚠️ **只读展示**，不新建复核真值。复核状态由后端的复核服务管理；
    本环境没有该库时如实标"不可得"（见 `review_store_status`）。
    """
    ev = DATA.attempt_evidence(attempt_id)
    if not ev:
        st.caption("未取到该作答的证据（可能已被清理，或查询失败）。")
        return

    st.markdown(
        f"**作答 `{ev['attempt_id']}`** ｜ 学生 `{ev['student_id']}` ｜ "
        f"题号 `{ev['problem_id']}` — {problem_title(ev['problem_id'])}"
    )
    st.caption(
        f"状态：`{ev['status']}` ｜ 当前节点 `{ev['current_node'] or '—'}` ｜ "
        f"开始 {ev['start_display']} ｜ 结束 {ev['end_display']} ｜ "
        f"最终判定：{'正确' if ev['final_correct'] else ('错误' if ev['final_correct'] is False else '未判定')}"
    )

    _e1, _e2 = st.columns(2)
    with _e1:
        st.markdown("**过程错误记录（mistakes）**")
        if ev["mistakes"]:
            st.dataframe(
                [{"节点": (m or {}).get("node", ""),
                  "错误性质": (m or {}).get("issue_type", ""),
                  "严重度": (m or {}).get("severity", ""),
                  "描述": (m or {}).get("issue", "")}
                 for m in ev["mistakes"]],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("本次作答没有过程错误记录。")
        st.markdown("**状态投影里的弱项 / 缺失推理**")
        st.caption(
            f"弱项：{'、'.join(ev['weak_concepts']) or '—'}；"
            f"缺失推理：{'、'.join(ev['missing_reasoning']) or '—'}"
        )
        _st = ev.get("state") or {}
        st.caption(
            f"已访问节点：{'、'.join(_st.get('visited_nodes') or []) or '—'} ｜ "
            f"系统启发式分数（非认知卡顿判定）：{_st.get('stuck_score', '—')}"
        )

    with _e2:
        st.markdown(f"**提示原文与判定依据（{len(ev['hints'])} 条）**")
        if ev["hints"]:
            for h in ev["hints"]:
                _eff = ("未判定" if h["effective"] is None
                        else ("有效" if h["effective"] else "无效"))
                with st.container(border=True):
                    st.markdown(
                        f"`{h['level']}` ｜ 节点 `{h['node'] or '—'}` ｜ "
                        f"性质 `{h['issue_type'] or '—'}` ｜ 判定：**{_eff}** ｜ {h['time']}"
                    )
                    st.caption("提示原文：" + (h["hint"] or "—"))
                    #: 判定依据是**可复核文本**（后端写入），必须展示 ——
                    #: 否则"有效/无效"就成了没有出处的结论。
                    st.caption("判定依据：" + (h["evidence"] or "（后端未记录依据）"))
        else:
            st.caption("本次作答没有提示记录。")

    with st.expander(f"逐轮分析轨迹与状态修订（{len(ev['traces'])} 轮）"):
        if ev["traces"]:
            st.dataframe(
                [{"轮次": t["round"], "模型": t["model"],
                  "耗时(ms)": t["latency_ms"], "时间": t["time"],
                  "修改前节点": (t["before_state"] or {}).get("current_node", "")}
                 for t in ev["traces"]],
                hide_index=True, width="stretch",
            )
            st.caption(
                "「状态修订」= 每轮分析**开始前**的状态快照。"
                "提示效果的三态判定就是比较这些快照得出的 ——"
                "所以这里能核对「为什么判为有效/无效」。"
            )
        else:
            st.caption("没有轨迹记录。")

    _rs = ev.get("review_store_status")
    if _rs == REVIEW_STORE_DETECTED:
        st.warning(
            "**检测到错误证据复核表，但尚未接入复核状态**："
            "「已确认 / 待复核 / 已撤销 / 已解决」这些状态**无法判断**。"
            "上面展示的是**未按复核决定处理的原始信号**，不是已确认的结论。"
        )
    elif _rs != REVIEW_STORE_INTEGRATED:
        st.warning(
            "**错误复核状态不可得**：本环境没有错误证据复核库，"
            "因此「已确认 / 待复核 / 已撤销 / 已解决」这些状态**无法判断**。"
            "上面展示的是**未经验证的历史与过程信号**，不是已确认的结论。"
        )

    # ================================================================
    # 原始材料（审阅第二批 #8 点名要求）
    #   「教师需要看到**原题**、**原图/OCR**、作答修订、实际提示、
    #     模型判断出处、**当前与过期复盘**，以及错误复核状态」
    # ================================================================
    st.divider()
    st.markdown("**原始材料（题目与逐轮输入）**")
    _src_rev = ev.get("source_revisions") or {}
    st.caption(
        f"证据修订身份：`{ev.get('evidence_revision')}`"
        f"（来源：{ev.get('evidence_revision_kind')}）——"
        "任一来源的修订号变化即变；这里**保留 source_id→revision 映射**，"
        "不用最大修订号代替身份。"
    )

    _m1, _m2 = st.columns([3, 2])
    with _m1:
        # ---- 原题 ----
        _meta = ev.get("problem_meta") or {}
        _title = _meta.get("title") or ""
        st.markdown(f"**原题**{'：' + _title if _title else ''}")
        _content = ev.get("problem_content") or _meta.get("content") or ""
        if _content:
            with st.container(border=True):
                st.markdown(_content)
            st.caption(
                "正文取自**作答发起时落库的原文快照**（`state.problem_content`）——"
                "比现查题库更可靠：题目后来被改过也能看到**当时那道题**。"
            )
        else:
            st.caption("未记录原题正文。")
        if _meta.get("knowledge_points"):
            st.caption("知识点：" + "、".join(_meta["knowledge_points"]))
        if _meta.get("difficulty"):
            st.caption(f"难度：{_meta['difficulty']}")

    with _m2:
        # ---- 原题配图 ----
        st.markdown("**原题配图**")
        _figs = ev.get("problem_figures") or []
        if _figs:
            for _i, _u in enumerate(_figs):
                try:
                    st.image(_u, caption=f"原题配图 {_i + 1}")
                except Exception:  # noqa: BLE001 图取不到不该让下钻整块崩
                    st.caption(f"配图 {_i + 1} 加载失败：`{_u}`")
        else:
            #: 审阅 T5：配图来自后端 `/problems/{id}` 的 figure_urls；
            #: 未取到时就如实说"未取到/本题无图"，不再引用本包不存在的
            #: "题库目录"，也不宣称图片闭环已完成。
            st.caption("本题无配图，或后端未返回可访问的配图 URL。")

    # ---- 逐轮原始输入 ----
    st.markdown("**逐轮原始输入（学生每次提交的原文）**")
    _inp = ev.get("student_inputs") or []
    if _inp:
        st.dataframe(
            [{"轮次": f"R{x['round']}", "提交原文": x["text"]} for x in _inp],
            hide_index=True, width="stretch",
        )
        st.caption(
            "这是**模型当轮看到的输入**（`payload.student_input`）——"
            "属于**输入侧证据**，与上面「过程错误记录」的结论侧证据互补。"
            "若经由照片识别，此处是**识别后（可能已人工更正）**的文本。"
        )
    else:
        st.caption("没有逐轮输入记录。")

    # ---- 媒体来源：原始 OCR 文本 / 更正文本 / 更正历史 ----
    _media = ev.get("media_sources") or []
    if _media:
        st.markdown("**照片 / 视频来源的 OCR 原文与更正（含修订历史）**")
        st.caption(
            "⚠️ 审阅 T5：这里展示的是**识别/更正后的文字**。"
            "下面的保留画面由教师只读接口提供；旧照片未保存时会显示缺失，"
            "不能根据OCR文字自行补造原图。"
        )
        for _m in _media:
            with st.container(border=True):
                st.markdown(
                    f"来源 `{_m['source_id']}` ｜ 当前修订号 "
                    f"`{_m['revision']}` ｜ "
                    + ("更正窗口已关闭" if _m["correction_closed"] else "更正窗口仍开放")
                )
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.caption("**OCR 原始文本**")
                    st.markdown(_m["original_text"] or "（空）")
                with _c2:
                    st.caption("**更正后文本（当前）**")
                    st.markdown(_m["corrected_text"] or "（空）")
                _hist = _m.get("revision_history") or []
                if len(_hist) > 1:
                    with st.expander(f"更正历史（{len(_hist)} 个版本）"):
                        st.dataframe(
                            [{"修订序号": k + 1,
                              "修订号": h.get("revision"),
                              "文本": h.get("text", "")}
                             for k, h in enumerate(_hist)],
                            hide_index=True, width="stretch",
                        )
                        st.caption(
                            "⚠️ 只有**最后一个版本**是当前有效证据，"
                            "之前各版本已被取代 —— 保留它们是供核对"
                            "「模型识别错在哪、人工改成了什么」。"
                        )
    else:
        st.caption("本次未读取到媒体来源记录；可能是文字提交，"
                   "也可能是旧记录缺少来源元数据。")

    # ---- 逐轮诊断与提示（审阅 T3）----
    #
    # 审阅 T3：旧实现把逐轮 diagnosis 直接命名为「当前复盘（最新有效）」
    # 与「过期复盘」。但这些字段是**逐轮诊断**，不是报告版本；最后一轮
    # **不**自动成为"当前有效复盘"，前面几轮也**不**因下一轮到来自动过期。
    # 所以这里改为**逐轮并列展示**，并明确"真正的复盘需从复核存储读取"。
    st.markdown("**逐轮诊断与提示**")
    _rounds = ev.get("round_diagnoses") or []
    if _rounds:
        st.dataframe(
            [{"轮次": f"R{x['round']}", "错误性质": x["issue_type"] or "—",
              "严重度": x["severity"] or "—", "诊断": x["issue"] or "—",
              "提示级别": x["hint_level"] or "—", "时间": x["time"]}
             for x in _rounds],
            hide_index=True, width="stretch",
        )
        for x in _rounds:
            if x.get("tutor_output"):
                with st.expander(f"第 {x['round']} 轮模型输出"):
                    st.caption(x["tutor_output"][:300])
        st.caption(
            "⚠️ 这是**每一轮的诊断与提示**（按轮次并列），不是「复盘版本」——"
            "最后一轮**不**自动成为「当前有效」，前面几轮也**不**因下一轮到来自动「过期」。"
            "实际报告版本在下方单独显示；错误的撤销/解决与报告版本是两种记录。"
        )
    else:
        st.caption("没有逐轮诊断记录。")

    _read = ev.get('teacher_evidence') or {}
    st.markdown('**保留的原图与视频帧**')
    _images = _read.get('images') or []
    if _images:
        from urllib.parse import urlencode, quote
        _selected = st.selectbox('选择证据画面', range(len(_images)),
            format_func=lambda i: f"{i + 1} · {_images[i]['media_type']} · {_images[i].get('timestamp_sec', '照片')}秒",
            key=f'teacher_image_{attempt_id}')
        _image = _images[_selected]
        _query = urlencode({'student_id': ev['student_id'], 'problem_id': ev['problem_id']})
        _url = (CFG.backend_url + '/teacher/attempts/' + quote(attempt_id, safe='')
                + '/images/' + _image['evidence_id'] + '?' + _query)
        st.image(_url, caption='原始保留画面；OCR更正不会改写图片', width='stretch')
    else:
        st.caption('原图未保留或暂不可读取；不能由图片缺失判为纯文字作答。')
    st.markdown('**已生成的复盘版本（只读，不触发生成）**')
    if _read.get('status') == 'unavailable':
        st.warning('教师证据接口不可得：' + _read.get('error', '读取失败'))
    for _report in _read.get('reports') or []:
        _label = '对应当前材料' if _report['material_status'] == 'current' else '材料已更正或续做，历史版本'
        with st.expander(f"{_label} · 输入修订{_report['input_revision']} · {_report['report_id'][:12]}"):
            st.caption('当前仅指材料身份匹配；结构与引用检查不代表物理语义正确，也未重新调用模型。')
            st.caption(f"来源修订：{_report['source_revisions']}；输入指纹：{_report['input_fingerprint']}")
            st.write(_report['review'].get('summary', ''))
            st.json(_report['review'], expanded=False)
    if not _read.get('reports'):
        st.caption('没有可读取的已生成复盘；不会把逐轮诊断当作报告。')
    if _read.get('issues'):
        st.warning('部分历史报告材料或引用校验未通过，已保留原件但不作为有效报告展示。')


def render_dimension(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    param: str,
    detail_rows: list[dict],
    detail_columns: dict[str, str],
    filter_field: str,
    note: str = "",
) -> None:
    """「饼图 + 排行 + 明细」的通用呈现。

    三个平行检索维度共用这一套版式 —— 否则要写三遍几乎相同的代码，
    改一处就得改三处（这类重复最容易出现"某个维度忘了同步"）。

    `filter_field` 是 `detail_rows` 里与 `category` 对应的字段名，
    用于点选后把明细收窄到该项。
    """
    clicked = None
    c1, c2 = st.columns([3, 2])
    with c1:
        if len(df) > 1:
            ev = st.altair_chart(
                pie_chart(df, category=category, value=value, param=param),
                on_select="rerun", selection_mode=param, key=f"pie_{param}",
                width="stretch",
            )
            clicked = picked(ev, param, category)
            st.caption("点击扇区可筛选下方明细。")
        else:
            st.caption("只有一项，无需饼图；直接看右侧明细。")
    with c2:
        st.dataframe(df.sort_values(value, ascending=False),
                     hide_index=True, width="stretch")

    st.subheader("排行")
    st.altair_chart(
        hbar(df, category=category, value=value, title=value), width="stretch",
    )

    options = [str(v) for v in df[category].tolist()]
    if clicked not in options:
        clicked = None
    sel = st.segmented_control(
        f"{category}（点图表或在此选择以筛选明细）",
        options, default=clicked, key=f"seg_{param}",
    ) or clicked

    title = f"明细（{sel}）" if sel else "明细（全部）"
    st.subheader(title)
    rows = detail_rows
    if sel:
        rows = [r for r in detail_rows if str(r.get(filter_field)) == str(sel)]
    if rows:
        st.dataframe(
            [{label: r.get(key, "") for key, label in detail_columns.items()}
             for r in rows],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("该范围内没有明细记录。")
    if note:
        st.caption(note)


# ============================ 侧边栏 ============================

with st.sidebar:
    st.header("系统状态")

    #: 配置错误**必须最先显示** —— 否则用户看到的只是"连不上"，
    #: 不知道原因是配置文件写错了。审阅要求"可显示的配置错误"。
    for _err in CFG.errors:
        st.error(f"配置错误：{_err}")
    if CFG.errors:
        st.caption("请看下方「连接配置」。**配置非法时不会悄悄改用别的数据源** ——"
                   "宁可停用，也不让统计结果来源不明。")

    _mode = DATA.mode
    if _mode == MODE_DB:
        st.success("后端已连接 · 数据库可读（支持全部筛选）")
    elif _mode == MODE_API:
        #: 后端通、数据库不通：界面上必须说清"能看什么、不能看什么"，
        #: 否则用户会以为筛选功能坏了。
        st.warning("后端已连接 · **数据库不可读**")
        #: 分层报因：连不上 / 结构不对 / 查询失败，用户该采取的行动不同。
        _kind = DATA.db_error_kind
        _hint = {
            DBERR_CONNECT: "数据库**连不上**：服务未启动，或地址/端口/账号口令不对。",
            DBERR_SCHEMA: "数据库**结构不兼容**：连上了，但缺少教师端需要的表或列"
                          "（可能接错了库）。",
            DBERR_QUERY: "数据库**查询失败**：权限不足或查询超时。",
        }.get(_kind, "")
        if _hint:
            st.caption(_hint)
        if DATA.db_error:
            st.caption(f"详情：{DATA.db_error}")
        st.caption("已降级为后端聚合接口：**时间 / 学生 / 班级筛选不可用**，仅全局口径。")
    else:
        st.error("后端与数据库均不可用")
        if DATA.db_error:
            st.caption(f"数据库原因：{DATA.db_error}")
        st.caption("请先启动后端，或检查下方「连接配置」。")

    #: 查询失败 ≠ 没有数据。**必须显式提示**，否则教师会以为学生的作答丢了。
    #: 审阅 T1：失败是**累计**的（成功不清除），正文会在全部统计完成后
    #: 再汇总一次；这里先提示"已发生失败"。
    if DATA.query_errors:
        st.warning("查询**失败**（结果可能不全，正文有汇总）")
        st.caption("原因：" + "；".join(
            f"{k or '查询'}：{v}" for k, v in DATA.query_errors.items()))

    #: 连接来源回显 —— 排障时第一时间要看的就是"它到底连到哪了"。
    #: 口令一律隐藏（`database_display` 已做掩码）。
    with st.expander("连接配置", expanded=(_mode != MODE_DB or bool(CFG.errors))):
        st.caption(f"后端地址：`{CFG.backend_url or '（配置非法，已停用）'}`")
        st.caption(f"数据库：`{DATA.database_display}`")
        st.caption(f"班级文件：`{CFG.classes_file}`")
        st.caption(f"来源：{CFG.sources.get('notes', '（无）')}")
        st.caption(
            "修改 `<包根>/teacher_config.json` 或设环境变量 "
            "`TEACHER_BACKEND_URL` / `TEACHER_DATABASE_URL` 后刷新即生效。"
        )

        #: ------------------------------------------------------------------
        #: 「DB / API 必须属于同一环境」（审阅第三批点名要求）
        #:
        #: 两个地址指向**不同环境**时，界面一切正常，但题目名来自一个实例、
        #: 统计数字来自另一个 —— 这是**静默错误**，最难发现。
        #: 所以这里主动交叉核对，并把结论摊在界面上（而不是只写在 README 里）。
        #: ------------------------------------------------------------------
        st.divider()
        st.caption("**环境一致性**（后端与数据库是否指向同一环境）")
        _env = env_consistency_cached(config_key(CFG), CFG.backend_url or "", CFG.database_url or "")
        _status = _env.get("status")
        if _status == "consistent":
            st.caption(f"✅ 一致：{_env.get('detail', '')}")
        elif _status == "mismatch":
            st.error(f"{_env.get('detail', '')}")
        else:
            #: 无法判定不算失败 —— 后端不在或演示版后端不返回该字段时正常出现。
            st.caption(f"⚪ 无法判定：{_env.get('detail', '（无）')}")

        #: 口径契约指纹：所有统计按此版本的口径计算。
        st.caption(f"统计口径契约：`{contract_fingerprint()}`（口径变更时此值必变）")

        #: ------------------------------------------------------------------
        #: 库侧只读授权「证明」—— 审阅：「未证明配置账户具有只读授权」。
        #: 应用侧只发 SELECT + 只读事务已做；这里把**账号在库里的真实权限**
        #: 摆出来（由数据库自己回答），让"只读"从声明变成可验证事实。
        #: ------------------------------------------------------------------
        st.caption("**数据库账号只读授权**")
        _acc = account_readonly_cached(config_key(CFG), CFG.database_url or "")
        _as = _acc.get("status")
        if _as == "readonly":
            st.caption(f"✅ {_acc.get('detail', '')}")
        elif _as == "writable":
            st.warning(_acc.get("detail", ""))
        else:
            st.caption(f"⚪ 无法判定：{_acc.get('detail', '（无）')}")

    st.divider()
    st.subheader("分析范围")
    filters_ok = DATA.filters_supported()
    scope = st.radio(
        "范围", [SCOPE_ALL, SCOPE_14D, SCOPE_RANGE, SCOPE_ONE],
        index=0, key="scope_mode", disabled=not filters_ok,
        label_visibility="collapsed",
    )
    if not filters_ok:
        st.caption("⚠️ 筛选需要数据库可读；当前仅支持「全部数据」。")

    f_start = f_end = f_attempt = None
    scope_label = SCOPE_ALL
    lo, hi = DATA.range_bounds() if filters_ok else ("", "")

    if filters_ok and scope == SCOPE_14D:
        _today = _dt.date.today()
        f_start, f_end = _today - _dt.timedelta(days=13), _today
        scope_label = f"最近 14 天（{f_start} ~ {f_end}）"
    elif filters_ok and scope == SCOPE_RANGE:
        _lo = _dt.date.fromisoformat(lo) if lo else _dt.date.today()
        _hi = _dt.date.fromisoformat(hi) if hi else _dt.date.today()
        _c1, _c2 = st.columns(2)
        with _c1:
            f_start = st.date_input("起", value=_lo, key="range_from")
        with _c2:
            f_end = st.date_input("止", value=_hi, key="range_to")
        if f_start and f_end and f_start > f_end:
            #: 起止选反了不静默纠正 —— 明确报错，否则用户会以为筛选没生效
            st.error("起始日期晚于结束日期，请调整。")
            f_start = f_end = None
            scope_label = "自定义时间段（参数无效）"
        else:
            scope_label = f"自定义 {f_start} ~ {f_end}"
    elif filters_ok and scope == SCOPE_ONE:
        _cands = DATA.attempts(limit=500)
        if _cands:
            _opts = {a["attempt_id"]: a for a in _cands}
            f_attempt = st.selectbox(
                "作答记录", list(_opts),
                format_func=lambda k: (
                    f"{_opts[k]['start_display']} · {_opts[k]['student_id']} · "
                    f"{_opts[k]['problem_id']}"
                    + (f" · 错误 {_opts[k]['mistake_count']}"
                       if _opts[k]["mistake_count"] else "")
                ),
                key="one_attempt",
            )
            scope_label = f"单次提交 {f_attempt}"
        else:
            st.caption("没有可选的作答记录。")
            scope_label = "单次提交（无记录）"

    # ---------------- 班级 + 学生（两级筛选） ----------------
    st.divider()
    st.subheader("班级与学生")

    #: 审阅 T7：班级配置错误（如"同一学生出现在多个班级"）**必须与数据库
    #: 在线状态分开显示** —— 旧实现把它放在 `filters_ok`（数据库可读）分支里，
    #: 数据库不可用时错误被隐藏，教师看不到"归属无法确定"的提示。
    if CLASS_ERR:
        st.warning(f"⚠️ 班级配置：{CLASS_ERR}")

    ROSTER = DATA.student_roster(start=f_start, end=f_end) if filters_ok else []
    ROSTER_IDS = [s["student_id"] for s in ROSTER]

    #: ------------------------------------------------------------------
    #: 演示数据「可筛除」（审阅第二批 #7 点名要求）
    #:
    #: 审阅原文：「演示 sim_* 与真实学生**明确分源、可筛除**，不默认混算」。
    #: 上一版只在班级名标了「（模拟数据）」，**没有筛除开关** —— 演示班会与
    #: 真实班并列，教师很容易把演示数据当成真实学情读。
    #: 现在给出显式开关；**默认不排除**（演示环境本来就要看演示数据），
    #: 但一旦勾上就彻底从名单里剔除，且界面会写明剔掉了几个。
    #: ------------------------------------------------------------------
    DEMO_IDS = DATA.demo_student_ids() if filters_ok else []
    exclude_demo = st.checkbox(
        "排除演示 / 测试数据" + (f"（{len(DEMO_IDS)} 名）" if DEMO_IDS else ""),
        value=False, key="exclude_demo",
        help="按学生 ID 前缀识别（`sim_*` 与 `tester`）。勾上后他们不会出现在"
             "任何统计里 —— 用于「只看真实学生」的场景。",
    )
    if exclude_demo and DEMO_IDS:
        _demo_set = set(DEMO_IDS)
        _before = len(ROSTER_IDS)
        #: ⚠️ **必须连 `ROSTER` 一起过滤** —— `CLASS_SUM` 等按班级聚合是基于
        #: `ROSTER` 算的；只过滤 `ROSTER_IDS` 会让"排除演示数据"在班级视图里失效。
        ROSTER = [s for s in ROSTER if s["student_id"] not in _demo_set]
        ROSTER_IDS = [s["student_id"] for s in ROSTER]
        _dropped = _before - len(ROSTER_IDS)
        if _dropped:
            st.caption(f"已排除 {_dropped} 名演示 / 测试学生"
                       f"（原有 {_before} 名，剩 {len(ROSTER_IDS)} 名）")

    #: ⚠️ **班级/学生筛选只在 DB 模式有意义。**
    #: API 模式（含 MODE_DOWN）下查询走的是后端全局聚合接口，
    #: 它**根本不接受学生集合**。旧实现照样把选择器摆出来，
    #: 教师选了班级、数字却一点没变 —— 看起来筛选生效了，实际没有。
    #: 所以这里**直接禁用**，并说明原因。
    if not filters_ok:
        st.caption("⚠️ 班级 / 学生筛选需要数据库可读，当前不可用（仅全局口径）。")
        STUDENT_IDS = None          #: None = 显式"全部"，交给后端接口
    else:
        #: 学生清单来自**当前时间范围**（计数随时间变化）；
        #: 班级成员来自配置文件（跨时间恒定）。取交集才是可分析的集合。
        CLASS_NAMES = [ALL_CLASSES] + sorted(CLASSES.keys())
        #: 审阅 T7「未分班学生」：有学生**不在任何班级里**时，必须额外提供
        #: 「未分班」入口 —— 否则这些学生会凭空消失（只能混在「全部班级」里，
        #: 无法单独查看）。旧实现只在 `CLASSES` 为空时才给「未分班」，
        #: 一旦配了班级，未分班学生就选不到了。
        _assigned = {m for ms in CLASSES.values() for m in ms}
        if any(sid not in _assigned for sid in ROSTER_IDS):
            CLASS_NAMES.append(UNCLASSED)
        #: 明确回显班级归属取自哪里 —— 「本地 classes.json 只能作为演示映射」，
        #: 教师需要知道当前看到的班级是演示映射还是库里的权威归属。
        st.caption(f"班级归属来源：{CLASS_SOURCE}")

        #: ⚠️ **默认值必须落在实际选项里。**
        #: 原实现无条件写 `default=[ALL_CLASSES]`，而班级配置缺失/损坏时
        #: 选项只有 `[UNCLASSED]` → Streamlit 抛
        #: `StreamlitDefaultNotInOptionsError`，整页打不开。
        #: 审阅已复现（补上 MODE_API 导入后紧接着就是这个错）。
        _default_classes = [ALL_CLASSES] if ALL_CLASSES in CLASS_NAMES else [CLASS_NAMES[0]]

        #: ⚠️ **配置里删掉了班级时，Streamlit 会静默清空选中值。**
        #:
        #: 实测：用户选了「甲班」→ 有人把甲班从 classes.json 删掉 → 刷新页面，
        #: 选中值变成 `[]`，于是下面的 `or _default_classes` 生效、
        #: **悄悄回退到「全部班级」** —— 界面上的数据范围从"甲班"变成"全部"，
        #: 而**没有任何提示**。教师会以为还在看甲班的数据。
        #:
        #: 审阅要求「班级删除…应保留**明确错误**和可用空状态」。
        #: 所以这里主动对比上一次的选中值，发现班级消失就明确告知，
        #: 而不是让它静默回退。
        _prev_chosen = st.session_state.get("_prev_chosen_classes") or []
        _dropped_classes = [c for c in _prev_chosen if c not in CLASS_NAMES]

        chosen_classes = st.multiselect(
            "班级（可多选）", CLASS_NAMES, default=_default_classes,
            key="class_filter",
            help="班级来自前端配置 frontend/classes.json（数据库无班级字段）",
        ) or _default_classes

        if _dropped_classes:
            st.warning(
                "⚠️ **之前选择的班级已从配置中删除**："
                + "、".join(f"「{c}」" for c in _dropped_classes)
                + "。\n\n"
                "已**回退到「全部班级」**（数据范围因此变大）——"
                "请确认这是你要的范围，并重新选择班级。"
            )
        #: 记录本次选择，供下一轮检测"班级被删"
        st.session_state["_prev_chosen_classes"] = list(chosen_classes)

        _pool: list[str] = []
        for _cn in chosen_classes:
            for _s in students_of_class(CLASSES, _cn, ROSTER_IDS):
                if _s not in _pool:
                    _pool.append(_s)
        _pool_set = set(_pool)
        _pool = [s for s in ROSTER_IDS if s in _pool_set]   # 保持与名单同序

        chosen_students = st.multiselect(
            "学生（可多选；留空 = 所选班级全部）", _pool,
            default=[], key="student_filter_multi",
        )
        #: 班级变化后旧的已选学生可能不在池里 —— 过滤掉，
        #: 否则会拿一个不属于当前班级的学生去查，结果与界面显示不符
        chosen_students = [s for s in chosen_students if s in _pool_set]

        #: ⚠️⚠️ **`[]` 与 `None` 语义不同，不能再用 `or` 混为一谈。**
        #:
        #: 旧实现 `STUDENT_IDS = chosen_students or _pool`：当所选班级
        #: **一个成员都没有**时 `_pool` 为空 → `STUDENT_IDS = []` →
        #: 而数据层把 `[]` 当"不筛选" → **查出全库记录**。
        #: 界面写着"0 名学生 / 某班"，表格里却是所有班的数据（审阅实测条件已复现）。
        #:
        #: 现在的语义：
        #:   · 选了具体学生 → 就这些（非空列表）
        #:   · 没选学生     → 所选班级的全部成员（可能是**空列表**，
        #:                    此时"该班没成员"是事实，必须查不到任何数据）
        #: 数据层对 `[]` 生成 `FALSE`，对 `None` 才表示"全体"。
        STUDENT_IDS = list(chosen_students) if chosen_students else list(_pool)
        if chosen_students:
            scope_label += f" · 已选 {len(chosen_students)} 名学生"
        elif len(chosen_classes) == 1 and chosen_classes[0] != ALL_CLASSES:
            scope_label += f" · {chosen_classes[0]}"
        st.caption(f"当前纳入分析：**{len(STUDENT_IDS)} 名学生**")
        if not STUDENT_IDS:
            #: 明确说清"是空集合、所以不会有数据"，而不是让人以为页面坏了。
            st.warning(
                "所选范围内**没有学生**，因此不会显示任何数据。"
                "（这是刻意的：空集合必须查出空结果，"
                "不能退化成「查全库」。）"
                "请检查班级配置的成员 ID 是否与数据库一致，"
                "或所选时间范围内是否确有作答。"
            )

    st.divider()
    st.subheader("数据说明")
    st.caption(
        "· **历史提及 ≠ 当前薄弱**：画像 `weak_concepts` 是**只增不减**的累计并集，"
        "学生后来掌握了也不会移除，也**不含**撤销/已解决状态；"
        "它**未经语义验证**，只能读作「历史上被提及过」。\n"
        "· **错误记录另算**：`mistakes` 是每次作答的**过程证据**"
        "（含出错节点与性质）。两者口径不同，页面已**分开展示**。\n"
        "· **不用最终答案判错**：最终答案正确也可能包含过程错误，"
        "按它会得到「零错题」。\n"
        "· **提示判定三态**：有效 / 无效 / **未判定**；未判定单列、不计入分母。"
        "「有效」是系统按节点/状态变化的**启发式判定**，不是教学因果效果。\n"
        "· **提示下发 ≠ 学生卡住**：主动求助同样会推高提示数，"
        "所以提示与错误记录分开列。\n"
        "· **分类与班级为前端规则**：学科分类与班级映射均由前端配置，非权威裁定。\n"
        "· 历史统计只读、不调用模型；本地核对决定需显式保存。"
    )

    st.divider()
    if st.button("刷新数据", width="stretch"):
        st.cache_resource.clear()
        st.rerun()


# ============================ 取数与聚合 ============================

ROWS = normalize_rows(
    DATA.weak_ranking(start=f_start, end=f_end,
                      student_ids=STUDENT_IDS, attempt=f_attempt)
)
COUNTS = DATA.counts(start=f_start, end=f_end,
                     student_ids=STUDENT_IDS, attempt=f_attempt)
HINTS = DATA.hint_stats(start=f_start, end=f_end,
                        student_ids=STUDENT_IDS, attempt=f_attempt)
STUCK = DATA.stuck_nodes(start=f_start, end=f_end,
                         student_ids=STUDENT_IDS, attempt=f_attempt)
PROB_RANK = DATA.problem_ranking(start=f_start, end=f_end,
                                 student_ids=STUDENT_IDS, attempt=f_attempt)
ISSUE_RANK = DATA.issue_type_ranking(start=f_start, end=f_end,
                                     student_ids=STUDENT_IDS, attempt=f_attempt)
WRONG = DATA.wrong_attempts(start=f_start, end=f_end,
                            student_ids=STUDENT_IDS, attempt=f_attempt)
#: **模型记录的过程错误**（`mistakes`）—— 与上面的"历史提及"(ROWS) 是两种信号。
ERR_RECORDS = DATA.error_record_ranking(start=f_start, end=f_end,
                                        student_ids=STUDENT_IDS, attempt=f_attempt)
#: 覆盖面的**精确**分子/分母（各自 COUNT DISTINCT，同源同筛选）。
#: 不用"各知识点学生数的最大值"—— 那会少算（审阅实测：2 人算成 1 人）。
COVERED = DATA.covered_students(start=f_start, end=f_end,
                                student_ids=STUDENT_IDS, attempt=f_attempt)
SCOPE_STUDENTS = DATA.scope_students(start=f_start, end=f_end,
                                     student_ids=STUDENT_IDS, attempt=f_attempt)
CURRENT_ERRORS = DATA.current_error_evidence(start=f_start, end=f_end,
                                             student_ids=STUDENT_IDS, attempt=f_attempt)
REVIEW_STATUS = DATA.review_store_status()
#: **范围指纹**：标识"这批数字是哪个范围算出来的"。
#: 审阅要求「返回同一个 aggregation/snapshot 标识」——
#: 本页的数字来自多次查询（知识点/提示/错题/人数），必须证明它们同源。
#: 指纹会印在每张诊断卡片上，界面并列展示时一眼可核验。
SCOPE_ID = scope_fingerprint(
    start=f_start, end=f_end, student_ids=STUDENT_IDS,
    attempt=f_attempt, extra=scope_label,
)
DISCS = discipline_totals(ROWS)
DISC_NAMES = [d["name"] for d in DISCS]
CLASS_SUM = (class_summaries(ROSTER, CLASSES, active_ids=[
    s["student_id"] for s in ROSTER if s["attempt_count"] > 0
]) if ROSTER else [])

PROBLEMS = DATA.problems()


def problem_title(pid: str) -> str:
    """题号 → 题目名（题号本身仍是主键，名字只作补充显示）。"""
    return (PROBLEMS.get(pid) or {}).get("title") or ""


# ============================ 页头 ============================

page_intro("教学处理台", "先核对书面证据，再处理系统判断；历史分布保留用于讲评。", "教师分析")
st.caption(
    f"分析范围：**{scope_label}**"
    + ("" if filters_ok else "（数据库不可读，仅全局口径）")
)
#: 口径提示放在最显眼处：教师端最容易出的错，是把"历史累计信号"
#: 读成"当前确认的错误"。审阅把这一点列为合入前必修。
with st.expander("统计口径与阅读说明"):
    st.caption(
        "⚠️ **口径**：本页「历史提及」来自学生画像的累计并集（只增不减，"
        "**未经语义验证**，也不含撤销/已解决状态）；「错误记录」来自每次作答的"
        "过程记录。两者**不是**同一件事，页面已分开展示。"
        + {
            REVIEW_STORE_UNAVAILABLE: "本环境**没有错误证据复核库**，因此"
                "「已确认错误」暂不可得 —— 标为**信息不足**。",
            REVIEW_STORE_DETECTED: "检测到错误证据复核表，但教师端**尚未接入"
                "复核状态**（仍按原始 mistakes 统计，未处理撤销/解决）。",
            REVIEW_STORE_INTEGRATED: "",
        }.get(REVIEW_STATUS, "")
    )


#: 审阅 T1：本次刷新里**累计**的查询失败，在这里统一汇总 ——
#: 一个统计失败、其他统计成功时，失败区域必须保持明确警告。
render_query_failures()

view = st.radio(
    "视图", [VIEW_TASKS, VIEW_DIST, VIEW_PARALLEL, VIEW_WRONG, VIEW_CLASS],
    horizontal=True, key="teacher_view", label_visibility="collapsed",
)

# ============================ ① 知识点分布 ============================

def view_distribution() -> None:
    st.header(VIEW_DIST)
    st.caption("**先看学科大类，点进去再看该模块内部**。饼图扇区按「提及次数」分配。")

    if not ROWS:
        data_hint()
        st.divider()
        render_diagnosis(
            build_diagnosis(weak_rows=[], hint_stats=HINTS, stuck_nodes=STUCK,
                            active_students=COUNTS["students"],
                            covered_students=COVERED,
                            attempt_count=COUNTS["attempts"],
                            scope_label=scope_label, scope_id=SCOPE_ID),
            heading="学情诊断",
        )
        return

    disc_df = pd.DataFrame(
        [{"学科": d["name"], "提及次数": d["mentions"], "知识点数": len(d["concepts"])}
         for d in DISCS]
    )
    st.subheader("学科大类分布")

    _l, _r = st.columns([3, 2])
    with _l:
        #: ⚠️ `selection_mode` 要填**图表 spec 里那个 selection 的名字**（`sel_disc`），
        #: 不是类型名 `"point"` —— 填类型名会报
        #: "Selection parameter 'point' is not defined in the chart spec"。
        _ev = st.altair_chart(
            pie_chart(disc_df, category="学科", value="提及次数", param="sel_disc"),
            on_select="rerun", selection_mode="sel_disc", key="pie_disc",
            width="stretch",
        )
        clicked_disc = picked(_ev, "sel_disc", "学科")
        st.caption("点击扇区即可下钻；也可用下方选择器。")
    with _r:
        st.dataframe(disc_df.sort_values("提及次数", ascending=False),
                     hide_index=True, width="stretch")

    if len(DISCS) == 1:
        st.info(
            f"当前范围内的历史提及**全部集中在「{DISCS[0]['name']}」**，"
            "已自动展开该模块的内部构成。"
        )

    #: 点击优先，选择器兜底（两者必须显示一致，否则会出现"选的是 A、看的是 B"）
    _default_disc = DISC_NAMES[0] if len(DISCS) == 1 else None
    _seg_disc = st.segmented_control(
        "模块（也可直接选择）", DISC_NAMES, default=_default_disc, key="seg_disc",
    )
    sel_disc = (clicked_disc if clicked_disc in DISC_NAMES else None) or _seg_disc

    if not sel_disc:
        st.info("请点击上方扇区或选择模块，查看其内部构成。")
    else:
        st.divider()
        st.subheader(f"「{sel_disc}」内部构成")
        sub_rows = concept_rows(ROWS, discipline=sel_disc)
        subs = subfield_totals(sub_rows)
        sub_names = [s["name"] for s in subs]
        clicked_sub = None
        sel_sub = None

        if len(subs) <= 1:
            #: 非力学学科的子领域统一为"综合"，细分不出东西 ——
            #: 直接跳到知识点明细，不摆一个单扇区饼图充数
            st.caption("该模块的知识点未再分领域，直接列出明细。")
        else:
            sub_df = pd.DataFrame(
                [{"知识领域": s["name"], "提及次数": s["mentions"],
                  "知识点数": len(s["concepts"])} for s in subs]
            )
            _l2, _r2 = st.columns([3, 2])
            with _l2:
                _ev2 = st.altair_chart(
                    pie_chart(sub_df, category="知识领域", value="提及次数",
                              param="sel_sub"),
                    on_select="rerun", selection_mode="sel_sub", key="pie_sub",
                    width="stretch",
                )
                clicked_sub = picked(_ev2, "sel_sub", "知识领域")
                st.caption("点击扇区可下钻到该领域的知识点。")
            with _r2:
                st.dataframe(sub_df.sort_values("提及次数", ascending=False),
                             hide_index=True, width="stretch")
            _seg_sub = st.segmented_control(
                "知识领域（也可直接选择）", sub_names, default=None, key="seg_sub",
            )
            sel_sub = (clicked_sub if clicked_sub in sub_names else None) or _seg_sub

        st.divider()
        _t = f"知识点明细（{sel_disc}" + (f" · {sel_sub}）" if sel_sub else " · 全部领域）")
        st.subheader(_t)
        detail = concept_rows(sub_rows, subfield=sel_sub) if sel_sub else sub_rows
        if detail:
            ddf = pd.DataFrame(
                [{"知识点": r["concept"], "提及次数": r["mentions"],
                  "涉及学生数": r["students"], "知识领域": r["subfield"]}
                 for r in detail]
            )
            _d1, _d2 = st.columns([3, 2])
            with _d1:
                st.altair_chart(
                    hbar(ddf, category="知识点", value="提及次数", title="提及次数"),
                    width="stretch",
                )
            with _d2:
                st.dataframe(ddf.sort_values("提及次数", ascending=False),
                             hide_index=True, width="stretch")
            st.caption(
                "「提及次数」= 该知识点被诊断出的累计次数；"
                "「涉及学生数」= 涉及的学生数（去重）。两者不等是正常的。"
            )
        else:
            st.caption("该领域下暂无知识点记录。")

    # ---- 信号来源对照（#4：历史提及 vs 模型记录的错误）----
    #
    # 审阅要求「分开展示」，并明确各自的性质与可得性。
    # 这一节存在的意义：教师最可能的误读是把上面的「历史提及」
    # 当成「当前确认的错误」，进而按错误的优先级安排讲评。
    st.divider()
    st.subheader("信号来源与可信度")
    _s1, _s2, _s3 = st.columns(3)
    with _s1:
        st.metric("历史提及（知识点）", len(ROWS),
                  help="来自画像 weak_concepts 的累计并集")
        st.caption(
            "**性质**：历史信号，只增不减、**未经语义验证**。\n\n"
            "**可用于**：找「哪些知识点常被提及」。\n\n"
            "**不可用于**：断言「当前薄弱」或「反复出错」——"
            "历史记录不会随掌握或复核自动清空；当前复核证据在独立区域显示。"
        )
    with _s2:
        st.metric("模型记录的错误（条）",
                  sum(r["count"] for r in ERR_RECORDS) if ERR_RECORDS else 0,
                  help="来自 attempts.mistakes 的过程证据")
        st.caption(
            "**性质**：每次作答的过程证据（含节点与性质）。\n\n"
            "**可用于**：定位「错在哪一步、错成什么性质」。\n\n"
            "**注意**：`mistakes` 里**没有知识点字段**，"
            "因此无法按知识点聚合 —— 不硬凑。"
        )
    with _s3:
        if REVIEW_STATUS == REVIEW_STORE_INTEGRATED:
            st.metric("错误复核库", "已接入")
            st.caption("可区分 待复核 / 已确认 / 已撤销 / 已解决。")
        elif REVIEW_STATUS == REVIEW_STORE_DETECTED:
            #: 审阅 T2：表在 ≠ 已读取复核决定。如实标注，不冒充 ready。
            st.metric("错误复核库", "检测到表，尚未接入", help="检测到复核表，但未接入复核状态")
            st.caption(
                "**检测到错误证据复核表，但教师端尚未接入复核状态**："
                "当前排行仍读原始 mistakes，**未处理已撤销 / 已解决**。\n\n"
                "因此**不提供**「已确认错误排行」——"
                "接入规范后端读接口后方可区分 待复核 / 已确认 / 已撤销 / 已解决。"
            )
        else:
            st.metric("错误复核库", "不可得", help="本环境未接入复核库")
            st.caption(
                "**信息不足**：本环境没有错误证据复核库，"
                "无法区分 待复核 / 已确认 / 已撤销 / 已解决。\n\n"
                "因此**不提供**「已确认错误排行」——"
                "那需要生产证据规则；本包只读数据库、不改后端，拿不到。"
            )
    if ERR_RECORDS:
        st.markdown("**错误记录按「性质 × 节点」分布**")
        st.dataframe(
            [{"错误性质": r["issue_type"], "节点": r["node"],
              "条数": r["count"], "涉及学生数": r["student_count"],
              "涉及题目数": r["problem_count"]}
             for r in ERR_RECORDS[:30]],
            hide_index=True, width="stretch",
        )
        st.caption(
            "这是**过程错误记录**的分布，与上方的「历史提及」是两种信号，"
            "不要相互换算或相加。"
        )

    st.divider()
    render_diagnosis(
        build_diagnosis(
            weak_rows=ROWS, hint_stats=HINTS, stuck_nodes=STUCK,
            active_students=SCOPE_STUDENTS, covered_students=COVERED,
            attempt_count=COUNTS["attempts"],
            scope_label=scope_label, scope_id=SCOPE_ID,
        ),
        heading="学情诊断",
    )


# ============================ ② 平行检索 ============================

def view_parallel() -> None:
    st.header(VIEW_PARALLEL)
    st.caption(
        "同一批数据、**平级的三个检索入口**：按知识点 / 按题目 / 按错误类型。"
        "换一个检索键，就从另一个角度定位问题。"
    )

    dim = st.segmented_control(
        "检索键", [DIM_KNOWLEDGE, DIM_PROBLEM, DIM_ISSUE],
        default=DIM_PROBLEM, key="parallel_dim",
    ) or DIM_PROBLEM

    if dim == DIM_KNOWLEDGE:
        if not ROWS:
            data_hint()
            return
        df = pd.DataFrame([
            {"知识点": r["concept"], "提及次数": r["mentions"],
             "涉及学生数": r["students"]}
            for r in ROWS
        ])
        detail = [
            {"知识点": r["concept"], "提及次数": r["mentions"],
             "涉及学生数": r["students"], "学科": r["discipline"],
             "知识领域": r["subfield"]}
            for r in ROWS
        ]
        render_dimension(
            df, category="知识点", value="提及次数", param="pdim_kp",
            detail_rows=detail, filter_field="知识点",
            detail_columns={"知识点": "知识点", "提及次数": "提及次数",
                            "涉及学生数": "涉及学生数", "学科": "学科",
                            "知识领域": "知识领域"},
            note="「提及次数」是累计次数；「涉及学生数」按学生去重。",
        )
        return

    if dim == DIM_PROBLEM:
        if not PROB_RANK:
            st.info("当前范围内没有作答记录。")
            return
        #: 检索量用**有问题次数**而非作答次数 —— 后者包含"做了且没出问题"的作答，
        #: 会把一道简单的题排到前面。作答次数另列一栏可查，两者都保留。
        df = pd.DataFrame([
            {"题号": r["problem_id"], "有问题次数": r["flagged_count"],
             "作答次数": r["attempt_count"], "错误点数": r["mistake_count"]}
            for r in PROB_RANK
        ])
        detail = [
            {"题号": r["problem_id"], "题目": problem_title(r["problem_id"]),
             "有问题次数": r["flagged_count"], "作答次数": r["attempt_count"],
             "错误点数": r["mistake_count"], "涉及学生数": r["student_count"]}
            for r in PROB_RANK
        ]
        render_dimension(
            df, category="题号", value="有问题次数", param="pdim_prob",
            detail_rows=detail, filter_field="题号",
            detail_columns={"题号": "题号", "题目": "题目",
                            "有问题次数": "有问题次数", "作答次数": "作答次数",
                            "错误点数": "错误点数", "涉及学生数": "涉及学生数"},
            note="「有问题次数」= 该题产生过错误或薄弱点的作答数（按过程判定）。",
        )
        st.divider()
        st.subheader("按题号看错题明细")
        _pids = [r["problem_id"] for r in PROB_RANK]
        _pid = st.segmented_control("题号", _pids, key="parallel_prob_drill")
        if _pid:
            st.markdown(f"**`{_pid}`** — {problem_title(_pid)}")
            _w = [w for w in WRONG if w["problem_id"] == _pid]
            if _w:
                st.dataframe(
                    [{"学生": w["student_id"], "时间": w["start_display"],
                      "错误类型": w["issue_types"] or "—",
                      "节点": w["nodes"] or "—",
                      "薄弱知识点": w["weak_text"] or "—",
                      "错误描述": (w["issues"] or "")[:120]}
                     for w in _w],
                    hide_index=True, width="stretch",
                )
            else:
                st.caption("该题在范围内没有错题记录。")
        else:
            st.caption("选择题号查看该题的具体错题明细。")
        return

    # dim == DIM_ISSUE
    if not ISSUE_RANK:
        st.info("当前范围内没有错误记录。")
        return
    df = pd.DataFrame([
        {"错误类型": r["issue_type"], "错误点数": r["count"]}
        for r in ISSUE_RANK
    ])
    detail = [
        {"错误类型": r["issue_type"], "错误点数": r["count"],
         "涉及学生数": r["student_count"], "涉及题目数": r["problem_count"]}
        for r in ISSUE_RANK
    ]
    render_dimension(
        df, category="错误类型", value="错误点数", param="pdim_issue",
        detail_rows=detail, filter_field="错误类型",
        detail_columns={"错误类型": "错误类型", "错误点数": "错误点数",
                        "涉及学生数": "涉及学生数", "涉及题目数": "涉及题目数"},
        note="⚠️ 这里是**错误性质**（concept = 概念 / method = 方法 等），"
             "**不是知识点** —— 要看知识点请切到「知识点」检索键。",
    )


# ============================ ③ 错题分析 ============================

def view_wrong() -> None:
    st.header(VIEW_WRONG)
    st.caption("回答「**哪道题、多少人、错在什么性质、记录集中在哪个节点**」。")

    conc = wrong_problem_concentration(WRONG)
    if not WRONG:
        st.info("当前范围未取得模型错误记录；不等于学生没有做错。")
        st.divider()
        render_diagnosis(
            build_wrong_diagnosis([], active_students=COUNTS["students"],
                                  scope_label=scope_label, scope_id=SCOPE_ID),
            heading="错题诊断",
        )
        return

    _w1, _w2, _w3 = st.columns(3)
    with _w1:
        st.metric("错题记录", conc["total_records"], help="一条 = 一次产生错误的作答")
    with _w2:
        st.metric("错误点数", conc["total_mistakes"])
    with _w3:
        st.metric("涉及题目数", len(conc["by_problem"]))

    st.divider()
    st.subheader("错题题号排行")
    st.caption(
        "按**涉及学生数**降序 —— 教学上「多少人都错在这题」比「总错了几次」更有意义"
        "（一个人反复错会把次数刷高）。"
    )
    wdf = pd.DataFrame([
        {"题号": r["problem_id"], "涉及学生数": r["student_count"],
         "错题记录数": r["records"], "错误点数": r["mistake_count"],
         "错误性质": r["types"] or "—"}
        for r in conc["by_problem"]
    ])
    _wa, _wb = st.columns([3, 2])
    with _wa:
        st.altair_chart(
            hbar(wdf, category="题号", value="涉及学生数", title="涉及学生数"),
            width="stretch",
        )
    with _wb:
        st.dataframe(wdf, hide_index=True, width="stretch")

    st.divider()
    st.subheader("错题明细（谁 · 哪道题 · 错在哪）")
    st.dataframe(
        [{"学生": w["student_id"], "题号": w["problem_id"],
          "题目": problem_title(w["problem_id"]),
          "时间": w["start_display"],
          "错误类型": w["issue_types"] or "—", "出错节点": w["nodes"] or "—",
          "薄弱知识点": w["weak_text"] or "—",
          "错误描述": (w["issues"] or "")[:110]}
         for w in WRONG],
        hide_index=True, width="stretch",
    )
    st.caption(
        "「错误描述」是该次作答记录到的错误说明。完整的逐轮过程"
        "（含 OCR 原文、逐步识别）请到学生端的「学习记录」查看。"
    )

    st.divider()
    _wl, _wr = st.columns(2)
    with _wl:
        st.subheader("出错环节（节点）")
        if STUCK:
            sdf = pd.DataFrame([
                {"节点": s["node"], "来自提示": s.get("from_hint", 0),
                 "来自错误记录": s.get("from_mistake", 0), "合计": s["count"]}
                for s in STUCK
            ])
            st.altair_chart(
                hbar(sdf, category="节点", value="合计", title="出现次数"),
                width="stretch",
            )
            st.dataframe(sdf, hide_index=True, width="stretch")
            #: ⚠️ 原文案是"口径：发提示即学生卡住"。审阅判定**越界**：
            #: 主动求助、连续求助、缺推导都不能证明认知卡顿。
            #: 现在只陈述两个来源的分布，并明确两者口径不同。
            st.caption(
                "两个来源**口径不同，刻意分开统计**："
                "**来自提示** = 在该节点下发过提示（反映**求助/被提示的分布**）；"
                "**来自错误记录** = mistakes 里记录的出错节点（过程证据）。"
                "「提示下发得多」**不代表**学生在该处认知卡住 ——"
                "主动求助同样会推高这个数，请结合提示原文与错误记录判断。"
            )
        else:
            st.caption("暂无节点记录。")
    with _wr:
        st.subheader("按题下钻")
        _pids = [r["problem_id"] for r in conc["by_problem"]]
        _pid = st.selectbox(
            "题号", _pids, key="wrong_problem_drill",
            format_func=lambda p: f"{p} — {problem_title(p)}",
        )
        _sel = [w for w in WRONG if w["problem_id"] == _pid]
        if _sel:
            st.dataframe(
                [{"学生": w["student_id"], "错误类型": w["issue_types"] or "—",
                  "节点": w["nodes"] or "—"} for w in _sel],
                hide_index=True, width="stretch",
            )
            _st = DATA.hint_stats(start=f_start, end=f_end,
                                  student_ids=STUDENT_IDS, problem_id=_pid)
            st.caption(
                f"该题提示 {_st.get('total', 0)} 条 —— "
                f"有效 {_st.get('effective', 0)} / 无效 {_st.get('ineffective', 0)} / "
                f"未判定 {_st.get('undecided', 0)}，"
                f"有效率 {pct(_st.get('effectiveness'))}。"
            )
        else:
            st.caption("该题暂无错题记录。")

    st.divider()
    render_diagnosis(
        build_wrong_diagnosis(WRONG, active_students=COUNTS["students"],
                              scope_label=scope_label, scope_id=SCOPE_ID),
        heading="错题诊断",
    )


# ============================ ④ 班级与个体 ============================

def view_class() -> None:
    st.header(VIEW_CLASS)
    st.caption(
        "回答「**各班的记录分布如何、记录集中在哪、班级整体诊断是什么**」，"
        "并可下钻到**原始证据**与个体。\n\n"
        "⚠️ 本页**不做班级能力排名** —— 各班作答量、题目难度、活跃覆盖都不同，"
        "记录多往往只是做得多。"
    )

    if not CLASSES:
        st.warning("未读到班级配置。")
        st.caption(
            f"班级由前端配置定义：`{CFG.classes_file}`。"
            "请按 `classes: {班级名: [学生ID, ...]}` 的格式填写。"
            + (f"当前错误：{CLASS_ERR}" if CLASS_ERR else "")
        )
        return
    st.caption(
        f"班级来源：`{CFG.classes_file.name}`（**前端配置**，数据库无班级字段）。"
        "修改该文件后刷新即可生效，无需重启服务。"
    )

    st.subheader("班级概览")
    if CLASS_SUM:
        cdf = pd.DataFrame([
            {"班级": c["name"], "在册人数": c["student_count"],
             "活跃人数": (c["active_count"] if c["active_count"] is not None else "—"),
             "活跃覆盖": (f"{c['coverage']:.0%}" if c.get("coverage") is not None else "—"),
             "作答次数": c["attempt_count"],
             "有错误记录的作答": c["problem_attempt_count"],
             "历史提及次数": c["weak_count"],
             "出错率": (f"{c['problem_rate']:.0%}" if c["problem_rate"] is not None else "—"),
             "提示数": c["hint_count"]}
            for c in CLASS_SUM
        ])
        st.dataframe(cdf, hide_index=True, width="stretch")
        #: ⚠️ 原文案把"人均薄弱点"说成"跨班可比"，并按它排名称"问题最多"。
        #: 审阅指出这**不具备可比性**：没有控制题目、难度、作答次数、
        #: 活跃覆盖与曝光时间 —— 积极作答的班反而累积更多历史信号。
        st.caption(
            "⚠️ 本表是**记录分布**，不是**能力或水平对比**。"
            "各班作答次数、题目难度、活跃覆盖都不同，"
            "**记录多很可能只是因为做得多**。"
            "系统刻意**不做人均排名** —— 那会把「更努力」读成「更差」。"
            "活跃覆盖低时，任何比率都不宜横向比较。"
        )
    else:
        st.info("当前时间范围内没有班级数据。")

    if CLASS_SUM:
        st.divider()
        render_diagnosis(compare_classes(CLASS_SUM, scope_label=scope_label),
                         heading="班级记录分布对照")

    st.divider()
    st.subheader("班级诊断（选一个班深入）")
    _opts = [c["name"] for c in CLASS_SUM] if CLASS_SUM else sorted(CLASSES.keys())
    _cn = st.selectbox("班级", _opts, key="class_diagnose")
    _members = students_of_class(CLASSES, _cn, ROSTER_IDS)

    if not _members:
        st.caption(f"「{_cn}」在**当前时间范围内**没有学生有数据。")
    else:
        _rows = normalize_rows(DATA.weak_ranking(
            start=f_start, end=f_end, student_ids=_members))
        _cnt = DATA.counts(start=f_start, end=f_end, student_ids=_members)
        _hint = DATA.hint_stats(start=f_start, end=f_end, student_ids=_members)
        _stuck = DATA.stuck_nodes(start=f_start, end=f_end, student_ids=_members)
        _wrong = DATA.wrong_attempts(start=f_start, end=f_end, student_ids=_members)
        #: 该班的精确覆盖面（各自 COUNT DISTINCT，与上面同一份筛选条件）
        _covered = DATA.covered_students(start=f_start, end=f_end, student_ids=_members)
        _scope = DATA.scope_students(start=f_start, end=f_end, student_ids=_members)
        #: 该班自己的范围指纹（与页头那个不同 —— 范围不同就该指纹不同）
        _CLASS_SCOPE_ID = scope_fingerprint(
            start=f_start, end=f_end, student_ids=_members,
            extra=f"班级「{_cn}」")

        _k1, _k2, _k3, _k4 = st.columns(4)
        with _k1:
            st.metric("学生数", _cnt["students"])
        with _k2:
            st.metric("作答次数", _cnt["attempts"])
        with _k3:
            st.metric("有模型错误记录的作答", _cnt["problem_attempts"])
        with _k4:
            st.metric("提示有效率", pct(_hint.get("effectiveness")))

        if _rows:
            _ddf = pd.DataFrame([
                {"知识点": r["concept"], "提及次数": r["mentions"],
                 "涉及学生数": r["students"], "知识领域": r["subfield"]}
                for r in _rows
            ])
            st.markdown("**该班知识点历史提及分布**")
            _g1, _g2 = st.columns([3, 2])
            with _g1:
                st.altair_chart(
                    pie_chart(_ddf, category="知识点", value="提及次数",
                              param="class_kp"),
                    on_select="ignore", width="stretch",
                )
            with _g2:
                st.dataframe(_ddf, hide_index=True, width="stretch")

        if _wrong:
            st.markdown("**该班错题题号**")
            _wc = wrong_problem_concentration(_wrong)
            st.dataframe(
                [{"题号": r["problem_id"], "题目": problem_title(r["problem_id"]),
                  "涉及学生数": r["student_count"], "错题记录数": r["records"],
                  "错误性质": r["types"] or "—"}
                 for r in _wc["by_problem"]],
                hide_index=True, width="stretch",
            )

        render_diagnosis(
            build_diagnosis(
                weak_rows=_rows, hint_stats=_hint, stuck_nodes=_stuck,
                active_students=_scope, covered_students=_covered,
                attempt_count=_cnt["attempts"],
                scope_label=f"班级「{_cn}」", scope_id=_CLASS_SCOPE_ID,
            ) + build_wrong_diagnosis(
                _wrong, active_students=_scope,
                scope_label=f"班级「{_cn}」", scope_id=_CLASS_SCOPE_ID,
            ),
            heading=f"「{_cn}」班级诊断",
        )
        #: #8 证据下钻：让"统计 → 学生 → attempt → 原始证据"这条链可达。
        if _wrong:
            st.divider()
            st.subheader("原始证据下钻（统计 → 作答记录 → 当前版本证据）")
            st.caption(
                "选一条错题记录，查看它的**状态投影、过程错误、每次提示的原文与"
                "判定依据、逐轮轨迹与状态修订**。"
                "这是核对「这个统计是怎么算出来的」的唯一入口 ——"
                "不新建一套复核真值，只读展示库里已有的证据。"
            )
            _ev_opts = {w["attempt_id"]: w for w in _wrong}
            _ev_id = st.selectbox(
                "作答记录", list(_ev_opts), key="class_evidence_pick",
                format_func=lambda k: (
                    f"{_ev_opts[k]['student_id']} · {_ev_opts[k]['problem_id']} · "
                    f"{_ev_opts[k]['start_display']}"
                ),
            )
            render_evidence(_ev_id)

    st.divider()
    st.subheader("个体诊断")
    if not STUDENT_IDS:
        st.caption("当前筛选下没有学生。")
        return
    _sid = st.selectbox(
        "学生", STUDENT_IDS, key="class_student_drill",
        format_func=lambda s: next(
            (f"{s}（作答 {x['attempt_count']} / 错题 {x['problem_attempt_count']}"
             f" / 弱项 {x['weak_count']}）"
             for x in ROSTER if x["student_id"] == s),
            s,
        ),
    )
    _d = DATA.student_detail(_sid, start=f_start, end=f_end)
    _h = _d.get("hints") or {}
    _weak = _d.get("weak_concepts") or []
    #: 个体范围指纹（单学生 + 当前时间范围）
    _STU_SCOPE_ID = scope_fingerprint(
        start=f_start, end=f_end, student_ids=[_sid], extra=f"学生 {_sid}")

    _s1, _s2, _s3, _s4 = st.columns(4)
    with _s1:
        st.metric("范围内作答", _d.get("attempt_count", 0))
    with _s2:
        st.metric("错题记录", len(_d.get("wrong") or []))
    with _s3:
        st.metric("范围内提示", _h.get("total", 0))
    with _s4:
        st.metric("提示有效率", pct(_h.get("effectiveness")))

    st.markdown("**薄弱知识点**")
    if _weak:
        st.markdown("\n".join(f"- {w}" for w in _weak))
        st.caption(
            "**跨时间累计**口径：一旦被诊断出过就会保留，"
            "**不随上方时间筛选变化**（这是学生画像本身的定义）。"
        )
    else:
        st.caption("暂无薄弱知识点记录。")

    _dw = _d.get("wrong") or []
    st.markdown("**该生错题题号**")
    if _dw:
        st.dataframe(
            [{"题号": w["problem_id"], "题目": problem_title(w["problem_id"]),
              "时间": w["start_display"], "错误类型": w["issue_types"] or "—",
              "节点": w["nodes"] or "—", "薄弱知识点": w["weak_text"] or "—"}
             for w in _dw],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("该生在此范围内没有错题记录。")

    _at = _d.get("attempts") or []
    if _at:
        with st.expander(f"全部作答记录（{len(_at)} 条）"):
            st.dataframe(
                [{"时间": a["start_display"], "题号": a["problem_id"],
                  "用时(秒)": a["duration_s"], "提示数": a["hint_records"],
                  "错误点数": a["mistake_count"], "弱项": a["weak_text"] or "—"}
                 for a in _at],
                hide_index=True, width="stretch",
            )

    render_diagnosis(
        build_diagnosis(
            weak_rows=normalize_rows(
                [{"knowledge_point": w, "mention_count": 1, "student_count": 1}
                 for w in _weak]),
            hint_stats=_h, stuck_nodes=[],
            #: 单个学生：分母 1；有弱项则覆盖 1，无则 0（不是"取最大值"）。
            active_students=1, covered_students=(1 if _weak else 0),
            attempt_count=_d.get("attempt_count", 0),
            #: ⚠️ **`_weak` 来自画像 `student_profiles.weak_concepts`，是跨时累计的**
            #: （并集只增不减），而 `attempt_count` 是**当前筛选范围**内的作答数。
            #: 两者时间口径不同，**不能拼进同一句** —— 审阅第二批 #6 明确禁止。
            #: 传 `lifetime` 后，"总体规模"会拆成两句分别标明口径。
            weak_scope=WEAK_SCOPE_LIFETIME,
            scope_label=f"学生 {_sid}", scope_id=_STU_SCOPE_ID,
        ) + build_wrong_diagnosis(
            _dw, active_students=1, scope_label=f"学生 {_sid}",
            scope_id=_STU_SCOPE_ID),
        heading=f"「{_sid}」个体诊断",
    )


# ============================ 分发 ============================


if view == VIEW_TASKS:
    from teacher_workbench import render_teacher_tasks
    render_teacher_tasks(DATA,CURRENT_ERRORS,CFG.backend_url,render_evidence,
                         start=f_start,end=f_end,student_ids=STUDENT_IDS,attempt=f_attempt)
    render_query_failures()
elif view == VIEW_DIST:
    view_distribution()
elif view == VIEW_PARALLEL:
    view_parallel()
elif view == VIEW_WRONG:
    view_wrong()
else:
    view_class()


with st.expander('历史统计与范围覆盖', expanded=view!=VIEW_TASKS):
    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
    with _m1:
        st.metric("覆盖学生", COUNTS["students"], help="该范围内**有过作答**的学生数")
    with _m2:
        st.metric("作答次数", COUNTS["attempts"])
    with _m3:
        st.metric("有过程错误记录的作答", COUNTS["problem_attempts"],
                  help="以 mistakes 为准（过程证据）。**不含**仅历史提及弱项、"
                       "但本次没有错误的作答 —— 那种写法会让"
                       "「历史上弱过」的学生每道题都被算成有问题")
    with _m4:
        st.metric("历史提及知识点", len(ROWS),
                  help="来自画像累计并集，只增不减、未经语义验证。"
                       "不是「当前薄弱知识点个数」")
    with _m5:
        st.metric("提示判定可判定占比", pct(
            (HINTS.get("effective", 0) + HINTS.get("ineffective", 0)) / HINTS["total"]
            if HINTS.get("total") else None),
            help="「有效+无效」占全部提示的比例。**未判定单列、不计入分母**；"
                 "这里的「有效」是系统按节点/状态变化的启发式判定，"
                 "不是教学效果的验证")

st.divider()
with st.expander('当前书面错误证据与复核状态', expanded=(view == VIEW_WRONG)):
    st.caption('独立于历史mistakes统计；只读服务端当前书面证据。确认者是本地操作者自报身份，不冒充教师认证或物理语义验证。')
    if CURRENT_ERRORS.get('review_store_status') != 'ready':
        st.warning('当前错误证据读取或复核状态不可得；不能解释成没有错误。')
        st.caption(CURRENT_ERRORS.get('error', '复核存储暂不可读'))
    else:
        _counts = CURRENT_ERRORS.get('counts') or {}
        st.caption(f"本次覆盖{CURRENT_ERRORS.get('attempt_count', 0)}次作答，其中{CURRENT_ERRORS.get('legacy_attempt_count', 0)}次旧口径记录不具备当前书面证据结构；下方数字不包含它们。")
        _columns = st.columns(4)
        for _column, (_key, _label) in zip(_columns, [('unreviewed','待核查'),('confirmed','本地确认'),('dismissed','已撤销'),('resolved','已解决')]):
            _column.metric(_label, _counts.get(_key, 0))
        st.caption('已撤销、已解决保留供追溯，不计当前活动错误；旧口径作答不强行转换成书面证据。')
        _evidence = CURRENT_ERRORS.get('evidence') or []
        if _evidence:
            st.dataframe([{'学生': e['student_id'], '作答': e['attempt_id'], '题目': e['problem_id'],
                           '证据ID': e['evidence_id'], '知识点': e['concept'], '书面问题': e['issue'],
                           '学生原文': e['student_text'], '来源': e.get('source_id'), '修订': e.get('source_revision'),
                           '复核状态': e['review_status'], '当前活动': e['active'],
                           '本地操作者': (e.get('review_decision') or {}).get('actor_label'),
                           '复核说明': (e.get('review_decision') or {}).get('note')} for e in _evidence],
                         hide_index=True, width='stretch')
        else:
            st.caption('当前范围没有可展示的书面错误证据；不代表所有作答已经验证正确。')

st.divider()
st.caption(
    "教师端历史数据只读、**不调用模型**；本地核对决定通过服务端显式保存。"
    "原题、逐轮诊断、OCR修订、保留图片与报告版本可在作答证据下钻查看。"
)
