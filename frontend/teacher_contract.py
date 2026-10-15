"""统一只读聚合契约 —— **历史统计口径来源；当前有效错误复用后端生产证据规则**。

## 为什么需要这个模块

审阅第三批原文：
> 「先**冻结统一只读聚合契约**；新增后端统计接口复用生产证据规则，
>   教师页面调用接口，**避免把规则再复制进 SQL**」

问题背景：教师端的统计分别来自知识点 / 提示 / 错题 / 人数等多条查询。
如果每条 SQL 各写一遍"什么算错误""什么算历史提及"，
**迟早出现两处口径不一致** —— 而且不一致时界面看起来完全正常。

（本项目已经踩过同类坑：`_has_mistakes` 曾把 `mistakes` 与 `weak_concepts`
混算，导致"历史上弱过"的学生每道题都被算成有问题。）

## 本模块做什么

1. **冻结口径**：`SIGNALS` / `METRICS` 用数据结构把每条口径写死，
   附上"它是什么 / 不是什么"的说明。
2. **单一来源**：SQL 谓词（`sql_error_records` / `sql_weak_mention` 等）
   只在本文件定义，`teacher_data.py` 引用它 —— **不再各写一遍**。
3. **可追溯**：`contract_fingerprint()` 给出契约指纹。
   统计结论带上它，就能回答"这个数字是按哪一版口径算的"。

## 边界（不夸大）

本包定位是**不改后端**，所以**没有**新增后端统计接口
（审阅设想的"教师页面调用接口"那一半做不到）。
本模块实现的是**前端侧的口径冻结与集中**：
在后端契约缺位的前提下，至少保证**本包内部**口径唯一。

> 若将来后端提供统一只读聚合接口，只需让 `teacher_data.py` 改为调用它，
> 本模块继续作为**口径对照表**（用来核对后端返回的口径是否与本包一致）。
"""
from __future__ import annotations

import hashlib
from typing import Final

# ================================================================ 信号

#: 两类**本质不同**的信号。审阅第一批 #4 要求「分开展示」。
SIGNAL_HISTORICAL = "historical_mention"
SIGNAL_ERROR_RECORD = "model_error_record"

SIGNALS: Final[dict[str, dict]] = {
    SIGNAL_HISTORICAL: {
        "name": "历史提及",
        "source": "attempts.weak_concepts / student_profiles.weak_concepts",
        "nature": "跨时间累计的知识点并集（**只增不减**）",
        "is": ["历史上被标记过什么知识点"],
        "is_not": [
            "不是已确认的当前薄弱点",
            "不是反复出错次数",
            "不含撤销 / 已解决状态",
            "未经语义验证",
        ],
        "time_scope": "lifetime",   #: 跨时累计 —— 见 WEAK_SCOPE_*
    },
    SIGNAL_ERROR_RECORD: {
        "name": "模型记录的过程错误",
        "source": "attempts.mistakes",
        "nature": "每次作答过程中模型记录的错误条目（含节点与性质）",
        "is": ["错在哪一步、错成什么性质"],
        "is_not": [
            "不含知识点字段（`mistakes` 里没有该字段，故无法按知识点聚合）",
            "不等于最终答案错误（演示数据 `final_correct` 可能全为 true）",
        ],
        "time_scope": "range",      #: 随筛选范围
    },
}

# ================================================================ 指标口径

#: 每个指标：来源、判据、时间口径、以及**容易读错的地方**。
METRICS: Final[dict[str, dict]] = {
    "students": {
        "name": "覆盖学生数",
        "sql": "count(DISTINCT a.student_id) FROM attempts a WHERE {where}",
        "time_scope": "range",
        "note": "范围内**有过作答**的去重学生数（不是表规模）",
    },
    "attempts": {
        "name": "作答次数",
        "sql": "count(*) FROM attempts a WHERE {where}",
        "time_scope": "range",
        "note": "",
    },
    "problem_attempts": {
        "name": "有过程错误记录的作答",
        "sql": "sum(CASE WHEN json_array_length(COALESCE(a.mistakes,'[]'::json)) > 0 "
               "THEN 1 ELSE 0 END) FROM attempts a WHERE {where}",
        "time_scope": "range",
        "note": "**只看 mistakes**（过程证据）。曾错误地 OR 上 weak_concepts，"
                "导致历史弱过的学生每道题都被算成有问题",
    },
    "weak_flagged": {
        "name": "被标记过弱项的作答",
        "sql": "sum(CASE WHEN json_array_length(COALESCE(a.weak_concepts,'[]'::json)) > 0 "
               "THEN 1 ELSE 0 END) FROM attempts a WHERE {where}",
        "time_scope": "range",
        "note": "**只看 weak_concepts**。与上一项**必须分开**，不得相加",
    },
    "covered_students": {
        "name": "弱项覆盖学生数（精确）",
        "sql": "count(DISTINCT a.student_id) FROM attempts a WHERE {where} "
               "AND json_array_length(COALESCE(a.weak_concepts,'[]'::json)) > 0",
        "time_scope": "range",
        "note": "**独立 COUNT(DISTINCT)**，不能拿「各知识点学生数的最大值」代替"
                "（实测那样会少算：2 个不重叠学生被算成 1 人）",
    },
    "hint_total": {
        "name": "提示条数",
        "sql": "count(*) FROM hint_records h JOIN attempts a ON h.attempt_id = a.id "
               "WHERE {where}",
        "time_scope": "range",
        "note": "",
    },
    "hint_effective": {
        "name": "提示判定 — 有效",
        "sql": "sum(CASE WHEN h.effective THEN 1 ELSE 0 END) ... WHERE {where}",
        "time_scope": "range",
        "note": "`effective` 是**系统按节点/状态变化的启发式判定**，"
                "不是教学因果效果。分母用**可判定**条数",
    },
    "hint_undecided": {
        "name": "提示判定 — 未判定",
        "sql": "sum(CASE WHEN h.effective IS NULL THEN 1 ELSE 0 END) ... WHERE {where}",
        "time_scope": "range",
        "note": "**单列，不计入分母**。当成「无效」会系统性低估提示效果",
    },
}

# ================================================================ 时间口径

#: 与 `teacher_logic.WEAK_SCOPE_*` 保持同义（此处再声明一遍便于契约自洽）。
SCOPE_RANGE = "range"
SCOPE_LIFETIME = "lifetime"

SCOPE_RULES: Final[dict[str, str]] = {
    SCOPE_RANGE: "按当前筛选范围过滤 —— 可与同范围指标（作答数等）同句陈述",
    SCOPE_LIFETIME: "跨时间累计 —— **不得**与当前范围指标拼进同一段诊断",
}

# ================================================================ SQL 谓词

#: 两个信号各自的**物理列名** —— 只在这里出现一次。
#: 换字段名（例如后端做了迁移）时只改这里，不必满文件搜。
COL_MISTAKES: Final[str] = "mistakes"
COL_WEAK_CONCEPTS: Final[str] = "weak_concepts"


def sql_json_len(alias: str, column: str = COL_MISTAKES) -> str:
    """JSON 数组的长度表达式（`json_array_length(COALESCE(<alias>.<col>,'[]'::json))`）。

    这是**所有**信号判据的原子构造 —— 谓词、求和、取值都基于它。
    集中在此的目的：`mistakes` / `weak_concepts` 的取值方式**只有一种写法**，
    不会出现"某处多写了个 COALESCE、空数组时整条查询 NULL 化"这类不一致。
    """
    if column not in (COL_MISTAKES, COL_WEAK_CONCEPTS):
        raise ValueError(f"未知信号列：{column!r}")
    return f"json_array_length(COALESCE({alias}.{column},'[]'::json))"


def sql_json_elements(alias: str, column: str = COL_MISTAKES) -> str:
    """JSON 数组展开为行（用于 `string_agg` 取错因 / 节点）。"""
    if column not in (COL_MISTAKES, COL_WEAK_CONCEPTS):
        raise ValueError(f"未知信号列：{column!r}")
    return f"json_array_elements(COALESCE({alias}.{column},'[]'::json))"


def sql_json_elements_text(alias: str, column: str = COL_WEAK_CONCEPTS) -> str:
    """JSON 数组展开为 text（用于 `string_agg` 拼知识点名）。"""
    if column not in (COL_MISTAKES, COL_WEAK_CONCEPTS):
        raise ValueError(f"未知信号列：{column!r}")
    return f"json_array_elements_text(COALESCE({alias}.{column},'[]'::json))"


def sql_error_records(alias: str = "a") -> str:
    """「本次作答有过程错误记录」的 SQL 谓词。

    契约项 `problem_attempts`。**只看 `mistakes`** ——
    不含 `weak_concepts`（那是跨时累计信号，见 SIGNALS）。
    """
    return f"{sql_json_len(alias, COL_MISTAKES)} > 0"


def sql_weak_mention(alias: str = "a") -> str:
    """「本次作答被标记过弱项」的 SQL 谓词。契约项 `weak_flagged`。"""
    return f"{sql_json_len(alias, COL_WEAK_CONCEPTS)} > 0"


def sql_covered_students(alias: str = "a") -> str:
    """「因弱项被覆盖」的 WHERE 谓词。契约项 `covered_students`。

    ⚠️ **返回的是谓词片段，不是完整查询。**
    指标本身是 `count(DISTINCT <alias>.student_id)` —— 那属于 SELECT 列表，
    不能塞进谓词里（写成本函数的旧版返回
    `count(DISTINCT a.student_id) AND json_array_length(...) > 0`，
    作为 WHERE 片段是**非法 SQL**）。故此处只返回过滤条件，
    调用方负责在外层写 `count(DISTINCT ...)`。

    它与 `sql_weak_mention` 的**过滤条件相同**，区别只在外层聚合方式；
    契约里保留独立函数是为了让「覆盖面必须独立 COUNT(DISTINCT)」
    这条要求在代码里有明确落点（见 `METRICS['covered_students']['note']`）。
    """
    return sql_weak_mention(alias)


# ================================================================ 指纹

def contract_fingerprint() -> str:
    """契约指纹 —— 口径定义变化就变。

    统计结论带上它，就能回答"这个数字是按哪一版口径算的"。
    与 `teacher_logic.scope_fingerprint()`（范围指纹）互补：
      · 范围指纹 → 「同一批数字是否**同范围**」
      · 契约指纹 → 「这批数字是按**哪版口径**算的」
    """
    payload = "|".join([
        *sorted(SIGNALS),
        *sorted(METRICS),
        *[f"{k}:{v['sql']}" for k, v in sorted(METRICS.items())],
        *[f"{k}:{v}" for k, v in sorted(SCOPE_RULES.items())],
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


#: 当前契约版本（人工维护；口径有实质变更时递增）
CONTRACT_VERSION = "1.0"
