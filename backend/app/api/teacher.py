"""P24 Teacher Mode 后端接口。

教师端要回答的问题，和学生学习端完全不同：
  · 不是"这个学生这一步对不对"，而是"**这一届学生在哪一步集体卡住**"
  · 不是"这道题的答案是什么"，而是"**这道题的哪个知识点反复出错**"

所以本模块全是**聚合**——全项目此前没有一条 `GROUP BY`。

## 数据来源
| 问题 | 来源 | 为什么 |
|---|---|---|
| 学生卡在哪些节点 | `hint_records.node` | 提示使用记录，不证明认知卡顿 |
| 反复出错的知识点 | `student_profiles.weak_concepts` | 跨学生跨题累积的薄弱点 |
| 提示有没有用 | `hint_records.effective`（P22 三态） | 教师最该看的：**哪类提示对这个班没用** |
| 作答轨迹 | `analysis_traces` / `attempts.state` | 单次作答的复盘 |

## 为什么把 `effective` 的 NULL 单独报出来
`effective` 是三态（True/False/NULL=未判定，见 `app/agent/hint_effect.py`）。
教师端如果只报"有效率"，把 NULL 混进分母，会**因为没测到而显得效果差**。
所以这里 `effective` / `ineffective` / `undecided` **三个数一起给**。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.services.learning_profile import profile_signals,current_profile_signals
from app.db.postgres import get_engine
from app.db.tables import (  # noqa: F401
    AnalysisTrace,
    Attempt,
    HintRecord,
    StudentProfile,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/teacher", tags=["teacher"])


def _rows(stmt) -> list:
    try:
        with Session(get_engine()) as session:
            return session.execute(stmt).all()
    except Exception as exc:  # noqa: BLE001 教师端查询不该把异常抛给前端
        logger.warning("教师端查询失败：%s: %s", type(exc).__name__, exc)
        return []


# ---------------------------------------------------------------- 1. 学生

@router.get("/students/{student_id}")
def teacher_student(student_id: str) -> dict:
    """一个学生的全貌：作答记录、薄弱点、提示历史与效果。"""
    with Session(get_engine()) as session:
        profile = session.get(StudentProfile, student_id)
        attempts = session.execute(
            select(Attempt).where(Attempt.student_id == student_id)
        ).scalars().all()

        attempt_ids = [a.id for a in attempts]
        hints = []
        if attempt_ids:
            hints = session.execute(
                select(HintRecord).where(HintRecord.attempt_id.in_(attempt_ids))
            ).scalars().all()

    # 提示效果**三态分开数**——把"未判定"混进分母会让效果显得更差
    effective = sum(1 for h in hints if h.effective is True)
    ineffective = sum(1 for h in hints if h.effective is False)
    undecided = sum(1 for h in hints if h.effective is None)
    decided = effective + ineffective

    return {
        "student_id": student_id,
        "attempt_count": len(attempts),
        **current_profile_signals(
            [(a.id, a.state) for a in attempts], (profile.weak_concepts if profile else []) or [],student_id),
        "weak_concepts": list((profile.weak_concepts if profile else None) or []),
        "hints": {
            "total": len(hints),
            "by_level": _count_by(hints, lambda h: f"L{h.hint_level}"),
            "effective": effective,
            "ineffective": ineffective,
            "undecided": undecided,
            # 分母是**可判定的条数**，与 P22 的口径一致
            "effectiveness": (effective / decided) if decided else None,
        },
        "attempts": [
            {
                "attempt_id": a.id,
                "problem_id": getattr(a, "problem_id", ""),
                "status": getattr(a, "status", ""),
                "finished": bool((getattr(a, "state", None) or {}).get("finished")),
                "current_node": (getattr(a, "state", None) or {}).get("current_node", ""),
                "visited_nodes": (getattr(a, "state", None) or {}).get("visited_nodes", []),
            }
            for a in attempts
        ],
    }


# ---------------------------------------------------------------- 2. 单次作答

@router.get("/attempts/{attempt_id}")
def teacher_attempt(attempt_id: str) -> dict:
    """一次作答的完整复盘：状态投影 + 轨迹 + 提示及效果。"""
    with Session(get_engine()) as session:
        attempt = session.get(Attempt, attempt_id)
        if attempt is None:
            raise HTTPException(status_code=404, detail=f"作答 {attempt_id} 不存在")
        traces = session.execute(
            select(AnalysisTrace)
            .where(AnalysisTrace.attempt_id == attempt_id)
            .order_by(AnalysisTrace.created_at)
        ).scalars().all()
        hints = session.execute(
            select(HintRecord).where(HintRecord.attempt_id == attempt_id)
            .order_by(HintRecord.id)
        ).scalars().all()

    state = getattr(attempt, "state", None) or {}
    return {
        "attempt_id": attempt_id,
        "student_id": getattr(attempt, "student_id", ""),
        "problem_id": getattr(attempt, "problem_id", ""),
        "status": getattr(attempt, "status", ""),
        "state": state,
        "traces": [
            {
                "created_at": t.created_at.isoformat() if t.created_at else "",
                "problem_id": getattr(t, "problem_id", ""),
                "before_state": getattr(t, "before_state", None),
                "payload": getattr(t, "payload", None),
            }
            for t in traces
        ],
        "hints": [
            {
                "id": h.id,
                "level": h.hint_level,
                "node": h.node,
                "hint": h.hint,
                "issue_type": h.issue_type,
                # 三态原样返回：None 就是"未判定"，**不要在这里折叠成 False**
                "effective": h.effective,
                "evidence": h.effective_evidence,
            }
            for h in hints
        ],
    }


# ---------------------------------------------------------------- 3. 题目的班级统计

@router.get("/problems/{problem_id}/stats")
def teacher_problem_stats(problem_id: str) -> dict:
    """**这道题上，全班卡在哪、错在哪、什么提示没用。**"""
    # 提示使用节点；提示使用不能证明学生卡住
    stuck = _rows(
        select(HintRecord.node, func.count().label("n"))
        .join(Attempt, Attempt.id == HintRecord.attempt_id)
        .where(Attempt.problem_id == problem_id, HintRecord.node != "")
        .group_by(HintRecord.node)
        .order_by(func.count().desc())
        .limit(10)
    )
    # 反复出现的错误类型
    issues = _rows(
        select(HintRecord.issue_type, func.count().label("n"))
        .join(Attempt, Attempt.id == HintRecord.attempt_id)
        .where(Attempt.problem_id == problem_id, HintRecord.issue_type != "")
        .group_by(HintRecord.issue_type)
        .order_by(func.count().desc())
        .limit(10)
    )
    attempt_count = _rows(
        select(func.count()).select_from(Attempt)
        .where(Attempt.problem_id == problem_id)
    )
    # 提示效果：三态分开数
    total = _rows(
        select(func.count()).select_from(HintRecord)
        .join(Attempt, Attempt.id == HintRecord.attempt_id)
        .where(Attempt.problem_id == problem_id)
    )
    eff = _rows(
        select(HintRecord.effective, func.count().label("n"))
        .join(Attempt, Attempt.id == HintRecord.attempt_id)
        .where(Attempt.problem_id == problem_id)
        .group_by(HintRecord.effective)
    )

    by_effect = {("undecided" if r[0] is None else ("effective" if r[0] else "ineffective")): r[1]
                 for r in eff}
    decided = by_effect.get("effective", 0) + by_effect.get("ineffective", 0)
    return {
        "problem_id": problem_id,
        "attempt_count": attempt_count[0][0] if attempt_count else 0,
        "frequent_help_nodes": [{"node": r[0], "count": r[1]} for r in stuck],
        "frequent_stuck_nodes": [],  # Deprecated: cognitive stuck is not measured.
        "stuck_measurement_status": "not_validated",
        # **键名如实叫 `issue_type`**：装的确实是错误类型（concept/method/…）。
        # 任务书示例里写的是 `knowledge_point`，但那是示例，不是这个字段的内容——
        # 叫错名字会让 C 端按知识点去解读错误类型。
        # 知识点维度的统计在 `/teacher/knowledge-points/stats`。
        "frequent_issues": [{"issue_type": r[0], "count": r[1]} for r in issues],
        "hint_count": total[0][0] if total else 0,
        "hint_effect": {
            "effective": by_effect.get("effective", 0),
            "ineffective": by_effect.get("ineffective", 0),
            # **未判定单列**：混进分母会因"没测到"而显得效果差
            "undecided": by_effect.get("undecided", 0),
            "effectiveness": (by_effect.get("effective", 0) / decided) if decided else None,
        },
    }


# ---------------------------------------------------------------- 4. 知识点总览

@router.get("/knowledge-points/stats")
def teacher_knowledge_stats(limit: int = 20) -> dict:
    """跨题目的薄弱知识点排行。

    `weak_concepts` 是 JSON 数组，用 `json_array_elements_text` 展开后再聚合
    ——**这是全项目第一条针对 JSON 列的 GROUP BY**。
    """
    from sqlalchemy import text

    rows = _rows_raw(
        "SELECT concept, COUNT(DISTINCT student_id) AS students, COUNT(*) AS n "
        "FROM student_profiles, json_array_elements_text(weak_concepts) AS concept "
        "GROUP BY concept ORDER BY students DESC, n DESC LIMIT :lim",
        {"lim": limit},
    )
    # Read JSON through the Python decoder: historical OCR can contain escaped
    # NULs, which PostgreSQL ->> rejects even for an unrelated key. Keep originals.
    from collections import Counter, defaultdict
    mentions = Counter()
    people = defaultdict(set)
    from app.services.error_evidence_reviews import decisions
    try:review_decisions=decisions();review_status='ready'
    except Exception:review_decisions=[];review_status='unavailable'
    for sid, aid, state in _rows(select(Attempt.student_id, Attempt.id, Attempt.state)):
        signals = profile_signals([(aid, state)],decisions=review_decisions)
        for evidence in signals["error_evidence"]:
            if evidence['review_status'] in {'dismissed','resolved'} or review_status!='ready':continue
            concept = evidence["concept"]
            mentions[concept] += 1
            people[concept].add(sid)
    observed = [(concept, len(people[concept]), count)
                for concept, count in mentions.items()]
    observed.sort(key=lambda row: (-row[1], -row[2], row[0]))
    observed = observed[:max(0, min(limit, 1000))]
    students = _rows_raw("SELECT COUNT(*) FROM student_profiles", {})
    return {
        "observed_error_knowledge_points": [
            {"knowledge_point": r[0], "student_count": r[1], "mention_count": r[2],
             "judgement_source": "model_diagnosis", "semantic_validated": False} for r in observed],
        'review_store_status':review_status,
        "data_status": "unverified_historical_signal",
        "description": "历史画像提及次数，不是已确认的薄弱或反复出错次数",
        "student_count": students[0][0] if students else 0,
        "knowledge_points": [
            {"knowledge_point": r[0], "student_count": r[1], "mention_count": r[2]}
            for r in rows
        ],
    }


# ---------------------------------------------------------------- 工具

def _count_by(items, key) -> dict:
    out: dict[str, int] = {}
    for it in items:
        k = key(it)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _rows_raw(sql: str, params: dict) -> list:
    from sqlalchemy import text

    try:
        with Session(get_engine()) as session:
            return list(session.execute(text(sql), params).all())
    except Exception as exc:  # noqa: BLE001
        logger.warning("教师端聚合失败：%s: %s", type(exc).__name__, exc)
        return []
