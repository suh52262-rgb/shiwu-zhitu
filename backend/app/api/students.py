"""学生接口：学习画像与作答历史（Phase 7/8）。"""
from fastapi import APIRouter, HTTPException

from app.db import persistence

router = APIRouter(prefix="/students", tags=["students"])


@router.get("/{student_id}/profile")
def get_student_profile(student_id: str) -> dict:
    """学生长期画像（薄弱知识点）+ 全部作答记录概要。"""
    try:
        profile = persistence.get_student_profile(student_id)
    except persistence.PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"PostgreSQL 不可用：{exc}") from exc
    if profile is None:
        raise HTTPException(status_code=404, detail=f"学生 {student_id} 不存在")
    return profile
