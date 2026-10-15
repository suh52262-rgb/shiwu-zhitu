"""Tutor 对话接口（占位，Phase 3 之后视需要实现）。"""
from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat_with_tutor(payload: dict) -> dict:
    """自由对话入口。TODO：与分级提示（hint）流程区分，后续阶段设计。"""
    return {"status": "not_implemented", "note": "Tutor 对话将在后续阶段实现"}
