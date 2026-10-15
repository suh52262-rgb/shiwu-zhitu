"""Read-only local teacher evidence API; ownership is not formal teacher login."""
from datetime import date
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from app.services import teacher_read

router = APIRouter(prefix='/teacher', tags=['teacher'])


@router.get('/environment')
def environment():
    try:
        return teacher_read.environment()
    except Exception as exc:
        raise HTTPException(503, '环境身份暂不可核对') from exc


@router.get('/error-evidence')
def error_evidence(start: date | None = None, end: date | None = None,
                   student_ids: list[str] | None = Query(None), empty_scope: bool = False,
                   attempt_id: str | None = None, problem_id: str | None = None):
    if start and end and start > end:
        raise HTTPException(422, '开始日期不能晚于结束日期')
    try:
        return teacher_read.scoped_errors(start=start, end=end,
            student_ids=[] if empty_scope else student_ids, attempt_id=attempt_id, problem_id=problem_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, '错误证据读取失败，不能解释成没有记录') from exc


@router.get('/attempts/{attempt_id}/evidence')
def attempt_evidence(attempt_id: str, student_id: str, problem_id: str):
    try:
        context = teacher_read.load_context(attempt_id, student_id, problem_id)
        return teacher_read.evidence(context)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, '作答证据暂不可得') from exc


@router.get('/attempts/{attempt_id}/images/{evidence_id}')
def image(attempt_id: str, evidence_id: str, student_id: str, problem_id: str):
    context = teacher_read.load_context(attempt_id, student_id, problem_id)
    item = next((item for item in teacher_read.image_catalogue(context) if item['evidence_id'] == evidence_id), None)
    if item is None:
        raise HTTPException(404, '图片证据不存在或不属于此作答')
    return FileResponse(item['_path'])
