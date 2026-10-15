from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from app.api import video
from app.services.photo_jobs import get_photo_jobs
from app.services.video_jobs import normalize_request_id

router = APIRouter(prefix='/photo/jobs', tags=['photo'])


@router.post('')
def create(file: UploadFile = File(...), student_id: str = Form(...),
           problem_id: str = Form(...), attempt_id: str = Form(''),
           request_hint: bool = Form(True), client_request_id: str = Form(...)):
    # A known UUID must replay even if its attempt has since finished. New jobs
    # validate and bind the target under the store lock in prepare_params.
    if not normalize_request_id(client_request_id):
        raise HTTPException(422, '照片任务需要有效的上传UUID')
    return get_photo_jobs().submit(file, dict(student_id=student_id, problem_id=problem_id,
        attempt_id=attempt_id, request_hint=request_hint, include_review=False),
        video.MAX_UPLOAD_BYTES, client_request_id)


@router.get('/by-request/{request_id}')
def find(request_id: str, student_id: str = Query(..., min_length=1),
         problem_id: str = Query(..., min_length=1)):
    return get_photo_jobs().get_by_request(request_id, student_id, problem_id)


@router.get('/{job_id}')
def get(job_id: str, student_id: str = Query(..., min_length=1),
        problem_id: str = Query(..., min_length=1)):
    return get_photo_jobs().get(job_id, student_id, problem_id)
