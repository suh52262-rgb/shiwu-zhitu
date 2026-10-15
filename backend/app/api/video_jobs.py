"""Refreshable background video jobs; ownership follows the existing local API model."""
from fastapi import APIRouter, File, Form, UploadFile, Query

from app.api import video
from app.services.video_jobs import get_video_jobs

router = APIRouter(prefix="/video/jobs", tags=["video"])


@router.post("")
def create_video_job(file: UploadFile = File(...), problem_id: str = Form(...),
                     student_id: str = Form(...), every_sec: float = Form(20.0),
                     attempt_id: str = Form(""), request_hint: bool = Form(False),
                     include_review: bool = Form(False),
                     client_request_id: str = Form("")) -> dict:
    """提交视频任务（契约 §2.2）。

    `client_request_id` 由页面在**一次上传动作开始时**生成：
    同一编号 + 相同内容与参数 ⇒ **返回原任务**（不产生第二次教学或模型调用）；
    内容或参数不同 ⇒ **409**。
    """
    video._validate_media_target(student_id, problem_id, attempt_id)
    return get_video_jobs().submit(file, {"student_id": student_id, "problem_id": problem_id,
        "every_sec": every_sec, "attempt_id": attempt_id, "request_hint": request_hint,
        "include_review": include_review}, video.MAX_UPLOAD_BYTES, client_request_id)


# ⚠️ 声明在 `/{job_id}` **之前**。虽然两条路径的段数不同、实际不会撞，
# 但"先声明更具体的"是免费的习惯 —— 哪天有人把 job_id 改成可选路径参数，
# 顺序就成了唯一的区别。
@router.get("/by-request/{client_request_id}")
def get_video_job_by_request(client_request_id: str,
                             student_id: str = Query(..., min_length=1),
                             problem_id: str = Query(..., min_length=1)) -> dict:
    """按上传编号找回任务 —— 创建请求超时后的**唯一正确动作**。

    契约 §2.2：超时（结果未知）时页面**先按编号查询，不得直接重传**。
    归属不符 ⇒ **404**（不是 403：403 会说漏"这个编号存在"）。
    """
    return get_video_jobs().get_by_request(client_request_id, student_id, problem_id)


@router.get("/{job_id}")
def get_video_job(job_id: str, student_id: str = Query(..., min_length=1),
                  problem_id: str = Query(..., min_length=1)) -> dict:
    return get_video_jobs().get(job_id, student_id, problem_id)
