"""Per-request progress hooks; synchronous callers are unaffected."""
from contextvars import ContextVar
import logging

logger = logging.getLogger(__name__)


progress_callback: ContextVar = ContextVar("video_progress_callback", default=None)
owned_upload_callback: ContextVar = ContextVar("video_owned_upload_callback", default=None)


def register_owned_upload(path) -> None:
    """Persist ownership before creating a second original upload; failures must propagate."""
    callback = owned_upload_callback.get()
    if callback is not None:
        callback(path)


def perception_progress_callback():
    """Capture the request callback before vision's thread pool loses ContextVars."""
    callback = progress_callback.get()

    def report(phase, _message="", **counts):
        if callback is None:
            return
        mapped = {"extracting": "extracting", "analyzing": "recognizing", "exporting": "exporting"}.get(phase)
        if mapped is None:
            return
        progress = {"completed": counts.get("analyze_done", 0),
                    "total": counts.get("analyze_total"),
                    "unit": "frames" if phase == "analyzing" else "stage"}
        try:
            callback(mapped, progress)
        except Exception:
            logger.exception("视频感知进度记录失败，分析继续")

    return report


def report_progress(phase: str, completed: int = 0, total: int | None = None,
                    unit: str = "stage", *, attempt_id: str = "") -> None:
    callback = progress_callback.get()
    if callback is not None:
        try:
            callback(phase, {"completed": completed, "total": total, "unit": unit,
                             **({"attempt_id": attempt_id} if attempt_id else {})})
        except Exception:
            # A progress-storage failure must not invalidate an already committed round.
            logger.exception("视频任务进度记录失败，分析继续")
