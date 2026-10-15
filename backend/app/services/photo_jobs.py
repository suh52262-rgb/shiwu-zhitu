"""Photos use the durable queue machinery; recognition is visible before teaching ends."""
import threading
from pathlib import Path
from fastapi import HTTPException
from app.services.video_jobs import VideoJobQueue


def _run_photo(file, **params):
    from app.api.video import analyze_image
    params.pop('_requested_attempt_id', None)
    return analyze_image(file=file, **params)


class PhotoJobQueue(VideoJobQueue):
    def __init__(self, root, capacity=2, runner=None):
        super().__init__(root, capacity, runner or _run_photo)

    def _upload(self, data):
        return super()._upload(data).with_name('source.image')

    @staticmethod
    def _same_params(existing, params):
        def original(value):
            value = dict(value)
            value['attempt_id'] = value.pop('_requested_attempt_id', value.get('attempt_id', ''))
            return value
        return original(existing.get('params') or {}) == original(params)

    def prepare_params(self, params):
        if self.runner is not _run_photo:  # injected isolated test runner
            return params
        from app.api.attempts import _resolve_attempt, AnalyzeRequest, get_attempt_store
        from app.services.problem_repository import get_problem_repository
        store = get_attempt_store()
        lock = store.attempt_lock(params['student_id'], params['problem_id'])
        # Never wait for an analysis while holding the queue lock: its progress
        # callback may need this same queue lock.
        if not lock.acquire(blocking=False):
            raise HTTPException(409, '当前作答仍在处理中，请等待后提交照片')
        try:
            problem = get_problem_repository().get_problem(params['problem_id'])
            if problem is None:
                raise HTTPException(404, '题目不存在')
            record = _resolve_attempt(AnalyzeRequest(student_id=params['student_id'],
                problem_id=params['problem_id'], attempt_id=params['attempt_id'], answer=''), problem)
            return dict(params, _requested_attempt_id=params['attempt_id'], attempt_id=record.attempt_id)
        finally:
            lock.release()

    def _progress(self, job_id, phase, progress):
        progress = dict(progress)
        recognized = progress.pop('recognized_text', None)
        with self.lock:
            if recognized is not None:
                data = self._read(job_id)
                if data['status'] == 'running':
                    data['recognized_text'] = recognized
                    self._write(data)
            return super()._progress(job_id, phase, progress)

    def get(self, job_id, student_id, problem_id):
        try:
            return super().get(job_id, student_id, problem_id)
        except HTTPException as exc:
            raise HTTPException(404 if exc.status_code == 403 else exc.status_code,
                                str(exc.detail).replace('视频', '照片')) from exc

    def submit(self, *args, **kwargs):
        try:
            return super().submit(*args, **kwargs)
        except HTTPException as exc:
            raise HTTPException(exc.status_code, str(exc.detail).replace('视频', '照片')) from exc


_queues = {}
_lock = threading.Lock()


def get_photo_jobs():
    from app.api.video import PROJECT_ROOT
    root = PROJECT_ROOT / 'data' / 'photo_jobs'
    with _lock:
        key = str(root.resolve())
        if key not in _queues:
            _queues[key] = PhotoJobQueue(root)
        return _queues[key]
