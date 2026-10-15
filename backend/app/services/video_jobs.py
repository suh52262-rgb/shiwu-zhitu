"""Bounded single-process video queue with durable progress, never automatic paid retry.

Run one application process. On process restart unfinished jobs become interrupted;
the teaching attempt may contain partial results and must be inspected before retry.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.services.graph_versions import atomic_json
from app.services.video_progress import progress_callback, owned_upload_callback

logger = logging.getLogger(__name__)
FINAL = {"succeeded", "failed", "interrupted"}

#: 上传编号必须是 UUID（契约 §2.2）。大小写不敏感，存的时候统一小写。
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                      r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def normalize_request_id(value) -> str:
    """校验并归一化 `client_request_id`；留空 = 本次不参与幂等。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if _UUID_RE.fullmatch(text) is None:
        raise HTTPException(422, "client_request_id 必须是 UUID")
    return text.lower()


def consume_upload(file: UploadFile, max_bytes: int,
                   target: Path | None = None) -> tuple[str, int]:
    """读一遍上传，**顺带算内容哈希** —— 判「是不是同一次上传」靠它。

    `target` 非空时同时落盘（正常创建那条路）；为空时**只哈希不写盘**：
    重复提交不该再占一份磁盘，更不该建第二个任务。
    """
    digest = hashlib.sha256()
    count = 0
    handle = target.open("wb") if target is not None else None
    try:
        while chunk := file.file.read(1024 * 1024):
            count += len(chunk)
            if count > max_bytes:
                raise HTTPException(413, "视频超过上传大小上限")
            digest.update(chunk)
            if handle is not None:
                handle.write(chunk)
    finally:
        if handle is not None:
            handle.close()
    if not count:
        raise HTTPException(400, "上传的文件是空的")
    return digest.hexdigest(), count


class VideoJobQueue:
    def __init__(self, root: Path, capacity: int = 3, runner=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(capacity)
        self.pending = queue.Queue()
        self.runner = runner
        self.worker = None
        self._recover()

    def _path(self, job_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise HTTPException(422, "无效的视频任务编号")
        target = (self.root / f"{job_id}.json").resolve()
        if target.parent != self.root:
            raise HTTPException(403, "视频任务路径不合法")
        return target

    def _read(self, job_id: str) -> dict:
        path = self._path(job_id)
        if not path.is_file():
            raise HTTPException(404, "视频任务不存在")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or data.get("job_id") != job_id
                    or not isinstance(data.get("status"), str)
                    or data["status"] not in FINAL | {"queued", "running"}):
                raise ValueError("bad job state")
            return data
        except (ValueError, OSError) as exc:
            raise HTTPException(409, "视频任务状态损坏") from exc

    def _write(self, data: dict):
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self._path(data["job_id"]), data)

    def _upload(self, data):
        folder = (self.root / data["job_id"]).resolve()
        if folder.parent != self.root:
            raise HTTPException(403, "视频暂存路径不合法")
        path = (folder / "source.video").resolve()
        if path.parent != folder:
            raise HTTPException(403, "视频暂存路径不合法")
        return path

    def _cleanup(self, data):
        try:
            path = self._upload(data)
            path.unlink(missing_ok=True)
            data["spool_removed"] = not path.exists()
            if path.parent.exists():
                path.parent.rmdir()
        except Exception:
            data["spool_removed"] = False
            logger.exception("视频任务暂存清理失败")
        # New jobs reliably register inner uploads before any write. Legacy
        # running/failed jobs cannot prove whether an untracked copy remains.
        known_absent = data.get("analysis_upload_registered") is False or data.get("status") == "queued"
        name = data.get("analysis_upload_name")
        inner_removed = True if known_absent else None
        if name is not None:
            try:
                from app.api.video import UPLOAD_DIR
                if not isinstance(name, str) or re.fullmatch(r"[0-9a-f]{32}\.[a-zA-Z0-9]{1,12}", name) is None:
                    raise ValueError("invalid registered upload")
                root = UPLOAD_DIR.resolve()
                target = (root / name).resolve()
                if target.parent != root:
                    raise ValueError("upload escapes root")
                target.unlink(missing_ok=True)
                inner_removed = not target.exists()
            except Exception:
                inner_removed = False
                logger.exception("视频分析原上传清理失败")
        result = data.get("result")
        inner_report = result.get("upload_removed") if isinstance(result, dict) else None
        data["analysis_upload_cleanup_verified"] = inner_removed
        # Preserve an explicit inner cleanup failure instead of overwriting it
        # with successful spool removal (even if a later retry removed the file).
        if inner_report is False:
            inner_removed = False
        elif inner_removed is None and inner_report is True:
            inner_removed = True
        data["analysis_upload_removed"] = inner_removed
        data["upload_removed"] = data.get("spool_removed") is True and inner_removed is True

    def _recover(self):
        for path in self.root.glob("*.json"):
            if re.fullmatch(r"[0-9a-f]{32}", path.stem) is None:
                continue
            try:
                data = self._read(path.stem)
                if data["status"] in {"queued", "running"}:
                    if data["status"] == "queued":
                        data.setdefault("analysis_upload_registered", False)
                    data.update(status="interrupted", phase="interrupted", result=None,
                                error={"status_code": 409, "message": "服务已重启，任务中断；可能已保存部分作答，请先查看作答记录。不会自动重跑。"})
                    self._cleanup(data)
                    self._write(data)
            except (HTTPException, OSError):
                logger.exception("恢复视频任务失败（%s）", path.name)

    @staticmethod
    def public(data):
        return {key: value for key, value in data.items()
                if key not in {"params", "filename", "content_type", "content_sha256",
                               "analysis_upload_name", "analysis_upload_registered"}}

    def get(self, job_id: str, student_id: str, problem_id: str):
        with self.lock:
            data = self._read(job_id)
            if data.get("student_id") != student_id or data.get("problem_id") != problem_id:
                raise HTTPException(403, "视频任务不属于指定学生和题目")
            return self.public(data)

    def _find_by_request(self, client_request_id: str) -> dict | None:
        """按上传编号找任务 —— **扫描任务文件，不维护第二份索引**。

        ⚠️ 为什么不做一张 `编号 → 任务号` 的索引表：那会多一个**真相源**，
        而它一定会和任务文件漂开（本项目的教训：两处默认值、各建一次的材料、
        生成与缓存两套判据）。任务文件本来就是持久的，
        **把它当唯一真相，重启后自然还在。**

        代价是每次按编号查询要扫一遍目录。任务文件都很小、数量是演示量级，
        而这条路径只在"创建结果未知"时才走 —— **这个代价换来的是不会漂。**
        """
        with self.lock:
            for path in sorted(self.root.glob("*.json")):
                if re.fullmatch(r"[0-9a-f]{32}", path.stem) is None:
                    continue
                try:
                    data = self._read(path.stem)
                except HTTPException:
                    continue        # 坏文件跳过，**不让它挡住找回**
                if data.get("client_request_id") == client_request_id:
                    return data
        return None

    @staticmethod
    def _same_params(existing: dict, params: dict) -> bool:
        """参数是否完全一致 —— 「内容或参数不同 ⇒ 409」里的那一半。"""
        return dict(existing.get("params") or {}) == dict(params or {})

    def get_by_request(self, client_request_id: str, student_id: str, problem_id: str):
        """按上传编号找回任务（契约 §2.2）。

        ⚠️ **归属不符返回 404，不是 403** —— 403 等于告诉对方"这个编号存在，
        只是不属于你"，那是**泄露存在性**。
        """
        request_id = normalize_request_id(client_request_id)
        if not request_id:
            raise HTTPException(422, "缺少上传编号")
        data = self._find_by_request(request_id)
        if (data is None or data.get("student_id") != student_id
                or data.get("problem_id") != problem_id):
            raise HTTPException(404, "没有这个编号对应的视频任务")
        return self.public(data)

    def _replay_if_known(self, request_id: str, params: dict, file: UploadFile,
                         max_bytes: int) -> dict | None:
        """这个编号是不是已经交过了？交过就**不占槽、不建任务**地回放原任务。

        ⚠️ 顺序是刻意的：**参数不同立刻 409，连上传都不读**；
        参数相同才读一遍算哈希（此时也**只哈希、不落盘**）。
        ⇒ 重复提交不占队列容量，所以**队列满时重复提交也能拿回原任务**，
          而不是收到一个会误导人的 429。
        """
        existing = self._find_by_request(request_id)
        if existing is None:
            return None
        if not self._same_params(existing, params):
            raise HTTPException(409, "这个上传编号已用于不同的题目或参数")
        digest, _ = consume_upload(file, max_bytes)          # 只哈希，不落盘
        if digest != existing.get("content_sha256"):
            raise HTTPException(409, "这个上传编号已用于另一份视频内容")
        logger.info("上传编号 %s 重复提交，回放原任务 %s", request_id, existing["job_id"])
        return self.public(existing.copy())

    def prepare_params(self, params: dict) -> dict:
        """Called once for a new upload under the queue lock, before execution."""
        return params

    def submit(self, file: UploadFile, params: dict, max_bytes: int,
               client_request_id: str = ""):
        request_id = normalize_request_id(client_request_id)
        if request_id:
            replay = self._replay_if_known(request_id, params, file, max_bytes)
            if replay is not None:
                return replay
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, "视频任务队列已满，请稍后重试")
        data = {"job_id": uuid.uuid4().hex, "student_id": params["student_id"],
                "problem_id": params["problem_id"], "attempt_id": params["attempt_id"],
                "status": "queued", "phase": "queued", "progress": {"completed": 0, "total": None, "unit": "stage"},
                "result": None, "error": None, "params": params,
                "filename": Path(file.filename or "video.mp4").name,
                "content_type": file.content_type or "video/mp4", "upload_removed": False,
                "analysis_upload_registered": False,
                "client_request_id": request_id or None,
                "created_at": datetime.now(timezone.utc).isoformat()}
        released = False
        try:
            path = self._upload(data)
            path.parent.mkdir()
            digest, _ = consume_upload(file, max_bytes, path)
            data["content_sha256"] = digest
            with self.lock:
                # ⚠️ **提交前再查一次**：两个并发请求带同一个编号时，
                # 上面那次预查可能都没查到，于是两边都会走到这里。
                # ⇒ 拿着锁再查一遍，晚到的那个回滚自己，回放先到的。
                if request_id:
                    racing = self._find_by_request(request_id)
                    if racing is not None:
                        self._cleanup(data)
                        self.slots.release()
                        released = True
                        # ⚠️ **内容与参数都要比。**
                        # 只比内容哈希的话，"同编号 + 同文件 + **不同参数**"的并发请求
                        # 会拿回原任务 —— 而它本该是 409（它要的是另一次处理）。
                        # 预查那条路（`_replay_if_known`）本来就比了参数，
                        # 这里漏掉就**只在这一条路上漏**，最难发现。
                        if (not self._same_params(racing, params)
                                or racing.get("content_sha256") != digest):
                            raise HTTPException(
                                409, "这个上传编号已用于另一次提交（内容或参数不同）")
                        logger.info("上传编号 %s 并发重复提交，回放原任务 %s",
                                    request_id, racing["job_id"])
                        return self.public(racing.copy())
                data['params'] = self.prepare_params(data['params'])
                data['attempt_id'] = data['params']['attempt_id']
                self._write(data)
                response = self.public(data.copy())
                self.pending.put(data["job_id"])
                if self.worker is None:
                    self.worker = threading.Thread(target=self._work, name="video-worker", daemon=True)
                    self.worker.start()
                return response
        except BaseException:
            if not released:
                self._cleanup(data)
                self.slots.release()
            raise

    def _progress(self, job_id, phase, progress):
        with self.lock:
            data = self._read(job_id)
            if data["status"] != "running":
                return
            progress = dict(progress)
            attempt_id = progress.pop("attempt_id", "")
            if attempt_id:
                data["attempt_id"] = attempt_id
            previous = data.get("progress") or {}
            if (data.get("phase") == phase and previous.get("unit") == progress.get("unit")
                    and previous.get("total") == progress.get("total")
                    and previous.get("completed", 0) > progress.get("completed", 0)):
                return
            data.update(phase=phase, progress=progress)
            self._write(data)

    def _register_upload(self, job_id, path):
        from app.api.video import UPLOAD_DIR
        root = UPLOAD_DIR.resolve()
        target = Path(path).resolve()
        if target.parent != root or re.fullmatch(r"[0-9a-f]{32}\.[a-zA-Z0-9]{1,12}", target.name) is None:
            raise HTTPException(422, "视频上传文件名不合法")
        with self.lock:
            data = self._read(job_id)
            if data["status"] != "running" or data.get("analysis_upload_name") not in (None, target.name):
                raise HTTPException(409, "视频任务上传归属已变化")
            data.update(analysis_upload_name=target.name, analysis_upload_registered=True)
            self._write(data)  # This failure must stop analyze_video before opening the file.

    def _work(self):
        while True:
            job_id = self.pending.get()
            data = None
            token = None
            upload_token = None
            try:
                with self.lock:
                    data = self._read(job_id)
                    if data["status"] in FINAL:
                        continue
                    data.update(status="running", phase="upload")
                    self._write(data)
                token = progress_callback.set(lambda phase, progress: self._progress(job_id, phase, progress))
                upload_token = owned_upload_callback.set(lambda path: self._register_upload(job_id, path))
                from app.api.video import analyze_video
                with self._upload(data).open("rb") as source:
                    from starlette.datastructures import Headers
                    upload = UploadFile(file=source, filename=data["filename"],
                        headers=Headers({'content-type': data['content_type']}))
                    result = (self.runner or analyze_video)(upload, **data["params"])
                with self.lock:
                    latest = self._read(job_id)
                    if latest["status"] not in FINAL:
                        latest.update(status="succeeded", phase="completed", result=result, error=None,
                                      attempt_id=result.get("attempt_id") or latest.get("attempt_id", ""))
                        data = latest
            except Exception as exc:
                logger.exception("视频后台任务失败（%s）", job_id)
                with self.lock:
                    try:
                        latest = self._read(job_id)
                        if latest["status"] not in FINAL:
                            message = exc.detail if isinstance(exc, HTTPException) else "视频处理失败，请查看作答记录后再重试"
                            latest.update(status="failed", phase="failed", result=None,
                                          error={"status_code": exc.status_code if isinstance(exc, HTTPException) else 502,
                                                 "message": str(message)})
                        data = latest
                    except HTTPException:
                        pass
            finally:
                if token is not None:
                    progress_callback.reset(token)
                if upload_token is not None:
                    owned_upload_callback.reset(upload_token)
                try:
                    if data is not None:
                        with self.lock:
                            # A recovered terminal marker must not be overwritten by
                            # an old worker's initial running snapshot during cleanup.
                            latest = self._read(job_id)
                            if latest['status'] in FINAL:
                                data = latest
                            self._cleanup(data)
                            self._write(data)
                except Exception:
                    logger.exception("视频任务终态记录失败（%s），重启后将如实标记中断", job_id)
                finally:
                    self.slots.release()
                    self.pending.task_done()


_queues: dict[str, VideoJobQueue] = {}
_guard = threading.Lock()


def get_video_jobs() -> VideoJobQueue:
    from app.api.video import PROJECT_ROOT
    root = PROJECT_ROOT / "data" / "video_jobs"
    with _guard:
        key = str(root.resolve())
        if key not in _queues:
            _queues[key] = VideoJobQueue(root)
        return _queues[key]
