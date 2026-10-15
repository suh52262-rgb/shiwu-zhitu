"""P25 Demo Playback：比赛现场用的**可控制回放**。

## 为什么是回放而不是实时跑

任务书的原话是"**完全不依赖真实视频识别的稳定演示模式**"，
并列了三种现场事故：网络不好、摄像头识别失败、视频模型 API 延迟。

只要现场还要调模型，这三种事故就都会让演示卡住或答错。
所以回放的是**预先录制好的每轮结果**——**一次模型都不调**，
现场断网也能跑完。

代价是"演示内容是真的还是编的"这个问题必须回答清楚：
**录制来自真实跑过的一轮**（`scripts/record_demo.py` 跑真实模型并落盘），
不是手写的假数据。这一点在接口响应里用 `recorded_at` / `source` 标明。

## 与 `pq_demo.py` 的区别

`pq_demo.py` 是**固定剧本**——一按就跑到底，没法停、没法跳、没法回退。
本模块是**播放器**：有状态机，支持 pause / resume / next / reset / speed。

## 状态机

    idle ──start──> playing ──pause──> paused ──resume──> playing
                      │  │                  │
                      │  └──next───────────┘   （单步，不改变播放状态）
                      └──finish──> finished
    任意状态 ──reset──> 回到第一帧（paused）

**`next` 不改变 playing/paused**：现场演示常常是"暂停着一步步讲"，
单步之后自动继续播会让讲的人失去节奏。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/demo", tags=["demo"])

#: 录制文件目录（相对仓库根）
RECORDINGS_DIR = Path("demo") / "recordings"

#: 允许的倍速。**只开放这三档**——现场用不到更细的，
#: 而任意倍速会让"这一帧该停多久"难以预期。
ALLOWED_SPEEDS = (1.0, 2.0, 4.0)

STATE_IDLE = "idle"
STATE_PLAYING = "playing"
STATE_PAUSED = "paused"
STATE_FINISHED = "finished"


# ---------------------------------------------------------------- 录制

@dataclass
class Recording:
    """一段可回放的演示。"""

    case_id: str
    title: str = ""
    problem_id: str = ""
    #: 真实录制时间与来源，**必须如实标明**——现场要能回答"这是真跑出来的吗"
    recorded_at: str = ""
    source: str = ""
    steps: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Recording":
        return cls(
            case_id=d.get("case_id", ""),
            title=d.get("title", ""),
            problem_id=d.get("problem_id", ""),
            recorded_at=d.get("recorded_at", ""),
            source=d.get("source", ""),
            steps=list(d.get("steps") or []),
        )


class PlaybackSession:
    """播放器状态机。**纯内存**——演示过程不需要持久化，重启即重来。"""

    def __init__(self, rec: Recording) -> None:
        self.rec = rec
        self.cursor = 0                 # 已播放到第几帧（0 = 还没播第一帧）
        self.state = STATE_IDLE
        self.speed = 1.0

    # ---- 控制 ----

    def start(self) -> None:
        if self.state in (STATE_IDLE, STATE_FINISHED):
            self.cursor = 0
            self.state = STATE_PLAYING

    def pause(self) -> None:
        if self.state == STATE_PLAYING:
            self.state = STATE_PAUSED

    def resume(self) -> None:
        if self.state == STATE_PAUSED:
            self.state = STATE_PLAYING

    def next(self) -> None:
        """单步前进。**不改变 playing/paused**（见模块 docstring）。"""
        if self.cursor < len(self.rec.steps):
            self.cursor += 1
        if self.cursor >= len(self.rec.steps) and self.state != STATE_IDLE:
            self.state = STATE_FINISHED

    def reset(self) -> None:
        self.cursor = 0
        self.state = STATE_PAUSED       # 回到开头**停住**，而不是立刻又播

    def set_speed(self, speed: float) -> None:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"倍速只能是 {ALLOWED_SPEEDS}，收到 {speed}")
        self.speed = speed

    # ---- 视图 ----

    @property
    def current(self) -> dict | None:
        if self.cursor <= 0 or self.cursor > len(self.rec.steps):
            return None
        return self.rec.steps[self.cursor - 1]

    def view(self) -> dict:
        """**累积视图**：Timeline 与节点路径随播放逐渐长出来。

        这是回放的意义所在——现场看到的应该是"一点点长出来"的过程，
        而不是一上来就把整段结果摊开。
        """
        played = self.rec.steps[: self.cursor]
        visited: list[str] = []
        for s in played:
            for n in s.get("visited_nodes") or []:
                if n not in visited:
                    visited.append(n)
        return {
            "case_id": self.rec.case_id,
            "title": self.rec.title,
            "problem_id": self.rec.problem_id,
            # 如实标明录制来源——现场要能回答"这是真跑出来的吗"
            "recorded_at": self.rec.recorded_at,
            "source": self.rec.source,
            "state": self.state,
            "speed": self.speed,
            "cursor": self.cursor,
            "total_steps": len(self.rec.steps),
            "current": self.current,
            "visited_nodes": visited,
            "timeline": [e for s in played for e in (s.get("timeline") or [])],
        }


# ---------------------------------------------------------------- 会话存储

_SESSIONS: dict[str, PlaybackSession] = {}


def load_recording(case_id: str, *, root: Path | None = None) -> Recording:
    """读录制文件。找不到时给出**可操作的**错误信息。"""
    base = root or RECORDINGS_DIR
    path = base / f"{case_id}.json"
    if not path.is_file():
        available = sorted(p.stem for p in base.glob("*.json")) if base.is_dir() else []
        raise HTTPException(
            status_code=404,
            detail=f"没有录制 {case_id!r}。可用：{available or '（无，先跑 scripts/record_demo.py）'}",
        )
    return Recording.from_dict(json.loads(path.read_text(encoding="utf-8")))


class ControlRequest(BaseModel):
    action: str = Field(..., description="start/pause/resume/next/reset/speed")
    speed: float | None = None


@router.get("/recordings")
def list_recordings() -> dict:
    """列出可用录制——现场选片用。"""
    base = RECORDINGS_DIR
    if not base.is_dir():
        return {"recordings": [], "hint": "还没有录制，先跑 scripts/record_demo.py"}
    out = []
    for p in sorted(base.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "case_id": p.stem,
                "title": d.get("title", ""),
                "problem_id": d.get("problem_id", ""),
                "recorded_at": d.get("recorded_at", ""),
                "source": d.get("source", ""),
                "steps": len(d.get("steps") or []),
            })
        except Exception as exc:  # noqa: BLE001 单个坏文件不该让列表整体失败
            out.append({"case_id": p.stem, "error": f"{type(exc).__name__}: {exc}"})
    return {"recordings": out}


@router.post("/playback/{case_id}")
def start_playback(case_id: str) -> dict:
    """选择案例并开始播放。"""
    session = PlaybackSession(load_recording(case_id))
    session.start()
    _SESSIONS[case_id] = session
    return session.view()


@router.get("/playback/{case_id}")
def playback_view(case_id: str) -> dict:
    session = _SESSIONS.get(case_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"案例 {case_id!r} 还没开始播放，先 POST /demo/playback/{case_id}",
        )
    return session.view()


@router.post("/playback/{case_id}/control")
def playback_control(case_id: str, req: ControlRequest) -> dict:
    """pause / resume / next / reset / speed。"""
    session = _SESSIONS.get(case_id)
    if session is None:
        raise HTTPException(status_code=404,
                            detail=f"案例 {case_id!r} 还没开始播放")
    action = (req.action or "").strip().lower()
    try:
        if action == "start":
            session.start()
        elif action == "pause":
            session.pause()
        elif action == "resume":
            session.resume()
        elif action == "next":
            session.next()
        elif action == "reset":
            session.reset()
        elif action == "speed":
            if req.speed is None:
                raise ValueError("speed 动作必须带 speed 参数")
            session.set_speed(float(req.speed))
        else:
            raise ValueError(
                f"未知动作 {action!r}；可用：start/pause/resume/next/reset/speed")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return session.view()
