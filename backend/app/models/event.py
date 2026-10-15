"""Learning Event 模型。

无论输入未来来自网页输入、手写笔迹、屏幕录制、Qwen3-VL 还是摄像头，
最终都统一转换为 LearningEvent，构成 Learning Timeline。
"""
from pydantic import BaseModel, Field

EVENT_TYPES = ("start", "write", "modify", "erase", "idle", "hint_request", "submit")


class LearningEvent(BaseModel):
    attempt_id: str = ""
    timestamp: float = 0.0        # 相对 attempt 开始的秒数
    event_type: str = "write"
    content: str = ""             # 如："将 N=mg 改为 N=mg cosθ"
    metadata: dict = Field(default_factory=dict)   # 预留多模态来源等扩展字段
