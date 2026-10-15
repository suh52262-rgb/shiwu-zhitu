"""视频关键帧判定的纯 CV 层（模型无关）。
"""

from app.vision.keyframe import (  # noqa: F401
    KIND_GRADIENT,
    KIND_KEYFRAME,
    KIND_SCENE,
    KIND_STATIC_REP,
    ChangeScore,
    KeyframePolicy,
    KeyframeState,
    auto_roi_box,
    classify,
    compare,
    crop_zoom,
    localized_score,
    ncc_distance,
    phash,
    probe_signatures,
    resize_keep,
    unsharp,
)

__all__ = [
    "KIND_GRADIENT", "KIND_KEYFRAME", "KIND_SCENE", "KIND_STATIC_REP",
    "ChangeScore", "KeyframePolicy", "KeyframeState",
    "auto_roi_box", "classify", "compare", "crop_zoom", "localized_score",
    "ncc_distance", "phash", "probe_signatures", "resize_keep", "unsharp",
]
