"""视频服务层与纯 CV 层的收敛护栏。

## 为什么需要这组测试

成员 C 交付的 `video_analysis.py` 里**又实现了一遍**那 9 个 CV 函数
（和他自己的 `analyze_video_opt.py`、以及我们抽出的 `keyframe.py` 三份）。
算法逐常量核对后确认一致，但**三份会各自演化**。

已收敛为：`keyframe.py` 是**唯一实现**，`service.py` 只做服务层。

**这组测试盯的就是"别再分叉"**——因为分叉的症状是隐蔽的：
两份都"能跑"，只有结果悄悄不同时才暴露。
"""
import inspect

import cv2
import numpy as np
import pytest

from app.vision import keyframe as kf
from app.vision import service as sv

#: 服务层名字 → keyframe 里的真名
CV_PAIRS = [
    ("_phash", "phash"),
    ("_probe_signatures", "probe_signatures"),
    ("_ncc_distance", "ncc_distance"),
    ("_localized_score", "localized_score"),
    ("_compare", "compare"),
    ("unsharp", "unsharp"),
    ("resize_keep", "resize_keep"),
    ("crop_zoom", "crop_zoom"),
    ("auto_roi_box", "auto_roi_box"),
]


@pytest.mark.parametrize("sname,kname", CV_PAIRS)
def test_cv_functions_are_the_same_object(sname, kname):
    """**同一个对象**，不是"两份实现碰巧结果一样"。"""
    assert getattr(sv, sname) is getattr(kf, kname), (
        f"service.{sname} 与 keyframe.{kname} 不是同一实现——CV 层又分叉了"
    )


def test_service_has_no_duplicate_cv_definitions():
    """服务层里**不许再出现** CV 函数的定义。

    这条是防"有人图方便又拷一份进去"。
    """
    src = inspect.getsource(sv)
    for name in ("def _compare", "def _phash", "def _probe_signatures",
                 "def _ncc_distance", "def _localized_score", "def unsharp",
                 "def resize_keep", "def crop_zoom", "def auto_roi_box"):
        assert name not in src, f"service.py 里出现了重复定义：{name}"


def test_service_imports_from_keyframe():
    src = inspect.getsource(sv)
    assert "from app.vision.keyframe import" in src


def test_keyframe_import_is_outside_optional_dependency_try():
    """⚠️ **keyframe 的导入必须在可选依赖的 try/except 之外。**

    放进 try 里的话，`keyframe` 自己 import cv2 失败时抛的 ImportError
    会被 `except ImportError` 吞掉，把"cv2 没装"变成"模块静默降级"——
    **错误被吃掉，症状变成别的东西**。
    （这是我第一版收敛脚本踩到的坑。）
    """
    src = inspect.getsource(sv)
    try_at = src.index("try:")
    import_at = src.index("from app.vision.keyframe import")
    except_at = src.index("except ImportError")
    assert not (try_at < import_at < except_at), \
        "keyframe 的导入落进了可选依赖的 try 块里"


def test_computation_is_identical():
    """端到端：同一个合成帧，两层算出的分数与 ROI 必须一致。"""
    base = np.full((720, 1280, 3), 240, np.uint8)
    cv2.putText(base, "HELLO", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (20, 20, 20), 6)
    cur = base.copy()
    cv2.putText(cur, "1234", (900, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1)

    a = sv._compare(sv._probe_signatures(cur), sv._probe_signatures(base))
    b = kf.compare(kf.probe_signatures(cur), kf.probe_signatures(base))
    assert a.total == pytest.approx(b.total, abs=1e-12)
    assert sv.auto_roi_box(base, cur) == kf.auto_roi_box(base, cur)


def test_service_layer_keeps_its_value():
    """**服务层的东西不能丢**——它不是纯粹的重复，有 C 的贡献在里面。"""
    for name in ("run_pipeline", "VideoAnalysisOptions", "VideoAnalysisError",
                 "extract_frames", "analyze_frames", "export_results",
                 "resolve_model", "build_vl_client", "_FrameSink"):
        assert hasattr(sv, name), f"服务层入口 {name} 丢了"


def test_service_docstring_states_the_convergence():
    """说明要留在文件里，否则下一个人会再拷一份进去。"""
    doc = sv.__doc__ or ""
    assert "收敛" in doc
    assert "keyframe" in doc
