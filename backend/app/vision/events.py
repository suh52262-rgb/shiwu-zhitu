"""把视频逐帧描述转成 `LearningEvent`（感知层闭环的最后一步）。

评审报告 P-3 指出的集成缺口：

> 本脚本产出的是**逐帧画面描述 JSON**，**没有任何一步**把它转成 LearningEvent。
> ……**逐帧描述是"快照"，而 LearningEvent 是"变化事件"**

任务书 4.3 节的协议要求：
> 无论未来学生行为来自：网页输入；手写笔迹；屏幕录制；视频；Qwen3-VL；
> **最终都转换为统一 LearningEvent**。……主 Agent 不直接处理原始视频，
> 而只消费 LearningEvent。

## 这一层的职责

```
关键帧（快照）          本模块            统一事件（变化）
  ts=12.0  "N=mg"   ──────────────→   t=12.0  modify  "将 N=mg 改为 N=mg cosθ"
  ts=18.0  "N=mg cosθ"
```

**上游**（`keyframe.py` + VL）只管"这一帧画面里有什么"，
**本模块**管"相比上一帧，学生做了什么"。**两者的区别是本质的**：
快照可以重复，变化不可重复。

## 三条纪律

1. **不伪造** —— 帧的文字取不到时，**不猜内容**。宁可记 `idle` 并在
   `metadata` 里标明缺口，也不写一个看起来完整的 `write`。
2. **不吞掉失败** —— VL 解析失败的帧要显式体现在 `metadata` 里
   （评审 P-2 批评过旧脚本"JSON 解析失败被静默当作成功"）。
3. **复用既有 Schema** —— 用 `app/models/event.py` 的 `LearningEvent`，
   不新造一套。`student_id` / `problem_id` 放 `metadata`
   （Schema 注释写明它是"预留多模态来源等扩展字段"）。
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.models.event import LearningEvent

#: 多模态来源标记。主 Agent 据此知道这条事件是"看视频看来的"。
SOURCE_VIDEO = "video"

#: 会随时间自己变、但**不是学生动作**的画面元素。
#:
#: 屏幕录制里这类东西会持续产生假 `write`：时钟、播放进度、光标闪烁、
#: 工具栏高亮。**不滤掉的话，时间线上会灌满噪声事件。**
NOISE_PATTERNS = (
    re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$"),          # 时钟 12:34
    # 日期：年份可能是 2 位或 4 位。第一版写 `\d{1,2}/...` 漏掉了 2026/10/04。
    re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*$"),   # 2026-10-04
    re.compile(r"^\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\s*$"), # 10/04/2026
    re.compile(r"^\s*[-—=]{3,}\s*$"),                        # 分隔线
)


def _is_noise(text: str) -> bool:
    return any(p.match(text) for p in NOISE_PATTERNS)


# ---------------------------------------------------------------- 文本归一

def normalize_lines(text: str) -> list[str]:
    """把一帧的 `text` 拆成行，去空行、去首尾空白。

    **保留顺序**——顺序本身是信息（第几行写了什么）。
    """
    if not text:
        return []
    out = []
    for line in str(text).replace("\r", "\n").split("\n"):
        s = line.strip()
        if s and not _is_noise(s):
            out.append(s)
    return out


@dataclass
class _Change:
    """两个快照之间的差异。"""

    kind: str          # 'add' | 'remove' | 'modify' | 'none'
    content: str       # 人类可读的中文描述
    added: int = 0
    removed: int = 0


def diff_snapshots(prev: list[str], cur: list[str]) -> _Change:
    """比较两次快照，产出**人类可读**的变化描述。

    `content` 的形态对齐 Schema 注释里的例子：
    `"将 N=mg 改为 N=mg cosθ"`。
    """
    if prev == cur:
        return _Change("none", "")

    if not prev and cur:
        joined = "；".join(cur)
        return _Change("add", f"写下：{joined}", added=len(cur))
    if prev and not cur:
        return _Change("remove", f"擦除：{'；'.join(prev)}", removed=len(prev))

    p, c = set(prev), set(cur)
    added = [x for x in cur if x not in p]
    removed = [x for x in prev if x not in c]

    # **按"改了什么"分类，不要落到默认分支**——
    # 第一版只在"增删各一条"时判 modify，其余全落到默认的 modify，
    # 于是"只增"和"只删"都被误判成 modify。
    if added and not removed:
        return _Change("add", f"写下：{'；'.join(added)}", added=len(added))
    if removed and not added:
        return _Change("remove", f"擦除：{'；'.join(removed)}", removed=len(removed))
    if len(added) == 1 and len(removed) == 1:
        return _Change("modify", f"将 {removed[0]} 改为 {added[0]}",
                       added=1, removed=1)
    parts = []
    if added:
        parts.append(f"写下：{'；'.join(added)}")
    if removed:
        parts.append(f"擦除：{'；'.join(removed)}")
    return _Change("modify", "；".join(parts), added=len(added), removed=len(removed))


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------- 主入口

def frames_to_learning_events(
    records: Iterable[dict[str, Any]],
    *,
    attempt_id: str = "",
    student_id: str = "",
    problem_id: str = "",
    t0: float = 0.0,
    source: str = SOURCE_VIDEO,
) -> list[LearningEvent]:
    """逐帧描述 → `LearningEvent` 列表。

    ## 参数
    · `records` —— 关键帧记录。每项至少要有 `ts` 与某种文字字段
      （`text` / `ocr` / `description`，按序取第一个非空的）
    · `t0` —— 会话起点（秒）。事件时间戳是**相对**它的，与 Schema 注释
      "相对 attempt 开始的秒数"一致
    · `student_id` / `problem_id` —— **放进 `metadata`**，不新造 Schema 字段

    ## 映射规则
    | 帧类型 / 变化 | event_type |
    |---|---|
    | 首帧 | `start` |
    | 新增文字 | `write` |
    | 文字被替换 | `modify` |
    | 文字被删 | `erase` |
    | `static_rep` 且无变化 | `idle` |
    | 无变化、非静止帧 | **不产出事件** |

    ## 诚实降级
    取不到文字的帧**不产出事件**，但在下一条事件的 `metadata.gaps` 里
    记下"第 N 帧无文字"——**留下缺口痕迹，而不是编一个内容出来**。
    """
    items = [r for r in records if isinstance(r, dict)]
    if not items:
        return []
    # 时间戳排序（乱序输入不该产出颠倒的事件顺序）
    items = sorted(items, key=lambda r: float(r.get("ts") or 0.0))

    base_meta: dict[str, Any] = {"source": source}
    if student_id:
        base_meta["student_id"] = student_id
    if problem_id:
        base_meta["problem_id"] = problem_id

    events: list[LearningEvent] = []
    prev_lines: list[str] | None = None
    prev_ts: float | None = None
    gaps: list[str] = []

    for idx, rec in enumerate(items):
        ts = float(rec.get("ts") or 0.0)
        raw = ""
        for key in ("text", "ocr", "description"):
            v = rec.get(key)
            if isinstance(v, str) and v.strip():
                raw = v
                break
        if not raw:
            # **不伪造**：记下缺口，不产出事件
            gaps.append(f"第 {idx + 1} 帧（t={ts:.1f}s）无可用文字")
            continue

        cur_lines = normalize_lines(raw)
        kind = rec.get("type") or rec.get("kind") or ""

        if prev_lines is None:
            events.append(LearningEvent(
                attempt_id=attempt_id, timestamp=max(0.0, ts - t0),
                event_type="start",
                content=f"开始作答：{'；'.join(cur_lines)}" if cur_lines else "开始作答",
                metadata={**base_meta, "frame_index": idx, "frame_type": kind},
            ))
        else:
            ch = diff_snapshots(prev_lines, cur_lines)
            if ch.kind == "none":
                # 画面没变：只有"静止补帧"才产出 idle，否则不是事件
                if kind == "static_rep":
                    events.append(LearningEvent(
                        attempt_id=attempt_id, timestamp=max(0.0, ts - t0),
                        event_type="idle", content="",
                        metadata={**base_meta, "frame_index": idx,
                                  "frame_type": kind,
                                  "idle_seconds": round(ts - (prev_ts or ts), 1)},
                    ))
                prev_lines, prev_ts = cur_lines, ts
                continue
            etype = {"add": "write", "remove": "erase",
                     "modify": "modify"}.get(ch.kind, "modify")
            events.append(LearningEvent(
                attempt_id=attempt_id, timestamp=max(0.0, ts - t0),
                event_type=etype, content=ch.content,
                metadata={**base_meta, "frame_index": idx, "frame_type": kind,
                          "lines_added": ch.added, "lines_removed": ch.removed},
            ))
        prev_lines, prev_ts = cur_lines, ts

    # 缺口痕迹挂到最后一条事件上（而不是丢弃）
    if gaps and events:
        events[-1].metadata["gaps"] = gaps
    return events


def to_timeline(events: list[LearningEvent]) -> list[dict]:
    """给前端/教师端的扁平时间线视图。"""
    return [
        {"t": round(e.timestamp, 1), "type": e.event_type, "content": e.content,
         "source": e.metadata.get("source", "")}
        for e in events
    ]
