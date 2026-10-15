"""把服务层的 `records` 桥接到 `LearningEvent`。

任务书 4.3 节：**"主 Agent 不直接处理原始视频，而只消费 LearningEvent。"**

## 一个必须处理的设计点：ROI 记录与整帧记录**同时刻**

抽帧阶段一张关键帧会产出两条记录：
  · 整帧（`frame_type=keyframe`），走内容理解提示词
  · ROI 裁剪放大图（`roi_tag` 非空），走 OCR 提示词

**两者 `timestamp_sec` 相同**——它们是**同一时刻的两种读数**，
不是两个时刻。若直接喂给 `frames_to_learning_events`，会产出成对的事件，
时间线上每写一笔都变成两条。

**处理**：按 `timestamp_sec` 分组，同组内**ROI 的 OCR 文字优先**
（它是针对变化区域的定向读数，比整帧更准），整帧文字作为补充。
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from app.models.event import LearningEvent
from app.vision.events import frames_to_learning_events



def _text_of(rec: dict) -> str:
    """从一条记录里取文字，按可靠度排序。

    ROI 记录走 OCR 提示词 → `text` 是逐行读数、另有 `ocr_lines`。
    整帧记录走内容理解 → `text` 未必有，`description` 更常见。
    """
    for key in ("text", "description"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v
    lines = rec.get("ocr_lines")
    if isinstance(lines, list) and lines:
        return "\n".join(str(x) for x in lines if str(x).strip())
    return ""


def merge_by_timestamp(records: Iterable[dict]) -> list[dict]:
    """同 `timestamp_sec` 的多条记录合并成一条快照。

    **不合并的后果**：一张关键帧 + 它的 ROI 会变成两条同刻事件，
    时间线上每写一笔都重复。这是本模块存在的理由。
    """
    groups: dict[float, dict[str, Any]] = {}
    order: list[float] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        ts = r.get("timestamp_sec")
        key = float(ts) if isinstance(ts, (int, float)) else 0.0
        if key not in groups:
            groups[key] = {"ts": key, "type": "", "text": "",
                           "error": None, "n": 0}
            order.append(key)
        g = groups[key]
        g["n"] += 1
        # 整帧记录提供 frame_type（ROI 记录的 type 是 'roi'，不代表时刻性质）
        ft = r.get("frame_type") or r.get("type") or ""
        if ft and ft != "roi" and not g["type"]:
            g["type"] = ft
        # **ROI 文字优先**：它是针对变化区域的定向读数
        is_roi = bool(r.get("roi_tag")) or r.get("frame_type") == "roi"
        t = _text_of(r)
        if t:
            if is_roi or not g["text"]:
                g["text"] = t
        # 错误要留住（不吞掉失败）
        if r.get("error") and not g["error"]:
            g["error"] = str(r["error"])[:200]
    return [groups[k] for k in sorted(order)]


def records_to_learning_events(
    records: Iterable[dict],
    *,
    attempt_id: str = "",
    student_id: str = "",
    problem_id: str = "",
    t0: float = 0.0,
    # 用 None 占位而不是直接写 SETTLE_WINDOW：
    # 默认参数在**函数定义时**求值，常量位置就变成硬约束。
    settle_window: Optional[int] = None,
) -> list[LearningEvent]:
    """`run_pipeline` 的 `records` → `LearningEvent` 列表。

    **先过一遍"稳定期"**（见下方 `settle_snapshots`）：只在内容写完并稳定
    之后才产出事件。`settle_window=0` 可关掉。

    ⚠️ **有 `error` 的记录不产出事件**——识别失败的帧内容未知，
    编一个出来比缺失更糟（会被下游当成真的）。缺口由 `events.py`
    记进 `metadata.gaps`。
    """
    merged = merge_by_timestamp(records)
    snaps = []
    for g in merged:
        if g["error"]:
            # 明确置空：让下游把它记成"缺口"，而不是当无事发生
            snaps.append({"ts": g["ts"], "text": "", "type": g["type"]})
            continue
        snaps.append({"ts": g["ts"], "text": g["text"], "type": g["type"]})
    _w = SETTLE_WINDOW if settle_window is None else settle_window
    if _w > 0:
        snaps = settle_snapshots(snaps, window=_w)
    return frames_to_learning_events(
        snaps, attempt_id=attempt_id, student_id=student_id,
        problem_id=problem_id, t0=t0,
    )


# ---------------------------------------------------------------- 稳定期
#
# ## 为什么需要它（有实测依据）
#
# 学生**逐笔书写**时，关键帧抓到的常常是**画到一半的字**，而 VL 不会说
# "没写完"，它会**给一个看起来合理的符号**。实测 60 帧：
#
# ```
# '由题知'        60/60 帧  100%  尾部连续 60  ← 写完并留住了
# '⊙' '⊕' 'Γ' '⊥' '√' 'Σ'   ≤6/60 帧  ≤10%  尾部连续 0   ← 闪一下就没
# 'F' '=' '_' '二'  17~39/60 帧  28~65%  尾部连续 0~18  ← **真的，但正在写**
# ```
#
# **注意第三行**：`F`、`=` 是学生正在写的真实公式，但它们**读数同样不稳定**
# （这帧读到 `F`，下帧读到 `F_k`，再下帧又没有）。
#
# ⇒ 这个信号测的**不是"真 vs 幻觉"，而是"写完了 vs 还没写完"**。
#
# **所以它不能用"滤掉低留存的"** —— 那会把答案本身滤掉。
# 正确用法是反过来的：**只对已经稳定的内容产出事件**。
# 正在写的内容不是被丢弃，而是**推迟到写完再报**。
# 幻觉符号永远稳定不下来，于是自动被排除——**是副产品，不是专门过滤**。

#: 只保留最近 K 帧读数的**共同前缀**。闪烁的读数活不过 K 帧。
#:
#: 为什么 K=3 就够：实测真实内容 60/60 帧稳定，而所有幻觉 ≤6/60 帧——
#: **中间有 10 倍余量**，K 取 3~10 都行。取 3 是保守选择（延迟最低）。
SETTLE_WINDOW = 3

#: 稳定前缀短于这个长度就不算"内容"——避免为一个标点产出事件。
SETTLE_MIN_CHARS = 2

#: 内容**变短**时要连续多少帧才认账（单调护栏）。
#:
#: 画布上的东西不会自己消失，所以"读数变短"更可能是**误读**而不是擦除。
#: 实测：单向/双向确认后仍有假的"擦除"事件
#: （`将 由题知 F 改为 由题知`——学生并没有擦掉那个 F）。
#: 要它连续变短这么多帧才认，误读就过不来，而真擦除仍然能被记录。
SHRINK_WINDOW = 8


def settled_prefix(texts: list[str], k: int) -> str:
    """最近 `k` 条读数的**共同前缀**。

    闪烁的读数（只在一两帧出现）不可能出现在全部 k 条里，
    于是被自然排除；而正在写的内容一旦稳定下来就会进入前缀。

    `k <= 0` 返回空串（表示"不启用"）——
    **不能让它落到 `texts[-0:]`**，那等于取整个列表，再 `[0]` 就越界。
    """
    if k <= 0:
        return ""
    window = [t for t in texts[-k:] if t]
    if len(window) < k:
        return ""
    prefix = window[0]
    for t in window[1:]:
        n = 0
        for a, b in zip(prefix, t):
            if a != b:
                break
            n += 1
        prefix = prefix[:n]
        if not prefix:
            return ""
    return prefix.strip()


def settle_snapshots(snaps: list[dict], *, window: int = SETTLE_WINDOW,
                     min_chars: int = SETTLE_MIN_CHARS,
                     confirm: bool = True,
                     monotonic: bool = True,
                     shrink_window: int = SHRINK_WINDOW) -> list[dict]:
    """把逐帧读数转成"已写完"的快照序列。

    每帧的文本替换为**到该帧为止最近 `window` 帧的共同前缀**。
    于是：
    · 写完并稳定的内容 → 立即进入，且之后一直保留
    · 正在写的内容 → 等它稳定后进入（**推迟，不是丢弃**）
    · 闪烁的误读 → 永远进不来

    ## `confirm=True`：再加一道**前向窗口**
    只有后向窗口不够——实测快照序列**自身仍会摇摆**：
    ```
    '由题知 F'      ← 稳定 4 帧，够长
    '由题知 F = ?'  ← 1 帧
    '由题知 F ='    ← 稳定 3 帧，够长
    '由题知 F'      ← **又退回去了**
    ```
    每段都比 `window=3` 长，后向窗口拦不住；结果是 22 条事件里大半是这种摆动。

    `confirm=True` 时，某帧的值还必须在**其后 `window` 帧里立得住**才算数——
    与后向窗口合起来即**双向去抖**：一个读数要前后都有支撑。
    """
    out = []
    seen: list[str] = []
    texts = [(s.get("text") or "").strip() for s in snaps]
    n = len(texts)
    if window <= 0:
        # **`window<=0` = 关闭稳定期**，原样返回。
        # 不能让它走下面的逻辑：`texts[-0:]` 等于取整个列表，
        # `settled_prefix` 里再取 `[0]` 就越界。
        return [dict(s) for s in snaps]
    #: 已确认过的**最长**内容。用于单调护栏（见 `monotonic` 说明）。
    high_water = ""
    shrink_run = 0
    for i, s in enumerate(snaps):
        seen.append(texts[i])
        p = settled_prefix(seen, window)
        if confirm and p and i + window <= n:
            # **前向确认**：还要在其后 window 帧里立得住。
            # 取两者的共同前缀——谁更保守听谁的。
            q = settled_prefix(texts[i:i + window], window)
            m = 0
            for a, b in zip(p, q):
                if a != b:
                    break
                m += 1
            p = p[:m]
        # 末尾凑不满前向窗口的帧：不能再确认，但**已稳定的内容不因此丢掉**

        if monotonic:
            # ⚠️ **不能写成 `if monotonic and high_water:`**——
            # 那样 `high_water` 初始为空时整个分支被跳过，
            # 而 `high_water = p` 的赋值**就在分支里**，于是它**永远填不上**，
            # 护栏形同虚设。（实测踩过：加上护栏后事件数一条没变。）
            if len(p) >= len(high_water):
                high_water = p
                shrink_run = 0
            else:
                # **内容变短了。** 画布上的东西不会自己消失，
                # 所以这更可能是误读而不是擦除——要它连续变短
                # `shrink_window` 帧才认账。
                shrink_run += 1
                if shrink_run < shrink_window:
                    p = high_water
                else:
                    high_water = p

        text = p if len(p) >= min_chars else ""
        out.append({**s, "text": text})
    return out
