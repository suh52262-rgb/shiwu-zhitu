"""知识点归位：**把诊断给出的 `concept` 对齐到题库的 `knowledge_points`**。

## 为什么必须在代码里做（而不只是改提示词）
`diagnose` 产出的 `concept` 会写进**学生画像**，并用于**推荐下一题** ——
推荐的判据是「画像里的知识点 ∩ 题库里各题的知识点」。

**实测的问题**：提示词只举例 `（如"机械能守恒条件"）`，**没有封闭词表**，
模型把例子当前缀自己加限定，于是同一个知识点被记成三条：
```
"机械能守恒条件（只有重力或弹力做功）"
"机械能守恒条件（除重力外是否有其他力做功）"
"机械能守恒条件（是否只有重力/弹力做功）"
```
⇒ 画像越滚越长、推荐匹配不上、教师端统计失真。

## 为什么不能只靠提示词
这轮的教训反复是"**提示词约束不住输出**"：
P14 那条"硬性要求一：一题多解必须画成分叉"就没管住。
**提示词只能"请求"对齐，代码才能"保证"对齐。**

## 归位规则（保守）
1. **精确命中**（归一化后） → 用题库原文
2. **包含关系** → 用题库原文（如 `机械能守恒条件（…）` 含 `机械能守恒`）
3. **都对不上** → **返回空串**（宁可漏记，不要记错）
"""
from __future__ import annotations

from typing import Iterable, Optional

#: 比对这些概念时剥掉的"限定尾巴"。**只剥括号与连接符后面的部分**，
#: 不做同义词归并 —— 那需要一张词表，是另一个量级的工作。
_CUT = ("（", "(", "——", "—", "：", ":", "，", ",")


def _norm(s: str) -> str:
    return "".join((s or "").split())


def _head(s: str) -> str:
    """取括号/顿号之前的"主干"，用于包含比对。"""
    t = _norm(s)
    for c in _CUT:
        i = t.find(c)
        if i > 0:
            t = t[:i]
    return t


def snap_concept(concept: str,
                 vocabulary: Optional[Iterable[str]]) -> str:
    """把 `concept` 归位到 `vocabulary` 里的某一项。

    **返回空串表示"对不上"** —— 调用方应当**丢弃**而不是保留原值，
    否则画像里又会混进词表外的标签，前功尽弃。
    """
    raw = (concept or "").strip()
    if not raw:
        return ""
    vocab = [v for v in (vocabulary or []) if (v or "").strip()]
    if not vocab:
        # **没有词表就不做任何加工** —— 原样返回，
        # 总比凭空造一个"看起来对齐"的值好。
        return raw

    n = _norm(raw)
    # ① 精确
    for v in vocab:
        if _norm(v) == n:
            return v
    # ②③ 包含 / 主干命中 —— **必须取最长的那个**。
    #
    # ⚠️ 原来是"按词表顺序返回第一个命中"，实测踩到：
    #   `'动量定理的矢量性与速度方向变化'` 同时包含 `动量` 与 `矢量性`，
    #   而 `动量` 排在前面 → **归成了 `动量`，太粗**。
    # 多个候选时**最长匹配最贴切** —— 这是 substring 匹配的标准做法。
    h = _head(raw)
    cands: list[str] = []
    for v in vocab:
        vn = _norm(v)
        if not vn:
            continue
        if vn in n or n in vn:
            cands.append(v)
        elif h and (vn in h or h in vn):
            cands.append(v)
    if cands:
        return max(cands, key=lambda v: len(_norm(v)))
    # ④ 对不上 → **丢弃**
    return ""


def snap_many(concepts: Iterable[str],
              vocabulary: Optional[Iterable[str]]) -> list[str]:
    """批量归位 + 去重（保持首次出现顺序）。"""
    out: list[str] = []
    for c in concepts or []:
        s = snap_concept(c, vocabulary)
        if s and s not in out:
            out.append(s)
    return out
