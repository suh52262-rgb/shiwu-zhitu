"""文字层里的「坏公式」识别：保留原文供审计，但**不向量化**。

## 为什么必须做

这版教材的文字层来自 OCR，**公式基本不可用**。实测前 120 页就有 64 行
这种内容被当作正文入库：

```
's = (vo +v)t =( ++at)'          ← 应为 s = v₀t + ½at²
'11.y.=9.460530×10l*m~10l°m.1010'
'1AU = 1.495 978 92 ×10m,'
```

它们和 VL 生成的正规 LaTeX **一起进了向量库**，检索时会被当作
「教材原文」返回给学生。**这是正确性问题，不是覆盖问题。**

## 处理方式：隔离，不是删除

`text` 原样保留在 PostgreSQL（审计、排障、将来换更好的 OCR 都还要用），
只把该块标为 `indexable=False`，`index_blocks` 跳过它。

**为什么不用 `is_generated` 表达**：那个字段的语义是"内容由模型生成"。
OCR 文字层是**扫描件自带**的，不是模型生成的——把两者混为一谈会让
「排除机器生成内容」这条检索规则误伤一大片真实教材文字。

## 判定依据（在真实页面上核对过阈值）

不是"含 `=` 就算公式"——那样会把 `'根据牛顿第二定律：'` 也误伤。
综合三个信号：
  · 数学符号与数字占比高
  · 汉字占比低（**但有少量汉字也必须能识别**，如
    `'若取to=0时V=Vo，S=S。=0,则'` 汉字占比 0.2）
  · 长度短
"""
from __future__ import annotations

import re

#: 数学符号。
#:
#: **必须包含 `+ - * /`**：它们是最基础的运算符，漏掉会让
#: `'s = (vo +v)t =( ++at)'` 这种最典型的坏公式检测不出来
#: （只有两个 `=` 命中，占比 0.09 远低于阈值）——踩过。
MATH_CHARS = set("=+-*/∫∑√±×÷²³₁₂αβγθωπΔ→←∂∞≈≠≤≥·<>%^_|")

#: 汉字
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

#: 疑似公式行里允许的字符（用于算"非公式字符"占比）
_ALLOWED = set("0123456789+-*/=<>().,;:[]{}^_|'\" \t")

#: 块短于这个长度才可能是公式碎片（长段落不判）
MAX_LEN = 60

#: 汉字占比超过这个值就认为是散文
MAX_CJK_RATIO = 0.30

#: **汉字绝对个数**达到这个值就认为是散文，不再当公式拦。
#:
#: 为什么要加绝对个数、而不只看占比：实测有 6 行正文被误伤，例如
#:   `'令(1.14)和(1.15)式中的g=0.,则有'`（7 个汉字 / 24 字 = 0.29，
#:    刚好压在 0.30 的阈值下）
#:   `'R=6370km，R。=R-2900km=3470km.将已知数据代入,可解出'`
#: 这些是**含行内公式的散文**，拦掉就是内容丢失。
#: 真正的坏公式汉字数普遍是 0~2（`'a='`、`'α - lα|= lim'`、`'787× 960'`），
#: 所以用一个绝对下限能干净地分开两者，不受长度影响。
MIN_CJK_AS_PROSE = 5

#: 数学符号+数字的占比下限
MIN_MATH_RATIO = 0.30


def looks_like_broken_formula(text: str) -> bool:
    """这一行是不是"被 OCR 弄坏的公式"。

    只判**扫描件的文字层**——原生文字页的排版工具会把公式作为正常文本
    导出，那种情况不该被拦（那里的公式虽然也可能不准，但不是 OCR 的锅，
    而且拦掉会误伤）。

    ## 阈值已人工抽查核对（见 docs/公式隔离率抽查报告.md）

    在这份教材上隔离 10.3%（2369/22976），抽查结论：

      · **100 条分层随机样本中，没有一条是散文**
      · 全量 2369 条里 **97.5% 一个汉字都没有**——中文教材里这种块
        不可能是散文，**核心判据在全量数据上成立**（不只是抽样成立）
      · 含 ≥3 汉字的只有 14 条（0.6%），多数是公式引导语
        （`'若取to=0时V=Vo…则'`），而它引导的公式在 VL 块里是全的
      · 被隔离块所在的 360 页中 **352 页（97.8%）仍有 VL 公式块**
        ——公式内容没丢，隔离等于零损失
      · 最坏情况（8 页无 VL 覆盖、19 条"唯一表示"）全是
        ISBN／网址／印刷规格／页码／索引条目，**没有一条是教学内容**

    唯一真实损失：约 4 条习题答案片段。不值得为它放宽判据——
    放宽会让大量真正的坏公式重新进入索引，方向是反的。

    ⚠️ 换教材（不同 OCR 引擎／字体／扫描质量）前必须**重新核对**，
    尤其是"汉字数分布"这一项——它最便宜，也最能说明判据是否仍成立。
    """
    stripped = (text or "").strip()
    if not stripped or len(stripped) > MAX_LEN:
        return False

    # 纯公式编号 `(1.15)` 是**有用**的检索锚点（正文会写"由式(1.15)"），
    # 而且它不"错"，所以放行。
    if re.fullmatch(r"[（(]\s*\d+[.\-–]\d+\s*[)）]", stripped):
        return False

    cjk = sum(1 for c in stripped if CJK_RE.match(c))
    # 汉字够多 → 是含行内公式的散文，不是坏公式
    if cjk >= MIN_CJK_AS_PROSE:
        return False
    if cjk / len(stripped) > MAX_CJK_RATIO:
        return False

    # **一个汉字都没有的行，在中文教材里不可能是散文**。
    # 这条比"数学符号占比"更可靠：`'s = (vo +v)t =( ++at)'` 里拉丁变量名
    # 占了大半，数学符号占比只有 0.23，卡在阈值下检测不出来（踩过）。
    has_math_marker = any(c in stripped for c in "=+-*/∫∑√±×÷^_")

    math = sum(1 for c in stripped if c in MATH_CHARS or c.isdigit())
    if cjk == 0 and has_math_marker:
        return True
    if math / len(stripped) < MIN_MATH_RATIO:
        return False

    # 必须真的像方程：有等号，或有明显的数学标记
    has_equation = "=" in stripped
    has_marker = any(c in stripped for c in "∫∑√±×÷^_")
    if not (has_equation or has_marker):
        return False

    # 非公式字符太多 → 多半是散文里夹了个等号
    weird = sum(1 for c in stripped
                if c not in _ALLOWED and c not in MATH_CHARS
                and not c.isalnum() and not CJK_RE.match(c)
                and c not in "，。；：、（）【】“”‘’—…·")
    return weird / len(stripped) < 0.15


def classify_text_line(text: str, *, is_scanned: bool) -> tuple[bool, str]:
    """返回 `(是否可向量化, 原因)`。

    原因会写进块的 `note` 字段，便于事后统计"到底拦掉了多少、都是什么"——
    一个静默的过滤器比不过滤更危险。
    """
    if not is_scanned:
        return True, ""
    if looks_like_broken_formula(text):
        return False, "扫描件文字层疑似坏公式，已隔离（原文保留供审计）"
    return True, ""
