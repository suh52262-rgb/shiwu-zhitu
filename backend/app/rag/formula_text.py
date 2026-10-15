"""公式 LaTeX → 可读文本（供向量化用）。

## 为什么必须做这一步

BGE 是**中文文本模型**，对 `\\frac{g}{2v_0^2\\cos^2\\alpha}` 这种记号的
语义表征很差——直接拿 LaTeX 去 embed，检索"抛物线轨迹方程"根本命中不了。

所以公式块要**存两份**：
  · `text`       = LaTeX 原样（给人看、给 LLM 用、供渲染）
  · `embed_text` = 转换后的可读形式（真正被向量化）

**转换只求"检索时能被找到"，不追求数学等价**——渲染用的始终是 `text`。

## 一个取舍

正则替换必然不完美（嵌套分式、矩阵、多行对齐都会留下痕迹）。
但这不影响正确性：`text` 里的 LaTeX 是完整的，
`embed_text` 只影响"这条能不能被检索到"，不影响"检索到之后显示什么"。
"""
from __future__ import annotations

import re

#: 常见 LaTeX 命令 → 可读符号。按教材实际出现的符号统计后逐步补。
SYMBOLS: dict[str, str] = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "θ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\upsilon": "υ", r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ",
    r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ",
    r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\cdot": "·", r"\ast": "∗", r"\circ": "°",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\equiv": "≡", r"\propto": "∝", r"\infty": "∞",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\leftrightarrow": "↔", r"\to": "→",
    r"\int": "∫", r"\iint": "∬", r"\oint": "∮", r"\sum": "∑", r"\prod": "∏",
    r"\partial": "∂", r"\nabla": "∇", r"\sqrt": "√",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
    r"\cot": "cot", r"\sec": "sec", r"\csc": "csc",
    r"\arcsin": "arcsin", r"\arccos": "arccos", r"\arctan": "arctan",
    r"\ln": "ln", r"\log": "log", r"\exp": "exp", r"\lim": "lim",
    r"\max": "max", r"\min": "min", r"\mathrm{d}": "d",
    r"\boldsymbol": "", r"\mathbf": "", r"\text": "",
}

#: 这些命令连同参数一起去掉（它们只影响排版，不影响含义）
_DROP_WITH_ARG = (r"\left", r"\right", r"\quad", r"\qquad", r"\displaystyle",
                  r"\limits", r"\!", r"\,", r"\;", r"\:", r"\ ")

#: 函数名：这些必须能当"词"读出来，所以要单独补空格
#: （`x\tan\alpha` 直接替换会粘成 `xtanα`，检索时就成了噪声）
_FUNCTIONS = (r"\arcsin", r"\arccos", r"\arctan", r"\sinh", r"\cosh", r"\tanh",
              r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
              r"\ln", r"\log", r"\exp", r"\lim", r"\max", r"\min", r"\det")

_GREEK_KEYS = sorted(
    (k for k in SYMBOLS if k not in _FUNCTIONS), key=len, reverse=True
)      # 长命令先替换，避免 \varpi 被 \var 截断


def latex_to_readable(latex: str) -> str:
    """把 LaTeX 转成中文文本模型能理解的形式。

    只做记号层面的替换，**不追求数学等价**——它的用途是检索，不是渲染。

    **替换顺序有讲究**（踩过坑）：
    1. 先去掉 `\\boldsymbol{}` / `\\text{}` 这类**零宽包装**。
       否则 `\\frac{d\\boldsymbol{P}}{dt}` 里外层花括号含嵌套，
       `\\{([^{}]*)\\}` 匹配不到，`\\frac` 就原样留下了（实测输出过
       `\\frac d P dt`）。
    2. 再处理分式/根式——此时花括号已经扁平。
    3. 函数名（`\\tan` 等）要**补空格**，否则 `x\\tan\\alpha` 会粘成 `xtanα`。
    4. 最后清掉残留的 `\\命令`、`&`、多余花括号。
    """
    if not latex:
        return ""
    s = latex.strip()

    # ---- 0. 公式定界符 ----
    s = re.sub(r"^\s*\$\$?", "", s)
    s = re.sub(r"\$\$?\s*$", "", s)

    # ---- 1. 零宽包装：只留内容（必须在 \frac 之前）----
    for _ in range(3):
        s = re.sub(r"\\(?:boldsymbol|mathbf|mathrm|mathit|text|operatorname)\s*\{([^{}]*)\}",
                   r"\1", s)

    # ---- 2. 分式与根式 ----
    for _ in range(4):                       # 反复几轮吃下浅层嵌套
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", s)

    # ---- 3. 上下标 ----
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^\1", s)
    s = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", s)

    # ---- 4. 环境 ----
    s = re.sub(r"\\begin\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\\end\s*\{[^{}]*\}", " ", s)
    s = s.replace("\\\\", " ")               # 换行
    s = s.replace("&", " ")                  # 对齐制表符

    # ---- 5. 只影响排版的命令 ----
    for cmd in _DROP_WITH_ARG:
        s = s.replace(cmd, " ")

    # ---- 6. 函数名要能当词读：\tan → " tan "，避免 xtanα ----
    for fn in _FUNCTIONS:
        name = SYMBOLS.get(fn) or fn.lstrip("\\")     # 表里没有的就用命令名本身
        s = re.sub(re.escape(fn) + r"(?![a-zA-Z])", f" {name} ", s)

    # ---- 7. 其余符号 ----
    for cmd in _GREEK_KEYS:
        s = s.replace(cmd, SYMBOLS[cmd])

    # ---- 8. 收尾 ----
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)       # 残留的未知命令
    s = s.replace("\\", " ")                 # 落单的反斜杠（\left\{ 之类）
    s = re.sub(r"[\^_]\s*(?=[\s,).]|$)", " ", s)   # 落单的 ^ _
    s = re.sub(r"\s+", " ", s).strip()
    return s


def formula_embed_text(
    latex: str,
    *,
    page: int | None = None,
    chapter: str = "",
    number: str = "",
) -> str:
    """公式块真正拿去向量化的串 = 可读形式 + 定位上下文。

    带上页码、章节与**公式编号**：编号是公式与正文互相引用的锚点
    （正文里会写"由式(1.15)"），带上它才能支持"式1.15"这类查询。
    """
    parts = []
    if number:
        parts.append(f"公式{number}")
    elif page:
        parts.append("公式")
    if page:
        parts.append(f"第{page}页")
    if chapter:
        parts.append(chapter)
    body = latex_to_readable(latex)
    if body:
        parts.append(body)
    return " ".join(parts)
