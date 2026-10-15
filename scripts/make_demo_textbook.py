"""生成内置演示教材 PDF（data/textbooks/大学物理演示教材.pdf）。

用途：在拿到真实教材 PDF 之前，用内容正确的演示教材打通 RAG 全流程。
真实教材到位后，直接放入 data/textbooks/ 重新运行 build_rag.py 即可。
运行：.venv/Scripts/python scripts/make_demo_textbook.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import fitz  # PyMuPDF  # noqa: E402

from app.config import PROJECT_ROOT  # noqa: E402

# 中文内容需要 CJK 字体：按优先级尝试 Windows 自带字体
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]

# 每个条目 = 一页（页码即 metadata.page，章节标题用于 chapter 归属）
SECTIONS = [
    (
        "《大学物理演示教材》",
        "视物知途项目内置演示教材\n\n"
        "说明：本 PDF 由项目自动生成，用于在接入真实教材之前演示 RAG 检索流程。\n"
        "内容为大学物理（力学部分）的标准表述，可支撑机械能相关问题的知识检索。\n"
        "拿到真实教材后，请将其放入 data/textbooks/ 目录并重新运行 build_rag.py。",
    ),
    (
        "第2章 牛顿运动定律",
        "2.1 牛顿第二定律\n\n"
        "物体的加速度与所受合外力成正比，与质量成反比，加速度方向与合外力方向相同："
        "F合 = ma。\n\n"
        "受力分析的一般步骤：第一步，确定研究对象；第二步，按先重力、再弹力、"
        "后摩擦力的顺序找出物体受到的每一个力；第三步，建立坐标系（斜面问题通常沿"
        "斜面方向与垂直斜面方向分解），列方程求解。\n\n"
        "2.2 滑动摩擦力\n\n"
        "滑动摩擦力大小为 f = μN，方向与相对滑动方向相反。式中 N 是接触面间的"
        "正压力。特别注意：在倾角为 θ 的斜面上，物体对斜面的正压力 N = mg·cosθ，"
        "而不是 mg。这是斜面问题中最常见的错误之一。",
    ),
    (
        "第4章 机械能守恒定律",
        "4.1 功与动能定理\n\n"
        "恒力做功：W = Fs·cosα，α 是力与位移方向的夹角。摩擦力方向与位移相反时做负功。\n\n"
        "动能定理：合外力对物体所做的功等于物体动能的变化量，"
        "W合 = ΔEk = (1/2)mv2² − (1/2)mv1²。\n\n"
        "动能定理是普遍规律，既适用于直线运动也适用于曲线运动，既适用于恒力也适用于"
        "变力。当机械能不守恒时，动能定理是分析问题的首选工具。",
    ),
    (
        "第4章 机械能守恒定律",
        "4.2 机械能守恒定律及其条件\n\n"
        "定律内容：在只有重力（或弹力）做功的情形下，物体的动能与重力势能（弹性势能）"
        "相互转化，机械能的总量保持不变。\n\n"
        "表达式：(1/2)mv1² + mgh1 = (1/2)mv2² + mgh2。\n\n"
        "守恒条件（判断依据）：只有重力或弹力做功，其他力不做功，或它们做功的代数和"
        "为零。判断机械能是否守恒，不是看物体是否受摩擦力以外的力，而是看这些力"
        "是否做功、做功之和是否为零。\n\n"
        "重要结论：若摩擦力对物体做了负功，则机械能减少，减少的机械能转化为内能，"
        "此时机械能不守恒，应改用动能定理分析。",
    ),
    (
        "第4章 机械能守恒定律",
        "4.3 斜面模型典型分析\n\n"
        "情景：质量为 m 的物体从高 h、倾角 θ 的粗糙斜面顶端由静止滑下，到达底端速度为 v。\n\n"
        "受力分析：重力 mg、支持力 N = mg·cosθ、滑动摩擦力 f = μN = μmg·cosθ，"
        "方向沿斜面向上。\n\n"
        "各力做功：重力做正功 W_G = mgh；支持力与速度方向垂直，不做功；"
        "摩擦力做负功 W_f = −f·s = −μmg·cosθ·s，其中斜面长 s = h / sinθ。\n\n"
        "由动能定理：mgh + W_f = (1/2)mv²，所以摩擦力做的功 "
        "W_f = (1/2)mv² − mgh（代入数值后为负）。\n\n"
        "结论：因为摩擦力做负功，下滑过程机械能不守恒；机械能的减少量等于克服"
        "摩擦力所做的功。",
    ),
    (
        "第4章 机械能守恒定律",
        "4.4 功能关系\n\n"
        "除重力（和弹力）以外的其他力对物体做的功，等于物体机械能的变化量："
        "W其它 = ΔE机械。这是判断机械能变化量的普适关系。\n\n"
        "相互摩擦的一对滑动摩擦力做功之和恒为负，其大小等于滑动摩擦力与相对位移的"
        "乘积，等于系统内能的增加量：Q = f·s相对。摩擦生热即来源于此。\n\n"
        "能量守恒定律：能量既不会凭空产生也不会凭空消失，只能从一种形式转化为另一种"
        "形式，或从一个物体转移到另一个物体，总量保持不变。",
    ),
]


def find_font() -> str:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise FileNotFoundError("未找到中文字体（simhei/msyh/simsun），请确认 Windows 字体目录")


def main() -> None:
    font_path = find_font()
    out_path = PROJECT_ROOT / "data" / "textbooks" / "大学物理演示教材.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    for title, body in SECTIONS:
        page = doc.new_page()  # A4
        page.insert_font(fontname="cjk", fontfile=font_path)
        page.insert_textbox(
            fitz.Rect(50, 60, 545, 790),
            f"{title}\n\n{body}",
            fontname="cjk",
            fontsize=12,
            align=0,
        )
    doc.save(str(out_path))
    doc.close()
    print(f"OK {out_path}")


if __name__ == "__main__":
    main()
