"""Reproduce the AI-authored source dataset; never regenerate a frozen batch in place."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("correct", "legal_equivalent", "missing_reasoning", "explicit_error", "ocr_ambiguity", "truncated_cross_round")


def problem(pid, split, statement, kps, solution, materials, notes, diagram=None):
    cases = []
    for category, rounds in zip(CATEGORIES, materials, strict=True):
        expected = {
            "judgment": "observed_error" if category == "explicit_error" else "uncertain" if category in {"ocr_ambiguity", "truncated_cross_round"} else "no_observed_error",
            "adjudication_notes": notes[category],
            "minimum_observed_errors": 1 if category == "explicit_error" else 0,
            "maximum_observed_errors": None if category == "explicit_error" else 0,
            "must_not_claim": ["认知卡顿、注意力、掌握程度或学习效果", "未展示的推理已经写出", "系统提示是学生作答", "信息缺失本身就是物理错误"],
            "allow_empty_findings": category in {"correct", "legal_equivalent", "missing_reasoning"},
        }
        cases.append({"id": f"{pid}_{category}", "category": category, "origin": "ai_authored", "rounds": rounds, "expected": expected})
    result = {"id": pid, "split": split, "origin": "ai_authored", "statement": statement, "knowledge_points": kps, "reference_solution": solution, "cases": cases}
    if diagram:
        result["diagram"] = f"diagrams/{diagram}.svg"
        result["diagram_origin"] = "ai_authored_svg_schematic"
        result["diagram_role"] = "problem_illustration_not_teaching_knowledge_graph"
    return result


def notes(correct, equivalent, missing, error, ocr, truncated):
    return dict(zip(CATEGORIES, [correct, equivalent, missing, error, ocr, truncated], strict=True))


PROBLEMS = [
    problem(
        "t4_dev_force", "development",
        "质量2 kg的物块在水平面上从静止开始向右运动，恒定水平拉力10 N向右，滑动摩擦力恒为4 N向左。物块向右移动3 m，求加速度和此时速率。忽略其他阻力。示意图只标力的方向，不按比例。",
        ["牛顿第二定律", "匀变速直线运动", "动能定理"],
        "取右为正，合力10-4=6 N，a=6/2=3 m/s²向右。v²=2as=18 m²/s²，速率v=3√2 m/s。等价：合外力功18 J=½mv²。",
        [
            ["取向右为正，合力F-f=10-4=6 N，a=6/2=3 m/s²。由v²=2as=2×3×3=18，v=3√2 m/s。"],
            ["取向左为正，a=-3 m/s²，位移Δx=-3 m。v_x²=2aΔx=18；因为物块向右，v_x=-3√2 m/s，速率3√2 m/s。"],
            ["加速度3 m/s²，方向向右；速率3√2 m/s。"],
            ["拉力和摩擦力都推动物块，因此合力10+4=14 N，a=7 m/s²，v=√42 m/s。"],
            ["取右为正，F合=10 ? 4=6 N（中间运算符识别不清），a=3 m/s²；v=3√2 m/s。"],
            ["取右为正，合力为6 N，a=3 m/s²。", "v²=2as=18；接下来写v="],
        ],
        notes("完整正确；不能指控缺少摩擦或错误。", "负加速度和负速度来自合法坐标选择；速率为正，不能判符号错误。", "结论正确但推理未展示；可以建议补合力方程，不能断言用了错误方法。", "明确把反向摩擦与拉力相加；应指出合力应为6 N，而不是14 N。", "算符不清，不能把问号确定为加号并指控错误；可请求核对算符。", "跨轮合并可得已写a和v²；最终开方未完成，不能说学生从未列运动学关系。"),
        "force",
    ),
    problem(
        "t4_dev_fall", "development",
        "质量0.50 kg的小球从距离地面5.0 m处由静止自由落下，忽略空气阻力，g=10 m/s²，地面为重力势能零点。求落地前瞬间速率，以及下落过程重力所做的功。",
        ["机械能守恒", "重力做功", "自由落体"],
        "v=√(2gh)=10 m/s；重力功W=mgh=0.50×10×5=25 J，为正。也可由h=½gt²得t=1 s，再v=gt。",
        [
            ["忽略空气阻力，mgh=½mv²，v=√(2×10×5)=10 m/s。重力与位移同向，W_G=mgh=25 J。"],
            ["以竖直向上为正，a=-10 m/s²，Δy=-5 m，v_y=-10 m/s，速率10 m/s；W_G=(-mg)Δy=25 J。"],
            ["落地前速率10 m/s；重力功25 J。"],
            ["小球向下而重力势能减少，所以重力功W_G=-mgh=-25 J；由能量关系速率仍为10 m/s。"],
            ["v=10 m/s；W_G=2S J（最后一位看不清，可能为5，也可能为字母S）。"],
            ["mgh=½mv²，代入得25=0.25v²。", "所以v²=100，重力做功W_G="],
        ],
        notes("两项正确且理由充分。", "合法上正坐标下速度为负，速率为正；重力功为正。", "未展示方法，不能把正确数值指为错误或证明已理解守恒。", "重力势能减小对应重力正功；-25 J是明确错误，速度结论正确不能一并否定。", "2S无法确定是否25；应核查字符，不应确定指控数值错。", "跨轮已推出v²=100，但未写速率和功的最终值；可提示完成，不能虚构学生已写25 J。"),
    ),
    problem(
        "t4_dev_circuit", "development",
        "理想12 V直流电源与R₁=2 Ω、R₂=4 Ω串联，忽略导线电阻。求电路电流、R₁两端电压及R₂的电功率。图中无其他支路。",
        ["串联电路", "欧姆定律", "电功率"],
        "R总=6 Ω，I=12/6=2 A；U₁=IR₁=4 V；P₂=I²R₂=16 W。U₂=8 V，P₂=U₂I同样有效。",
        [
            ["串联总电阻R=2+4=6 Ω，I=12/6=2 A。U₁=2×2=4 V，P₂=2²×4=16 W。"],
            ["串联分压U₁:U₂=1:2，所以U₁=4 V、U₂=8 V。I=U₂/R₂=2 A，P₂=U₂I=16 W。"],
            ["I=2 A，U₁=4 V，P₂=16 W。"],
            ["串联电流I=12/(2+4)=2 A，U₁=4 V；R₂的功率P₂=I²/R₂=4/4=1 W。"],
            ["I=2 A，U₁=4 V，P₂=I² ? R₂=16 W（乘除符号识别不清）。"],
            ["R总=6 Ω，I=2 A。", "U₁=4 V，P₂=I²R₂=2²×"],
        ],
        notes("正确完整。", "分压法合法，不必按先求电流的参考顺序书写。", "结论正确，但计算依据未展示。", "功率公式明确错用I²/R；正确I²R=16 W，其他两项正确。", "不得把不明算符确定为除号；最终16 W正确，应核查OCR。", "已列正确功率公式但乘数与结果截断；不得说完全没有功率关系。"),
        "circuit",
    ),
    problem(
        "t4_dev_collision", "development",
        "光滑水平面上，1 kg小车以6 m/s向右运动，与静止的2 kg小车碰撞后粘在一起，碰撞时间很短且系统所受外力冲量可忽略。求碰后共同速度（大小和方向）及机械能损失。",
        ["动量守恒", "完全非弹性碰撞", "机械能损失"],
        "取右正，1×6=(1+2)v，v=2 m/s向右。初动能18 J，末动能6 J，损失12 J。粘连碰撞机械能不守恒。",
        [
            ["水平外力冲量可忽略，动量守恒：1×6=3v，v=2 m/s向右。E初=½×1×6²=18 J，E末=½×3×2²=6 J，损失12 J。"],
            ["取左为正，p初=-6 kg·m/s，3v=-6，v=-2 m/s，即2 m/s向右。ΔE_k=6-18=-12 J，所以机械能损失12 J。"],
            ["共同速度2 m/s向右，机械能损失12 J。"],
            ["碰撞后粘在一起但机械能守恒，½×1×6²=½×3v²，所以v=√12 m/s，机械能损失为0。"],
            ["动量方程1×6=3v，v=2 m/s向右；机械能损失=18 ? 6=12 J，运算符无法辨清。"],
            ["动量守恒，v=2 m/s向右。初动能18 J。", "碰后动能½×3×2²=6 J；损失为"],
        ],
        notes("正确。", "速度负号和ΔE负号均合法；损失量为正，不能指控12 J与-12 J矛盾。", "正确结论不能因省略证明判错。", "粘连碰撞错误地用机械能守恒，导致错误速度和零损失。", "结果支持减法但算符未清，不得确定是加法并批错。", "两轮已有所需动能数值，但损失最终值未写；可提示相减而非宣称已完成。"),
    ),
    problem(
        "t4_hold_projectile", "holdout",
        "小球从高20 m的平台边缘以5 m/s水平向右抛出，忽略空气阻力，g=10 m/s²，落地点与平台底部位于同一水平地面。求飞行时间、水平射程和落地前瞬间速率。示意图不按比例。",
        ["平抛运动", "运动的合成", "匀变速直线运动"],
        "竖直初速为零，20=½×10t²，t=2 s。x=5t=10 m。水平速度5 m/s、竖直速度大小20 m/s，合速率√(5²+20²)=5√17 m/s。",
        [
            ["20=½gt²，t=2 s。x=v₀t=10 m。v_x=5 m/s，v_y=gt=20 m/s向下，速率√(5²+20²)=5√17 m/s。"],
            ["取竖直向上为正，-20=½(-10)t²得t=2 s；x=10 m。落地v=(5,-20) m/s，|v|=5√17 m/s。"],
            ["飞行2 s，水平射程10 m，落地速率5√17 m/s。"],
            ["下落时间t=2 s，水平射程10 m。水平速度5 m/s加竖直速度20 m/s，落地速率5+20=25 m/s。"],
            ["t=2 s，x=10 m；v=5√1? m/s，根号内最后一位看不清。"],
            ["20=5t²得t=2 s，x=10 m。", "落地速度分量v_x=5，v_y=-20（上正），合速率√("],
        ],
        notes("完整正确；竖直速度已标向下。", "二维分量和上正坐标合法，不应把-20判成向上运动。", "正确结论未展示推导。", "垂直速度分量不能以标量大小直接相加；25 m/s明确错，应使用平方和开方。", "根号内字符不明，只能核查；不能认定写了错误常数。", "已写正确分量，合成表达截断；不能当作25 m/s或已经完成。"),
        "projectile",
    ),
    problem(
        "t4_hold_buoyancy", "holdout",
        "重力为3.0 N、体积为2.0×10⁻⁴ m³的实心物块，由竖直细绳从上方悬挂并完全浸没在水中，保持静止，不接触容器。水密度1000 kg/m³，g=10 m/s²，忽略绳的重力及体积。求浮力和绳的拉力。",
        ["阿基米德原理", "受力平衡"],
        "完全浸没排水体积等于物块体积，F浮=ρgV=2 N向上。静止时T+F浮=G，T=1 N向上。",
        [
            ["完全浸没，V排=2.0×10⁻⁴ m³。F浮=1000×10×2.0×10⁻⁴=2 N，向上。静止满足T+F浮=3 N，所以T=1 N。"],
            ["取向下为正，3-F浮-T=0；由ρgV得F浮大小2 N，所以T大小1 N，两者均向上。"],
            ["浮力2 N，绳的拉力1 N，方向都向上。"],
            ["物块静止，所以浮力一定等于重力，F浮=3 N；拉力为0 N。"],
            ["F浮=ρgV=2 N；T=3 ? 2=1 N，减号或加号识别不清。"],
            ["F浮=ρgV=2 N。", "竖直平衡T+2=3，所以T="],
        ],
        notes("正确。", "合法取下正，大小与方向表达自洽。", "结论正确，过程缺失不等于错误。", "静止不能忽略绳拉力；排水体积给定使浮力2 N而非3 N。", "最终1 N正确且算符不清；应核查，不得确定作了加法。", "跨轮平衡式已正确但最后拉力未完成，不得虚构已写1 N。"),
    ),
]


def svg(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="360" viewBox="0 0 800 360"><rect width="800" height="360" fill="white"/><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 Z" fill="#263238"/></marker></defs><g fill="#263238" stroke="#263238" stroke-width="3" font-family="sans-serif" font-size="22">' + body + '</g><text x="25" y="335" font-family="sans-serif" font-size="16" fill="#555">AI-authored schematic; not to scale; labels are stated in the question.</text></svg>\n'


DIAGRAMS = {
    "force": '<path d="M70 240H730"/><rect x="310" y="130" width="160" height="110" fill="#e8f1fa"/><text x="345" y="185" stroke="none">m=2 kg</text><path d="M470 170H670" marker-end="url(#arrow)"/><text x="520" y="140" stroke="none">F=10 N</text><path d="M310 210H150" marker-end="url(#arrow)"/><text x="145" y="180" stroke="none">f=4 N</text><path d="M320 275H570" marker-end="url(#arrow)"/><text x="380" y="305" stroke="none">s=3 m</text>',
    "circuit": '<path d="M120 180V90H290 M410 90H490 M610 90H690V250H120V210" fill="none"/><rect x="290" y="65" width="120" height="50" fill="#e8f1fa"/><rect x="490" y="65" width="120" height="50" fill="#e8f1fa"/><path d="M80 180H160 M95 210H145"/><text x="285" y="45" stroke="none">R1=2 Ω</text><text x="490" y="45" stroke="none">R2=4 Ω</text><text x="180" y="205" stroke="none">U=12 V</text><text x="55" y="168" stroke="none">+</text><text x="55" y="232" stroke="none">−</text>',
    "projectile": '<path d="M80 100H280V280H730" fill="none"/><circle cx="285" cy="95" r="8"/><path d="M295 95H440" marker-end="url(#arrow)"/><text x="330" y="65" stroke="none">v0=5 m/s</text><path d="M285 100Q460 115 610 280" stroke-dasharray="8 8" fill="none"/><path d="M210 115V265" marker-end="url(#arrow)"/><text x="85" y="210" stroke="none">h=20 m</text><path d="M295 300H600" marker-end="url(#arrow)"/><text x="435" y="325" stroke="none">x=?</text>',
}


def main():
    (ROOT / "diagrams").mkdir(parents=True, exist_ok=True)
    for name, body in DIAGRAMS.items():
        (ROOT / "diagrams" / f"{name}.svg").write_text(svg(body), encoding="utf-8")
    data = {"schema_version": "t4_source_v1", "origin": "ai_authored", "authored_date": "2026-10-09", "purpose": "engineering_acceptance_not_teacher_or_student_research", "problems": PROBLEMS}
    (ROOT / "dataset.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
