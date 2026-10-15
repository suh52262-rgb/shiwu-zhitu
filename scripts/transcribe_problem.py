"""把题目截图转成题面文本（P14 自动建图的图像入口）。

为什么需要它：P14 的生成器只接受**文本**（`problem_statement` / `reference_solution`），
而新题往往只有一张截图或照片。本脚本用 VL 模型补上"截图 → 文本"这一步，
之后就能直接交给 `scripts/generate_graph.py` 建图。

设计边界（与项目"核心 Agent 不依赖具体模型"的原则一致）：
这是**输入适配工具**，不是 Agent 的一部分。自动建图本身仍然只依赖文本，
换任何 OCR / 多模态方案都不影响 P14 与 LangGraph 工作流。

用法：
    # 打印转写结果
    .venv/Scripts/python scripts/transcribe_problem.py 题目截图.png

    # 直接落到题目目录，供 generate_graph.py 使用
    .venv/Scripts/python scripts/transcribe_problem.py 题目截图.png --out-dir data/problems/phy_003

环境变量（见 .env.example）：
    QWEN3VL_API_KEY    必填，DashScope 兼容模式的 Key
    QWEN3VL_BASE_URL   选填，默认 https://dashscope.aliyuncs.com/compatible-mode/v1
    QWEN3VL_MODEL      选填，默认依次尝试 qwen3-vl-plus / qwen-vl-max / qwen-vl-plus
"""
import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

# 控制台默认 GBK，输出里有 ✓/⚠ 等字符，先重配编码（与 run_eval.py 一致）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# 关键：必须在**模块导入时**就把 .env 灌进 os.environ。
# pydantic-settings 只把值填进它自己的 Settings，**不会**写 os.environ，
# 而本模块是用 os.getenv 读 QWEN3VL_* 的；被其它脚本 import 时（如
# generate_graph.py --image）若只在 main() 里 load_dotenv，就会读不到 Key。
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv 随 pydantic-settings 一起安装
    pass

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 候选模型链统一在 app.services.vision 里管理——**不要在这里另写一份**。
# 本项目已被模型变更打过两次脸（qwen-vl-max 插图定位不可用、
# qwen3-vl-flash 于 2026-10-10 下架），模型名必须是配置而非代码。
sys.path.insert(0, str(ROOT / "backend"))
try:
    from app.services.vision import VISION_FALLBACKS as CANDIDATE_MODELS
except ImportError:  # pragma: no cover - 后端不可导入时退回内置列表
    CANDIDATE_MODELS = ("qwen3-vl-30b-a3b-instruct", "qwen3-vl-plus", "qwen3-vl-flash")

PROMPT = """请把这张题目截图转写成纯文本，供后续程序自动生成"解题步骤图"使用。

要求：
1. 完整保留题干与所有小问的编号、分值、文字；
2. 数学符号用 LaTeX 行内写法（如 $v_r$、$\\frac{Nm}{M+Nm}v_r$、$\\sum_{n=1}^{N}$）；
3. 图中的"解析/答案"部分请单独用一行 `【参考答案】` 开头，与原题分开；
4. 不要添加任何你自己的解释、点评或补充设问；
5. 直接输出转写结果，不要写"以下是转写"之类的开场白。

输出格式：
【题目】
<题干与小问>

【参考答案】
<图中的解析过程>
"""


class TranscribeError(RuntimeError):
    """转写失败（未配置 Key / 模型不可达 / 图片读不出来）。"""


def _image_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        raise TranscribeError(f"不支持的图片类型 {mime}（支持 png/jpeg/webp/gif）")
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode()}"


def transcribe_raw(image_path: Path, model: str, api_key: str, base_url: str) -> str:
    """调用 VL 模型，返回原始转写文本。"""
    from openai import OpenAI  # openai SDK 已是项目依赖

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        temperature=0.1,
        timeout=180,
    )
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise TranscribeError(f"{model} 返回了空 choices")
    return choices[0].message.content or ""


def transcribe_problem_image(
    image_path: str | Path,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    """截图 → (题面, 参考答案)。

    依次尝试候选模型；全部失败则抛 `TranscribeError`（附带每个模型的原因，
    避免"就是不行"这种无从下手的报错）。
    """
    path = Path(image_path)
    if not path.is_file():
        raise TranscribeError(f"找不到图片：{path}")

    api_key = api_key or os.getenv("QWEN3VL_API_KEY", "")
    if not api_key:
        raise TranscribeError(
            "未配置 QWEN3VL_API_KEY。题目截图需要 VL 模型转写；"
            "也可手工把题面写进 problem.json 的 content 字段后直接建图。"
        )
    base_url = base_url or os.getenv("QWEN3VL_BASE_URL", DEFAULT_BASE_URL)
    models = [model] if model else [os.getenv("QWEN3VL_MODEL") or CANDIDATE_MODELS[0],
                                    *CANDIDATE_MODELS[1:]]

    errors: list[str] = []
    for name in dict.fromkeys(models):          # 去重且保序
        try:
            text = transcribe_raw(path, name, api_key, base_url)
        except Exception as exc:  # noqa: BLE001 逐个候选尝试，最后统一汇报
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        if text.strip():
            return split_transcript(text)
        errors.append(f"{name}: 返回空内容")

    raise TranscribeError("所有候选模型都失败：\n  - " + "\n  - ".join(errors))


def split_transcript(text: str) -> tuple[str, str]:
    """把 VL 输出按【题目】/【参考答案】标记切开。

    模型偶尔不按格式输出：那就整段当作题面，参考答案留空——
    P14 的提示词本身就支持"未提供参考答案"。
    """
    body = text
    for marker in ("【题目】", "[题目]"):
        if marker in body:
            body = body.split(marker, 1)[1]
            break
    for marker in ("【参考答案】", "[参考答案]"):
        if marker in body:
            statement, solution = body.split(marker, 1)
            return statement.strip(), solution.strip()
    return body.strip(), ""


def main() -> int:
    parser = argparse.ArgumentParser(description="题目截图 → 题面文本（供自动建图使用）")
    parser.add_argument("image", help="截图或照片路径")
    parser.add_argument("--model", help="指定 VL 模型（默认依次尝试若干候选）")
    parser.add_argument("--out-dir", help="写入该目录的 statement.txt / solution.txt")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本串联）")
    args = parser.parse_args()

    try:
        statement, solution = transcribe_problem_image(args.image, model=args.model)
    except TranscribeError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    if args.json:
        import json

        print(json.dumps({"statement": statement, "solution": solution}, ensure_ascii=False))
    else:
        print(f"题面（{len(statement)} 字）\n{'-' * 60}\n{statement}")
        if solution:
            print(f"\n参考答案（{len(solution)} 字）\n{'-' * 60}\n{solution}")
        else:
            print("\n（未识别出参考答案，建图时会由模型自行推导解题过程）")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "statement.txt").write_text(statement + "\n", encoding="utf-8", newline="\n")
        (out / "solution.txt").write_text(solution + "\n", encoding="utf-8", newline="\n")
        print(f"\n✓ 已写入 {out}/statement.txt 与 {out}/solution.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
