"""题目截图转写工具的测试（P14 的图像入口）。

只测**纯逻辑**与错误路径，不发起任何网络请求：
VL 调用本身依赖外部 Key 与网络，放进单测只会让"没配 Key 的机器"红一片。
真正需要联网的那一步由 `scripts/transcribe_problem.py` 手工运行验证。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_transcriber():
    spec = importlib.util.spec_from_file_location(
        "transcribe_problem", ROOT / "scripts" / "transcribe_problem.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["transcribe_problem"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def transcriber():
    return _load_transcriber()


# ---------------------------------------------------------------- 输出切分

def test_split_transcript_separates_statement_and_solution(transcriber):
    text = "【题目】\n质量为 M 的小车……\n\n【参考答案】\n由动量守恒 Mv + Nm(v-v_r)=0"
    statement, solution = transcriber.split_transcript(text)

    assert statement.startswith("质量为 M 的小车")
    assert "【题目】" not in statement          # 标记本身不该混进题面
    assert solution.startswith("由动量守恒")
    assert "【参考答案】" not in solution


def test_split_transcript_without_solution(transcriber):
    statement, solution = transcriber.split_transcript("【题目】\n只有题面，没有解析")

    assert statement == "只有题面，没有解析"
    assert solution == ""


def test_split_transcript_without_markers_falls_back_to_whole_text(transcriber):
    """模型不按格式输出时，整段当题面——P14 的提示词本就支持"无参考答案"。"""
    statement, solution = transcriber.split_transcript("模型忘了加标记，直接给了题面")

    assert statement == "模型忘了加标记，直接给了题面"
    assert solution == ""


def test_split_transcript_accepts_halfwidth_brackets(transcriber):
    statement, solution = transcriber.split_transcript("[题目]\n题面\n[参考答案]\n解析")

    assert statement == "题面"
    assert solution == "解析"


def test_split_transcript_strips_surrounding_whitespace(transcriber):
    statement, solution = transcriber.split_transcript("【题目】\n\n  题面  \n\n【参考答案】\n\n  解析  \n")

    assert statement == "题面"
    assert solution == "解析"


# ---------------------------------------------------------------- 错误路径

def test_missing_api_key_gives_actionable_error(transcriber, monkeypatch, tmp_path):
    """没配 Key 时必须说清"怎么办"，而不是抛一个裸 KeyError。"""
    monkeypatch.delenv("QWEN3VL_API_KEY", raising=False)
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(transcriber.TranscribeError) as excinfo:
        transcriber.transcribe_problem_image(image)

    message = str(excinfo.value)
    assert "QWEN3VL_API_KEY" in message
    assert "problem.json" in message          # 给出绕开 VL 的办法


def test_missing_image_file_is_reported(transcriber, monkeypatch):
    monkeypatch.setenv("QWEN3VL_API_KEY", "sk-fake")
    with pytest.raises(transcriber.TranscribeError, match="找不到图片"):
        transcriber.transcribe_problem_image("D:/不存在的目录/没有这张图.png")


def test_unsupported_image_type_is_rejected(transcriber, monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN3VL_API_KEY", "sk-fake")
    doc = tmp_path / "题目.pdf"
    doc.write_bytes(b"%PDF-1.4")

    with pytest.raises(transcriber.TranscribeError, match="不支持的图片类型"):
        transcriber.transcribe_problem_image(doc)


# ---------------------------------------------------------------- 提示词约束

def test_prompt_forbids_adding_extra_questions(transcriber):
    """提示词必须明确禁止模型自行补充设问——否则生成的图会包含题目没有的步骤。"""
    assert "不要添加" in transcriber.PROMPT
    assert "【参考答案】" in transcriber.PROMPT
