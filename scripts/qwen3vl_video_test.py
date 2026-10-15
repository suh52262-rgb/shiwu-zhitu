"""Qwen 视频理解能力测试（成员 C「视频过程感知」方向预研）。

流程：本地视频 → OpenCV 均匀抽帧 → base64 JPEG → DashScope 兼容接口（OpenAI 协议）
→ 视觉模型按时间顺序解析视频内容。

用法：
    QWEN3VL_API_KEY=sk-xxx python scripts/qwen3vl_video_test.py <视频路径> [模型名]

说明：
- 密钥从环境变量读取，禁止硬编码
- 默认模型 qwen3.7-flash，若该模型不支持视频输入则自动回退 qwen3-vl-flash
- 与主框架的关系：主 Agent 只消费 LearningEvent，本脚本属于「原始视频 → 事件」的
  前置感知层实验（任务书第 4.3 节），不接入 LangGraph 工作流
"""
import base64
import os
import sys
from pathlib import Path

import cv2

BASE_URL = os.getenv("QWEN3VL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_MODELS = ["qwen3.7-flash", "qwen3-vl-flash"]  # 依次尝试
MAX_FRAMES = 10
TARGET_WIDTH = 1280  # 抽帧后缩放，控制请求体积


def extract_frames(video_path: str, n: int = MAX_FRAMES) -> list[str]:
    """均匀抽 n 帧，返回 base64 JPEG data URL 列表。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("无法读取视频帧（文件损坏或编码不支持）")
    data_urls: list[str] = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / n))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > TARGET_WIDTH:
            frame = cv2.resize(frame, (TARGET_WIDTH, int(h * TARGET_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode()
            data_urls.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return data_urls


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
    models = [sys.argv[2]] if len(sys.argv) > 2 else DEFAULT_MODELS
    api_key = os.getenv("QWEN3VL_API_KEY")
    if not api_key:
        sys.exit("请设置环境变量 QWEN3VL_API_KEY")
    if not Path(video).exists():
        sys.exit(f"视频不存在：{video}")

    frames = extract_frames(video)
    print(f"视频：{video}")
    print(f"抽帧：{len(frames)} 帧（宽 {TARGET_WIDTH}px，JPEG q80）")
    payload_kb = sum(len(u) for u in frames) // 1024
    print(f"请求体积：约 {payload_kb} KB（base64）")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    prompt = (
        "这是一段屏幕录制视频按时间顺序均匀抽取的帧。请按时间顺序描述视频内容："
        "1. 画面里是什么应用/界面；2. 用户做了什么操作（输入了什么文字、发生了什么变化）；"
        "3. 如果能看到正在输入公式或文字，请指出具体内容。"
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}],
        }
    ]

    last_error = None
    for model in models:
        print(f"\n== 尝试模型：{model} ==")
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, timeout=180
            )
            print(f"[{model}] 解析结果：\n")
            print(resp.choices[0].message.content)
            print(f"\nVIDEO_PARSE_OK（模型：{model}）")
            return
        except Exception as exc:  # noqa: BLE001 尝试下一个候选模型
            last_error = exc
            msg = str(exc)[:300]
            print(f"  失败：{msg}")
            if "stage" not in msg and "Ark" not in msg:
                continue
    sys.exit(f"所有候选模型均失败，最后错误：{last_error}")


if __name__ == "__main__":
    main()
