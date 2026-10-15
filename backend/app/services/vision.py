"""视觉模型（VL）调用：模型选型与候选链回退。

**为什么要有这个模块，而不是在脚本里写死模型名。**

本项目已经被"模型变更"打过两次脸：

1. `qwen-vl-max` / `qwen-vl-plus`——实测在**插图定位**上完全不可用：
   给的 bbox 裁出来是图注和正文，根本没有图（见 docs/证据-命门测试/）。
   它们比新模型便宜一半，但便宜没用，产出的是垃圾。
2. `qwen3-vl-flash`——2026-10-10 00:00（北京时间）**下架**
   （阿里云公告《部分老旧模型延期下线通知》，notice-id 118420）。
   而它曾经是性价比最优选。

所以模型名必须是**配置**，不是代码。默认值 + 候选链放在这里一处，
切换模型只改环境变量 `QWEN3VL_MODEL`，或改下面这一个元组。

## 选型依据：10 页横评（不同版式，见 docs/PDF导入方案.md）

**单页测试不足以定论**——第一轮只用第 47 页，得出的排序后来被 10 页推翻了。
10 页覆盖：基准页 / 表格页 / 多图页 / 公式密集页。

| 模型 | JSON | 图 | 第146页 | 均耗时 | 输入tok/页 | 平均IoU | 校准页IoU |
|---|---|---|---|---|---|---|---|
| **`deepseek-flash`**（关思考） | **10/10** | 17 | **4** | **1.8 s** | **1198** | 0.841 | 0.647 |
| `deepseek-chat` | 10/10 | 17 | 4 | 2.0 s | 1198 | 0.818 | 0.509 |
| `qwen3-vl-30b-a3b-instruct` | 9/10 | 13 | **0 ← 全漏** | 3.9 s | 2782 | **0.942** | 0.909 |
| `glm-4.6v` | 9/10 | 15 | 4 | 39.3 s | 6326 | 0.939 | **0.943** |
| `glm-4.5v` | 9/10 | 15 | 4 | 39.1 s | 5860 | 0.911 | 0.857 |

（校准页 = 第 47 页，有**人工核对过**的正解，图1-17 bbox = [495,293,842,392]。
第一版的"共识指标"是**同义反复**、会把所有模型都报成 100%，
是拿校准页才发现的——指标本身也要被校准。）

**四条关键教训**（都违反直觉）：

1. **"公式认得好"与"位置指得准"是两种能力，而且会分离。**
   `qwen3-vl-30b-a3b` 的 bbox 最准（0.942），却**在第 146 页把 4 张插图全漏了**；
   `deepseek-flash` 的框偏松（0.841）但**一张不漏**。
   **漏检比框松严重得多**——所以综合可靠性让 deepseek 排第一。

2. **"更专的 OCR 模型"不等于更好。**
   `qwen-vl-ocr` 把整段中文散文也当成公式吐出来，还把提示语混进公式——
   它是纯 OCR 模型，**不理解"只提取公式"这条指令**。

3. **"调用没报错"不等于"看见了图"。**
   `deepseek-v4-pro` 会**接受**图片却不报错，只在 reasoning 里
   嘀咕"显示[Unsupported Image]…可能无法看到图片"。
   判断视觉能力**必须问一个只有看见图才能答对的问题**
   （探针：「页面右上角那条公式的编号是什么？」→ 正解 `(1.14)`）。

4. **思考模式不只是慢，是更差。**
   `deepseek-flash` 默认开思考。同一页：默认 22 s / 6151 输出 token /
   bbox IoU **0.36**；关掉后 2.7 s / 291 token / IoU **0.56**。
   所以 `call_vision` 对 deepseek 默认注入 `NO_THINKING`。

**还有一条护栏规则**（比选模型更耐用）：
页面文字层有 `图X-Y` 图注、模型却报 0 张图 → 疑似漏检。
10 页里它准确抓到了第 146 页那一次，且**零误报**。
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ---------------------------------------------------------------- 供应商

@dataclass(frozen=True)
class Provider:
    """一个 OpenAI 兼容的视觉模型供应商。"""

    name: str
    base_url: str
    key_env: str

    def api_key(self) -> str:
        return os.getenv(self.key_env, "").strip()


#: 候选链可以跨供应商——**这正是"不锁定单一模型/厂商"的落点**。
#: 各家都提供 OpenAI 兼容端点，所以接入成本只是加一行。
PROVIDERS: dict[str, Provider] = {
    "dashscope": Provider(
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "QWEN3VL_API_KEY",
    ),
    "zhipu": Provider(
        "zhipu",
        "https://open.bigmodel.cn/api/paas/v4",
        "ZHIPU_API_KEY",
    ),
    "deepseek": Provider(
        "deepseek",
        "https://api.deepseek.com",
        "DEEPSEEK_API_KEY",
    ),
}


@dataclass(frozen=True)
class Candidate:
    """候选链的一项：用哪个供应商的哪个模型。"""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


#: 主力模型。10 页不同版式实测：JSON 10/10、无漏检、最快（1.8s）、最省（1198 tok/页）。
#:
#: ⚠️ 必须**关掉思考模式**（见 `NO_THINKING`）。它默认开启，开启时会把输出预算
#: 烧在思考上：实测同一页 22s / 6151 输出 token / bbox IoU 0.36，
#: 关掉后 2.7s / 291 token / IoU 0.56——不只是慢，是**更差**。
DEFAULT_VISION_MODEL = "deepseek-flash"

#: 关掉 DeepSeek 思考模式的参数（供应商专有）。
NO_THINKING: dict = {"thinking": {"type": "disabled"}}

#: 候选链，按顺序尝试。
#:
#: 排序依据（10 页横评 + 第 47 页人工核对正解，详见模块文档）：
#: 1. `deepseek-flash`——**综合最稳**：JSON 10/10、无漏检、1.8s、1198 tok/页。
#:    代价是 bbox 精度略逊（平均 IoU 0.841 vs qwen 的 0.942）
#: 2. `qwen3-vl-30b-a3b-instruct`——**bbox 最准**（平均 IoU 0.942），
#:    但第 146 页**把 4 张插图全漏了**。适合做插图精修，不适合当唯一来源
#: 3. `zhipu/glm-4.6v`——质量与 qwen 相当（0.939）且无漏检，但**慢 10 倍**（39s）
#: 4. `zhipu/glm-4.5v`——同上但更慢、垃圾页更多
#: 5. `dashscope/qwen3-vl-flash`——2026-10-10 下架，留链尾救急
VISION_FALLBACKS: tuple[Candidate, ...] = (
    Candidate("deepseek", "deepseek-flash"),
    Candidate("dashscope", "qwen3-vl-30b-a3b-instruct"),
    Candidate("zhipu", "glm-4.6v"),
    Candidate("zhipu", "glm-4.5v"),
    Candidate("dashscope", "qwen3-vl-flash"),
)


class VisionUnavailable(RuntimeError):
    """所有候选视觉模型都不可用。异常消息里带每个候选的失败原因。"""


def vision_candidates() -> list[Candidate]:
    """返回本次要尝试的候选列表（去重保序）。

    `QWEN3VL_MODEL` 优先，格式 `provider/model` 或裸模型名
    （裸名按 dashscope 解释），设置后就**只用它**，方便排障与影子对比。
    """
    override = os.getenv("QWEN3VL_MODEL", "").strip()
    if override:
        if "/" in override:
            provider, model = override.split("/", 1)
            return [Candidate(provider, model)]
        return [Candidate("dashscope", override)]
    return list(dict.fromkeys(VISION_FALLBACKS))


def vision_models() -> list[str]:
    """候选链里的模型名（兼容旧调用方）。"""
    return [c.model for c in vision_candidates()]


def resolve_candidate(model: str) -> list[Candidate]:
    """把调用方给的模型名解析成候选。

    支持两种写法：
    · `"provider/model"` —— 明确指定供应商
    · `"model"`          —— **先在已知候选链里找它的供应商**

    ## 为什么第二种必须查候选链
    原来裸名一律按 `dashscope` 解释，于是：
    ```
    call_vision(img, prompt, model="deepseek-flash")
      → 请求 dashscope/deepseek-flash → 404 The model does not exist
    ```
    **而 `vision_models()` 返回的就是裸名**——调用方从那里拿到名字再传回来
    必然失败，**往返不成立**。（实测踩到。）

    未知的裸名仍然按 `dashscope` 解释，保持历史行为。
    """
    m = (model or "").strip()
    if not m:
        return vision_candidates()
    if "/" in m:
        return [Candidate(*m.split("/", 1))]
    known = [c for c in VISION_FALLBACKS if c.model == m]
    return known[:1] if known else [Candidate("dashscope", m)]


def available() -> bool:
    """只要有**任意一个**候选的 key 配好了就算可用。"""
    return any(PROVIDERS[c.provider].api_key() for c in vision_candidates())


def sniff_mime(data: bytes) -> str:
    """从字节头判断图片 MIME。**默认按 PNG，但 JPEG 要认出来。**

    为什么必须认：`call_vision` 原先**硬编码 `image/png`**，
    而视频帧是 JPEG（关键帧按 q95 编码，见 `app/vision/service.py`）。
    把 JPEG 字节标成 PNG 是在赌供应商会嗅探——**有的会，有的直接报错**。
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return "image/png"


def call_vision(
    image: bytes | Path | str,
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 180.0,
    temperature: float = 0.1,
    extra_body: dict | None = None,
    mime: str | None = None,
) -> tuple[str, dict]:
    """把一张图交给视觉模型，返回 (模型输出文本, 元信息)。

    依次尝试候选链；全部失败抛 `VisionUnavailable`，
    消息里列出每个候选的失败原因（避免"就是不行"这种无从下手的报错）。

    `extra_body` 用于供应商专有参数。目前唯一的用途是**关掉 DeepSeek 的思考模式**
    （`{"thinking": {"type": "disabled"}}`）：它默认开启，且开启时不接受
    `temperature`。不关掉的话 `deepseek-flash` 会把输出预算烧在思考上、
    返回空的 `content`——实测踩过这个坑。

    `mime` 留空则**从字节头嗅探**（见 `sniff_mime`）。
    调用方明确知道格式时可显式传入，避免嗅探误判。

    元信息含 `provider` / `model` / `elapsed_s` / `prompt_tokens` /
    `completion_tokens`，便于记账与排障。
    """
    import time

    from openai import OpenAI

    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    else:
        data = image
    b64 = base64.b64encode(data).decode()
    use_mime = mime or sniff_mime(data)

    if model:
        candidates = resolve_candidate(model)
    else:
        candidates = vision_candidates()

    errors: list[str] = []
    for candidate in candidates:
        from app.services.request_execution import bounded_timeout
        request_timeout = bounded_timeout(timeout)
        provider = PROVIDERS.get(candidate.provider)
        if provider is None:
            errors.append(f"{candidate}: 未知供应商 {candidate.provider}")
            continue
        key = provider.api_key()
        if not key:
            errors.append(f"{candidate}: 未配置 {provider.key_env}")
            continue

        from app.services.llm.circuit import AccountCircuit
        from app.services.llm.base import LLMError, classify_exception
        guard = AccountCircuit(candidate.provider, provider.base_url, key)
        try:
            ticket = guard.before_call()
        except LLMError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        client = OpenAI(api_key=key, base_url=provider.base_url, max_retries=0)
        started = time.perf_counter()
        kwargs: dict = {}
        # DeepSeek 的模型默认开思考模式，会把输出预算烧在思考上、拿不到正文。
        # 实测同一页：默认 22s / 6151 输出 token / bbox IoU 0.36；
        # 关掉 2.7s / 291 token / IoU 0.56。**不只是慢，是更差。**
        if candidate.provider == "deepseek" and extra_body is None:
            kwargs["extra_body"] = NO_THINKING
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            resp = client.chat.completions.create(
                model=candidate.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         # **不要写死 image/png**：视频帧是 JPEG，
                         # 标错格式是在赌供应商会嗅探（有的会，有的报错）。
                         "image_url": {"url": f"data:{use_mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=temperature,
                timeout=request_timeout,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 逐个候选尝试，最后统一汇报
            guard.failed(classify_exception(exc), ticket)
            reason = f"{type(exc).__name__}: {str(exc)[:140]}"
            errors.append(f"{candidate}: {reason}")
            logger.warning("视觉模型 %s 调用失败：%s", candidate, reason)
            continue

        text = resp.choices[0].message.content or ""
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
        if not text.strip():
            # 有些模型"接受"图片却看不见（如 deepseek-v4-pro），
            # 只在 reasoning 里嘀咕；也有的是思考模式烧光了输出预算。
            # 这里显式当成失败，避免静默产出空结果。
            hint = "（疑似思考模式烧光了输出预算）" if reasoning else \
                   "（该模型可能不支持图片输入）"
            errors.append(f"{candidate}: 返回空内容{hint}")
            continue

        usage = getattr(resp, "usage", None)
        meta = {
            "provider": candidate.provider,
            "model": candidate.model,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "image_kb": round(len(data) / 1024),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "reasoning_chars": len(reasoning),
        }
        return text, meta

    raise VisionUnavailable("所有候选视觉模型都失败：\n  - " + "\n  - ".join(errors))


#: 官方单价（美元 / 百万 token，取低峰/标准档）。价格会变，只用于量级估算。
PRICES: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.15, 0.6),
    "deepseek-chat": (0.15, 0.6),
    "qwen3-vl-30b-a3b-instruct": (0.108, 0.43),
    "qwen3-vl-plus": (0.144, 1.434),
    "qwen3-vl-flash": (0.022, 0.215),
    "glm-4.6v": (0.14, 0.42),
    "glm-4.5v": (0.28, 0.84),
}


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str | None = None) -> float:
    """按官方价估算单次调用成本（美元）。

    未知模型返回 0——**宁可记账缺失，也不要报一个假数字**。
    """
    target = model or (vision_models() or [""])[0]
    target = target.split("/", 1)[-1]          # 容忍 provider/model 写法
    price = PRICES.get(target)
    if not price:
        return 0.0
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
