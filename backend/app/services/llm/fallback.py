"""结构化输出的降级链（P20，任务书第三阶段 P20 节）。

任务书要求的行为：

    第一次解析失败 → 自动重试 → 仍失败 → fallback parser → 仍失败 → 返回安全默认状态

设计原则（这条比实现本身更重要）：

> **降级是"不改状态并如实告知"，不是"编造一个看起来合理的值"。**

反例：`analyze_step` 解析不出来时，如果给 `has_obvious_error` 填一个默认的 `False`，
系统就会判定"学生本轮推进正确" → 不给任何提示 → 学生明明卡住了却得不到帮助，
而且**没有任何人知道出过错**。这比直接报错更糟。

因此本模块只负责两件事：
1. `lenient_parse`：从半截/脏输出里**尽力抢救**真实字段（抢救到的是真的）；
2. `safe_default`：实在抢救不出来时，构造一个**显式标记为降级**的默认实例，
   由调用方决定"是否采用"。**是否采用由节点判断**——对"判断类"字段（是否出错、
   匹配到哪个节点），节点应当选择不采用，只记录降级原因。
"""
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

#: 从 ```json ... ``` 代码块里取内容
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _slice_object(text: str) -> str:
    """取最外层 {...}。模型常在 JSON 前后写解释文字。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _loads_candidates(text: str) -> list[Any]:
    """按"从严格到宽松"依次尝试解析，返回所有成功的候选。

    能走到这里说明严格解析已经失败过，所以这里不追求一次成功，而是多试几种常见畸形。
    """
    base = _slice_object(_strip_fences(text)).strip()
    single_to_double = re.sub(r"'([^']*)'", r'"\1"', base)

    attempts: list[str] = [
        base,
        # 尾逗号（模型很爱写）
        re.sub(r",\s*([}\]])", r"\1", base),
        # 单引号当字符串定界符
        single_to_double,
        re.sub(r",\s*([}\]])", r"\1", single_to_double),
        # Python 风格字面量：True / False / None 不是合法 JSON
        re.sub(
            r"\b(True|False|None)\b",
            lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)],
            single_to_double,
        ),
        # 中文全角标点
        base.replace("“", '"').replace("”", '"').replace("，", ",").replace("：", ":"),
    ]

    out: list[Any] = []
    for candidate in attempts:
        try:
            out.append(json.loads(candidate))
        except (ValueError, TypeError):
            continue
    return out


def _salvage_scalars(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """正则抢救：从文本里按字段名捞 字符串/数字/布尔/字符串数组。

    只在 JSON 整体解析失败时兜底。捞到的是**原文里的真实内容**，不是编造的。
    值的引号可能是双引号也可能是单引号（模型两种都会写）。
    """
    found: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        key = rf'["\']?{re.escape(name)}["\']?\s*[:：]\s*'
        m = re.search(key + r'"([^"]*)"', text) or re.search(key + r"'([^']*)'", text)
        if m:
            found[name] = m.group(1)
            continue
        # 布尔要先于数字匹配：否则 "true" 不会被数字正则吃掉，但顺序写错会漏
        m = re.search(key + r"(true|false|True|False)", text)
        if m:
            found[name] = m.group(1).lower() == "true"
            continue
        m = re.search(key + r"([-\d.]+)", text)
        if m:
            raw = m.group(1)
            found[name] = float(raw) if "." in raw else int(raw)
            continue
        # 字符串数组：["a", "b"] 或 ['a', 'b']
        m = re.search(key + r"\[([^\]]*)\]", text, re.S)
        if m:
            items = re.findall(r'"([^"]*)"', m.group(1)) or re.findall(r"'([^']*)'", m.group(1))
            if items:
                found[name] = items
    return found


def lenient_parse(text: str, schema: type[T]) -> tuple[T | None, list[str]]:
    """宽松解析：尽力从脏输出里抢救出一个合法实例。

    返回 `(实例 或 None, 修复说明)`。修复说明会写进 Trace，便于发现
    "模型老是输出坏 JSON" 这类需要改提示词或换模型的问题。
    """
    if not text or not text.strip():
        return None, ["模型返回了空内容"]

    notes: list[str] = []
    for payload in _loads_candidates(text):
        if not isinstance(payload, dict):
            continue
        cleaned, dropped = _drop_invalid_fields(payload, schema)
        if dropped:
            notes.append(f"丢弃了校验不通过的字段：{'、'.join(dropped)}")
        try:
            return schema.model_validate(cleaned), notes
        except ValidationError:
            continue

    salvaged = _salvage_scalars(text, schema)
    if salvaged:
        cleaned, dropped = _drop_invalid_fields(salvaged, schema)
        try:
            return schema.model_validate(cleaned), [*notes, "整体 JSON 无法解析，按字段正则抢救"]
        except ValidationError:
            notes.append(f"字段抢救后仍不合法（丢弃 {'、'.join(dropped) or '无'}）")
    return None, [*notes, "未能从输出中抢救出任何合法字段"]


def _drop_invalid_fields(payload: dict, schema: type[BaseModel]) -> tuple[dict, list[str]]:
    """逐个剔除校验不过的字段，让"大部分对、个别字段脏"的输出仍可用。"""
    candidate = {k: v for k, v in payload.items() if k in schema.model_fields}
    dropped: list[str] = [k for k in payload if k not in schema.model_fields]
    for _ in range(len(candidate)):
        try:
            schema.model_validate(candidate)
            return candidate, dropped
        except ValidationError as exc:
            bad = _first_bad_field(exc)
            if not bad or bad not in candidate:
                return candidate, dropped
            candidate.pop(bad)
            dropped.append(bad)
    return candidate, dropped


def _first_bad_field(exc: ValidationError) -> str | None:
    for err in exc.errors():
        loc = err.get("loc") or ()
        if loc and isinstance(loc[0], str):
            return loc[0]
    return None


def safe_default(schema: type[T]) -> T:
    """安全默认实例：所有字段用类型零值，**必填字段也不放过**。

    存在的意义是"最后兜底不抛异常"。调用方拿到它时必须知道这是降级产物——
    因此**不要**把它当成真实判断使用（见模块 docstring）。
    """
    zeros: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        zeros[name] = _zero_for(field.annotation)
    try:
        return schema.model_validate(zeros)
    except ValidationError:
        # 连零值都不合法（例如 Literal 限定取值）→ 用模型自己的默认值构造
        return schema.model_construct(**zeros)


def _zero_for(annotation: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is str:
        return ""
    if origin in (list, tuple, set) or annotation in (list, tuple, set):
        return []
    if origin is dict or annotation is dict:
        return {}
    return None
