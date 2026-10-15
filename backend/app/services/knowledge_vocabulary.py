"""受控词表：各题的知识点**只能从中选**。

## 为什么需要
推荐的判据是「画像知识点 ∩ 题库知识点」。而实测下来：
· 现有 8 道题共 72 条知识点，其中 **6 组是同义分叉**（`机械能守恒` /
  `机械能守恒条件` / `机械能守恒的适用条件` 是同一个东西）
· `phy_003` 的自动生成图还把**"做题技巧"标成了知识点**
  （`代数化简`、`递推关系`、`求和运算`）

**那些分叉不是一次意外，是"每题各自写"这个做法的必然结果。**
不清根的话，**加新题（如教材例题 466 道）会批量重演**。

## 设计
· `data/knowledge_vocabulary.json` 持有一张表 + 一张别名表
· `write_problem()` 对**表外标签报警**（可用 `allow_new_concepts=True` 放行）
· `snap_concept` 继续兜底 LLM 的自由措辞

## ⚠️ 词表本身需要领域判断
**哪些算同一个知识点，不能由机器定** —— 本轮那几组合并都是人拍板的。
所以增删词表**要人工确认**，本模块只负责"挡住没确认过的"。
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: 词表文件位置（`data/knowledge_vocabulary.json`）
VOCAB_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge_vocabulary.json"


class OutOfVocabulary(RuntimeError):
    """题目的知识点里出现了词表外的标签。"""


@lru_cache(maxsize=4)
def load_vocabulary(path: str | None = None) -> dict:
    """读词表。**载入失败不抛异常，返回空表** ——

    词表是治理工具，不是运行前提。文件坏了不该让整个题库打不开。
    """
    p = Path(path) if path else VOCAB_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("受控词表载入失败（%s）：%s —— 本次不做词表校验", p, exc)
        return {}


def vocabulary(path: str | None = None) -> list[str]:
    """规范名列表。"""
    return sorted((load_vocabulary(path).get("concepts") or {}).keys())


def aliases(path: str | None = None) -> dict[str, str]:
    """别名 → 规范名。"""
    return dict(load_vocabulary(path).get("aliases") or {})


def dropped(path: str | None = None) -> list[str]:
    """**明确不承认**的标签（已被判定为非知识点，如"代数化简"）。

    留在文件里是为了：① 记录这个判断；② 以后误加回来时能被挡住。
    """
    return list(load_vocabulary(path).get("dropped") or [])


def canonical(name: str, path: str | None = None) -> str:
    """把名字归到规范名。别名→规范名；**不认识的逐字返回**。"""
    n = (name or "").strip()
    if not n:
        return ""
    vocab = load_vocabulary(path)
    return (vocab.get("aliases") or {}).get(n, n)


def validate(kps: Iterable[str], path: str | None = None) -> dict:
    """校验一组知识点。

    返回 `{"ok", "unknown", "dropped", "known"}`。
    · `unknown` = 表外且不是别名的 → **新概念，需要人工确认后才进表**
    · `dropped` = 在"明确不承认"名单里的 → **应直接删除**
    """
    vocab = load_vocabulary(path)
    concept_set = set((vocab.get("concepts") or {}).keys())
    alias_map = vocab.get("aliases") or {}
    drop_set = set(vocab.get("dropped") or [])
    if not concept_set:
        return {"ok": True, "unknown": [], "dropped": [], "known": []}

    known, unknown, bad = [], [], []
    for k in kps or []:
        s = (k or "").strip()
        if not s:
            continue
        if s in concept_set or s in alias_map:
            known.append(canonical(s, path))
        elif s in drop_set:
            bad.append(s)
        else:
            unknown.append(s)
    return {"ok": not unknown and not bad,
            "unknown": unknown, "dropped": bad, "known": known}


def assert_in_vocabulary(kps: Iterable[str], *,
                         allow_new: bool = False,
                         path: str | None = None) -> None:
    """不在词表里就抛 `OutOfVocabulary`。

    `allow_new=True` 时**只对"明确不承认"的标签报错** ——
    新概念应当允许，但必须是有意为之（并在提交信息里写明理由）。
    """
    r = validate(kps, path)
    if r["dropped"]:
        raise OutOfVocabulary(
            f"这些标签已被判定**不是知识点**，不该出现在题目里：{r['dropped']}。"
            f"（见 data/knowledge_vocabulary.json 的 `dropped`）"
        )
    if r["unknown"] and not allow_new:
        raise OutOfVocabulary(
            f"这些标签不在受控词表里：{r['unknown']}。\n"
            f"**受控词表的意义就是挡住没确认过的新概念** ——\n"
            f"  ① 确认它确实是新知识点 → 加进 "
            f"{VOCAB_PATH.name} 的 `concepts`，再录题\n"
            f"  ② 它其实是某个已有概念的另一种说法 → 加进 `aliases`\n"
            f"  ③ 确认它不是知识点 → 加进 `dropped`\n"
            f"（确实要临时放行可传 allow_new_concepts=True，但不该成为常态）"
        )
