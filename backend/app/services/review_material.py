"""复盘材料的**来源**与形状 —— 单一真相源（契约 §2.5）。

## 为什么单独一个模块
材料的必需字段、`scope` 取值、指纹算法，
**生产路径（视频运行）与评测路径（合成文本样本）必须逐字一致**。
留在 `api/video.py` 里，评测脚本就得再写一份 —— 复制出来的那份迟早漂开，
而它漂开的表现是"某条材料校验不过"，很难查。

## 两种来源（契约 §2.5）
```
video_run_snapshot        真实视频流程产出
synthesized_text_sample   **评测用合成文本材料**（T4）
```
⚠️ **来源写在数据里**，并会随 `POST /video/review` 的响应回带为 `review_scope` ——
所以一份合成材料产出的复盘报告，**从响应就能看出不是真实视频复盘**。

⚠️ **没有 HTTP 写入口**：写入材料的能力只在服务端函数与一个 CLI 里。
那是一个"写入复盘材料"的动作，不该暴露在 HTTP 面上。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: 真实视频流程产出（语义不变）。
MATERIAL_SCOPE_VIDEO = "video_run_snapshot"
#: 评测用合成文本材料。**它的存在本身就是来源标记。**
MATERIAL_SCOPE_SYNTHESIZED = "synthesized_text_sample"
#: 允许的全部取值。**不在集合里的一律拒绝**（契约不变式 Ⅰ）。
ALLOWED_MATERIAL_SCOPES = frozenset({MATERIAL_SCOPE_VIDEO, MATERIAL_SCOPE_SYNTHESIZED})

#: 材料必需的字段（少一个就拒）。
REQUIRED_FIELDS = frozenset({
    "student_id", "problem_id", "attempt_id", "problem", "rounds",
    "source_revisions", "graph_version", "graph_snapshot", "timeline_length", "scope",
})


def scope_allowed(scope) -> bool:
    """这个来源值允许吗。**未知值一律不允许** —— 放行它就等于来源可以随便写。"""
    return isinstance(scope, str) and scope in ALLOWED_MATERIAL_SCOPES


def fingerprint(data: dict) -> str:
    """材料的指纹：**排序 + 紧凑**地序列化后取 sha256。

    排序与紧凑不是风格问题 —— 不排序的话，同一份内容只要键序不同就会算出不同指纹，
    于是缓存永远不命中、或者同一材料被当成两份。
    """
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def valid_input(data) -> bool:
    """存下来的 JSON 必须**先符合生产者的形状**，下游才敢取字段。

    ⚠️ `scope` 从"必须等于 video_run_snapshot"放宽为"必须属于允许集合"（契约 §2.5），
    其余检查**一条没减**。
    """
    if not isinstance(data, dict):
        return False
    if not REQUIRED_FIELDS.issubset(data):
        return False
    problem, rounds, revisions = data["problem"], data["rounds"], data["source_revisions"]
    return (
        all(isinstance(data[key], str) and data[key] for key in ("student_id", "problem_id", "attempt_id"))
        and isinstance(problem, dict)
        and isinstance(problem.get("statement"), str)
        and isinstance(problem.get("knowledge_points"), list)
        and all(isinstance(kp, str) for kp in problem["knowledge_points"])
        and isinstance(rounds, list)
        and all(isinstance(r, dict) and isinstance(r.get("answer"), str) for r in rounds)
        and isinstance(revisions, dict)
        and all(isinstance(key, str) and key and type(value) is int and value >= 0
                for key, value in revisions.items())
        and isinstance(data["graph_version"], str)
        and (data["graph_snapshot"] is None or isinstance(data["graph_snapshot"], dict))
        and type(data["timeline_length"]) is int and data["timeline_length"] >= 0
        and scope_allowed(data["scope"])
    )


def _build_payload(*, scope: str, student_id: str, problem_id: str, attempt_id: str,
                   problem, rounds: list[dict], state) -> dict:
    """**两条路径共用**的构造器 —— 形状只在这里写一次。

    `source_revisions` / `graph_version` / `graph_snapshot` / `timeline_length` /
    `media_source_ids` 全部**从目标作答的实时状态推出** ——
    因为 `/video/review` 的 `validate_current_input()` 会拿它们与实时作答对账。
    """
    from app.services.video_review_versions import timeline_items

    versions = {str(r.get("source_id")): r.get("source_revision", 0)
                for r in rounds if r.get("source_id")}
    return {
        "student_id": student_id, "problem_id": problem_id, "attempt_id": attempt_id,
        "problem": {"statement": getattr(problem, "statement", "") or getattr(problem, "content", ""),
                    "knowledge_points": list(getattr(problem, "knowledge_points", []) or [])},
        "rounds": rounds, "source_revisions": versions,
        "graph_version": getattr(state, "graph_version", "") or next(
            (r.get("graph_version") for r in reversed(rounds) if r.get("graph_version")), ""),
        "graph_snapshot": getattr(state, "graph_snapshot", None),
        "timeline_length": len(getattr(state, "timeline", []) or []),
        "timeline_snapshot": timeline_items(state) if state else [],
        "media_source_ids": [str(item.get("source_id")) for item in
                             (getattr(state, "media_sources", []) or [])],
        "scope": scope,
    }


def write_material(out_dir: Path, payload: dict) -> Path:
    """把材料原子写入 `review_input.json`（含指纹）。"""
    from app.services.graph_versions import atomic_json

    target = out_dir / "review_input.json"
    atomic_json(target, {"input": payload, "fingerprint": fingerprint(payload)})
    return target


def save_video_material(out_dir: Path, student_id: str, problem_id: str,
                        attempt_id: str, rounds: list[dict]) -> Path:
    """**视频流程**写材料（`scope=video_run_snapshot`，语义与改动前一致）。"""
    from app.services.attempt_store import get_attempt_store
    from app.services.problem_repository import get_problem_repository

    problem = get_problem_repository().get_problem(problem_id)
    record = get_attempt_store().load_attempt(attempt_id) if attempt_id else None
    payload = _build_payload(scope=MATERIAL_SCOPE_VIDEO, student_id=student_id,
                             problem_id=problem_id, attempt_id=attempt_id,
                             problem=problem, rounds=rounds,
                             state=record.state if record else None)
    return write_material(out_dir, payload)


def write_synthesized_material(out_dir: Path, *, student_id: str, problem_id: str,
                               attempt_id: str, rounds: list[dict]) -> Path:
    """为**评测样本**写一份合成文本材料（契约 §2.5）。

    ⚠️ **刻意没有 `scope` 参数**：
    合成路径**不可能**产出 `video_run_snapshot`，
    也就不可能把评测样本伪装成真实视频复盘（契约不变式 Ⅲ）。

    ⚠️ 目录**必须已存在**。请先 `build_synthesized_payload` 校验成功再建目录 ——
    反过来的话，一次"作答不存在"的失败会在 `data/video_runs/` 里留下**空目录**（实测踩过）。
    """
    payload = build_synthesized_payload(student_id=student_id, problem_id=problem_id,
                                        attempt_id=attempt_id, rounds=rounds)
    return write_material(out_dir, payload)


def build_synthesized_payload(*, student_id: str, problem_id: str,
                              attempt_id: str, rounds: list[dict]) -> dict:
    """构造合成材料并**校验**，但**不落盘**（契约 §2.5）。

    ⚠️ 与写入分开是刻意的：调用方应当**先构造成功、再建目录**，
    这样失败路径上什么都不会留下。

    ⚠️ 题干与图**取自真实题库与真实作答状态**，只有"学生写了什么"来自样本。
    作答、学生、题目三者必须真实对应 —— 否则这里就抛错，什么都不写。
    """
    from app.services.attempt_store import get_attempt_store
    from app.services.problem_repository import get_problem_repository

    problem = get_problem_repository().get_problem(problem_id)
    if problem is None:
        raise ValueError(f"题目不存在：{problem_id}")
    record = get_attempt_store().load_attempt(attempt_id)
    if record is None:
        raise ValueError(f"作答不存在：{attempt_id}")
    if record.student_id != student_id or record.problem_id != problem_id:
        raise ValueError("作答不属于指定学生和题目")
    payload = _build_payload(scope=MATERIAL_SCOPE_SYNTHESIZED, student_id=student_id,
                             problem_id=problem_id, attempt_id=attempt_id,
                             problem=problem, rounds=rounds, state=record.state)
    if not valid_input(payload):
        raise ValueError("构造出的合成材料未通过形状校验（这是内部错误，请报 issue）")
    return payload
