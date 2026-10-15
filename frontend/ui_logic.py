"""学生端的**纯逻辑**：把"后端响应 → 界面该显示什么"从渲染里拆出来。

## 为什么单独一个模块
`streamlit_app.py` 里原来这些判断都写在渲染路径里，**没法测**。
后果已经出现过一次：新学生（404）被显示成"画像暂不可用（**数据库未连接**）"，
而且那条"新学生"的 `except httpx.HTTPStatusError` **是死代码**
（没调 `raise_for_status()`，httpx 根本不抛）。

**336 行的学生端零测试覆盖**，这类"契约变了、界面不报错、只显示一句错话"
的问题不会有任何东西拦住它。

## 约定
本模块**不导入 streamlit、不发网络请求** —— 只做判断。
调用方负责拿响应、负责渲染。
"""
from __future__ import annotations

from typing import Any, Optional

#: 画像面板的三种状态
MODE_NEW = "new"          # 新学生，还没有记录
MODE_OK = "ok"            # 正常渲染
MODE_ERROR = "error"      # 拿不到，说明原因


def problems_from_payload(status: int, payload: Any) -> Optional[dict[str, dict]]:
    """后端 `GET /problems` 的响应 → `{problem_id: item}`。

    **返回 `None` 表示"这个响应用不了"**，调用方应回退到扫本地题库。
    返回 `{}` 也表示用不了（空题库不该把本地那份盖掉）。

    ⚠️ 后端返回的是 `{"items": [...], "total": N}`，**不是裸数组**。
    这个形状写错过一次（探针里），所以在这里钉死。
    """
    if status != 200 or not isinstance(payload, dict):
        return None
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return None
    out: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        pid = item.get("problem_id") or item.get("id")
        if pid:
            out[str(pid)] = item
    return out or None


def profile_view(status: int, payload: Any) -> dict:
    """后端 `GET /students/{id}/profile` 的响应 → 界面该显示什么。

    返回 `{"mode": ..., "caption": ..., "stats": {...}}`。

    ## 三种情况必须分开
    · **404 → 新学生**。这是**每个学生的第一个状态**，不是错误。
      原来它被显示成"数据库未连接" —— 数据库好得很。
    · **其它非 200** → 把**状态码显出来**，便于排查（不要笼统写"不可用"）。
    · **200** → 正常。字段缺失要兜底，不要 KeyError。
    """
    if status == 404:
        return {"mode": MODE_NEW, "caption": "新学生，暂无画像", "stats": {}}

    if status != 200:
        return {"mode": MODE_ERROR,
                "caption": f"画像暂不可用（后端返回 {status}）",
                "stats": {}}

    if not isinstance(payload, dict):
        return {"mode": MODE_ERROR,
                "caption": "画像暂不可用（响应格式异常）", "stats": {}}

    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        # 200 但字段不对——**不能崩，也不能假装正常**
        attempts = []
    weak = payload.get("observed_error_concepts") if payload.get("profile_policy_version") else []
    if not isinstance(weak, list):
        weak = []
    done = sum(1 for a in attempts
               if isinstance(a, dict) and a.get("final_correct"))
    return {"mode": MODE_OK, "caption": "",
            "stats": {"attempt_count": len(attempts), "done": done,
                      "weak": [str(w) for w in weak]}}


def health_view(status: int, payload: Any) -> dict:
    """后端 `GET /health` → 侧边栏状态该显示什么。"""
    if status == 200 and isinstance(payload, dict) and payload.get("status"):
        return {"ok": True, "text": f"后端已连接（{payload['status']}）"}
    return {"ok": False,
            "text": f"后端已连接但返回异常（{status}）" if status == 200
                    else f"后端未连接（HTTP {status}）"}


def connection_error_text(base_url: str, exc: BaseException) -> str:
    """连不上后端时侧边栏该显示什么（含启动命令提示）。"""
    return (f"后端未连接：{base_url}\n\n{exc}\n\n"
            "请先启动：`cd backend && uvicorn app.main:app --reload`")


# Session transitions share the same implementation in the UI and behavior tests.
#: 按上传编号找回的**三种**结果 —— 必须分开，否则会把"查不通"说成"没建成"。
#:
#: ⚠️ 这是一个**真实缺陷**的落点：旧版把三者一律返回 `False`，
#: 于是网络异常 / 服务端 500 都被显示成
#: 「没有找到任务 ⇒ **可以重新上传，不会重复**」——
#: 而那时任务**可能已经建成并开始付费处理了**。
#:
#: 放在 `ui_logic` 而不是渲染层，是为了**测试能直接引用同一份常量** ——
#: 两边各写一份字符串，改一处就会静默失配。
RECOVER_FOUND = "found"        # 找到了，已接管
RECOVER_ABSENT = "absent"      # 404：确认没有这个编号的任务 ⇒ **可以重传**
RECOVER_UNKNOWN = "unknown"    # 网络 / 5xx / 响应不可用 ⇒ **结果仍未知，不得说可以重传**


def remember_request_id(state, request_id: str, query_params=None) -> None:
    """把上传编号**持久留在会话（并尽量留在 URL）里**，供后面按编号查询。

    ⚠️ 只写在提示文案里不够：学生一刷新页面，编号就没了 ——
    而"结果未知"恰恰是最需要过后再查一次的情形。

    `query_params` 由调用方传（Streamlit 的 `st.query_params`）——
    **不在这一层 import streamlit**，否则它就测不动了。
    """
    state["video_job_request_id"] = request_id
    if query_params is not None:
        query_params["video_request"] = request_id


def forget_request_id(state, query_params=None) -> None:
    """**放弃这一次上传**：清掉已钉住的编号，下一次上传生成新编号（契约 ⑤）。

    ## 为什么必须有这个出口
    契约 T0 不变式⑤的原话：
    > 编号复用**只对"同一次上传"成立**。页面**必须**提供「放弃这次上传、开始新的一次」的出口
    > （清除已钉住的编号 ⇒ 下一次上传生成新编号）。
    > ⚠️ 否则用户**换一个视频就永远 409**：编号被钉住 → 服务端按内容判冲突 → 409
    >    → 提示"用同一编号重试" → 再试还是 409（**死循环**）。

    实测确认过（2026-10-09）：编号同时写进 URL，**刷新也甩不掉**，
    `clear_media` 也不清它 —— 演示时想试第二个视频就会卡死。

    ## ⚠️ 什么时候**不许**调它
    **结果未知时清编号 = 打开重复付费的门**（契约 §2.2：查询失败不重新上传）。
    所以只在两种情形下调：
      ① 用户**明确**点了「放弃这次上传」（他知道那次结果不再找回）；
      ② 服务端已确认那次上传**没建成**（`RECOVER_ABSENT` ⇒ 不存在重复付费风险）。
    """
    state.pop("video_job_request_id", None)
    state.pop("video_job_request_absent_confirmed", None)
    if query_params is not None:
        # 编号写在 URL 里 ⇒ 不清 URL 的话，刷新页面又会把它捡回来。
        try:
            query_params.pop("video_request", None)
        except Exception:  # noqa: BLE001  老版本 Streamlit 的 query_params 只读
            pass


def clear_media(state) -> None:
    state.pop("video_result", None)
    state.pop("video_review_error", None)
    state.pop("video_job_error", None)
    state.pop("recommendations", None)
    # 照片这条路的失败原因与「重试用的上传内容」也是**绑定当前材料**的，
    # 切题/换媒体时必须一起丢，否则会拿着上一位学生的照片重试。
    state.pop("photo_error", None)
    state.pop("photo_retry", None)
    for key in list(state):
        if str(key).startswith("ocr_"):
            state.pop(key, None)
    state["upload_version"] = state.get("upload_version", 0) + 1


def video_review_request(state, student_id: str, problem_id: str) -> dict | None:
    """Bind a review to the visible run and its current OCR revisions."""
    result = state.get("video_result") or {}
    if (state.get("active_session") != (student_id, problem_id)
            or result.get("media_type") != "video"
            or not result.get("run_id") or not result.get("attempt_id")
            or result.get("attempt_id") != state.get("attempt_id")
            or result.get("student_id") != student_id or result.get("problem_id") != problem_id):
        return None
    return {"student_id": student_id, "problem_id": problem_id,
            "attempt_id": result["attempt_id"], "run_id": result["run_id"],
            "input_revision": result.get("input_revision", 0),
            "refresh_input": bool(result.get("review_refresh_required", result.get("ocr_corrected", False))),
            "_ui_correction_generation": result.get("correction_generation", 0),
            "_ui_source_revisions": {r["source_id"]: r.get("source_revision", 0)
                                     for r in result.get("rounds", []) if r.get("source_id")}}


def record_photo_failure(state, reason: str, name: str, data: bytes, mime: str) -> None:
    """照片失败：**原因**与**重试入口**分开存。

    ⚠️ 原因**不写进 `tutor["hint"]`** —— `hint` 是分级提示的载体
    （契约 §1.1 不变式①：空 `hint` 是 L0 的设计语义，错误归因不能挤进去）。
    把错误塞进 hint，学生就会把「处理失败」当成一条教学提示读。

    重试要能**原样重发**同一张照片：学生已经拍过一次了，
    让他为了重试再选一次文件是白折腾。所以这里只留最小的三件。
    """
    state["photo_error"] = reason
    state["photo_retry"] = {"name": name, "data": data, "mime": mime}
    # 上一次的照片结果已由 `clear_media` 丢掉；教学状态**不进不退** ——
    # 这一轮没有产生新分析，就不该假装有。


# ================================================================ 复盘 v3 的三类
#
# 契约 §2.3 把 `findings.category` **固定三选一**，并给死含义：
# ```
# observed_error   有具体原文支持的错误判断
# uncertain        信息不足或识别可疑，需要核查
# suggestion       可选书写或练习建议
# ```
#
# ⚠️ **三类不是"错题的三种程度"** —— 它们要求学生做**不同的事**：
# ```
# 观察到的问题 → 去看那处原文，判断自己是不是真错了
# 需要核查     → 去核对识别文本（**我可能看错了**），先别信下面的结论
# 可选练习     → 不是错，是可以做得更好
# ```
# 把三类都写成「做错了」，就等于把「我可能看错了」和「你可以写得更好」
# 一并说成了「你错了」—— **那是错误归因，本项目的红线。**
CATEGORY_LABELS = {
    "observed_error": "观察到的问题",
    "uncertain": "需要核查",
    "suggestion": "可选练习",
}
CATEGORY_HINTS = {
    "observed_error": "有原文支持的判断；请对着下面的证据自己核一遍。",
    "uncertain": "信息不足，**或我可能看错了识别文本** —— 先核对原稿，再决定信不信。",
    "suggestion": "**不是错**，是可以做得更好的地方。",
}
#: 显示顺序 —— **`uncertain` 放最前**。
#:
#: 理由：它是在给**整份报告打折**（「我可能看错了」）。
#: 排在具体问题后面，学生读完一堆指控才看到"这些可能都不算数"，
#: **等于先用错误结论说服了他。**
CATEGORY_ORDER = ("uncertain", "observed_error", "suggestion")

#: `validation.issues[].kind` 的人话（契约 §2.3 只定义了校验，没定义文案）
VALIDATION_ISSUE_TEXT = {
    "dangling_reference": "有一条判断引用了**不存在的证据** —— 它找不到依据",
    "model_written_count": "报告里出现了**模型自己写的次数**（次数应由服务端统计）",
    "unresolved_placeholder": "有统计占位符**没有被填充**",
}


def review_categories(review: dict) -> dict:
    """把 v3 复盘按契约的三类分组，并把 `evidence_ids` **解析成服务端原文**。

    ## 为什么是纯函数
    渲染层（`streamlit_app.py`）测不动，而"分组对不对、引用指不指得回原文"
    **恰恰是这里唯一会错的地方**。⇒ 把它挪出来，直接测。

    ## 三条刻意的行为
    ```
    ① 引用原文**只从服务端证据目录取** —— 模型不能改写（契约 §2.3②），
       所以视图里出现的每一句原文，都是服务端填的那一份。
    ② 引用查不到就**如实标出来**，不静默丢掉 —— 契约⑦说这种报告不该被缓存，
       但前端不能假设后端永远拦得住：**看不见的坏引用比看得见的更危险。**
    ③ `category` 出现第四种值时**单独收进 `unknown`**，不硬塞进三类里
       —— 契约说"固定三选一"，真出现了就说明有一边错了，得让人看见。
    ```
    返回 `is_v3=False` 时，调用方走 v1/v2 的老渲染（历史报告不伪装成带证据报告）。
    """
    if not isinstance(review, dict):
        return {"is_v3": False, "groups": [], "unknown": [],
                "validation": {}, "validation_issues": []}
    # ⚠️ 判据是**结构版本号**，不是"有没有 findings"：
    #    契约⑥允许 findings 为空列表（无明确问题），那是合法 v3，不该退回老渲染。
    if not str(review.get("review_schema_version") or "").strip():
        return {"is_v3": False, "groups": [], "unknown": [],
                "validation": {}, "validation_issues": []}

    raw_findings = review.get("findings")
    findings = raw_findings if isinstance(raw_findings, list) else []
    evidence = {}
    for item in (review.get("evidence") or []):
        if isinstance(item, dict) and item.get("id"):
            evidence[str(item["id"])] = item

    grouped: dict[str, list] = {name: [] for name in CATEGORY_ORDER}
    unknown: list[dict] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        cited = []
        for eid in (finding.get("evidence_ids") or []):
            item = evidence.get(str(eid))
            cited.append({"id": str(eid), "found": item is not None,
                          "round": (item or {}).get("round"),
                          "revision": (item or {}).get("revision"),
                          "text": (item or {}).get("text") or ""})
        entry = {"where": (finding.get("where") or "").strip(),
                 "what": (finding.get("what") or "").strip(),
                 "why": (finding.get("why") or "").strip(),
                 "downstream": (finding.get("downstream") or "").strip(),
                 "evidence": cited}
        category = finding.get("category")
        if category in grouped:
            grouped[category].append(entry)
        else:
            unknown.append({**entry, "category": category})

    groups = [{"category": name, "label": CATEGORY_LABELS[name],
               "hint": CATEGORY_HINTS[name], "items": grouped[name]}
              for name in CATEGORY_ORDER]

    raw_validation = review.get("validation")
    validation = raw_validation if isinstance(raw_validation, dict) else {}
    issues = []
    for issue in (validation.get("issues") or []):
        if not isinstance(issue, dict):
            continue
        kind = issue.get("kind")
        issues.append({"kind": kind, "detail": issue.get("detail") or "",
                       "text": VALIDATION_ISSUE_TEXT.get(
                           kind, issue.get("detail") or "未知校验问题")})
    return {"is_v3": True, "groups": groups, "unknown": unknown,
            "validation": validation, "validation_issues": issues}


def can_generate_review(state, student_id: str, problem_id: str) -> bool:
    """**现在**能不能为当前材料发起一次复盘生成（契约 §2.1 的 `review_available`）。

    ⚠️ 它与 `review_status` **正交**，两者互不蕴含 —— 这是最容易写错的地方：
    ```
    not_requested + review_available=True   ⇒ 能（视频：先看提示，回头再要复盘）
    ready         + review_available=False  ⇒ 不能（照片：报告已出，但没有延后入口）
    ```
    ⚠️ 它也**不看** `review_refresh_required` —— 那个字段说的是
    「手上这份旧报告需要更新」，而不是「现在能生成」。
    照片更正后正是「需要更新、却无处可生成」⇒ **只提示，不给按钮**。
    """
    if video_review_request(state, student_id, problem_id) is None:
        return False
    return (state.get("video_result") or {}).get("review_available") is not False


def apply_video_review(state, request: dict, payload: dict) -> None:
    """Reject stale results without changing the teaching trajectory or editor."""
    from copy import deepcopy
    if (video_review_request(state, request["student_id"], request["problem_id"]) != request
            or payload.get("attempt_id") != request["attempt_id"]
            or payload.get("run_id") != request["run_id"]
            or payload.get("input_revision") not in (
                request["input_revision"], request["input_revision"] + int(request["refresh_input"]))
            or not payload.get("generation_fingerprint")
            or not payload.get("review_input_fingerprint")
            or payload.get("review_source_revisions") != request["_ui_source_revisions"]
            or payload.get("review_status") != "ready"
            or not isinstance(payload.get("review"), dict)):
        raise ValueError("复盘响应与当前视频不匹配，已保留原进度。")
    result = deepcopy(state["video_result"])
    for key in ("review", "review_status", "review_elapsed_sec", "review_input_fingerprint",
                "review_scope", "review_source_revisions", "graph_version", "input_revision", "generation_fingerprint"):
        if key in payload:
            result[key] = deepcopy(payload[key])
    result["review_refresh_required"] = False
    state["video_result"] = result
    state.pop("video_review_error", None)


def reset_session(state, now: float) -> None:
    clear_media(state)
    state.update(timeline=[], tutor={}, attempt_id=None, attempt_status="in_progress", db_record=None,
                 start_ts=now, last_action_ts=now)


def current_attempt_id(state):
    """Terminal attempts stay visible but must never be used for continuation."""
    tutor = state.get("tutor") or {}
    if (state.get("attempt_status", "in_progress") != "in_progress"
            or tutor.get("finished") or tutor.get("status", "in_progress") != "in_progress"):
        return None
    return state.get("attempt_id")


def correction_unavailable(state, result: dict, source: dict) -> str:
    """Explain why an OCR source cannot replace the current analysis."""
    if not source.get("source_id"):
        return "此历史记录没有更正来源标识，请重新提交照片或视频。"
    if not current_attempt_id(state):
        return "本次作答已结束，无法更正；请重新开始后提交。"
    if result.get("attempt_id") != current_attempt_id(state):
        return "此识别记录属于另一份作答，无法更正。"
    usable = [r for r in result.get("rounds", []) if "error" not in r]
    if not usable or usable[-1].get("source_id") != source["source_id"]:
        return "仅可更正最近一步；更早步骤已有后续分析。"
    return ""


def apply_ocr_correction(state, payload: dict, source_id: str) -> None:
    """Replace projections using the server result, retaining the original OCR text."""
    from copy import deepcopy
    result = state.get("video_result") or {}
    if (payload.get("attempt_id") != current_attempt_id(state)
            or payload.get("source_id") != source_id
            or not isinstance(payload.get("analysis"), dict)
            or not isinstance(payload.get("timeline"), list)):
        raise ValueError("更正响应与当前作答不匹配，已保留原进度。")
    rounds = deepcopy(result.get("rounds") or [])
    source = next((r for r in rounds if r.get("source_id") == source_id), None)
    if source is None:
        raise ValueError("当前识别记录已改变，已保留原进度。")
    revision = payload.get("source_revision")
    previous_revision = source.get("source_revision", 0)
    if type(revision) is not int or revision < previous_revision:
        raise ValueError("更正响应与当前识别版本不匹配，已保留原进度。")
    if revision == previous_revision and result.get("ocr_corrected"):
        if source.get("answer") == payload.get("corrected_text"):
            return  # An idempotent correction must not invalidate a newer review.
        raise ValueError("更正响应与当前识别版本不匹配，已保留原进度。")
    analysis = deepcopy(payload["analysis"])
    source.update(analysis)
    source.update(original_answer=payload["original_text"], answer=payload["corrected_text"],
                  source_revision=payload["source_revision"])
    timeline = [{"timestamp": e.get("timestamp", 0), "type": e.get("event_type", e.get("type", "submit")),
                 "content": e.get("content", "")} for e in payload["timeline"]]
    # The former review/summary describes the superseded analysis and must not be reused.
    usable = [r for r in rounds if "error" not in r]
    updated = {**result, "rounds": rounds, "review": None,
               "weak_concepts": sorted({c for r in usable for c in r.get("weak_concepts", [])}),
               "off_graph_rounds": sum(bool(r.get("round_off_graph")) for r in usable),
               "ocr_corrected": True}
    history = deepcopy(result.get("review_history") or [])
    if result.get("review"):
        # 本轮决定：旧报告**按原版本保留并标明过期**，
        # 但**不得作为当前结论展示**（`review` 紧接着被置空）。
        # `superseded` 就是那个「独立字段」——
        # **过期与否不从 `review_status` 里读**，三值只描述当前材料的生成状态。
        entry = {key: deepcopy(result[key]) for key in (
            "review", "input_revision", "generation_fingerprint", "review_input_fingerprint",
            "review_source_revisions", "review_scope") if key in result}
        entry["superseded"] = True
        history.append(entry)
    # 契约 §2.1：`review_status` **只有三个值**（not_requested | ready | failed），
    # 前端**不得**引入第四个（曾用过的 `invalidated` 已去掉）。
    # 更正后旧报告失效，对**更正后的新材料版本**确实尚未生成过复盘 ⇒ `not_requested`；
    # 「过期、可重新生成」由 `review_refresh_required` + `review_history` 表达。
    updated.update(review_status="not_requested", review_refresh_required=True,
                   correction_generation=result.get("correction_generation", 0) + 1,
                   review_history=history)
    for key in ("review_elapsed_sec", "review_input_fingerprint", "generation_fingerprint",
                "review_source_revisions", "review_scope"):
        updated.pop(key, None)
    state.update(tutor=analysis, timeline=timeline, video_result=updated,
                 attempt_status="completed" if analysis.get("finished") else "in_progress", db_record=None)
    state.pop("recommendations", None)
    state.pop("video_review_error", None)


def switch_session(state, student_id: str, problem_id: str, now: float) -> bool:
    """Keep independent snapshots per student/problem; clear uploaded files on every switch."""
    from copy import deepcopy
    key = (student_id, problem_id)
    old = state.get("active_session")
    if old == key:
        return False
    snapshots = state.setdefault("session_snapshots", {})
    if old is not None:
        snapshots[old] = deepcopy({name: state.get(name) for name in
            ("timeline", "tutor", "attempt_id", "attempt_status", "db_record", "start_ts", "last_action_ts")})
    reset_session(state, now)
    if key in snapshots:
        state.update(deepcopy(snapshots[key]))
    state["active_session"] = key
    return True


def restore_attempt(state, record: dict, student_id: str, problem_id: str) -> bool:
    """Only restore the selected student's in-progress record; timeline comes from the server."""
    if (record.get("student_id") != student_id or record.get("problem_id") != problem_id
            or record.get("status") != "in_progress"):
        return False
    state["attempt_id"] = record["attempt_id"]
    state["attempt_status"] = record["status"]
    state["timeline"] = [{"timestamp": e.get("timestamp", 0),
                          "type": e.get("event_type", "submit"),
                          "content": e.get("content", "")}
                         for e in record.get("events", [])]
    server_state = record.get("state")
    # Do not combine a stale local diagnosis with a newer server projection.
    tutor = dict(server_state) if isinstance(server_state, dict) else {
        "graph_mode": (state.get("tutor") or {}).get("graph_mode", False)}
    if "diagnosed_issue" in tutor:
        tutor["issue"] = tutor["diagnosed_issue"]
    if "last_step_match" in tutor:
        tutor["step_match"] = tutor["last_step_match"]
    for name in ("current_node", "current_method", "visited_nodes", "weak_concepts", "stuck_score", "status"):
        if name in record:
            tutor[name] = record[name]
    tutor["finished"] = False
    hints = record.get("hints") or []
    if hints and not isinstance(server_state, dict):
        tutor.update(hints[-1])
    state["tutor"] = tutor
    state["db_record"] = record
    return True


def media_notices(result: dict) -> dict:
    """Facts about coverage and storage; unknown removal never implies deletion."""
    warnings = []
    coverage = result.get("coverage") or {}
    if coverage.get("truncated"):
        warnings.append(f"视频仅分析了部分范围：覆盖至 {coverage.get('covered_until_sec', '未知')} 秒，"
                        f"总时长 {coverage.get('video_total_sec', '未知')} 秒；"
                        f"原因：{coverage.get('truncate_reason') or '处理范围受限'}。复盘仅适用于已覆盖内容。")
    if coverage.get("failed_frame_count"):
        warnings.append(f"有 {coverage['failed_frame_count']} 帧识别失败，复盘可能缺少这些时刻的内容。")
    remaining = result.get("remaining_snapshots") or 0
    if remaining or result.get("teaching_stop_reason") == "analysis_error":
        warnings.append(f"教学分析提前停止（{result.get('teaching_stop_reason') or '未知原因'}），"
                        f"还有 {remaining} 个快照未批改。")
    if result.get("media_type") == "image":
        retention = [r.get('image_evidence_status') for r in result.get('rounds', []) if r.get('image_evidence_status')]
        if result.get('image_evidence_status'):
            retention.append(result['image_evidence_status'])
        if 'retained' in retention:
            storage = '原图已保留供作答核对；识别文本、批改记录等分析产物也可能保留。'
        elif retention:
            storage = '原图保留未成功或格式不支持；识别文本、批改记录等分析产物可能保留。'
        else:
            storage = '原图保留状态未提供；不能承诺未落盘。识别文本、批改记录等分析产物可能保留。'
    elif result.get("upload_removed") is True:
        storage = "原视频已从服务器删除；识别文本、批改记录及其他分析产物仍可能保留。"
    elif result.get("upload_removed") is False:
        storage = "原视频尚未确认删除；识别文本、批改记录及其他分析产物仍可能保留。"
    else:
        storage = "原视频删除状态未知；识别文本、批改记录及其他分析产物可能保留。"
    return {"warnings": warnings, "storage": storage}


# Video workers run independently of Streamlit reruns. These guards bind results
# to a user's explicit input generation, rather than to a particular page render.
def invalidate_video_job(state) -> None:
    key = state.get("active_session")
    if key is not None:
        generations = dict(state.get("video_job_generations") or {})
        generations[key] = generations.get(key, 0) + 1
        state["video_job_generations"] = generations


def video_job_context(state, student_id: str, problem_id: str) -> dict:
    return {"student_id": student_id, "problem_id": problem_id,
            "generation": (state.get("video_job_generations") or {}).get((student_id, problem_id), 0)}


def rollback_video_job_start(state, context: dict) -> None:
    """A failed admission leaves the prior task queryable, without undoing later actions."""
    key = (context["student_id"], context["problem_id"])
    if (state.get("active_session") == key and video_job_context(state, *key) == context
            and context["generation"] > 0):
        generations = dict(state.get("video_job_generations") or {})
        generations[key] = context["generation"] - 1
        state["video_job_generations"] = generations


def retain_video_job(state, context: dict, payload: dict) -> dict:
    from copy import deepcopy
    key = (context["student_id"], context["problem_id"])
    if (state.get("active_session") != key
            or video_job_context(state, *key) != context
            or payload.get("student_id") != key[0] or payload.get("problem_id") != key[1]
            or not isinstance(payload.get("job_id"), str) or not payload["job_id"]
            or payload.get("status") not in {"queued", "running", "succeeded", "failed", "interrupted"}):
        raise ValueError("视频任务与当前作答不匹配，原进度已保留。")
    jobs = deepcopy(state.get("pending_video_jobs") or {})
    previous = jobs.get(key)
    if previous and previous.get("generation") == context["generation"] and previous["job_id"] != payload["job_id"]:
        raise ValueError("视频任务已改变，原进度已保留。")
    if previous and previous.get("generation") == context["generation"]:
        if (previous.get("status") in {"succeeded", "failed", "interrupted"}
                and payload["status"] != previous["status"]):
            raise ValueError("视频任务已结束，忽略迟到的旧进度。")
        if (isinstance(previous.get("updated_at"), str) and isinstance(payload.get("updated_at"), str)
                and payload["updated_at"] < previous["updated_at"]):
            raise ValueError("忽略迟到的视频任务进度。")
    job = {**deepcopy(payload), "generation": context["generation"]}
    if previous and previous.get("job_id") == job["job_id"]:
        job["applied"] = previous.get("applied", False)
    jobs[key] = job
    state["pending_video_jobs"] = jobs
    return job


def current_video_job(state, student_id: str, problem_id: str) -> dict | None:
    key = (student_id, problem_id)
    job = (state.get("pending_video_jobs") or {}).get(key)
    if (state.get("active_session") != key or not job
            or job.get("generation") != video_job_context(state, *key)["generation"]):
        return None
    return job


def apply_video_job_result(state, context: dict, payload: dict, now: float) -> bool:
    from copy import deepcopy
    # Validate a final payload before retaining it, so malformed/foreign results
    # cannot replace the last queryable job while leaving teaching state unchanged.
    result = payload.get("result")
    if payload.get("status") == "succeeded":
        if (not isinstance(result, dict) or result.get("student_id") != context["student_id"]
                or result.get("problem_id") != context["problem_id"] or not result.get("attempt_id")
                or (payload.get("attempt_id") and payload["attempt_id"] != result["attempt_id"])):
            raise ValueError("视频结果归属不匹配，原进度已保留。")
        if not isinstance(result.get("rounds"), list):
            raise ValueError("视频结果格式异常，原进度已保留。")
        usable = [r for r in result["rounds"] if isinstance(r, dict) and "error" not in r]
        if not usable:
            raise ValueError("这段视频里没有可用的作答内容，请确认纸面文字清晰。")
    job = retain_video_job(state, context, payload)
    if job["status"] != "succeeded" or job.get("applied"):
        return False
    last = deepcopy(usable[-1])
    state.update(tutor=last, attempt_id=result["attempt_id"],
                 attempt_status="completed" if last.get("finished") else "in_progress",
                 db_record=None, last_action_ts=now, video_result=deepcopy(result))
    state["timeline"] = [*state.get("timeline", []), {
        "timestamp": round(now - state.get("start_ts", now), 1), "type": "submit",
        "content": f"（视频作答，识别到 {result.get('submitted')} 步）"}]
    state["pending_video_jobs"][(context["student_id"], context["problem_id"])]["applied"] = True
    return True
