"""录完上传的视频作答与按需复盘端点。

感知层复用 run_pipeline，教学复用 analyze_attempt，合并同刻整帧与 ROI。
视频分析是同步长请求，耗时随视频、解码、模型调用及重试变化，
响应包含各阶段实测计时；不承诺固定每帧速度或总处理时间。

复盘默认从视频分析主请求分离，通过 /video/review 按需同步生成，
也可显式 include_review=true 兼容即时复盘。另有 /video/jobs 单进程后台队列，
支持查询进度和跨页面刷新恢复，进程重启会标记中断；当前没有断点续传。
阻塞处理使用同步 def，由 FastAPI 放入线程池，避免阻塞事件循环。
"""
from __future__ import annotations

import math
import json
import re
import logging
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.api.attempts import AnalyzeRequest, analyze_attempt, validate_explicit_attempt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video", tags=["video"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "data" / "video_runs"
UPLOAD_DIR = RUNS_DIR / "uploads"

#: 上传体积上限。**手机录 3 分钟约 100~300 MB**，留够余量但挡住异常大文件。
MAX_UPLOAD_BYTES = 800 * 1024 * 1024

#: 复盘生成状态的**全部取值**（契约 §2.1，2026-10-09 T0 裁决）。
#:
#: ⚠️ **曾经只有两个值**：`ready if review_result else "available"` ——
#: 那把「学生没要复盘」和「要了但生成失败」压成了同一个 `available`，
#: 而这两件事的处置完全不同（前者正常，后者该提示可重试）。
#: `available` 表达的是「材料还在、可以稍后生成」，**不是生成状态** ——
#: 它现在独立成 `review_available` 布尔字段，不再占用这里。
REVIEW_NOT_REQUESTED = "not_requested"
REVIEW_READY = "ready"
REVIEW_FAILED = "failed"


def _review_requested(value, default: bool) -> bool:
    """请求侧 `Form` 默认值的**唯一**解包口。

    ⚠️ 这个函数的存在是因为一个实测踩过的坑：
    `include_review: bool = Form(False)` 在 **HTTP 调用**时得到真正的 `bool`，
    但**直接以 Python 调用端点函数时得到的是 `Form` 对象**（测试与脚本都这么调）——
    于是 `if include_review is True:` 会静默为假，**默认值等于没有**。

    ⇒ 声明了什么默认值，就在这里显式补回来，**不靠猜**。
    """
    return value if isinstance(value, bool) else default


def _review_status(requested: bool, generated: bool) -> str:
    """三值判定的**唯一**入口（契约 §2.1）。

    `requested=False` ⇒ `not_requested`（**不是** `failed`）；
    `requested=True` 且没产出报告 ⇒ `failed`（**不得**回落成 `not_requested`）。
    """
    if not requested:
        return REVIEW_NOT_REQUESTED
    return REVIEW_READY if generated else REVIEW_FAILED



def _snapshots_from(results: dict, every_sec: float) -> list[dict]:
    """把感知层结果按节奏切成快照。

    **复用 `merge_by_timestamp`** —— ROI 与整帧同一时刻要合并（ROI 文字优先），
    否则同一时刻会被读两遍。
    """
    from app.vision.bridge import merge_by_timestamp

    # 失败 ROI 可能仍带有残留文字，不能覆盖同刻成功整帧。
    # 失败证据由 coverage.failed_frames 保留，不作为教学输入。
    successful = []
    for record in results.get("frames") or []:
        if not isinstance(record, dict) or record.get("error"):
            continue
        if record.get("has_content") is False or record.get("has_text") is False:
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            lines = record.get("ocr_lines") or []
            text = "\n".join(line for line in lines if isinstance(line, str)) if isinstance(lines, list) else ""
        if not text.strip():
            continue
        # Scene descriptions are perception metadata, never student answers.
        successful.append({**record, "text": text, "description": ""})
    merged = sorted(merge_by_timestamp(successful),
                    key=lambda g: g["ts"])
    out: list[dict] = []
    last = -1e9
    for g in merged:
        if g["ts"] - last >= every_sec and (g.get("text") or "").strip():
            out.append(g)
            last = g["ts"]
    return out


def _validate_media_target(student_id: str, problem_id: str, attempt_id: str) -> None:
    """昂贵识别前只读校验；教学入口仍须在锁内重新校验。"""
    from app.services.problem_repository import get_problem_repository

    validate_explicit_attempt(attempt_id, student_id, problem_id)
    if get_problem_repository().get_problem(problem_id) is None:
        raise HTTPException(404, f"题目 {problem_id} 不存在")


class VideoReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    student_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    input_revision: int = Field(default=0, ge=0)
    refresh_input: bool = False


def _fingerprint(data: dict) -> str:
    """⚠️ **转发到 `services/review_material`** —— 那里是形状与指纹的唯一真相源。

    留这个名字是因为既有调用方与测试都在用它（`video._fingerprint`）；
    真正实现只有一份，生产路径与评测路径共用。
    """
    from app.services import review_material

    return review_material.fingerprint(data)


def _valid_review_input(data) -> bool:
    """⚠️ **转发到 `services/review_material`**（同上：单一真相源）。

    `scope` 由"必须等于 `video_run_snapshot`"放宽为"必须属于允许集合"
    （契约 §2.5，2026-10-09 T0），其余检查一条没减。
    """
    from app.services import review_material

    return review_material.valid_input(data)


#: v3 才有的四个字段。**没有 v3 标记的报告，出口一律摘掉它们。**
_V3_ONLY_FIELDS = ("review_schema_version", "evidence", "findings", "validation")


def _validated_review_payload(payload) -> dict:
    from app.services.review_builder import ReviewReport

    report = ReviewReport.model_validate(payload, strict=True)
    if not report.timeline.strip():
        raise ValueError("empty review timeline")
    data = report.model_dump()

    # ⚠️ **这就是契约「历史 v1/v2 不伪装成带证据报告」的落点。**
    # 字段必须挂在 `ReviewReport` 上（否则 v3 报告带不出来），
    # 而 `model_dump()` 会**一视同仁**地加给每一条响应 —— 包括缓存里的旧报告。
    # ⇒ 出口按 `review_schema_version` 判定：有版本号保留，没有则**全部摘掉**（不是给空值）。
    #
    # **为什么放这里**：这是缓存读与新生成本**共同的出口**；
    # 放上游会漏掉缓存那条路。
    if not (data.get("review_schema_version") or "").strip():
        for _k in _V3_ONLY_FIELDS:
            data.pop(_k, None)
    return data


def _reference_defects(payload: dict, source_revisions: dict | None = None) -> list[dict]:
    """**重算**这份报告的引用缺陷 —— 生成路径与缓存路径**共用这一个判据**。

    ⚠️ **不读 `payload["validation"]`。**
    缓存里存着上一版代码写的 `validation`；信它，就等于信写它时的代码 ——
    而缺陷恰恰可能是那时的代码漏掉的。所以这里从 `evidence` + `findings`
    **重新算一遍**（`review_builder.reference_defects`）。
    """
    from app.services.review_builder import reference_defects

    if not isinstance(payload, dict):
        return [{"kind": "malformed", "detail": "报告不是对象"}]
    evidence = payload.get("evidence")
    findings = payload.get("findings")
    if not isinstance(evidence, list) or not isinstance(findings, list):
        # v1/v2 报告没有这两个结构 ⇒ 无从判引用，**跳过而不是判它坏**
        # （契约：历史报告保留读取能力，不伪装成带证据报告）。
        return []
    return reference_defects(evidence, findings, source_revisions)


#: 哪些结构缺陷**放行**（即不阻断成功报告）。**默认空集 —— 这是刻意的。**
#:
#: ## 为什么是"空的黑名单"而不是"已知种类白名单"
#: 旧版写的是 `_BLOCKING_REFERENCE_KINDS = {dangling_reference, …}` —— 一个**白名单**：
#: 只有列进去的种类才阻断。后果是 `model_written_count` / `unresolved_placeholder`
#: 被挡在门外，理由写的是「这是文体问题，不是结构不合法；惩罚过重」。
#:
#: ⚠️ **那是对契约 ⑦ 的窄读**：契约看的是 `validation.valid`，不是"错得重不重"。
#:    实证代价（T4 开发集 2026-10-09）：**44/72 份报告 `valid=false` 仍被算作成功**，
#:    可用率被报成 100%，而结构合格的只有 28 份。
#:
#: ⚠️ 白名单更深的毛病是**形态**：将来任何一个**新**的 issue 种类都会**静默不阻断** ——
#:    缺陷能长期存在，靠的就是这个。⇒ 一律 **fail-closed**：
#:    **`structural_defects()` 返回非空就阻断**；要放行某类，必须**显式**写进下面这个集合，
#:    并在注释里写明理由（**举证责任在放行方**）。
_NON_BLOCKING_ISSUE_KINDS: frozenset[str] = frozenset()


def _structural_defects(payload: dict,
                        source_revisions: dict | None = None) -> list[dict]:
    """**重算**这份报告的全部结构缺陷（引用 + 统计 + 占位符）。

    ⚠️ 与 `review_builder.structural_defects` 是**同一处实现** ——
    生成路径与缓存读取路径各写一遍，就会像上一轮那样**漂开**（生成挡、读取不挡）。

    ⚠️ **证据池取自报告自己的 `evidence` 字段**（服务端生成的，契约 §2.8）——
    这样缓存路径与生成路径**用的是同一个池子**，不必再各传一份（各传一份就会漂开）。
    """
    from app.services.review_builder import structural_defects

    pool = payload.get("evidence") if isinstance(payload, dict) else None
    return [d for d in structural_defects(payload, source_revisions, evidence_pool=pool)
            if d.get("kind") not in _NON_BLOCKING_ISSUE_KINDS]


def _run_path(run_id: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise HTTPException(422, "无效的视频运行编号")
    root = RUNS_DIR.resolve()
    target = (root / run_id).resolve()
    if target.parent != root:
        raise HTTPException(403, "视频运行路径不合法")
    return target


def _save_review_input(out_dir: Path, student_id: str, problem_id: str,
                       attempt_id: str, rounds: list[dict]) -> None:
    """**视频流程**写复盘材料 —— 转发到 `services/review_material`（单一真相源）。

    ⚠️ 构造器与指纹都在那边；这里只保留名字，免得打断既有调用方与测试
    （`test_video_jobs` 会 monkeypatch 它，`test_video_lazy_review` 直接调它）。
    """
    from app.services import review_material

    review_material.save_video_material(out_dir, student_id, problem_id, attempt_id, rounds)


@router.post("/review")
def review_video(req: VideoReviewRequest) -> dict:
    """复盘服务端保存的视频轨迹；不推进作答，不接受客户端轨迹。

    默认针对指定固定材料。显式刷新仅接受服务端记录的同视频 OCR 更正。
    单进程内同一 run 串行；多进程部署需要共享任务锁。
    """
    from app.services.attempt_store import get_attempt_store
    from app.services.graph_versions import atomic_json, graph_lock
    from app.services.review_builder import review_trajectory
    from app.services.video_review_versions import generation_context, refresh_material, timeline_items

    record = get_attempt_store().load_attempt(req.attempt_id)
    if record is None:
        raise HTTPException(404, "作答不存在")
    if record.student_id != req.student_id or record.problem_id != req.problem_id:
        raise HTTPException(403, "作答不属于指定学生和题目")
    out_dir = _run_path(req.run_id)
    with graph_lock(out_dir):
        source = out_dir / "review_input.json"
        if source.resolve().parent != out_dir:
            raise HTTPException(403, "复盘材料路径不合法")
        if not source.is_file():
            raise HTTPException(404, "视频复盘材料不存在")
        try:
            saved = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or not _valid_review_input(saved.get("input")):
                raise ValueError("invalid review input shape")
            data = saved["input"]
            fingerprint = _fingerprint(data)
            if fingerprint != saved["fingerprint"]:
                raise ValueError("input fingerprint mismatch")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise HTTPException(409, "视频复盘材料损坏，请重新上传") from exc
        if any(data.get(key) != getattr(req, key) for key in ("student_id", "problem_id", "attempt_id")):
            raise HTTPException(403, "视频运行不属于指定作答")
        original = data
        revision_dir = out_dir / "review_inputs"
        available = sorted(int(p.stem) for p in revision_dir.glob("*.json") if p.stem.isdigit())
        latest_revision = max(available, default=0)
        if req.refresh_input and req.input_revision != latest_revision:
            raise HTTPException(409, "复盘材料版本已更新，请刷新后重试")
        revision = req.input_revision
        if revision:
            try:
                saved = json.loads((revision_dir / f"{revision}.json").read_text(encoding="utf-8"))
                if not _valid_review_input(saved.get("input")) or _fingerprint(saved["input"]) != saved["fingerprint"]:
                    raise ValueError("invalid material")
                data = saved["input"]
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                raise HTTPException(409, "复盘材料版本不存在或已损坏") from exc
            if any(data.get(key) != getattr(req, key) for key in ("student_id", "problem_id", "attempt_id")):
                raise HTTPException(403, "复盘材料版本不属于指定作答")
        if req.refresh_input:
            current = get_attempt_store().load_attempt(req.attempt_id)
            if current is None or current.student_id != req.student_id or current.problem_id != req.problem_id:
                raise HTTPException(409, "作答已变化，请刷新后重试")
            refreshed = refresh_material(original, data, current.state)
            if refreshed != data:
                revision = latest_revision + 1
                data = refreshed
        fingerprint = _fingerprint(data)
        # Re-check current evidence before both model generation and cache return.
        # Finished attempts are valid; corrected or extended inputs are not this run.
        def validate_current_input():
            current = get_attempt_store().load_attempt(req.attempt_id)
            if current is None:
                raise HTTPException(404, "作答不存在")
            if current.student_id != req.student_id or current.problem_id != req.problem_id:
                raise HTTPException(403, "作答不属于指定学生和题目")
            state = current.state
            sources = {str(item.get("source_id")): item.get("revision", 0)
                       for item in (getattr(state, "media_sources", []) or [])}
            if (any(sources.get(key) != revision for key, revision in data["source_revisions"].items())
                    or len(getattr(state, "timeline", []) or []) != data.get("timeline_length", 0)
                    or ("timeline_snapshot" in data and timeline_items(state) != data["timeline_snapshot"])
                    or ("media_source_ids" in data and list(sources) != data["media_source_ids"])):
                raise HTTPException(409, "作答材料已更正或续做，该视频复盘材料已过期")
        validate_current_input()
        try:
            generation, generation_fingerprint, pinned_llm = generation_context()
        except Exception as exc:
            raise HTTPException(502, "复盘配置暂时不可用，可重试") from exc
        cache_key = _fingerprint({"input": fingerprint, "generation": generation_fingerprint})
        archive = out_dir / "review_reports" / f"{cache_key}.json"
        legacy_cache = out_dir / "review_report.json"
        cache = archive
        # Legacy reports lack generation provenance and are deliberately cache misses.
        # Keep the original report intact for audit; use versioned reports from now on.
        if cache.resolve().parent not in (out_dir, (out_dir / "review_reports").resolve()):
            raise HTTPException(403, "复盘缓存路径不合法")
        result = None
        if cache.is_file():
            try:
                existing = json.loads(cache.read_text(encoding="utf-8"))
                if (isinstance(existing, dict)
                        and existing.get("review_input_fingerprint") == fingerprint
                        and existing.get("generation_fingerprint") == generation_fingerprint
                        and existing.get("input_revision") == revision
                        and type(existing.get("review_elapsed_sec")) in (int, float)
                        and math.isfinite(existing["review_elapsed_sec"])
                        and existing["review_elapsed_sec"] >= 0):
                    existing["review"] = _validated_review_payload(existing.get("review"))
                    # ⚠️ **缓存读取与新生报告走同一道结构关口**（fail-closed）。
                    # 原先只有生成路径查 `dangling_reference`，缓存路径只查形状 ——
                    # ⇒ 一份**旧代码写下的**（或从别处来的）带坏引用/坏统计的报告，
                    #   会在缓存里被一路放行、当成功结果返回。
                    # **重算，不读缓存里的 validation**（见 `_structural_defects`）。
                    _cache_bad = _structural_defects(
                        existing["review"], data.get("source_revisions"))
                    if _cache_bad:
                        from app.services.review_builder import classify_defects

                        logger.warning("缓存的复盘结构校验未通过（%s），**当缓存未命中处理**：%s",
                                       classify_defects(_cache_bad), _cache_bad)
                    else:
                        result = existing
            except (ValueError, OSError):
                pass  # A broken cache is replaceable; immutable input is authoritative.
        cached = result is not None
        if result is None:
            validate_current_input()
            start = time.monotonic()
            try:
                report = review_trajectory(SimpleNamespace(**data["problem"]), data["rounds"],
                                           llm=pinned_llm,
                                           source_revisions=data.get("source_revisions"))
            except Exception as exc:
                logger.exception("视频复盘失败")
                raise HTTPException(502, "复盘暂时失败，可重试") from exc
            if report is None:
                raise HTTPException(502, "复盘暂时失败，可重试")
            if getattr(report, '_raw_generation', None) is not None:
                atomic_json(out_dir / 'review_raw_generations' / f'{cache_key}.json', report._raw_generation)
            try:
                report_data = _validated_review_payload(report.model_dump())
            except (AttributeError, TypeError, ValueError) as exc:
                raise HTTPException(502, "复盘暂时失败，可重试") from exc
            # ⚠️ **结构不合格 ⇒ 不缓存成功报告**（契约 ⑦，fail-closed）
            # 含 `model_written_count` / `unresolved_placeholder` ——
            # 契约看的是 `validation.valid`，不是"错得重不重"（2026-10-09 澄清）。
            #
            # 判据与缓存路径**同一个函数**（`_structural_defects`）——
            # 两条路各写一遍，就会像上一轮那样**漂开**：生成时挡、读取时不挡。
            _bad = _structural_defects(report_data, data.get("source_revisions"))
            if _bad:
                # **失败信息要分类**（用户第 4 条）：说清是哪一类不合格。
                # ⚠️ 但**不说"物理有问题"** —— 规则判不了物理，`classify_defects` 也只分两类。
                from app.services.review_builder import classify_defects

                _category = classify_defects(_bad)
                logger.warning("复盘结构校验未通过（%s），**不写缓存**：%s", _category, _bad)
                raise HTTPException(
                    502, f"复盘结构校验未通过（{_category}），可重试；"
                         "这**不代表物理判断已被验证**")
            result = {"review": report_data, "review_input_fingerprint": fingerprint,
                      "input_revision": revision, "generation_fingerprint": generation_fingerprint,
                      "generation": generation, "cache_key": cache_key,
                      "review_elapsed_sec": round(time.monotonic() - start, 2)}
            validate_current_input()  # Correction may occur during the slow model call.
            if revision and not (revision_dir / f"{revision}.json").exists():
                atomic_json(revision_dir / f"{revision}.json",
                            {"input": data, "fingerprint": fingerprint, "input_revision": revision})
            atomic_json(archive, result)
            if revision == 0 and not legacy_cache.exists():
                atomic_json(legacy_cache, result)
        validate_current_input()
        # 这是**显式生成**端点：走到这里就是「请求了且成功」⇒ `ready`（契约 §2.1）。
        # 失败一律在上面的 HTTPException 里抛出，**不会**在这里回落成别的值。
        return {**result, "cached": cached, "review_status": REVIEW_READY,
                "review_available": True, "run_id": req.run_id,
                "attempt_id": req.attempt_id, "review_scope": data["scope"],
                "student_id": req.student_id, "problem_id": req.problem_id,
                "review_source_revisions": data["source_revisions"],
                "graph_version": data["graph_version"]}


@router.post("/analyze")
def analyze_video(
    file: UploadFile = File(..., description="学生录的作答视频"),
    problem_id: str = Form(...),
    student_id: str = Form(...),
    every_sec: float = Form(20.0),
    attempt_id: str = Form(""),
    request_hint: bool = Form(False),
    include_review: bool = Form(False),
) -> dict:
    """上传视频 → 感知及逐帧教学 → 返回轨迹和按需复盘 run_id。"""
    from app.vision.service import VideoAnalysisOptions, run_pipeline
    from app.services.video_progress import report_progress, register_owned_upload, perception_progress_callback

    # ⚠️ **分阶段计时** —— 学生实测 412s，而我自己拼出来的估算是 119s，
    # **差了 293s，而那 293s 我根本没量过**（上传/存盘/冷启动/连起来跑）。
    # **估的和实测的差三倍，就不该再估了。**
    _t: dict[str, float] = {}
    _t0 = time.time()
    _validate_media_target(student_id, problem_id, attempt_id)
    report_progress("upload")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    token = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{token}{suffix}"
    # Background jobs must persist this association before an abrupt exit could
    # leave an untracked original video. Synchronous callers have no callback.
    register_owned_upload(video_path)

    response = None
    try:
        # ---- 存盘（顺带挡体积）----
        written = 0
        try:
            with video_path.open("wb") as fh:
                while chunk := file.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413, f"视频超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限")
                    fh.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"写入上传文件失败：{exc}") from exc

        if written == 0:
            raise HTTPException(400, "上传的文件是空的")

        _t["① 接收上传"] = time.time() - _t0
        out_dir = RUNS_DIR / token
        logs: list[str] = []

        # ---- 感知层 ----
        _t1 = time.time()
        report_progress("perception")
        # ---- ⚠️ **显式传 `probe_every=8`，不动全局默认值** ----
        #
        # ## 为什么在这里传而不是改默认值
        # 实测抽帧占感知层的 **97%**（`extract_frames` 523s vs `analyze_frames` 14s），
        # 而 `probe_every` 2→8 让这一段 **2.6x**（533s → 206s），
        # 端到端 **1.94x**（375s → 194s），
        # 且**教学结论一致**（`key_concept` 相同、卡点同指）。
        #
        # **但不改全局默认值**，理由三条：
        # ① 默认值影响**所有入口**（CLI 脚本、别的调用方），而**我只验过这一条路**
        # ② `service.py` L422 有个常量注释写着"取 30：`probe_every=2` 时约等于
        #    视频开头 2 秒" —— **它按 2 反推的**。改成 8 会让那个窗口在视频时间上
        #    变长约 4 倍，而它喂的是场景切换判定与自适应校准，**我没测过那种耦合**
        # ③ 而且默认值**有两处**（`keyframe.py` L255 与 `service.py` L174），
        #    只改一处会不一致
        #
        # **显式传参 = 影响范围明确、可回退。**
        #
        # ⚠️ **验证边界**：只在**一个视频**上验过
        # （`真实例子作答02`，5 分钟、手写、场景平稳）。
        # 镜头晃动大 / 频繁擦改的素材可能真漏变化点 ——
        # **而"漏了"这件事，已证明从文本里测不出来**（OCR 抖动太大）。
        _opts = VideoAnalysisOptions()
        _opts.probe_every = 8
        try:
            summary = run_pipeline(video_path, out_dir, _opts,
                         log=lambda m: logs.append(str(m)),
                         progress_cb=perception_progress_callback())
        except Exception as exc:  # noqa: BLE001
            logger.exception("视频感知层失败")
            raise HTTPException(
                502, f"视频解析失败：{type(exc).__name__}: {exc}") from exc

        _t["② 感知层"] = time.time() - _t1
        _t2 = time.time()
        report_progress("snapshots")

        # ---- 切快照 ----
        #
        # ⚠️ **从 `results.json` 读，不要用 `run_pipeline` 的返回值。**
        # 实测：`run_pipeline` 返回的是 `summary`（另一套形状），
        # 里面**没有 `frames`** —— 而 `frames` 在它导出的 `results.json` 里（实测 158 条）。
        # 我第一版拿 summary 去 `.get("frames")`，**拿到空**。
        # **导出物才是对外契约**，用它更稳，也不用猜 summary 的字段名。
        rj = out_dir / "results.json"
        if not rj.is_file():
            raise HTTPException(502, f"感知层没有产出 results.json（{out_dir.name}）")
        try:
            results = json.loads(rj.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"results.json 读不出来：{exc}") from exc

        snaps = _snapshots_from(results, max(1.0, every_sec))
        candidate_snapshot_count = len(snaps)
        duplicate_observations = []
        if request_hint is not True:
            from app.vision.teaching_snapshots import coalesce_snapshots
            snaps, duplicate_observations = coalesce_snapshots(snaps)
        if not snaps:
            raise HTTPException(
                422, "这段视频里没有识别到可提交的作答内容。"
                     "确认画面里写的内容清晰、且不是空镜。")

        _t["③ 切快照"] = time.time() - _t2
        _t3 = time.time()

        # ---- 逐帧驱动教学链路（**和文字作答同一条代码路**）----
        rounds: list[dict] = []
        cur_attempt = attempt_id
        teaching_stop_reason = "exhausted"
        report_progress("teaching", 0, len(snaps), "snapshots")
        for i, g in enumerate(snaps, 1):
            req = AnalyzeRequest(student_id=student_id, problem_id=problem_id,
                                 answer=g["text"].strip(), attempt_id=cur_attempt,
                                 request_hint=request_hint)
            req._media_type = "video"
            req._media_timestamp = g["ts"]
            try:
                resp = analyze_attempt(req)
            except Exception as exc:  # noqa: BLE001
                # 预检与实际分析之间可能被重置/结束。首轮尚无结果时保留
                # 入口的 4xx 契约；已提交的轮次则按部分轨迹如实返回。
                if not rounds and isinstance(exc, HTTPException):
                    raise
                logger.exception("第 %d 轮驱动失败", i)
                rounds.append({"round": i, "ts": g["ts"],
                               "answer": (g.get("text") or "").strip(), "error":
                               f"{type(exc).__name__}: {exc}"})
                teaching_stop_reason = "analysis_error"
                break
            d = resp.model_dump()
            cur_attempt = d.get("attempt_id") or cur_attempt
            # ⚠️ **必须把"学生实际写了什么"一起记下来。**
            # `AnalyzeResponse` 里没有它（那是回给学生看的，不含他的原文），
            # 而复盘要判"他是真会还是凑对" —— **唯一依据就是这一行**。
            # 第一版我漏了这个，复盘照样出得来、看着还挺通顺，
            # 只是卡点描述全变成"或者…或者…"（在猜）。
            rounds.append({"round": i, "ts": g["ts"],
                           "answer": (g.get("text") or "").strip(), **d})
            report_progress("teaching", i, len(snaps), "snapshots", attempt_id=cur_attempt)
            if d.get("finished") or d.get("status") == "finished":
                teaching_stop_reason = "completed"
                break

        _t["④ 驱动教学链路"] = time.time() - _t3
        _t4 = time.time()

        # Save immutable server-side input before any optional expensive review.
        ok = [r for r in rounds if "error" not in r]
        _save_review_input(out_dir, student_id, problem_id, cur_attempt, ok)
        review_result = None
        # 契约 §2.1：`review_available` = 「服务端材料仍在、**可以稍后**请求生成」，
        # 与生成状态正交。**没有可用轮次时不许声称可复盘** ——
        # 那时 `cur_attempt` 可能为空，而 `POST /video/review` 要求 `attempt_id` 非空。
        review_available = bool(ok and cur_attempt)
        want_review = _review_requested(include_review, False)
        if want_review:
            report_progress("review")
            try:
                review_result = review_video(VideoReviewRequest(
                    student_id=student_id, problem_id=problem_id,
                    attempt_id=cur_attempt, run_id=token))
            except Exception:  # noqa: BLE001
                # ⚠️ **收全部异常，不只 HTTPException。**
                # 教学轨迹已经算完并落盘，复盘失败**不该把它一起毁掉** ——
                # 而契约要求「请求生成但失败 ⇒ 如实返回 `failed`」，
                # **抛出 502 就永远返回不了那个 failed**。
                logger.warning("即时视频复盘失败，轨迹仍可用", exc_info=True)
        _t["⑤ 离线复盘"] = time.time() - _t4 if want_review else 0.0
        _t["合计"] = time.time() - _t0

        response = {
            "media_type": "video",
            "run_id": token,
            "review_status": _review_status(want_review, review_result is not None),
            "review_available": review_available,
            "review_scope": "video_run_snapshot",
            "student_id": student_id,
            "problem_id": problem_id,
            "coverage": summary.get("coverage") or {},
            "teaching_stop_reason": teaching_stop_reason,
            "remaining_snapshots": len(snaps) - len(ok),
            "last_submitted_ts": ok[-1]["ts"] if ok else None,
            "attempt_id": cur_attempt,
            "rounds": rounds,
            "snapshot_count": len(snaps),
            "candidate_snapshot_count": candidate_snapshot_count,
            "duplicate_snapshots_skipped": len(duplicate_observations),
            "duplicate_observations": duplicate_observations,
            "perception_timings": summary.get("timings") or {},
            "submitted": len(ok),
            "off_graph_rounds": sum(1 for r in ok if r.get("round_off_graph")),
            "visited_nodes": sorted({n for r in ok
                                     for n in (r.get("visited_nodes") or [])}),
            "weak_concepts": sorted({w for r in ok
                                     for w in (r.get("weak_concepts") or [])}),
            "video_run_dir": str(out_dir.relative_to(PROJECT_ROOT)),
            "upload_removed": False,
            # **分阶段耗时**，供排查"怎么这么慢"。学生实测 412s 而估算只有 119s，
            # 差的那 293s 就藏在这几项里。
            "timings": {k: round(v, 2) for k, v in _t.items()},
            # 复盘：**放在轨迹前面**，学生先看结论。
            # `None` = 复盘没跑出来（不影响轨迹可用）。
            "review": review_result["review"] if review_result else None,
            "log_tail": logs[-8:],
        }
        return response
    finally:
        # 清理失败只记录，不覆盖解析/教学的原始异常。
        removed = False
        try:
            video_path.unlink(missing_ok=True)
            removed = not video_path.exists()
        except Exception:
            logger.exception("删除上传文件失败（%s）", video_path)
        if response is not None:
            response["upload_removed"] = removed


@router.post("/analyze_image")
def analyze_image(
    file: UploadFile = File(..., description="学生卡住时拍的作答照片"),
    problem_id: str = Form(...),
    student_id: str = Form(...),
    attempt_id: str = Form(""),
    request_hint: bool = Form(False),
    include_review: bool = Form(True),
) -> dict:
    from app.config import get_settings
    from app.services.request_execution import request_execution
    settings = get_settings()
    roles = settings.photo_role_options if settings.llm_provider == 'deepseek' else {}
    with request_execution(settings.photo_analysis_budget_s, roles=roles):
        return _analyze_image(file, problem_id, student_id, attempt_id, request_hint, include_review)


def _analyze_image(file, problem_id, student_id, attempt_id, request_hint, include_review):
    """**一张照片 → 一次批改。** 与 `/analyze`（视频）返回**同一种形状**。

    ## 为什么加这一条
    视频那条路的成本主要是**解码**（855 帧 → 254 秒），而**卡住时的求助
    根本不需要过程** —— 学生举着纸拍一张就够了。
    ```
    视频   5 分钟 → 855 帧 → **254 秒**
    图片   1 张   →   1 次 VL 调用 → **约 2 秒**
    ```
    ⇒ 这不是"视频的简化版"，是**另一条更贴切的入口**：
    学生要的不是"回放我怎么做题的"，是"我卡在这了，你看一眼"。

    ## 复用而不是另起
    · 识别：`call_vision` + `DEFAULT_PROMPT`（那段提示词本就是为
      "学生手写作答画面"写的，输出 `{text, description, has_content}`）
    · 教学：`analyze_attempt` —— **和文字作答、和视频快照完全同一条代码路**
    · 复盘：`review_trajectory`
    ⇒ 前端拿到的东西和视频那条**形状一致**，渲染代码不用分叉。
    """
    from app.api.attempts import AnalyzeRequest, analyze_attempt
    from app.services.problem_repository import get_problem_repository
    from app.services.review_builder import review_trajectory
    from app.vision.service import DEFAULT_PROMPT, call_vision

    _t: dict[str, float] = {}
    _t0 = time.time()
    _validate_media_target(student_id, problem_id, attempt_id)

    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(400, "上传的文件是空的")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限")
    _t["① 接收上传"] = time.time() - _t0

    # ---- 识别（一次 VL 调用）----
    _t1 = time.time()
    try:
        text, meta = call_vision(raw, DEFAULT_PROMPT,
                                 mime=file.content_type or "image/jpeg")
    except Exception as exc:  # noqa: BLE001
        logger.exception("图片识别失败")
        raise HTTPException(
            502, f"图片识别失败：{type(exc).__name__}: {exc}") from exc
    _t["② 识别"] = time.time() - _t1

    # `DEFAULT_PROMPT` 要求返回 JSON；**模型偶尔会包 Markdown 代码块**，
    # 所以两种都要能吃。解析不出来就退回用原始文本 —— 那也比如实为空好。
    answer = ""
    if isinstance(text, str) and text.strip():
        s = text.strip()
        if s.startswith("```"):
            s = s.split("```")[1] if "```" in s[3:] else s[3:]
            s = s.removeprefix("json").strip()
        try:
            answer = (json.loads(s) or {}).get("text") or ""
        except Exception:  # noqa: BLE001
            answer = text
    answer = (answer or "").strip()

    if not answer:
        raise HTTPException(
            422, "这张图里没有识别到作答内容。"
                 "确认字迹清晰、画面里确实有解题过程。")

    from app.services.video_progress import progress_callback
    callback = progress_callback.get()
    if callback is not None:
        callback('teaching', {'completed': 1, 'total': 1, 'unit': 'images',
                             'recognized_text': answer})

    # ---- 教学链路（一轮）----
    _t2 = time.time()
    req = AnalyzeRequest(student_id=student_id, problem_id=problem_id,
                         answer=answer, attempt_id=attempt_id,
                         request_hint=request_hint)
    req._media_type = "image"
    try:
        resp = analyze_attempt(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("图片批改失败")
        raise HTTPException(
            502, f"批改失败：{type(exc).__name__}: {exc}") from exc
    d = resp.model_dump()
    _t["③ 批改"] = time.time() - _t2
    from app.services.teacher_read import retain_photo
    d['image_evidence_status'] = retain_photo(raw, source_id=d.get('source_id', ''),
        student_id=student_id, problem_id=problem_id, attempt_id=d.get('attempt_id', ''))
    rounds = [{"round": 1, "ts": 0.0, "answer": answer, **d}]

    # ---- 复盘（**按需**，契约 §2.1）----
    #
    # ⚠️ 这是「照片即时辅导」的提速开关。
    # 学生卡住时举着一张纸要的是**下一步提示**，不是一整段过程复盘；
    # 旧实现无条件生成复盘 ⇒ 他得等复盘跑完才看到任何东西。
    # **默认仍是 `true`**（旧调用方行为不变），新页面显式传 `false`。
    want_review = _review_requested(include_review, True)
    report = None
    _t3 = time.time()
    if want_review:
        try:
            prob = get_problem_repository().get_problem(problem_id)
            report = review_trajectory(prob, rounds) if prob else None
            # ⚠️ **照片路径原先绕过结构闸门**（2026-10-09 收口）：
            #    视频路径的即时复盘走 `review_video()` ⇒ 过闸门；
            #    照片路径直接把 `report.model_dump()` 返回 ⇒ **结构不合格也当成功给出去**。
            #    这是"两条路各写一遍就会漂开"的第 N 次 —— 而这次是**照片绕过了**。
            #
            # ⚠️ 但照片路径**没有重试路径**（`review_available=false`，不落盘、无 run_id），
            #    所以不能像视频那样抛 502 让学生重试 —— 那只会让他连识别原文都拿不到。
            #    ⇒ 按契约 §2.1 的三态：**不过闸门就不返回成功结果**
            #      （`review=None` + `review_status=failed`），
            #      识别原文与即时提示照常返回（§2.1 不变式①：复盘失败不得击穿批改结果）。
            if report is not None:
                from app.services.review_builder import classify_defects

                defects = _structural_defects(report.model_dump())
                if defects:
                    logger.warning(
                        "照片复盘结构不合格（%s），按契约不返回成功结果：%s",
                        classify_defects(defects), [d.get("kind") for d in defects])
                    report = None
        except Exception:  # noqa: BLE001
            # 契约 §2.1 不变式①：**复盘失败不得击穿已经算好的批改结果**。
            # 旧实现让它抛成 502 —— 学生连「识别到了什么、下一步做什么」都拿不到。
            logger.warning("照片复盘失败，批改结果仍可用", exc_info=True)
            report = None
    _t["④ 复盘"] = time.time() - _t3 if want_review else 0.0
    _t["合计"] = time.time() - _t0

    return {
        "media_type": "image",
        "student_id": student_id,
        "problem_id": problem_id,
        "attempt_id": d.get("attempt_id") or attempt_id,
        "rounds": rounds,
        "snapshot_count": 1,
        "submitted": 1,
        "off_graph_rounds": 1 if d.get("round_off_graph") else 0,
        "visited_nodes": d.get("visited_nodes") or [],
        "weak_concepts": d.get("weak_concepts") or [],
        "recognized_text": answer,
        "teaching_timings_ms": d.get('node_timings_ms', {}),
        "review": report.model_dump() if report is not None else None,
        "review_status": _review_status(want_review, report is not None),
        # 照片没有复盘输入目录及 `run_id`；核对原图的留存是独立证据机制 ⇒
        # 当前**不存在**「稍后生成」的路径。如实说 `false`，
        # 前端就不会给学生挂一个按不动的按钮（契约 §2.1）。
        "review_available": False,
        "timings": {k: round(v, 2) for k, v in _t.items()},
        "upload_removed": True,   # 本函数已消费临时输入；核对原图副本按证据策略保留
        "log_tail": [],
    }


from app.services.ocr_correction import CorrectionRequest, correct_ocr


@router.post("/correct_ocr")
def correct_ocr_endpoint(req: CorrectionRequest) -> dict:
    return correct_ocr(req)
