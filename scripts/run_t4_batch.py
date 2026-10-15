"""Freeze AI-authored T4 samples and run complete cases against isolated services.

Dry runs are plans, never model results. Real runs use disposable databases and
the application's FastAPI routes in-process; they never contact the demo server.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
CATEGORIES = {"correct", "legal_equivalent", "missing_reasoning", "explicit_error",
              "ocr_ambiguity", "truncated_cross_round"}


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("origin") != "ai_authored":
        raise ValueError("This protocol requires explicitly AI-authored samples")
    problems = data.get("problems", [])
    if len(problems) != 6 or len({p["id"] for p in problems}) != 6:
        raise ValueError("Require six distinct problems")
    if sorted(p["split"] for p in problems) != ["development"] * 4 + ["holdout"] * 2:
        raise ValueError("Require four development and two problem-isolated holdout problems")
    ids = set()
    for p in problems:
        if not p.get("statement") or not p.get("reference_solution"):
            raise ValueError("Problem lacks statement/reference solution")
        cases = p.get("cases", [])
        if len(cases) != 6 or {c["category"] for c in cases} != CATEGORIES:
            raise ValueError("Each problem requires exactly the six T4 categories")
        for c in cases:
            if c["id"] in ids or not c.get("expected"):
                raise ValueError("Duplicate case or missing predeclared expectation")
            ids.add(c["id"])
            if not c.get("rounds") or not all(isinstance(r, str) and r.strip() for r in c["rounds"]):
                raise ValueError("Each round must contain nonempty verbatim sample text")
        if p.get("diagram"):
            target = (Path(path).parent / p["diagram"]).resolve()
            if not target.is_relative_to(Path(path).parent.resolve()) or not target.is_file():
                raise ValueError("Diagram missing or outside dataset directory")
    return data


def source_files(dataset):
    files = [Path(dataset).resolve(), Path(__file__).resolve()]
    files.extend(Path(dataset).parent.glob("*.py"))
    for base in (ROOT / "backend/app", ROOT / "prompts", ROOT / "frontend"):
        files.extend(p for p in base.rglob("*") if p.suffix in {".py", ".md", ".html", ".js", ".css"})
    data = load_dataset(dataset)
    files.extend(Path(dataset).parent / p["diagram"] for p in data["problems"] if p.get("diagram"))
    for p in (ROOT / "requirements.txt", ROOT / "scripts/accept_real_videos.py"):
        files.append(p)
    return {str(p.resolve().relative_to(ROOT)): digest(p) for p in sorted(set(files))}


def model_configuration():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from app.config import get_settings
    from app.agent.prompt_registry import resolve_version
    from urllib.parse import urlsplit, urlunsplit
    s = get_settings()
    u = urlsplit(s.deepseek_base_url)
    return {"provider": "deepseek", "model": "deepseek-flash",
            "base_url": urlunsplit((u.scheme, u.netloc.rsplit("@", 1)[-1], u.path, "", "")),
            "temperature": s.llm_temperature, "timeout": s.llm_timeout,
            "max_retries": s.llm_max_retries, "review_prompt": os.getenv('PROMPT_REVIEW_VERSION','v3'),
            "review_timeout":s.review_timeout,"review_max_retries":s.review_max_retries,
            "prompt_versions": {role: (os.getenv('PROMPT_REVIEW_VERSION','v3') if role == "review" else resolve_version(role))
                                for role in ("review", "diagnosis", "tutor", "analyze_step", "step_matcher")},
            "rag": "empty disposable corpus; not a RAG comparison",
            "rag_material_policy": os.getenv("RAG_MATERIAL_POLICY", s.rag_material_policy),
            "teaching_graph": "graphless; SVG question diagrams are not knowledge graphs",
            "diagram_scope": "SVGs accompany source/reviewer packets; models receive self-contained text only"}


def freeze(dataset, path, reviews, seed=20261009):
    if path.exists():
        raise ValueError("Freeze output exists; never overwrite a frozen batch")
    data = load_dataset(dataset)
    reports = [json.loads(Path(p).read_text(encoding="utf-8-sig")) for p in reviews]
    if len(reports) != 2 or len({r.get("reviewer_id") for r in reports}) != 2:
        raise ValueError("Require two distinct independent AI sample reviews")
    sample_hash = digest(dataset)
    for r in reports:
        if (r.get("dataset_sha256") != sample_hash or r.get("verdict") != "accept"
                or r.get("provenance") != "independent_ai_review"
                or set(r.get("case_ids", [])) != {c["id"] for p in data["problems"] for c in p["cases"]}):
            raise ValueError("Reviews must accept this exact dataset and cover all cases")
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dataset": str(Path(dataset).resolve().relative_to(ROOT)),
                "files": source_files(dataset), "configuration": model_configuration(),
                "review_files": {str(Path(p).resolve().relative_to(ROOT)): digest(p) for p in reviews},
                "seed": seed, "cases": 36, "repetitions": 3, "planned_runs": 108,
                "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                 if d.metadata["Name"]},
                "origin": "ai_authored; independently AI-reviewed; no teacher/student research",
                "holdout_policy": "On first holdout failure retire this holdout as development evidence"}
    holdout_sources = sorted((p["statement"], digest(Path(dataset).parent / p["diagram"])
                              if p.get("diagram") else None)
                             for p in data["problems"] if p["split"] == "holdout")
    manifest["holdout_identity"] = hashlib.sha256(json.dumps(
        holdout_sources, ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest["holdout_question_identities"] = [hashlib.sha256(json.dumps(
        p, ensure_ascii=False).encode("utf-8")).hexdigest() for p in holdout_sources]
    write(path, manifest)
    return manifest


def verify(manifest):
    for rel, expected in {**manifest["files"], **manifest["review_files"]}.items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f"Frozen file drift: {rel}")
    if model_configuration() != manifest["configuration"]:
        raise ValueError("Frozen model configuration drift")


def evidence_digest(batch):
    files = list((batch / "runs").glob("*.json")) + list((batch / "review_packets").glob("*"))
    mapping = batch / "restricted_mapping.json"
    if mapping.exists():
        files.append(mapping)
    content = {str(p.relative_to(batch)): digest(p) for p in sorted(files) if p.is_file()}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def check_development_gate(gate_path, manifest_path, data):
    """Require actual full development records and two locked independent reviews."""
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    expected_sha = digest(manifest_path)
    if gate.get("manifest_sha256") != expected_sha or gate.get("passed") is not True:
        raise ValueError("Development gate does not pass for this manifest")
    batch = (ROOT / gate["batch_dir"]).resolve()
    if not batch.is_relative_to(ROOT):
        raise ValueError("Batch evidence outside workspace")
    summary_path = batch / "summary.json"
    if digest(summary_path) != gate.get("summary_sha256"):
        raise ValueError("Development summary drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("evidence_sha256") != evidence_digest(batch):
        raise ValueError("Raw development outputs/review packet drift")
    expected = {(c["id"], i) for p in data["problems"] if p["split"] == "development"
                for c in p["cases"] for i in range(1, 4)}
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (batch / "runs").glob("*.json")]
    if (summary.get("dry_run") is not False or summary.get("smoke") is not False
            or len(records) != 72 or {(r["case_id"], r["repeat"]) for r in records} != expected
            or any(r.get("manifest_sha256") != expected_sha or r.get("split") != "development"
                   or r.get("status") not in {"succeeded", "failed"} for r in records)
            or sum(r["status"] == "succeeded" for r in records) / 72 < .9):
        raise ValueError("Gate requires all 72 actual development records and >=90% availability")
    reviewers = set()
    adjudication = _load_adjudication(gate, manifest_path, summary_path, batch)
    for entry in gate.get("review_reports", []):
        report_path = (ROOT / entry["path"]).resolve()
        if not report_path.is_relative_to(ROOT) or digest(report_path) != entry["sha256"]:
            raise ValueError("Independent review evidence drift")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics", {})
        # ⚠️ **有裁决时，严重指标与 verdict 以裁决为准**（见 `_load_adjudication`）——
        #    初评的 verdict 是**裁决前**的意见，拿它当验收条件等于"不许裁决改变结论"。
        #    但**绑定与锁定条件对初评照查**：证据链不能因为有了裁决就放松。
        hard = (report.get("manifest_sha256") != expected_sha
                or report.get("summary_sha256") != digest(summary_path)
                or report.get("evidence_sha256") != evidence_digest(batch)
                or report.get("provenance") != "independent_ai_review"
                or report.get("initial_ratings_locked") is not True
                or set(report.get("blind_item_ids", [])) != {f"Q{i:03d}" for i in range(1, 73)})
        if hard:
            raise ValueError("Independent development review does not meet the frozen engineering gate")
        if adjudication is None:
            # 无裁决：沿用原来的严格条件（初评必须自己就达标）
            if (report.get("verdict") != "accept"
                    or any(metrics.get(k) != 0 for k in ("severe_false_accusations",
                           "unsupported_cognitive_claims", "invalid_displayed_references",
                           "serious_physics_contradictions"))
                    or metrics.get("explicit_error_total") != 12
                    or metrics.get("explicit_error_recognized", 0) < 10):
                raise ValueError("Independent development review does not meet the frozen engineering gate")
        else:
            _require_review_is_the_adjudicated_one(report, adjudication)
        # ⚠️ 两条路都要记 reviewer_id —— 第一版我把它放进了 else 分支，
        #    结果**没有裁决的旧路径**里 reviewers 永远是空的，
        #    连"需要两份独立评审"这条都会误报（两个原有测试当场红了）。
        reviewers.add(report.get("reviewer_id"))
    if adjudication is not None:
        _require_adjudication_passes(adjudication, gate)
    if len(reviewers - {None, ""}) != 2:
        raise ValueError("Require two independent locked AI development reviews")


#: 裁决后**允许**残留的分歧字段 —— 只允许**不进门槛判据**的评分项。
#: ⚠️ 严重指标那四个布尔**一个都不许残留**（它们就是门槛本身）。
_NON_GATING_DISAGREEMENT_SUFFIX = "_1to5"

#: `unresolved_disagreement_fields` 里**允许出现的说明键**（不是字段，是文档）。
#: ⚠️ 实测踩到过一次：裁决者把 `note` 和真字段并列写进了那个映射，
#: 于是一个**说明**被当成了"未裁决的门槛字段"。
#: ⇒ 白名单**显式列出**说明键；其余任何键都必须以 `_1to5` 结尾。
#:    （不写成"忽略所有不认识的键" —— 那会让人把 `unsupported_cognitive_claim`
#:      换个名字藏进去。）
_DOC_KEYS_IN_DISAGREEMENTS = frozenset({"note", "notes", "explanation", "comment"})


def _load_adjudication(gate, manifest_path, summary_path, batch):
    """把"裁决后通过"接进验收链（2026-10-09 新增）。

    ## 为什么必须改这里
    原先 `check_development_gate` **只认初评**：它要求每份评审报告的
    `verdict == "accept"` 且四项严重指标为 0。⇒ 一旦裁决**推翻**了某条标记，
    「裁决后通过」的文件**仍然过不了自动关口** ——
    而守住留出集的正是这道关口。**裁决就等于写了个没人看的文件。**

    ## 这道接入是**严格**的（不是"看到 passed=true 就放行"）
    ```
    ① 裁决文件必须在工作区内、且哈希与 gate_final.adjudication_sha256 一致
    ② provenance 必须是 ai_integrator_adjudication（不许匿名）
    ③ adjudicator 必须显式自称 **root_integrator_not_a_third_blind_reviewer**
       —— 它不是第三名盲评者，这一点必须写在文件里而不是靠口头约定
    ④ initial_reviews_preserved 必须为 true，且**初评文件必须仍是裁决时的那两份**
       （逐份比对 original_initial_sha256；B 那种"重填版"必须显式登记）
    ⑤ 裁决必须绑定同一份 manifest / summary / evidence
    ⑥ **残留分歧只允许评分项**（`*_1to5`）；四个严重指标的布尔一个都不许残留
    ⑦ 裁决后的四项严重指标必须全 0，且明确错误识别 ≥ 10/12
    ```
    返回裁决 dict；没有裁决时返回 None（走原来的严格路径）。
    """
    if not gate.get("adjudication_sha256"):
        return None
    rel = gate.get("disagreements_resolved_in") or gate.get("adjudication_path")
    if not rel:
        raise ValueError("Adjudicated gate must name the adjudication file")
    path = (ROOT / rel).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Adjudication evidence missing or outside workspace")
    if digest(path) != gate["adjudication_sha256"]:
        raise ValueError("Adjudication evidence drift")
    adj = json.loads(path.read_text(encoding="utf-8"))
    if not str(adj.get("provenance", "")).startswith("ai_integrator_adjudication"):
        raise ValueError("Adjudication must declare ai_integrator_adjudication provenance")
    if adj.get("adjudicator") != "root_integrator_not_a_third_blind_reviewer":
        raise ValueError("Adjudication must declare it is not a third blind review")
    if adj.get("initial_reviews_preserved") is not True:
        raise ValueError("Adjudication must preserve the initial locked reviews")
    if (adj.get("manifest_sha256") != digest(manifest_path)
            or adj.get("summary_sha256") != digest(summary_path)
            or adj.get("evidence_sha256") != evidence_digest(batch)):
        raise ValueError("Adjudication bindings do not match this batch")
    unresolved = (gate.get("unresolved_disagreement_fields") or {})
    bad = [k for k in unresolved
           if k not in _DOC_KEYS_IN_DISAGREEMENTS
           and not k.endswith(_NON_GATING_DISAGREEMENT_SUFFIX)]
    if bad:
        raise ValueError(f"Unresolved disagreements include gating fields: {sorted(bad)}")
    return adj


def _require_review_is_the_adjudicated_one(report, adjudication):
    """初评必须是**裁决所针对的那一份**（哈希登记在裁决文件里）。"""
    known = set((adjudication.get("original_initial_sha256") or {}).values())
    if not known:
        raise ValueError("Adjudication must register the initial review hashes")
    if report.get("initial_ratings_sha256") not in known:
        raise ValueError("Adjudication does not cover this review's locked ratings")


def _require_adjudication_passes(adjudication, gate):
    """裁决后的门槛判据（四项全 0 + 识别 ≥10/12）。"""
    counts = adjudication.get("adjudicated_boolean_counts") or {}
    expected = ("severe_false_accusation", "unsupported_cognitive_claim",
                "invalid_displayed_reference", "serious_physics_contradiction")
    if set(counts) != set(expected) or any(counts[k] != 0 for k in expected):
        raise ValueError("Adjudicated severe metrics are not all zero")
    rec = adjudication.get("explicit_error_recognition_after_adjudication") or {}
    if rec.get("total") != 12 or rec.get("recognized", 0) < 10 or rec.get("met") is not True:
        raise ValueError("Adjudicated explicit-error recognition below the frozen threshold")



def plan(data, split, seed):
    items = [{"case_id": c["id"], "problem_id": p["id"], "split": p["split"],
              "category": c["category"], "repeat": repeat, "status": "planned"}
             for p in data["problems"] if split == "all" or p["split"] == split
             for c in p["cases"] for repeat in range(1, 4)]
    random.Random(seed).shuffle(items)
    return items


def execute_cases(items, execute, out, manifest_sha, stop_when=None):
    """Persist a started record before execution; never silently retry failed cases."""
    records = []
    for item in items:
        path = out / "runs" / f"{item['case_id']}-{item['repeat']}.json"
        if path.exists():
            raise ValueError("Run already exists; use a new batch directory")
        if stop_when is not None and stop_when():
            record = {**item, "status": "blocked", "executed": False,
                      "manifest_sha256": manifest_sha,
                      "error": {"type": "AccountCircuitOpen", "detail": "Local operator recovery required; no new case or model request executed"}}
            write(path, record)
            records.append(record)
            continue
        record = {**item, "manifest_sha256": manifest_sha, "status": "running"}
        write(path, record)
        try:
            record.update(execute(item))
        except Exception as exc:
            record.update(status="failed", error={"type": type(exc).__name__, "detail": str(exc)[:1000]})
        write(path, record)
        records.append(record)
    return records


def review_packets(records, data, out, seed):
    """Hide model, sequence, split and expected labels; preserve failures for review."""
    order = list(records)
    random.Random(seed + 1).shuffle(order)
    lookup = {c["id"]: (p, c) for p in data["problems"] for c in p["cases"]}
    mapping = {}
    for n, record in enumerate(order, 1):
        blind_id = f"Q{n:03d}"
        p, c = lookup[record["case_id"]]
        mapping[blind_id] = {k: record[k] for k in ("case_id", "repeat", "split", "category")}
        review = (record.get("review") or {}).get("response", {}).get("review")
        diagram = None
        if p.get("diagram"):
            source = ROOT / "evaluation/t4_demo" / p["diagram"]
            diagram = f"{blind_id}{source.suffix}"
            target = out / "review_packets" / diagram
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        write(out / "review_packets" / f"{blind_id}.json", {
            "blind_item_id": blind_id, "origin": "ai_authored_sample_and_ai_generated_output",
            "statement": p["statement"], "reference_solution": p["reference_solution"],
            "diagram": diagram, "student_material": c["rounds"],
            "workflow_status": record["status"],
            "teaching": [{"hint": t["response"].get("hint"), "issue": t["response"].get("issue"),
                          "degraded": t["response"].get("degraded"),
                          # ⚠️ **验收③**：评审材料要展示**实际投递的替代提示**及**替换标记**，
                          #    并让人看得出"这句不是模型原文"。
                          #    `hint` 就是**实际投递给学生的**那一句（替换后是保守提示）；
                          #    `hint_substituted` 标明它被替换过；`hint_substitution_reasons`
                          #    给出**为什么**（判据命中项）。
                          #    ⚠️ **原模型提示不进包** —— 它留在审计记录里
                          #    （`safety_substitutions[].original`），
                          #    因为评审要评的是**学生实际看到的东西**。
                          "hint_substituted": bool(t["response"].get("safety_substitutions")),
                          "hint_substitution_reasons": [
                              r for s in (t["response"].get("safety_substitutions") or [])
                              for r in (s.get("reasons") or [])]}
                         for t in record.get("teaching", []) if isinstance(t.get("response"), dict)],
            "review": review,
            "instructions": "Independent AI review only. Lock initial ratings before comparing. "
                            "Do not infer student cognition; structure validation is not physics verification.",
            # ⚠️ **计分规则必须随包投递**（2026-10-11 用户裁定，流程缺陷修复）：
            # 背景：v16 留出集盲评里，一名评审把 `answer_leakage` 的「越高越差」
            # 方向误用到了另外三个「越高越好」字段 ⇒ **36/36 方向颠倒** ✗
            # 事后实测：36 个匿名包里「越高越好 / 越高越差 / 无泄题 / rubric」等
            # **全部 0 命中**，`instructions` 也不含方向 ⇒ 规则只在 docs 表格里，
            # **没随包投递** ✗
            # ⚠️ 但**不能**据此断定它是那次错误的**唯一根因** —— 只能说
            # **流程缺陷 + 可能促因** ✓（用户口径）
            # ⇒ 修复原则（用户原话）：**匿名只隐藏版本身份，不能隐藏评分规则** ✓
            "scoring_rules": {
                "criterion_version": "2026-10-11-v1",
                "unit_of_analysis": {
                    "correctness_1to5": "评**报告**对不对（复算/结论核对），不是评材料本身；"
                                        "学生材料里的错误由 material_has_definite_error 承载，不在此重复扣分",
                    "targeting_1to5": "评**提示有没有打中本轮缺口**",
                    "actionability_1to5": "评**下一步是否可执行**；"
                                          "⚠️ next_step 里出现答案数值属 leakage 射程，不在此扣分",
                    "answer_leakage_1to5": "**只衡量即时提示 `teaching[].hint`** 是否提前给出"
                                           "学生尚未得到的完整答案",
                },
                "direction": {
                    "correctness_1to5": "越高越好（5 = 最好）",
                    "targeting_1to5": "越高越好（5 = 最好）",
                    "actionability_1to5": "越高越好（5 = 最好）",
                    "answer_leakage_1to5": "**越高越差（1 = 最好）**",
                },
                "range": "1–5 的整数",
                "anchor": {
                    "answer_leakage_1to5": "无泄题 = 1（点规律/点错误方向/要求自检单位/"
                                           "复述学生已写出的数 ⇒ 不算泄题）",
                },
                "leakage_scope": {
                    "measures": "**只有 `teaching[].hint`**（合同口径：即时提示）",
                    "not_measured": "`issue` / `sticking_points` / `next_step` / `answer_check` / "
                                    "`encouragement` 等**事后复盘字段**；"
                                    "其中**同一字段内明写「核对时重建」**者不计泄题",
                    "note": "⚠️ `teaching[].issue` 在**认知归因判据（§2.10）**射程内，"
                            "但那**不管算不算泄题** —— 两条判据的射程不要混用",
                },
                "ratable_rule": "`ratable = (workflow_status == \"succeeded\")`；"
                                "失败项四项记 `null`（不补零），"
                                "`explicit_error_detected=false`（**`review` 为 null 时一律 false**）",
                "severe_flags": ["severe_false_accusation", "unsupported_cognitive_claim",
                                 "invalid_displayed_reference", "serious_physics_contradiction"],
            }})
    write(out / "restricted_mapping.json", mapping)


@contextmanager
def capture_requests():
    """Count physical provider invocations, including failed network attempts.

    _complete is exactly one SDK request (SDK retries disabled in this provider).
    No API keys, headers, or gateway query strings are recorded.
    """
    from app.services.llm.openai_compatible import OpenAICompatibleProvider
    from app.services.llm.base import LLMProvider
    original = OpenAICompatibleProvider._complete
    original_retry = LLMProvider._with_retry
    original_structured = LLMProvider.generate_structured
    calls = []
    local = threading.local()
    def structured(self, *args, **kwargs):
        previous = getattr(local, "structured", None)
        local.structured = {"id": uuid.uuid4().hex, "pass": 0}
        try:
            return original_structured(self, *args, **kwargs)
        finally:
            local.structured = previous
    def retry_group(self, *args, **kwargs):
        previous = getattr(local, "group", None)
        local.group = {"id": uuid.uuid4().hex, "attempt": 0}
        structure = getattr(local, "structured", None)
        if structure is not None:
            structure["pass"] += 1
        try:
            return original_retry(self, *args, **kwargs)
        finally:
            local.group = previous
    def measured(self, messages):
        group = getattr(local, "group", None)
        structure = getattr(local, "structured", None)
        if group is not None:
            group["attempt"] += 1
        row = {"model": self.model, "messages": messages, "status": "running",
               "retry_group": group["id"] if group else None,
               "network_attempt": group["attempt"] if group else 1,
               "structured_group": structure["id"] if structure else None,
               "structured_pass": structure["pass"] if structure else 1}
        calls.append(row)
        start = time.monotonic()
        try:
            text, usage = original(self, messages)
            row.update(status="succeeded", response=text, tokens=usage.as_dict())
            return text, usage
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, tokens=None)
            raise
        finally:
            row["elapsed_sec"] = time.monotonic() - start
    OpenAICompatibleProvider._complete = measured
    LLMProvider._with_retry = retry_group
    LLMProvider.generate_structured = structured
    try:
        yield calls
    finally:
        OpenAICompatibleProvider._complete = original
        LLMProvider._with_retry = original_retry
        LLMProvider.generate_structured = original_structured


def real_run(data, items, out, manifest_sha):
    if data.get('fixture_only'):
        raise ValueError('Synthetic unit-test fixtures cannot be used for real model evaluation')
    from accept_real_videos import disposable_services
    # Imports that initialize settings/DB occur only after disposable env is set.
    with disposable_services():
        from app.config import get_settings
        s = get_settings()
        # model_configuration loaded settings before disposable env: explicitly update DB fields.
        s.postgres_host = os.environ["POSTGRES_HOST"]
        s.postgres_port = int(os.environ["POSTGRES_PORT"])
        s.postgres_user = os.environ["POSTGRES_USER"]
        s.postgres_password = os.environ["POSTGRES_PASSWORD"]
        s.postgres_db = os.environ["POSTGRES_DB"]
        s.qdrant_url = os.environ["QDRANT_URL"]
        s.llm_provider, s.llm_model = "deepseek", "deepseek-flash"
        for role in ("step_matcher", "analyze_step", "diagnosis", "tutor", "review"):
            setattr(s, role + "_model", "deepseek-flash")
        os.environ["PROMPT_REVIEW_VERSION"] = model_configuration()['review_prompt']
        os.environ["PROBLEMS_DIR"] = str(out / "problems")
        for p in data["problems"]:
            # No gold labels/reference solution are included in the system's problem object.
            write(out / "problems" / p["id"] / "problem.json",
                  {"problem_id": p["id"], "statement": p["statement"],
                   "knowledge_points": p.get("knowledge_points", []), "metadata": {"origin": "ai_authored"}})
        from app.db.postgres import get_engine
        from app.db.tables import Base
        Base.metadata.create_all(get_engine())
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api import video
        from app.services.review_material import build_synthesized_payload, write_material
        from app.services.attempt_store import get_attempt_store
        from app.services.llm import clear_cache
        clear_cache()
        video.RUNS_DIR = out / "materials"
        cases = {c["id"]: c for p in data["problems"] for c in p["cases"]}
        # Every repeat gets a different student, preventing implicit reuse of an open attempt.
        def execute(item):
            teaching, calls, review = [], [], None
            student, attempt = "t4_" + uuid.uuid4().hex, ""
            result = {"status": "failed", "teaching": teaching, "model_requests": calls}
            started = time.monotonic()
            with capture_requests() as captured:
                try:
                    for answer in cases[item["case_id"]]["rounds"]:
                        req = {"student_id": student, "problem_id": item["problem_id"],
                               "answer": answer, "request_hint": True, "attempt_id": attempt}
                        response = client.post("/attempt/analyze", json=req)
                        payload = response.json()
                        teaching.append({"request": req, "http_status": response.status_code, "response": payload})
                        response.raise_for_status()
                        attempt = payload["attempt_id"]
                        state = get_attempt_store().load_attempt(attempt).state
                        teaching[-1]["state_snapshot"] = state.model_dump(mode="json")
                    rounds = [{"round": i, "ts": None, "answer": t["request"]["answer"]}
                              for i, t in enumerate(teaching, 1)]
                    material = build_synthesized_payload(student_id=student, problem_id=item["problem_id"],
                                                         attempt_id=attempt, rounds=rounds)
                    run_id = uuid.uuid4().hex
                    write_material(video.RUNS_DIR / run_id, material)
                    response = client.post("/video/review", json={"student_id": student,
                        "problem_id": item["problem_id"], "attempt_id": attempt, "run_id": run_id})
                    review = {"http_status": response.status_code, "response": response.json()}
                    response.raise_for_status()
                    if any(t["response"].get("degraded") for t in teaching):
                        raise ValueError("Teaching degraded; not a successful complete case")
                    result["status"] = "succeeded"
                except Exception as exc:
                    result["error"] = {"type": type(exc).__name__, "detail": str(exc)[:1000]}
                finally:
                    calls.extend(captured)
                    result.update(review=review, elapsed_sec=time.monotonic() - started,
                        physical_requests=len(calls), failed_requests=sum(c["status"] == "failed" for c in calls),
                        network_retries=sum(c["network_attempt"] > 1 for c in calls),
                        structure_retries=len({c["retry_group"] for c in calls if c["structured_pass"] > 1}),
                        reported_tokens=sum((c.get("tokens") or {}).get("total_tokens", 0) for c in calls),
                        token_usage_incomplete=any(not (c.get("tokens") or {}).get("total_tokens") for c in calls))
            return result
        with TestClient(app) as client:
            from app.services.llm.factory import get_llm
            return execute_cases(items, execute, out, manifest_sha,
                                 stop_when=lambda: get_llm('tutor').request_guard.status()['blocked'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/t4_demo/dataset.json")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--sample-reviews", nargs=2, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=["all", "development", "holdout"], default="development")
    parser.add_argument("--development-gate", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One development case only; never a quality gate")
    args = parser.parse_args()
    if args.freeze:
        if not args.sample_reviews:
            parser.error("Freeze requires two independent sample reviews")
        freeze(args.dataset, args.freeze, args.sample_reviews)
        print("Frozen 36 cases / 108 planned runs; no model calls")
        return
    if not args.manifest or not args.out:
        parser.error("Require --manifest and fresh --out")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    verify(manifest)
    data = load_dataset(ROOT / manifest["dataset"])
    if data.get('fixture_only') and not args.dry_run:
        parser.error('Synthetic unit-test fixtures cannot reserve holdouts or run real models')
    if not args.dry_run and args.split == "holdout":
        # Holdout gets one generation batch. Physics grading follows separately;
        # neither a crash nor a failed grade authorizes calling a rerun blind.
        claims = [ROOT / "artifacts/t4_holdout_claims" / (identity + ".json")
                  for identity in manifest["holdout_question_identities"]]
        claims[0].parent.mkdir(parents=True, exist_ok=True)
    else:
        claims = []
    items = plan(data, args.split, manifest["seed"])
    if args.smoke:
        if args.split != "development":
            parser.error("Smoke is development-only")
        items = [next(i for i in items if i["category"] == "correct")]
    if not args.dry_run and args.split != "development":
        if args.split == "all" or not args.development_gate:
            parser.error("Real holdout runs require a development gate; all is dry-run only")
        check_development_gate(args.development_gate, args.manifest, data)
    out = args.out.resolve()
    if out.exists():
        parser.error("Output must be a fresh directory")
    if any(claim.exists() for claim in claims):
        parser.error("A holdout question was already reserved/exposed; use new holdout questions")
    for claim in claims:
        with claim.open("x", encoding="utf-8") as stream:
            json.dump({"manifest_sha256": digest(args.manifest), "out": str(out),
                       "claimed_at": datetime.now(timezone.utc).isoformat()}, stream)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.manifest, out / "manifest.json")
    write(out / "plan.json", {"dry_run": args.dry_run, "smoke": args.smoke, "items": items})
    if args.dry_run:
        records = execute_cases(items, lambda item: {"status": "planned", "executed": False},
                                out, digest(args.manifest))
    else:
        try:
            records = real_run(data, items, out, digest(args.manifest))
        except Exception as exc:
            # Service startup failures still account for the whole scheduled denominator.
            # Existing started/completed records remain intact; no paid execution is repeated.
            records = []
            for item in items:
                path = out / "runs" / f"{item['case_id']}-{item['repeat']}.json"
                if path.exists():
                    record = json.loads(path.read_text(encoding="utf-8"))
                else:
                    record = {**item, "status": "failed", "executed": False,
                              "manifest_sha256": digest(args.manifest),
                              "error": {"type": type(exc).__name__, "detail": "Batch infrastructure failure"}}
                    write(path, record)
                records.append(record)
            write(out / "infrastructure_failure.json", {"type": type(exc).__name__,
                  "message": "Infrastructure failed; no automatic rerun; inspect local service logs"})
        review_packets(records, data, out, manifest["seed"])
    categories = {}
    for r in records:
        group = categories.setdefault(r["category"], {"records": 0, "succeeded": 0, "failed": 0, "blocked": 0})
        group["records"] += 1
        if r["status"] in {"succeeded", "failed", "blocked"}:
            group[r["status"]] += 1
    # ⚠️ **验收④（用户 2026-10-10）**：替换次数、替换案例数**必须单独报告**。
    #    理由：替换后的提示**不是模型原始输出** —— 若把"替换后通过"读成
    #    "模型输出合格"，那是把**兜底的功劳记在模型头上**。
    #    ⇒ 这里给三个数：发生替换的**轮次数**、受影响的**案例数**、以及它们的占比；
    #      案例层面的质量结果由聚合器按同一口径另行统计（不混进总数）。
    _sub_rounds = 0
    _sub_cases = 0
    for r in records:
        _n = sum(len(((t.get("response") or {}).get("safety_substitutions")) or [])
                 for t in (r.get("teaching") or []))
        if _n:
            _sub_rounds += _n
            _sub_cases += 1
    write(out / "summary.json", {"dry_run": args.dry_run, "smoke": args.smoke, "records": len(records),
        "evidence_sha256": evidence_digest(out),
        "succeeded": sum(r["status"] == "succeeded" for r in records),
        "failed": sum(r["status"] == "failed" for r in records),
        "blocked": sum(r["status"] == "blocked" for r in records),
        "by_category": categories,
        "physical_requests": sum(r.get("physical_requests", 0) for r in records),
        "network_retries": sum(r.get("network_retries", 0) for r in records),
        "structure_retries": sum(r.get("structure_retries", 0) for r in records),
        "reported_tokens": sum(r.get("reported_tokens", 0) for r in records),
        # ---- 安全替换：**单独报，不混进上面的数**（验收④）----
        "hint_substituted_rounds": _sub_rounds,
        "hint_substituted_cases": _sub_cases,
        "hint_substituted_case_ratio": round(_sub_cases / len(records), 4) if records else 0.0,
        "hint_substitution_note": (
            "被替换的提示**不是模型原始输出**；这些案例的通过不得解释为"
            "模型原始输出合格。原模型提示与原因保留在 runs/*.json 的 "
            "teaching[].response.safety_substitutions（审计用）。"),
        "claim": "AI evaluation only; no student learning-effect conclusion"})
    print(f"Saved {len(records)} records to {out}")


if __name__ == "__main__":
    main()
