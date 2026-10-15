"""复盘材料的**来源**契约（§2.5）：合成文本材料可用，但**不可能伪装成真实视频复盘**。

五个不变式（契约里逐条列了）：
```
Ⅰ  两个允许的 scope 都能通过校验；**未知 scope 仍被拒**
Ⅱ  合成材料产出的报告，响应里 review_scope == "synthesized_text_sample"
Ⅲ  写入口**不提供覆盖 scope 的参数** ⇒ 合成路径不可能产出 video_run_snapshot
Ⅳ  ④⑤ 不一致时仍然 409（合成材料**没有**豁免材料一致性对账）
Ⅴ  未知 scope 的材料**既不生成也不缓存**
```
"""
import inspect
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import video
from app.services import attempt_store, problem_repository, review_builder
from app.services import review_material, video_review_versions

#: 真实视频路径的既有夹具，**直接复用**而不是复制一份 ——
#: 复制出来的那份迟早和 `test_video_lazy_review` 里的漂开。
from test_video_lazy_review import review_env  # noqa: F401


def _report_payload():
    return {"timeline": "report", "sticking_points": [], "key_concept": "momentum",
            "next_step": "check direction", "encouragement": "continue", "answer_check": None}


@pytest.fixture
def synth_env(monkeypatch, tmp_path):
    """一份**合成文本材料**的环境：作答真实存在、题干与图来自真实存储（这里是桩）。"""
    monkeypatch.setattr(video, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(video, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(video, "UPLOAD_DIR", tmp_path / "runs" / "uploads")
    state = SimpleNamespace(media_sources=[], timeline=[{"type": "submit"}],
                            graph_version="graph-1", graph_snapshot={"nodes": []})
    record = SimpleNamespace(student_id="stu_t4", problem_id="phy_002",
                             state=state, status="finished")
    monkeypatch.setattr(attempt_store, "get_attempt_store", lambda: SimpleNamespace(
        load_attempt=lambda key: record if key == "att-1" else None))
    monkeypatch.setattr(problem_repository, "get_problem_repository", lambda: SimpleNamespace(
        get_problem=lambda pid: SimpleNamespace(statement="原题干", knowledge_points=["动量守恒"])
        if pid == "phy_002" else None))
    monkeypatch.setattr(video_review_versions, "generation_context",
                        lambda: ({"model": "test"}, "test-generation", None))
    calls = []

    def generate(problem, rounds, **_kwargs):
        calls.append((problem.statement, rounds[0]["answer"]))
        return SimpleNamespace(model_dump=lambda: _report_payload())

    monkeypatch.setattr(review_builder, "review_trajectory", generate)

    run_id = "b" * 32
    root = video.RUNS_DIR / run_id
    root.mkdir(parents=True, exist_ok=True)
    path = review_material.write_synthesized_material(
        root, student_id="stu_t4", problem_id="phy_002", attempt_id="att-1",
        rounds=[{"round": 1, "ts": 0.0, "answer": "P、Q 开始时一起运动，可以看作一个整体。"}])
    req = video.VideoReviewRequest(student_id="stu_t4", problem_id="phy_002",
                                   attempt_id="att-1", run_id=run_id)
    return SimpleNamespace(req=req, root=root, path=path, calls=calls, run_id=run_id)


def _rewrite(path, mutate):
    """改完材料**重算指纹**再写回 —— 否则测的是"指纹不对"，而不是想测的那条规则。"""
    saved = json.loads(path.read_text(encoding="utf-8"))
    mutate(saved["input"])
    saved["fingerprint"] = review_material.fingerprint(saved["input"])
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- Ⅰ 允许集合
def test_both_allowed_scopes_pass_validation():
    base = {"student_id": "s", "problem_id": "p", "attempt_id": "a",
            "problem": {"statement": "题干", "knowledge_points": ["kp"]},
            "rounds": [{"answer": "作答"}], "source_revisions": {},
            "graph_version": "", "graph_snapshot": None, "timeline_length": 0}
    for scope in (review_material.MATERIAL_SCOPE_VIDEO,
                  review_material.MATERIAL_SCOPE_SYNTHESIZED):
        assert review_material.valid_input({**base, "scope": scope}) is True, scope


@pytest.mark.parametrize("scope", ["", "unknown", "video_run_snapshot ", "SYNTHESIZED_TEXT_SAMPLE",
                                   None, 123, "synthesized_text_sample_v2"])
def test_unknown_scope_is_rejected(scope):
    """**放行未知来源，就等于来源可以随便写。**"""
    data = {"student_id": "s", "problem_id": "p", "attempt_id": "a",
            "problem": {"statement": "题干", "knowledge_points": []},
            "rounds": [], "source_revisions": {},
            "graph_version": "", "graph_snapshot": None, "timeline_length": 0,
            "scope": scope}
    assert review_material.valid_input(data) is False


# ---------------------------------------------------------------- Ⅱ 来源随响应回带
def test_synthesized_review_reports_its_own_scope(synth_env):
    """合成材料产出的报告**必须自证来源** —— 否则它看起来和真实视频复盘一模一样。"""
    result = video.review_video(synth_env.req)
    assert result["review_scope"] == review_material.MATERIAL_SCOPE_SYNTHESIZED
    assert result["review_status"] == video.REVIEW_READY
    assert synth_env.calls == [("原题干", "P、Q 开始时一起运动，可以看作一个整体。")]


def test_the_video_path_still_reports_the_video_scope(review_env):
    """**别把生产路径的来源也改了** —— 它必须还是 video_run_snapshot。"""
    result = video.review_video(review_env[0])
    assert result["review_scope"] == review_material.MATERIAL_SCOPE_VIDEO


# ---------------------------------------------------------------- Ⅲ 写入口没有 scope 参数
def test_the_synthesized_writer_cannot_be_asked_for_another_scope():
    """⚠️ 这是**结构性**保证：没有这个参数，就不存在"传错"或"传坏"的可能。"""
    params = inspect.signature(review_material.write_synthesized_material).parameters
    assert "scope" not in params, f"合成写入口出现了 scope 参数：{list(params)}"


def test_the_synthesized_writer_always_stamps_the_synthesized_scope(synth_env):
    saved = json.loads(synth_env.path.read_text(encoding="utf-8"))
    assert saved["input"]["scope"] == review_material.MATERIAL_SCOPE_SYNTHESIZED


def test_the_video_writer_always_stamps_the_video_scope(monkeypatch, tmp_path, synth_env):
    """生产路径同理：它也只能写出 video_run_snapshot。"""
    params = inspect.signature(review_material.save_video_material).parameters
    assert "scope" not in params
    root = tmp_path / "runs" / ("c" * 32)
    root.mkdir(parents=True, exist_ok=True)
    path = review_material.save_video_material(root, "stu_t4", "phy_002", "att-1",
                                               [{"answer": "x"}])
    assert json.loads(path.read_text(encoding="utf-8"))["input"]["scope"] == \
        review_material.MATERIAL_SCOPE_VIDEO


# ---------------------------------------------------------------- Ⅳ 一致性对账不豁免
def test_synthesized_material_is_not_exempt_from_the_consistency_check(synth_env):
    """合成材料只改变"材料从哪来"，**不改变"材料要对得上实时作答"**。"""
    _rewrite(synth_env.path, lambda data: data.__setitem__("timeline_length", 99))
    with pytest.raises(HTTPException) as exc:
        video.review_video(synth_env.req)
    assert exc.value.status_code == 409
    assert synth_env.calls == [], "材料对不上账，却还是调了模型"


def test_synthesized_material_source_revisions_must_match(synth_env):
    _rewrite(synth_env.path, lambda data: data.__setitem__("source_revisions", {"ghost": 3}))
    with pytest.raises(HTTPException) as exc:
        video.review_video(synth_env.req)
    assert exc.value.status_code == 409
    assert synth_env.calls == []


# ---------------------------------------------------------------- Ⅴ 未知来源不生成不缓存
def test_unknown_scope_neither_generates_nor_caches(synth_env):
    _rewrite(synth_env.path, lambda data: data.__setitem__("scope", "video_run_snapshot_v2"))
    with pytest.raises(HTTPException) as exc:
        video.review_video(synth_env.req)
    assert exc.value.status_code == 409
    assert synth_env.calls == [], "未知来源的材料竟然调了模型"
    assert not list((synth_env.root / "review_reports").glob("*.json")), "未知来源被缓存了"
    assert not (synth_env.root / "review_report.json").exists()


# ---------------------------------------------------------------- 构造与落盘分开
def test_building_the_payload_never_touches_the_disk(synth_env, tmp_path):
    """⚠️ **先构造、再建目录**。

    第一版 CLI 是先 `mkdir` 再校验，于是"作答不存在"这种失败
    会在 `data/video_runs/` 里**留下一个空目录**（实测留过一个）。
    ⇒ 把"构造"与"落盘"分开，失败路径上什么都不留。
    """
    target = tmp_path / "runs" / ("d" * 32)
    review_material.build_synthesized_payload(
        student_id="stu_t4", problem_id="phy_002", attempt_id="att-1",
        rounds=[{"answer": "x"}])
    assert not target.exists(), "只构造就建了目录"

    with pytest.raises(ValueError):
        review_material.build_synthesized_payload(
            student_id="stu_t4", problem_id="phy_002", attempt_id="不存在",
            rounds=[{"answer": "x"}])
    assert not target.exists(), "失败的构造留下了目录"


def test_building_rejects_a_mismatched_attempt(synth_env):
    """作答、学生、题目**三者必须真实对应** —— 不对应就什么都不写。"""
    with pytest.raises(ValueError, match="不属于"):
        review_material.build_synthesized_payload(
            student_id="别人", problem_id="phy_002", attempt_id="att-1",
            rounds=[{"answer": "x"}])
    with pytest.raises(ValueError, match="题目不存在"):
        review_material.build_synthesized_payload(
            student_id="stu_t4", problem_id="不存在的题", attempt_id="att-1",
            rounds=[{"answer": "x"}])


# ---------------------------------------------------------------- 形状：其余检查一条没减
@pytest.mark.parametrize("field,value", [
    ("student_id", ""), ("problem_id", ""), ("attempt_id", ""),
    ("problem", {"statement": 1, "knowledge_points": []}),
    ("rounds", [{"answer": 5}]), ("source_revisions", {"k": "not-int"}),
    ("graph_version", None), ("graph_snapshot", []), ("timeline_length", -1),
])
def test_relaxing_the_scope_did_not_relax_anything_else(field, value):
    """⚠️ 这次改动的风险就是"顺手把别的检查也放松了" —— 逐字段钉住。"""
    data = {"student_id": "s", "problem_id": "p", "attempt_id": "a",
            "problem": {"statement": "题干", "knowledge_points": []},
            "rounds": [{"answer": "x"}], "source_revisions": {},
            "graph_version": "", "graph_snapshot": None, "timeline_length": 0,
            "scope": review_material.MATERIAL_SCOPE_SYNTHESIZED}
    data[field] = value
    assert review_material.valid_input(data) is False, field
