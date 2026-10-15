"""v3 证据结构的测试（**本次只做"壳"**）。

## 本次做了什么 / 没做什么
```
做    evidence（服务端生成）/ findings（从现有结构映射）/ validation（含引用检查）
      + 接进 review_trajectory + 让 `_missed`/`_stray` **第一次能被看见**
没做  模型侧仍输出 sticking_points；`findings` 是**服务端映射**来的，
      **不是模型按证据 ID 引用生成的**
      ⇒ 引用校验现在只能校"服务端自己填的引用"，
        **还不是真正的"模型不许编造引用"**
```
"""
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.services.review_builder import (  # noqa: E402
    CATEGORY_OBSERVED,
    CATEGORY_SUGGESTION,
    CATEGORY_UNCERTAIN,
    REVIEW_SCHEMA_VERSION,
    AnswerCheck,
    ReviewReport,
    StickingPoint,
    _is_a_real_suspicion,
    build_evidence_catalog,
    build_findings,
    build_material,
    build_review_user,
    build_validation,
    compute_trajectory_stats,
    count_downgraded_claims,
    format_evidence_catalog,
    reference_defects,
    review_schema_version,
    structural_defects,
    unsupported_claims,
)

ROUNDS = [{"round": 4, "ts": 62.9, "answer": "(1) I=1.98*10^4 N·s"},
          {"round": 5, "ts": 83.7, "answer": ""},
          {"round": 8, "ts": 144.2, "answer": "1.98×10^4，方向与Δv一致"}]


def _report(**kw):
    base = dict(timeline="他先算冲量。", key_concept="冲量",
                next_step="重做第(1)问", encouragement="第(2)问公式对了。",
                sticking_points=[StickingPoint(where="第(1)问", what="速率相减",
                                               why="要先定正方向")])
    base.update(kw)
    return ReviewReport(**base)


# ================================================================ evidence：服务端生成

def test_evidence_is_generated_by_the_server_verbatim():
    """**原文逐字取自轨迹** —— 这是契约「引用原文由服务端填充，模型不能自行改写」的落点。"""
    ev = build_evidence_catalog(ROUNDS)
    texts = [e["text"] for e in ev]
    assert "(1) I=1.98*10^4 N·s" in texts
    assert all(e["source"] == "student_answer" for e in ev)


def test_evidence_skips_empty_rounds_instead_of_faking_them():
    """空轮**不产出证据**，而不是造一条空证据。"""
    ids = [e["id"] for e in build_evidence_catalog(ROUNDS)]
    assert "E5.1" not in ids, "第 5 轮是空的，不该有证据"
    assert ids == ["E4.1", "E8.1"]


def test_evidence_ids_are_stable_and_use_real_round_numbers():
    """ID 稳定、且**用真实轮号**（与统计、与轨迹正文同一套）。"""
    assert [e["id"] for e in build_evidence_catalog(ROUNDS)] ==\
           [e["id"] for e in build_evidence_catalog(ROUNDS)]
    assert [e["round"] for e in build_evidence_catalog(ROUNDS)] == [4, 8]


def test_evidence_revision_is_null_when_unknown_not_fabricated():
    """没有修订号就**如实为 None**，不编一个。"""
    assert all(e["revision"] is None for e in build_evidence_catalog(ROUNDS))
    ev = build_evidence_catalog([{"round": 1, "answer": "x", "revision": 2}])
    assert ev[0]["revision"] == 2


def test_evidence_carries_the_real_source_id():
    """契约 §2.3：证据要有「**来源**及修订号」。

    ⚠️ `source` 那个字段是**种类**（恒为 `student_answer`），**不是来源** ——
    没有 `source_id` 就验不了「跨来源引用」和「更正前引用」。
    """
    ev = build_evidence_catalog([{"round": 1, "answer": "x", "source_id": "src_a",
                                  "source_revision": 3}])
    assert ev[0]["source_id"] == "src_a"
    assert ev[0]["revision"] == 3
    # 没有来源标识时**如实为 None**，不拿 `source` 顶替。
    assert all(e["source_id"] is None for e in build_evidence_catalog(ROUNDS))


# ================================================================ 预算与省略

def test_a_single_line_longer_than_the_budget_is_recorded_as_omitted():
    """⚠️ **截断了就要说自己截了。**

    实测过一个 **13000 字的单行**输入：旧实现只保留 12000 字，却报
    `omitted_chars=0`、`omitted_lines=0`、`truncated=false` ——
    **截断了却说自己没截**，比不截更危险：读的人会据此以为材料是完整的。
    （成因：硬切第一行后 `break`，而"丢掉的行"从 `lines[1:]` 数起，
     **被切掉的那截尾巴没进账**。）
    """
    rounds = [{"round": 1, "ts": 1.0, "answer": "字" * 13000}]
    material = build_material(rounds)
    item = material["items"][0]
    assert len(item["text"]) == material["budget"], "应当保留到预算上限"
    assert item["omitted_chars"] == 13000 - material["budget"], "省略字数没记账"
    assert item["omitted_lines"] >= 1
    assert item["mid_line_cut"] is True, "切在行中间要显式说出来"
    assert material["truncated"] is True, "**截断了却报 truncated=false**"


def test_a_mid_line_cut_is_declared_in_the_prompt():
    """提示词也要说清"这是半句" —— 否则模型会把没写完当成"学生写到这就停了"。"""
    user = build_review_user("题干", [], [{"round": 1, "ts": 1.0, "answer": "字" * 13000}])
    assert "被截断在行中间" in user
    assert "截断不代表学生只写到这" in user


def test_a_fitting_answer_is_not_marked_as_truncated():
    """没超预算就别乱标截断 —— 否则那个标记会失去意义。"""
    material = build_material([{"round": 1, "ts": 1.0, "answer": "短作答"}])
    assert material["truncated"] is False
    assert material["items"][0]["mid_line_cut"] is False
    assert material["items"][0]["omitted_chars"] == 0


# ================================================================ 引用重算：生成与缓存共用

def test_reference_defects_recomputes_instead_of_trusting_validation():
    """**本轮的核心改动**：判据从 `evidence` + `findings` **重算**。

    缓存里存着上一版代码写的 `validation`。信它，就等于信写它时的代码 ——
    而缺陷恰恰可能是那时的代码漏掉的。
    """
    evidence = [{"id": "E1.1", "source_id": "s1", "revision": 0}]
    findings = [{"category": CATEGORY_OBSERVED, "where": "x", "what": "y",
                 "evidence_ids": ["E9.9"]}]
    kinds = {d["kind"] for d in reference_defects(evidence, findings)}
    assert kinds == {"dangling_reference"}, "只依赖 evidence+findings 本身，不看 validation"


def test_unsupported_claim_is_a_defect_not_a_silent_pass():
    """一条「观察到的问题」没有任何可用引用 ⇒ **是缺陷，不是通过**。"""
    kinds = {d["kind"] for d in reference_defects(
        [{"id": "E1.1"}], [{"category": CATEGORY_OBSERVED, "what": "他错了",
                            "evidence_ids": []}])}
    assert "unsupported_claim" in kinds


def test_cross_source_and_stale_revisions_are_detected():
    """契约 T2 验收：**跨来源引用**与**更正前引用**不能成为有效报告。"""
    evidence = [{"id": "E1.1", "source_id": "s1", "revision": 0},
                {"id": "E2.1", "source_id": "s2", "revision": 1}]
    findings = [{"category": CATEGORY_SUGGESTION, "where": "x", "what": "y",
                 "evidence_ids": []}]
    # 材料当前记录：s1 已更正到修订 2；s2 根本不在材料里
    kinds = {d["kind"] for d in reference_defects(
        evidence, findings, {"s1": 2})}
    assert "stale_reference" in kinds, "引用了更正前的修订"
    assert "unknown_source" in kinds, "证据里的来源不在材料里"

    # ⚠️ **不给 `source_revisions` 就跳过这两条** —— 判不了就不判，不猜。
    kinds_wo = {d["kind"] for d in reference_defects(evidence, findings)}
    assert "stale_reference" not in kinds_wo and "unknown_source" not in kinds_wo


def test_downgraded_claims_do_not_make_every_report_invalid():
    """降级**不进 `issues`**：它是如实标注，不是缺陷。

    放进 issues 会让每一份（默认 v2 提示词下的）报告都 `valid=false` ——
    而那是**常态不是异常**，只会让 `valid` 失去意义。
    """
    ev = build_evidence_catalog(ROUNDS)
    findings = build_findings(_report(), compute_trajectory_stats(ROUNDS), ev)
    v = build_validation(ev, findings, stats=compute_trajectory_stats(ROUNDS))
    assert v["downgraded_claims"] == 1
    assert v["valid"] is True
    assert not any(i["kind"] == "unsupported_claim" for i in v["issues"])


# ================================================================ 版本绑定

def test_schema_version_is_bound_to_the_actual_prompt_version():
    """契约 §2.3 + T2：报告版本要绑**实际跑过的提示词版本**。

    默认提示词是 **v2**，而 v2**不要求模型引用证据 ID** ⇒ 那些报告的引用全空。
    旧版无论跑哪版都标一个 `"v3"`，读的人会以为"这份报告有真正的引用校验"。
    """
    assert review_schema_version("v2") == "v3+prompt-v2"
    assert review_schema_version("v3") == "v3+prompt-v3"
    assert review_schema_version("").endswith("prompt-unknown")
    # 结构版本仍是可判定的（前端与出口摘除都看这个）
    assert review_schema_version("v2").startswith(REVIEW_SCHEMA_VERSION)


def test_validation_carries_the_prompt_version_for_traceability():
    ev = build_evidence_catalog(ROUNDS)
    v = build_validation(ev, [], prompt_version="v2")
    assert v["prompt_version"] == "v2"
    assert v["validator_version"] == REVIEW_SCHEMA_VERSION


# ================================================================ findings：映射

def test_findings_map_to_the_three_fixed_categories():
    """三类**固定**，不许自造。"""
    fi = build_findings(_report(answer_check=AnswerCheck(
        student_answer="a", derived_from_work="b", consistent="不一致",
        note="差一个符号", ocr_caution="可能是识别问题")),
        compute_trajectory_stats(ROUNDS),
        # ⚠️ 引用**必须有**才判得了：没有引用的判断会被降级（见下一条）。
        # 这里给一条真实证据，让三类都能出现。
        [{"id": "E4.1", "text": "a"}])
    cats = {f["category"] for f in fi}
    assert cats <= {CATEGORY_OBSERVED, CATEGORY_UNCERTAIN, CATEGORY_SUGGESTION}
    assert CATEGORY_OBSERVED in cats and CATEGORY_UNCERTAIN in cats\
        and CATEGORY_SUGGESTION in cats


def test_findings_allow_empty_list():
    """契约⑥：**无明确问题时允许空列表**，不强制造卡点。"""
    fi = build_findings(ReviewReport(timeline="x", key_concept="",
                                     next_step="", encouragement="",
                                     sticking_points=[]),
                        compute_trajectory_stats(ROUNDS))
    assert fi == []


def test_a_claim_without_a_reference_is_not_called_an_observed_error():
    """**本轮的硬规则**：契约把 `observed_error` 定义成「**有具体原文支持**的错误判断」。

    ⇒ 一条没有任何引用的判断，按定义就**不是** observed_error。
    旧版允许它挂着"错误"的类别、引用却是空的 —— 于是
    `valid=true` **不能保证每条指控都有依据**。
    """
    fi = build_findings(_report(), compute_trajectory_stats(ROUNDS))   # sticking_point 无引用
    assert CATEGORY_OBSERVED not in {f["category"] for f in fi}, "无引用的判断不该叫 observed_error"
    claim = next(f for f in fi if f["category"] == CATEGORY_UNCERTAIN)
    assert "找不到对应的原文引用" in claim["what"]
    assert claim["evidence_ids"] == []
    assert count_downgraded_claims(fi) == 1


def test_a_claim_citing_a_nonexistent_id_stays_observed_error():
    """⚠️ **不能一并降级**：引用了一个**不存在**的 ID 是另一种缺陷。

    那是"结论指不回去"（`dangling_reference`），要**拦住并拒绝缓存**；
    降级会把它洗成"没引用"，**反而把那个缺陷藏起来**。
    """
    rep = _report(sticking_points=[StickingPoint(where="第(1)问", what="速率相减",
                                                 why="先定正方向", evidence_ids=["E9.9"])])
    fi = build_findings(rep, compute_trajectory_stats(ROUNDS), [{"id": "E4.1", "text": "x"}])
    assert fi[0]["category"] == CATEGORY_OBSERVED
    assert fi[0]["evidence_ids"] == ["E9.9"]
    kinds = {d["kind"] for d in reference_defects([{"id": "E4.1"}], fi)}
    assert "dangling_reference" in kinds


def test_the_conclusion_check_gets_cited_from_the_servers_own_quote():
    """「结论核对」那条**本来就有据可依** —— 原文是服务端逐字填的。

    旧版只是**没把引用挂上去**，于是它也变成了"无依据的指控"。
    """
    ev = [{"id": "E8.1", "source_id": "s1", "revision": 0, "round": 8, "ts": 1.0,
           "text": "F = 1.65e5 N"}]
    fi = build_findings(_report(sticking_points=[], answer_check=AnswerCheck(
        student_answer="F = 1.65e5 N", derived_from_work="F = I/Δt",
        consistent="不一致", note="量级不对")), compute_trajectory_stats(ROUNDS), ev)
    claim = next(f for f in fi if f["category"] == CATEGORY_OBSERVED)
    assert claim["evidence_ids"] == ["E8.1"]
    assert count_downgraded_claims(fi) == 0


def test_mapped_findings_declare_that_they_have_no_real_references():
    """⚠️ **如实**：映射来的 findings **填不出真实引用**时，`evidence_ids` 为空。

    不编一个 ID 进去充数 —— 那会让 `validation` 看起来"验过了"。
    """
    fi = build_findings(_report(), compute_trajectory_stats(ROUNDS))
    assert all(f["evidence_ids"] == [] for f in fi)


# ================================================================ validation

def test_validation_carries_the_two_things_that_were_logger_only():
    """⭐ **本次最实在的一处**：`_missed` / `_stray` 上一版只进 logger，
    前端与人审都看不到。现在它们进 `validation.issues`。"""
    v = build_validation(build_evidence_catalog(ROUNDS), [], 
                         missed=["9.99×10^9"], stray=["四轮"],
                         stats=compute_trajectory_stats(ROUNDS))
    kinds = {i["kind"] for i in v["issues"]}
    assert "unresolved_placeholder" in kinds
    assert "model_written_count" in kinds
    assert v["valid"] is False


def test_validation_scope_says_it_is_not_physics():
    """契约⑧：**不得显示成「物理语义已验证」**。

    所以把这句话**写进数据本身** —— 免得前端拿到 `valid: true` 就当成物理没问题。
    """
    v = build_validation([], [])
    assert v["valid"] is True
    assert "不代表物理语义已验证" in v["scope"]


def test_validation_rejects_dangling_references():
    v = build_validation([{"id": "E1.1"}], [{"evidence_ids": ["E9.9"]}])
    assert v["valid"] is False
    assert any(i["kind"] == "dangling_reference" for i in v["issues"])


# ================================================================ 兼容与接线

def test_schema_has_the_four_v3_fields():
    """四个字段**必须挂在 `ReviewReport` 上** —— 否则 v3 报告带不出来。"""
    for k in ("review_schema_version", "evidence", "findings", "validation"):
        assert k in ReviewReport.model_fields, f"{k} 不在 schema 上"


def test_old_reports_are_stripped_at_the_exit():
    """⭐ **这条才是关键，也是我第一次栽的地方。**

    字段挂在 schema 上 ⇒ `model_dump()` 会**一视同仁**地加给每一条响应，
    **包括缓存里那些 v2 时代生成的报告** ——
    那会让一份旧报告看起来像 v3，只是 `evidence` 为空。
    而契约写着「**历史 v1/v2 不伪装成带证据报告**」。

    ⇒ 出口（`api/video.py::_validated_review_payload`）按版本号判定：
      有 ⇒ 保留；没有 ⇒ **四个字段全部摘掉**（不是给空值）。

    **⚠️ 这正是我第一次提交时打挂 21 条缓存测试的原因** ——
    当时我只加了字段，没做出口摘除。
    """
    from app.api.video import _validated_review_payload

    old_style = {"timeline": "他先算冲量。", "key_concept": "冲量",
                 "next_step": "重做", "encouragement": "公式对了",
                 "sticking_points": []}
    out = _validated_review_payload(old_style)
    for k in ("review_schema_version", "evidence", "findings", "validation"):
        assert k not in out, f"旧报告不该带 {k}"


def test_v3_reports_keep_the_four_fields():
    """真跑 v3 的报告**要带着**它们 —— 摘除不能摘过头。"""
    from app.api.video import _validated_review_payload

    v3_style = {"timeline": "他先算冲量。", "key_concept": "冲量",
                "next_step": "重做", "encouragement": "公式对了",
                "sticking_points": [],
                "review_schema_version": "v3", "evidence": [{"id": "E1.1"}],
                "findings": [], "validation": {"valid": True}}
    out = _validated_review_payload(v3_style)
    assert out["review_schema_version"] == "v3"
    assert out["evidence"] == [{"id": "E1.1"}]
    assert out["validation"] == {"valid": True}


def test_the_stripping_list_covers_exactly_the_four():
    """摘除清单**恰好四个** —— 多摘会丢真字段，少摘会漏伪装。"""
    from app.api.video import _V3_ONLY_FIELDS

    assert set(_V3_ONLY_FIELDS) == {"review_schema_version", "evidence",
                                    "findings", "validation"}


def test_three_builders_are_available_for_the_serialization_step():
    """三个构造函数**保留**（有测试、可直接用），等序列化层就绪时接上。"""
    from app.services import review_builder as rb

    for kw in ("build_evidence_catalog", "build_findings", "build_validation"):
        assert callable(getattr(rb, kw))


def test_schema_version_constant():
    assert REVIEW_SCHEMA_VERSION == "v3"

# ================================================================ v3 模型侧：引用证据 ID

def test_sticking_point_can_carry_evidence_ids():
    """**加这一个字段是本次的全部** —— 不重写报告结构，只让它能「指回去」。"""
    sp = StickingPoint(where="a", what="b", why="c")
    assert sp.evidence_ids == [], "**默认空** —— 旧报告与不填的模型都不炸"
    sp2 = StickingPoint(where="a", what="b", why="c", evidence_ids=["E4.1"])
    assert sp2.evidence_ids == ["E4.1"]


def test_reference_check_is_no_longer_vacuous():
    """⭐ **这条是本次的意义所在。**

    改之前：`findings` 是服务端映射来的 ⇒ `evidence_ids` **恒为空**
    ⇒ `build_validation` 的引用校验**永远通过，等于没校验**。

    改之后：模型填的 ID 会一路带到 `findings` ⇒ **编造的 ID 真的会被抓出来。**
    """
    ev = build_evidence_catalog(ROUNDS)
    st = compute_trajectory_stats(ROUNDS)

    real = ReviewReport(timeline="t", key_concept="", next_step="", encouragement="",
                        sticking_points=[StickingPoint(where="a", what="b", why="c",
                                                       evidence_ids=["E4.1"])])
    fake = ReviewReport(timeline="t", key_concept="", next_step="", encouragement="",
                        sticking_points=[StickingPoint(where="a", what="b", why="c",
                                                       evidence_ids=["E9.9"])])

    assert build_validation(ev, build_findings(real, st), stats=st)["valid"] is True
    v_bad = build_validation(ev, build_findings(fake, st), stats=st)
    assert v_bad["valid"] is False
    assert any(i["kind"] == "dangling_reference" for i in v_bad["issues"])


def test_empty_evidence_ids_stay_empty_and_valid():
    """**留空是诚实的** —— 不许因为「没引用」就判无效。

    提示词里明说了：「没有证据支持的判断，`evidence_ids` 留空，
    **空引用是诚实的；编引用是错的。留空不会让你的报告变差。**」
    ⇒ 校验必须与此一致，否则就是在**逼模型编引用**。
    """
    ev = build_evidence_catalog(ROUNDS)
    st = compute_trajectory_stats(ROUNDS)
    rep = ReviewReport(timeline="t", key_concept="", next_step="", encouragement="",
                       sticking_points=[StickingPoint(where="a", what="b", why="c")])
    v = build_validation(ev, build_findings(rep, st), stats=st)
    assert v["valid"] is True, "留空不该被判无效"


def test_catalog_is_fed_into_the_prompt_with_ids():
    """**防「写了没接」**：目录必须在交给模型的 user 消息里，且带 ID。"""
    u = build_review_user("题干", [], ROUNDS)
    assert "【证据目录】" in u
    assert "[E4.1]" in u and "[E8.1]" in u
    assert "只能用这里的 ID" in u


def test_catalog_rendering_refuses_to_let_the_model_rewrite():
    """契约②：**原文由服务端填，模型不能自行改写** —— 所以提示词要明说。"""
    blk = format_evidence_catalog(build_evidence_catalog(ROUNDS))
    assert "不要编 ID" in blk
    assert "不要改写上面的原文" in blk
    assert "(1) I=1.98*10^4 N·s" in blk, "原文要逐字给出"


def test_catalog_handles_no_evidence():
    """没有证据时如实说没有，**不给模型一个空目录去编**。"""
    assert "没有可用证据" in format_evidence_catalog([])


def test_v3_prompt_exists_and_keeps_v2_content():
    """v3 是**在 v2 之上追加**的 —— v2 的结论核对等能力不能丢。"""
    pr = Path(__file__).resolve().parents[1] / "prompts/review"
    v2 = (pr / "v2.md").read_text(encoding="utf-8")
    v3 = (pr / "v3.md").read_text(encoding="utf-8")
    assert "v3 新增" in v3 and "引用证据 ID" in v3
    head = v2.splitlines()[0].strip()
    assert head and head in v3, "v3 丢了 v2 的开头"
    assert len(v3) > len(v2)


def test_v3_is_implemented_but_not_the_default():
    """⚠️ **v3 已实现并测过，但默认仍是 v2** —— 这是有意的，不是忘了切。

    契约（`docs/契约-媒体与复盘接口-冻结.md`）写着：
    > **v3 通过开发集后再切换默认。**

    ⚠️ **2026-10-09 更新理由**（这一节此前写「而那个开发集还没建」——**已过期**）：
    ```
    · 开发集**已建、已跑四轮**；第四轮经第三方裁决后 **passed=true**（可用率 70/72、
      四项严重指标全 0、明确错误识别 10/12）。
    · **但留出集仍是 0/36、未解锁** ⇒ 按契约，**切默认要等留出集也通过**。
    ⇒ 结论没变（仍然不该切 v3），**变的是依据**。
      「还没建」→「建了、开发集过了、留出集还没过」是三个不同阶段，
      写错依据比写错结论更难被发现。
    ```

    **切回 v3 的条件**：开发集通过（已达成）**且 36 次留出集也通过**；
    临时启用可用环境变量 `PROMPT_REVIEW_VERSION=v3`。

    **这条测试是刻意钉住"当前默认"的** ——
    将来留出集过了、有人切到 v3，**这条会失败，提醒他同时更新这里的理由**。
    """
    import os
    from pathlib import Path

    from app.agent.prompt_registry import DEFAULT_VERSIONS

    assert DEFAULT_VERSIONS["review"] == "v2", (
        "默认被改成 v3 了 —— 开发集已通过（裁决后），但**留出集仍是 0/36、未解锁**；"
        "按契约，切默认要等留出集也通过。若确要切，请同时更新 prompt_registry 与本测试的理由。")
    # v3 提示词**确实存在**（不是"没做"，是"没启用"）
    assert (Path(__file__).resolve().parents[1] / "prompts/review/v3.md").is_file(), "v3 提示词不见了"
    # ⚠️ **理由本身也要被盯住**：代码里那段「为什么不切默认」的注释，
    #    必须写明**当前真正生效的条件**（要等留出集），而不是停在旧阶段。
    #    实测踩到过：注释写着「那个开发集还没建」，而开发集早已跑过四轮。
    #    ⚠️ 不要断言「不许出现『还没建』」—— 我更正时**引用了那句旧话**，
    #      那样的断言会把"记录更正"本身判成违规（我第一版就这么错了）。
    registry_src = ((Path(__file__).resolve().parents[1] / "backend/app/agent/prompt_registry.py")
                    .read_text(encoding="utf-8"))
    assert "留出集" in registry_src, "prompt_registry 没说清「要等留出集」这个当前条件"
    assert "开发集" in registry_src, "prompt_registry 没交代开发集的状态"
    # 可临时启用
    assert os.environ.get("PROMPT_REVIEW_VERSION") in (None, "v2", "v3")

# ================================================================ 复核指出的两个阻断项

def test_v3_structures_are_built_even_without_answer_check():
    """⭐⭐ **复核指出的回归，而且我实测确认了它。**

    我原先用行号把「服务端填充」和「v3 三个结构」两块**插进了
    `if out.value.answer_check is not None:` 的体内** —— 于是：

    ```
    answer_check 为 None  → v3=''   evidence=0   ❌ **静默退回 v2 外观**
    answer_check 非空     → v3='v3' evidence=1   ✅
    ```

    **为什么真机跑没发现**：那次 `answer_check` 恰好非空。
    **但它是可选字段** —— 模型不给的时候，整份报告就没有证据结构，
    而出口摘除又把空字段摘掉，**看起来就像一份正常的 v2 报告**。

    ⇒ 这条测试钉住「**v3 结构必须独立于 `answer_check`**」。
    """
    from types import SimpleNamespace

    from app.services import review_builder as rb

    class _Out:
        def __init__(self, v):
            self.value, self.degraded, self.reasons = v, False, []

    class _LLM:
        def __init__(self, rep):
            self.rep = rep

        def chat_structured_resilient(self, *a, **k):
            return _Out(self.rep)

    def _mk(**kw):
        base = dict(timeline="他先算冲量。", key_concept="", next_step="重做",
                    encouragement="公式对了", sticking_points=[])
        base.update(kw)
        return rb.ReviewReport(**base)

    rounds = [{"round": 1, "ts": 1.0, "answer": "I=1.98×10^4"}]
    prob = SimpleNamespace(statement="t", knowledge_points=[])

    for label, rep in (
        ("answer_check 为 None", _mk()),
        ("answer_check 非空", _mk(answer_check=rb.AnswerCheck(
            student_answer="a", derived_from_work="b", consistent="一致"))),
    ):
        got = rb.review_trajectory(prob, rounds, llm=_LLM(rep))
        assert got is not None, label
        # ⚠️ 版本串现在**绑了实际提示词版本**（默认 v2）——
        # "是不是 v3 结构"要看前缀，不能拿全等去比。
        assert got.review_schema_version.startswith("v3"), f"{label}：没生成 v3 结构"
        # ⚠️ 2026-10-10 起是 **2** 条：学生作答 `E1.1` + **题面 `S1`**
        #    （契约 §2.8③ 来源分池 —— 题面必须能独立被引用，不许拿学生的话冒充题面）。
        assert len(got.evidence) == 2, f"{label}：证据目录条数不对（应为 学生作答 + 题面）"
        assert {e["source"] for e in got.evidence} == {"student_answer", "statement"}
        assert got.validation, f"{label}：validation 没填"


def test_the_two_v3_blocks_are_not_inside_a_conditional():
    """**静态钉住那个缩进错误** —— 光靠上面那条测不出"为什么错"。

    两块必须是 `try` 层（8 空格），**不能在 `if answer_check is not None:` 体内**。
    """
    import inspect

    from app.services import review_builder as rb

    src = inspect.getsource(rb.review_trajectory)
    for marker in ("_stats = compute_trajectory_stats(rounds)",
                   "out.value.review_schema_version = REVIEW_SCHEMA_VERSION"):
        line = next(l for l in src.splitlines() if l.strip().startswith(marker.split(" = ")[0]))
        indent = len(line) - len(line.lstrip())
        assert indent == 8, f"{marker} 缩进 {indent}，应在 try 层（8）"


def test_structural_defects_block_the_success_cache(monkeypatch):
    """**契约⑦ 2026-10-09 澄清**：`valid=false`（**含**文体/统计类）⇒ 不缓存成功报告。

    ## 这条测试的前身是本仓库里一个**反向**的钉子
    旧版钉的是：`dangling_reference` 挡，而 `model_written_count` /
    `unresolved_placeholder` **不挡**，理由写的是「那是文体问题，不是结构不合法；
    挡掉会因模型多写一句轮次而整个复盘失败，**惩罚过重**」。

    ⚠️ **那是对契约⑦的窄读**：契约看的是 `validation.valid`，不是"错得重不重"。
       实证代价（T4 开发集）：**44/72 份 `valid=false` 仍被算作成功**，
       可用率被报成 100%，而结构合格的只有 28 份。

    ⚠️ 更深的毛病是**形态**：旧判据是"已知种类白名单"，
       ⇒ **将来任何一个新种类都会静默不阻断**。现在改成 fail-closed。
    """
    from app.api import video as vmod
    from app.services import review_builder

    def payload(**over):
        base = {"timeline": "他把每一步都写清楚了", "sticking_points": [],
                "key_concept": "动量守恒", "next_step": "继续", "encouragement": "稳",
                "answer_check": None, "review_schema_version": "v3+prompt-v3",
                "evidence": [], "findings": [], "validation": {}}
        base.update(over)
        return base

    # ① 模型自写次数 ⇒ 阻断（旧版放行）
    bad = vmod._structural_defects(payload(timeline="他一共写了 3 次"), {})
    assert [d["kind"] for d in bad] == ["model_written_count"]
    # ② 未命中的占位符 ⇒ 阻断（旧版放行）
    bad = vmod._structural_defects(payload(timeline="见 [[value:第7轮]]"), {})
    assert [d["kind"] for d in bad] == ["unresolved_placeholder"]
    # ③ 干净报告 ⇒ 不阻断
    assert vmod._structural_defects(payload(), {}) == []


def test_an_unknown_defect_kind_still_blocks(monkeypatch):
    """**fail-closed**：判据**不是**白名单 —— 将来新增的种类默认就该被挡住。

    这条测的是**形态**，不是某一个种类：白名单形态正是那处缺陷能长期存在的原因。
    """
    from app.api import video as vmod
    from app.services import review_builder

    assert vmod._NON_BLOCKING_ISSUE_KINDS == frozenset(), "放行集合必须为空（举证责任在放行方）"
    monkeypatch.setattr(review_builder, "structural_defects",
                        lambda *a, **k: [{"kind": "brand_new_kind", "detail": "将来新增"}])
    assert vmod._structural_defects({}) == [{"kind": "brand_new_kind", "detail": "将来新增"}],\
        "未知种类被静默放行了 —— 又回到了白名单形态"


@pytest.mark.parametrize("kinds,expected", [
    (["dangling_reference"], "引用不合法"),
    (["unknown_source"], "引用不合法"),
    (["model_written_count"], "统计字段不合法"),
    (["unresolved_placeholder"], "统计字段不合法"),
    (["dangling_reference", "model_written_count"], "引用不合法、统计字段不合法"),
    (["brand_new_kind"], "结构不合法（未分类）"),
])
def test_failure_message_says_which_category(kinds, expected):
    """**失败要分类**（用户第 4 条）：说清是哪一类不合格，而不是一句笼统的"校验未通过"。"""
    from app.services.review_builder import classify_defects

    assert classify_defects([{"kind": k} for k in kinds]) == expected


def test_the_category_never_claims_physics_was_checked():
    """⚠️ **规则判不了物理** —— 分类里不许出现"物理"，否则就是冒充。"""
    from app.services.review_builder import classify_defects

    for kinds in (["dangling_reference"], ["model_written_count"], ["brand_new_kind"]):
        assert "物理" not in classify_defects([{"kind": k} for k in kinds])


# ---------------------------------------------------------------- 「没有可疑处」不是一条待核查项
#
# ⚠️ 由来：T4 评测里**每一份**报告都带一条 `uncertain / 识别` 的待核查项，
# 内容却是「未发现同一数字前后读法矛盾…未见可疑识别点」——
# 旧版只要 `ocr_caution` 非空就产出一条，**凭空给学生多一条要看的**。
# ⇒ 模型侧（提示词：没有可疑处就留空）+ 服务端（此处兜底）两边都收。

def test_a_caution_saying_nothing_is_wrong_is_not_a_finding():
    for quiet in ("未发现同一数字前后读法矛盾；2、18、6、12 量级均合理",
                  "未见可疑识别点", "没有发现读法冲突", "识别清晰",
                  "⚠️ 未发现可疑处", "：未见异常",
                  "", "   ", None):
        assert _is_a_real_suspicion(quiet) is False, f"{quiet!r} 被当成了可疑"


def test_a_real_caution_is_still_kept():
    """反过来：**真报了可疑**必须留下 —— 那是学生要用来自查的东西。"""
    for real in ("第 4 轮的 1.98 与第 8 轮的 198 读数不一致，可能漏了小数点",
                 "这里出现 ?，无法确认是 49 还是 4",
                 "同一位数字前后两次读法不同（1.49 / 1.94）"):
        assert _is_a_real_suspicion(real) is True, f"{real!r} 被误当成'没有可疑'"


def test_the_prompt_tells_the_model_to_leave_it_empty():
    """模型侧也要说 —— 只在服务端挡是治标（它会一直写，我们一直丢）。"""
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "没有可疑处就留空" in prompt, "提示词没让它留空"
    assert "写上去会被当成一条待核查项" in prompt, "没说清写了会怎样"


def test_the_prompt_states_the_new_blocking_consequence():
    """⚠️ **后果变了，提示词必须跟上**（契约⑦ 改成 fail-closed 之后）。

    旧提示词写的是"查出来会记进 validation，**会被看见**" ——
    那是在告诉模型"这只是一条记录"。现在它会**让整份复盘不合格、要重跑**。
    **按旧的轻微后果写提示词，等于把模型置于一条它不知道严重的规则之下。**

    ⚠️ 这里**曾经断言提示词里要有「44/72」**，后来删掉了 ——
    因为那个数字**已被推翻**：独立复核重放归档证明，44 份里 **42 份的唯一命中
    是普通副词「一次」**（判据误报），不是模型的违规。
    **把已被推翻的统计写进提示词，等于向模型传播错误信息。**
    """
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "整份复盘不合格" in prompt and "要重跑一次" in prompt
    assert "44/72" not in prompt, "又把那个已被推翻的数字写回去了"


def test_the_prompt_states_the_counting_boundary():
    """**P1-A（独立复核）**：提示词原先自相矛盾 ——
    标 ✅ 的示例「在第 7 轮之后被改写」本身就写在"不许自己写轮次"的禁区内。

    判据其实**允许单轮定位**（那是可追溯性要的东西），但提示词从没说过这条边界，
    而它又宣称标准"覆盖全篇"。⇒ 现在必须把边界写清。
    """
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "边界要分清" in prompt
    assert "单个轮次的**定位**" in prompt, "没说清单轮定位是允许的"
    assert "**多轮列表**" in prompt and "**任何计数**" in prompt, "没列清禁止的形态"



def test_the_prompt_covers_physics_self_consistency():
    """第 1 条的**可写部分**：评测暴露的两处，提示词里原本**没有**对应要求。

    ⚠️ 说清边界：**另两处（负号方向、向量同向）提示词早就写了，模型照样违反** ——
    所以"再写一遍规则"不是修复，补的是**原本缺失**的这两条。
    """
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "自己算通" in prompt, "缺「等式要算得通」这条"
    assert "先核条件" in prompt, "缺「说守恒之前先核条件」这条"
    assert "机械能有损失" in prompt and "不能" in prompt, "缺「能量损失推不出动量不守恒」"
    assert "P总=24 W" in prompt, "没把真实的错误案例写进去"


def test_the_prompt_gives_an_operational_rewrite_pattern():
    """⚠️ **不是再加禁令，而是给可操作的写法**（2026-10-09 新增）。

    两批开发集证明：**只列禁令不管用**（v3 早就写了「不许写没有犹豫/已经掌握」，
    模型照样违反 10 处与 6 处）。所以补的是**固定写法**：

        先逐字引（引号里必须是原文里确实有的字），**再只评引出来的那一句**

    并且点名本批实测出来的两种越界形态 —— 它们**在两种判法下都算越界**：
      · 「这两条公式你用得是对的」（原文只有数字，没有公式）
      · 「解出 v=−2 m/s 后你没有丢掉负号」（原文里没有这个式子）
    """
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "先逐字引，再只评引出来的那一句" in prompt
    assert "指不回原文" in prompt or "指不回去" in prompt, "缺自检那句"
    assert "这两条公式你用得是对的" in prompt, "没把实测的越界形态写进去"
    assert "没有丢掉负号" in prompt, "没把实测的越界形态写进去"
    # ⚠️ 「没有发生某事」那一族**已经定案**（契约 ⑦-3 第③条）。
    #    这里原先断言的是"边界正在裁决中"那句**保留态度** ——
    #    裁决落地后那句被我撤掉，改成硬规则。**测试随之改成断言"已定案"。**
    assert "已经定案" in prompt and "没有发生过什么" in prompt
    assert "边界正在裁决中" not in prompt, "裁决已落地，不该再留保留态度"
    # 而且必须给出**替换写法**（只禁止不给写法 = 模型只能删掉整段）
    assert "主语是" in prompt, "没给「对材料的描述」这种替代写法"


def test_the_prompt_names_the_hardest_actual_form():
    """⚠️ **目前被拒最多的一族，必须在提示词里点名到"具体措辞"**（2026-10-09 新增）。

    第三轮裁决后为真的 6 条里，5 条是同一形态：**「你用的 / 你用对了 / 你注意到了」**
    —— 它们长得像表扬，所以最难自己发现。裁决者给的可操作改法是
    **把归因的对象从"学生"改成"系统"**（「我按…核对」「从数值上判断…」）。
    """
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    for form in ("你用的", "你用对了", "你注意到了", "你是按", "你抓住了"):
        assert form in prompt, f"没点名最容易犯的形态：{form}"
    assert "把归因的对象从" in prompt, "没给裁决者那条改法（归因从学生改到系统）"
    assert "我按" in prompt and "从数值上判断" in prompt, "没给合规的改写模板"
    # 一句话判据本身也要在
    assert "你写的 X\" 可以" in prompt or "你写的 X" in prompt, "缺那句一句话判据"


# ================================================================ 归属给题面的引文（可上闸门的那一路）

_STMT = "碰撞时间很短且系统所受外力冲量可忽略，碰后两车粘在一起。"


def test_a_misquoted_statement_is_detected():
    """★ **这一路能上闸门**：题面是**服务端给的**（ground truth），归属词无歧义。

    实测（v7/v8/v9 三批）：归属给题面的引文一共 3 个，**3 个都不逐字** ⇒ 精确率 3/3、零误报。
    真实形态只有一种：**把「且」改成「、」**（Q001 / Q035 / Q072 三条都是这个）。
    """
    from app.services.review_builder import unfounded_statement_quotes

    got = unfounded_statement_quotes("题目说「碰撞时间很短、外力冲量可忽略」，正是为了让…", _STMT)
    assert got and "碰撞时间很短" in got[0]
    # 逐字引用 ⇒ 不许报
    assert unfounded_statement_quotes("题干给的「碰撞时间很短且系统所受外力冲量可忽略」", _STMT) == []


@pytest.mark.parametrize("text,why", [
    ("你写的「F合=6 N」是对的。", "不是题面引文（归属给学生的那一路不做闸门）"),
    ("题目说「随便一句题面里没有的话」", "⚠️ 这句会被判违规 —— 见下面的边界说明"),
])
def test_statement_quote_scope(text, why):
    """边界：**只看"归属给题面"的引文**，别的一概不管。

    ⚠️ 第二条**确实会报** —— 那是**故意的**：题面是服务端给的，
    凡声称"题目说 X"而 X 不在题面里，都是引用不实（多字/少字/换词都算）。
    这一路的代价（误杀）实测为 0，所以它敢当闸门；**学生材料那一路不敢**。
    """
    from app.services.review_builder import unfounded_statement_quotes

    got = unfounded_statement_quotes(text, _STMT)
    if "随便一句" in text:
        assert got, "声称题目说了一句题面里没有的话，必须报"
    else:
        assert got == [], f"不该报（{why}）"


def test_no_statement_means_no_check():
    """题面缺失时**不猜**：没有 ground truth 就不判（宁可不报，也不误杀）。"""
    from app.services.review_builder import unfounded_statement_quotes

    assert unfounded_statement_quotes("题目说「随便什么」", "") == []
    assert unfounded_statement_quotes("", _STMT) == []


def test_a_handwritten_quote_that_matches_the_source_is_allowed():
    """🔴 **第十轮实测后的修正**：归属框架 + 引号，**但那段字在源里找得到** ⇒ 放行。

    ```
    原实现只看"有没有归属框架 + 引号" ⇒ fail-closed 后**误杀术语**：
      v12 有 4/7 条失败出自这个判据，被拒的是「取右为正」「粘在一起」「损失为0」
      —— 术语/题面词组，**引文准确、无害**。
    我 v8 就量过这个判据精度约一半，**却仍设成阻断** —— 那是方法错误：
    量出精度不够之后必须**据此改设计**。
    ⇒ 判据落到**可判定的那一步**：不是猜"它是不是引文"，而是核对"**这段字在不在源里**"。
    ```
    实测（5 批 342 份）：修正后只标 6 处，含 v10 两处**已裁决真阳性**；v11/v12 为 **0**。
    """
    from app.services.review_builder import unattributed_quotes
    pool = [{"id": "S1", "source": "statement",
             "text": "碰撞时间很短且系统所受外力冲量可忽略，碰后两车粘在一起"},
            {"id": "E1.1", "source": "student_answer",
             "text": "取右为正，合力F-f=10-4=6 N，v=2 m/s 向右"}]
    # 引文准确（术语/题面词组/学生原话）⇒ 放行
    for ok in ["你写的「取右为正」这一句对", "题目说「粘在一起」",
               "你写的「2 m/s向右」", "损失「损失为0」"]:
        assert unattributed_quotes(ok, pool) == [], f"误杀：{ok}"
    # 那段字在源里**找不到** ⇒ 这才是"字面不实"，必须报
    for bad in ["你写的「合力 6 N」对", "题目说「移动3 m 时的速率」",
                "学生用到了「损失＝碰前动能减碰后动能」这个关系"]:
        assert unattributed_quotes(bad, pool), f"漏报：{bad}"


# ================================================================ 契约 §2.9：路径/次序断言

_CONC = [{"id": "E1.1", "source": "student_answer", "text": "I=2 A，U₁=4 V，P₂=16 W。"}]
_STEPS = [{"id": "E1.1", "source": "student_answer",
           "text": "取右为正，合力F-f=10-4=6 N，a=6/2=3 m/s²"}]
_Q066 = "从三问的先后顺序看，是先求串联电流，再由电流求 R₁ 的电压，最后求 R₂ 的电功率。"


def test_path_claim_is_blocked_only_when_the_material_has_no_steps():
    """★ 契约 §2.9：**看材料完备度，不看措辞**。

    ⚠️ **用户纠正过我的方案**：我原想"要求每一步都能引用" —— **那不够**：
    三条结果分别能引用，**不等于**"先算电流、再据此算电压"成立。
    前者证"这些字在纸上"，后者说"这些数**出自**那条路径"= 生成过程断言。
    """
    from app.services.review_builder import evidence_has_intermediate_steps, path_claim_defects
    assert not evidence_has_intermediate_steps(_CONC), "只有结论值，不该判成有步骤"
    assert path_claim_defects(_Q066, _CONC, "timeline"), "Q066 那句必须被拦"
    # 材料写了步骤 ⇒ **同一句式合规**（那是在复述纸面）
    assert evidence_has_intermediate_steps(_STEPS)
    assert path_claim_defects(_Q066, _STEPS, "timeline") == []
    # 没池子 ⇒ 不判（旧报告不误报）
    assert path_claim_defects(_Q066, None, "timeline") == []


@pytest.mark.parametrize("text,why", [
    ("无法从材料确认他是怎么算出这些数的。", "**明确拒绝推断** —— 这正是要鼓励的写法"),
    ("材料里看不出先算哪一个。", "同上"),
    ("中间的代入过程没有留在材料里，无法核对推导路径。", "同上"),
])
def test_explicit_refusals_are_not_flagged(text, why):
    """⚠️ 守卫①：拒绝推断的句子**必须放行**（v14 首版实测：Q064 被误抓）。"""
    from app.services.review_builder import path_claim_defects
    assert path_claim_defects(text, _CONC, "timeline") == [], f"误杀：{why}"


@pytest.mark.parametrize("text", [
    "先写动量守恒式 1×6=(1+2)v，再解出共同速度。",
    "先取向右为正，再列出合力表达式。",
])
def test_suggestions_in_next_step_are_not_flagged(text):
    """⚠️ 守卫②：`next_step` **按设计就是面向行动的建议**（「先写…再…」＝让他这么做），
    不是断言他做过 ⇒ **整类排除**（v14 实测：不排除会误杀 Q002/Q029）。"""
    from app.services.review_builder import path_claim_defects
    assert path_claim_defects(text, _CONC, "next_step") == []
    # ⚠️ 但**同句写在 `timeline` 里**就是断言 ⇒ 要报（字段决定它是不是断言）
    assert path_claim_defects(text, _CONC, "timeline"), "timeline 里的同句是断言，必须报"


@pytest.mark.parametrize("text,pool,why", [
    ("材料依次列出：I=2 A、U₁=4 V、P₂=16 W。", _CONC, "**纸面排列顺序**可以说（只描述纸面）"),
    ("三个结果分别与分压法、U₂²/R₂ 都相容。", _CONC, "数值相容性陈述，不断言次序"),
    ("第 1 轮交了三个结论，第 2 轮改写了 U₁。", _CONC, "提交顺序（服务端可得），不是计算顺序"),
])
def test_paper_order_and_submission_order_are_allowed(text, pool, why):
    """三类顺序**必须分开**（契约 §2.9）：提交顺序可由服务端定；
    纸面排列顺序可以描述；**只有"推导依赖关系"在没有步骤时不许断言**。"""
    from app.services.review_builder import path_claim_defects
    assert path_claim_defects(text, pool, "timeline") == [], f"误杀：{why}"


def test_the_timeline_is_written_by_the_server_not_the_model():
    """★ 契约 §2.9 的**主结构改动**：`timeline` 由服务端生成，**整段替换模型那份**。

    ⇒ v13 唯一卡点 Q066（模型在 timeline 里反推计算顺序）**结构性消失** ——
      **不依赖任何判据**，所以也不受"判据精度不够"的影响。
    """
    from app.services.review_builder import build_material, build_server_timeline
    m = build_material([{"round": 1, "answer": "I=2 A，U₁=4 V，P₂=16 W。"}],
                       statement="串联电路，求电流、R₁电压、R₂功率")
    out = build_server_timeline(m)
    # ① 只写提交顺序 + 可见原文
    assert "第1轮" in out and "I=2 A" in out and "提交顺序" in out
    # ② **不叙述计算过程**（Q066 那种句子不可能出现在这里）
    assert "先求" not in out and "再由" not in out and "最后求" not in out
    # ③ 常设免责说的是**本段自身**（不是"材料没写"—— 那会是无条件事实断言）
    assert "本段不推断实际计算顺序" in out
    assert "材料未展示计算步骤" not in out, "别把'材料没写'写成无条件事实断言"
    # ④ 系统复算**指向独立重建区**，不混进时间线
    assert "结论核对" in out


def test_server_timeline_counts_student_rounds_not_statement_or_evidence_fragments():
    from app.services.review_builder import build_server_timeline
    material={'items':[
        {'source':'statement','round':None,'text':'题面'},
        {'source':'student_answer','round':3,'text':'第一段书写'},
        {'source':'student_answer','round':3,'text':'同轮第二段书写'},
        {'source':'student_answer','round':7,'text':'下一轮书写'},
    ]}
    result=build_server_timeline(material)
    assert result.startswith('材料展示 2 轮学生提交')
    assert '· 题面：题面' in result and '第3轮' in result and '第7轮' in result
    only_statement=build_server_timeline({'items':[material['items'][0]]})
    assert only_statement.startswith('材料展示 0 轮学生提交')


def test_a_path_claim_migrating_to_another_field_is_still_caught():
    """★ **覆盖全部字段**（契约 §2.9，用户验收第 4 条）。

    真实形态（v13 归档实测）：**某条的 `timeline` 明写"无法确认计算顺序"，
    而它的 `answer_check.note` 却断言了顺序** —— 路径断言**搬了个字段**。

    ⚠️ **它目前只记录、不阻断**（跨 8 批实测：29 处命中里很大一部分落在
    `answer_check.derived_from_work`，而那正是契约指定的系统重建区域 ⇒ 精度不够）。
    ⇒ 断言的是**判据能看见它**，不是"它已经阻断"。收窄后再改成阻断。
    """
    from app.services.review_builder import path_claim_defects
    assert path_claim_defects(_Q066, _CONC, "encouragement"), "encouragement 里的路径断言没看见"
    assert path_claim_defects("再由同一电流得 U₁=4 V。", _CONC, "answer_check.note"),\
        "answer_check.note 里的路径断言没看见"


def test_the_hint_fallback_also_catches_a_misquoted_statement(monkeypatch):
    """⚠️ **误引题面也走同一道本地兜底**（2026-10-10）。

    为什么：这一类（idr）**恰好落在 hint 上**，而 hint 有本地兜底 ⇒
    拦掉之后**学生看不到误引**、评审也看不到 ⇒ 它不再进投递产物。
    ⚠️ 但它**不是**"归因判据"那一族，别把两者的覆盖混为一谈。
    """
    from app.agent.graph import CONSERVATIVE_HINT, tutor_graph
    from app.agent.schemas import Hint
    from fakes import install_fake_llm

    import test_agent_graph as G

    install_fake_llm(monkeypatch, hint=Hint(
        hint="题目说「碰撞时间很短、外力冲量可忽略」，所以不能用机械能守恒。"))
    result = tutor_graph.invoke({**G.BASE_INPUT, "problem_content": _STMT})

    assert result["hint"] == CONSERVATIVE_HINT, "误引题面的提示没有被换掉"
    subs = result["safety_substitutions"]
    assert subs and "unfounded_statement_quotes" in subs[-1]["detector"]
    assert any("碰撞时间很短" in r for r in subs[-1]["reasons"]), "没记下被误引的那一句"
    assert result["degraded_reasons"] == [], "误引题面是**文本替换**，不是判断降级"


# ================================================================ 契约 §2.8：引文结构化

def _pool():
    from app.services.review_builder import build_material, material_evidence
    return material_evidence(build_material(
        [{"round": 1, "answer": "取右为正，合力F-f=10-4=6 N，a=3 m/s²"},
         {"round": 2, "answer": "由v²=2as=2×3×3=18，v=3√2 m/s"}],
        statement="物块向右移动3 m，求加速度和此时速率"))


def test_statement_is_its_own_source_pool():
    """**来源分池**（契约 §2.8③）：题面与学生作答不能互相冒充。"""
    pool = _pool()
    srcs = {e["id"]: e["source"] for e in pool}
    assert srcs.get("S1") == "statement", "题面没有独立成池"
    assert all(v == "student_answer" for k, v in srcs.items() if k != "S1")


def test_reference_solution_never_enters_the_pool():
    """🔴 **参考答案是评测专用，绝不进教学/复盘输入**（契约 §2.8③）。

    实测现状：后端教学与复盘路径**不碰** `reference_solution`
    （只有离线题库工具 `graph_generator.py` 用）⇒ 这条守卫把它钉住。
    """
    from app.services.review_builder import ALLOWED_EVIDENCE_SOURCES, structural_defects
    assert "reference_solution" not in ALLOWED_EVIDENCE_SOURCES
    bad = structural_defects({"timeline": "x"}, evidence_pool=[
        *(_pool()), {"id": "R1", "source": "reference_solution", "text": "参考答案"}])
    assert [d["kind"] for d in bad] == ["unauthorized_evidence_source"]


def test_the_text_form_spares_the_model_from_counting_offsets():
    """★ **文本式引用记号**（2026-10-10 实测后加的第二种写法）。

    🔴 **为什么加**：v11 首次只教数值式，模型**认真在用但偏移算错了**
    （`⟦E1.1:18-27⟧` 越界，该证据只有 25 字符）⇒ 被正确拒收，
    但**那是接口设计的错** —— 要求 LLM 稳定数到"第 18 个码点"不可执行。
    ⇒ 改成**模型抄那段字，服务端自己定位**。
    """
    from app.services.review_builder import citation_defects, render_citations
    pool = [{"id": "E1.1", "source": "student_answer",
             "text": "取右为正，合力F-f=10-4=6 N，a=3 m/s²"}]

    # ① 抄对 ⇒ 通过，且**渲染出来的是原文**（不是模型那份）
    out, bad = render_citations("你写的⟦E1.1:合力F-f=10-4=6 N⟧是对的", pool)
    assert not bad and out == "你写的合力F-f=10-4=6 N是对的"
    # ② **只差空白** ⇒ 允许（契约 §2.7③ 的唯一等价），且仍渲染回**原文**
    out2, bad2 = render_citations("你写的⟦E1.1:合力 F-f = 10-4=6 N⟧是对的", pool)
    assert not bad2 and out2 == "你写的合力F-f=10-4=6 N是对的"
    # ③ 抄错一个字 ⇒ **拒**
    assert [d["kind"] for d in citation_defects("⟦E1.1:合力F-f=10+4=6 N⟧", pool)]\
        == ["quote_not_in_evidence"]
    # ④ 证据里没有的一段 ⇒ 拒
    assert [d["kind"] for d in citation_defects("⟦E1.1:他没写过的句子⟧", pool)]\
        == ["quote_not_in_evidence"]


def test_a_repeated_fragment_takes_the_first_occurrence():
    """**重复片段取第一处**（契约 §2.8①，确定性规则，不是隐式行为）。"""
    from app.services.review_builder import render_citations
    pool = [{"id": "D", "source": "student_answer", "text": "a=3；后来 a=3"}]
    assert render_citations("⟦D:a=3⟧", pool)[0] == "a=3"


@pytest.mark.parametrize("token,kind", [
    ("⟦E9.9:0-3⟧", "unknown_evidence"),          # 证据 ID 不存在
    ("⟦E1.1:0-999⟧", "range_out_of_bounds"),     # 偏移越界
    ("⟦E1.1:8-3⟧", "range_out_of_bounds"),       # 起止颠倒
])
def test_bad_citations_are_rejected(token, kind):
    """契约 §2.8①：**非法引用不得投递或缓存**。"""
    from app.services.review_builder import citation_defects
    assert [d["kind"] for d in citation_defects(f"你写的{token}", _pool())] == [kind]


def test_a_stale_revision_is_rejected():
    """**过期修订 ⇒ 拒**（契约 §2.8①）：那段字**可能已经被学生改过**，引用不再成立。"""
    from app.services.review_builder import citation_defects
    pool = [{"id": "E1.1", "source": "student_answer", "source_id": "src1",
             "revision": 2, "text": "a=3 m/s²"}]
    assert citation_defects("⟦E1.1:0-3⟧", pool, {"src1": 2}) == []          # 修订一致
    assert [d["kind"] for d in citation_defects("⟦E1.1:0-3⟧", pool, {"src1": 3})]\
        == ["stale_revision"]                                                # 已过期
    # ⚠️ 没有 source_id 或没有当前修订表 ⇒ **不判**（宁可不报，也不误杀）
    assert citation_defects("⟦E1.1:0-3⟧", pool, None) == []
    assert citation_defects("⟦E1.1:0-3⟧", [{"id": "E1.1", "source": "student_answer",
                                            "revision": 2, "text": "a=3 m/s²"}]) == []


def test_a_valid_citation_renders_the_original_text():
    from app.services.review_builder import render_citations
    out, bad = render_citations("你写的⟦E1.1:5-19⟧是对的", _pool())
    assert not bad
    assert out == "你写的合力F-f=10-4=6 N是对的"


def test_offsets_index_the_truncated_text_not_the_original():
    """**上下文截断**：偏移的坐标系是**已投递的那份** `evidence.text`（可能被预算截断）。

    ⇒ 截断掉的部分**取不到**（越界被拒），而不是"偷偷回到未截断的原文去取"。
    """
    from app.services.review_builder import citation_defects, render_citations
    pool = [{"id": "T1", "source": "student_answer", "text": "前半句",
             "omitted_chars": 40, "mid_line_cut": True}]
    assert len(pool[0]["text"]) == 3
    assert citation_defects("⟦T1:0-3⟧", pool) == []          # 投递范围内 ⇒ 可以引
    assert [d["kind"] for d in citation_defects("⟦T1:0-9⟧", pool)] == ["range_out_of_bounds"]
    assert render_citations("⟦T1:0-3⟧", pool)[0] == "前半句"


def test_offsets_disambiguate_a_repeated_fragment():
    """**重复片段**：同一段字出现两次时，**偏移决定取哪一处**（这正是用偏移而非 ID 的理由）。"""
    from app.services.review_builder import render_citations
    text = "a=3 m/s²，后来他又写 a=3 m/s²，两处相同"
    pool = [{"id": "D1", "source": "student_answer", "text": text}]
    first = text.index("a=3 m/s²")
    second = text.index("a=3 m/s²", first + 1)
    assert render_citations(f"⟦D1:{first}-{first + 8}⟧ 与 ⟦D1:{second}-{second + 8}⟧",
                            pool)[0] == "a=3 m/s² 与 a=3 m/s²"
    assert first != second, "夹具没造出重复片段"


def test_unicode_offsets_are_code_points():
    """**Unicode 偏移规则**：0 基**码点**索引（不是 UTF-16 单元）。

    `a´b😀c` 在 Python 里是 **5 个码点**（a、组合符、b、emoji、c），
    而在 UTF-16 里 emoji 占 **2 个单元** ⇒ 若按 UTF-16 数，`0-4` 会被算成越界。
    """
    from app.services.review_builder import citation_defects, render_citations
    pool = [{"id": "U1", "source": "student_answer", "text": "a\u0301b\U0001F600c"}]
    assert len(pool[0]["text"]) == 5, "夹具的码点长度不是 5"
    assert citation_defects("⟦U1:0-5⟧", pool) == []
    assert render_citations("⟦U1:0-4⟧", pool)[0] == "a\u0301b\U0001F600"
    assert [d["kind"] for d in citation_defects("⟦U1:0-6⟧", pool)] == ["range_out_of_bounds"]


def test_handwritten_attribution_quotes_are_blocked_but_terms_are_not():
    """**禁止模型自写归属引文；但普通术语引号必须放行**（契约 §2.8②）。

    ⚠️ 判据是**有没有归属框架**，不是"有没有引号" ——
    把所有引号都判成引用，会把正常写法一起扫掉。
    """
    from app.services.review_builder import structural_defects
    base = {"timeline": "两问都在第一轮里给出。", "sticking_points": [],
            "key_concept": "", "next_step": "", "answer_check": None,
            "evidence": [], "findings": []}
    blocked = structural_defects({**base, "encouragement": "你写的「合力 6 N」对"},
                                 evidence_pool=_pool())
    assert [d["kind"] for d in blocked] == ["unattributed_quote"]
    allowed = structural_defects({**base, "encouragement": "「机械能守恒」在这题不成立，取向右为正。"},
                                 evidence_pool=_pool())
    assert allowed == [], "普通术语引号被误判成引用"


def test_old_reports_without_the_new_structure_are_not_counted_as_violations():
    """**验收第 1 条**：旧报告缺新结构 ⇒ **不因无法迁移就统计为误报**。

    ⇒ 不传证据池时，引用类缺陷**一概不判**（历史报告保留读取能力）。
    """
    from app.services.review_builder import structural_defects
    old = {"timeline": "无", "sticking_points": [], "key_concept": "", "next_step": "",
           "encouragement": "你写的「合力 6 N」对", "answer_check": None,
           "evidence": [], "findings": []}
    assert structural_defects(old) == [], "没传池子却按新结构判了旧报告"


def test_the_prompt_states_the_verbatim_quote_rule():
    """⚠️ **引号是承诺**（2026-10-10 新增）。

    第六轮评审 A 标出 3 条 `invalid_displayed_reference`，都是**把话塞进学生嘴里**：
    ```
    Q001 hint 引题面「碰撞时间很短、外力冲量可忽略」——题面原文是「…很短且系统所受外力冲量可忽略」
    Q037「学生用到了「损失＝碰前动能减碰后动能」」——原文只有运算符不明的「18 ? 6」
    Q065「你写的是「初始动能 18 J」」——原文是「初动能 18 J」
    ```
    ⚠️ 我**量过机械判定的可行性**（v7/v8 两批）：
      全量引号 → 85/298 不在材料里，**几乎全是正当术语/转述** ⇒ 不可用；
      只查**归属给学生的**引文 → 召回约 1~2/3、精度约一半 ⇒ **也不够格当阻断闸门**
      （真当闸门会拒掉每批 5~9 份）。⇒ **所以这一条走提示词，不走闸门。**
    """
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "引号里必须是逐字原文" in prompt
    assert "引号是承诺" in prompt, "缺那句一句话判据"
    assert "别把概括塞进引号" in prompt, "没给可操作的替代写法"


def test_the_prompt_says_the_standard_covers_the_whole_report():
    """第 2 条：归因与计数的标准**覆盖全篇**，不只错误列表。"""
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "覆盖全篇" in prompt
    for field in ("timeline", "encouragement", "answer_check.note", "ocr_caution",
                  "next_step", "sticking_points"):
        assert field in prompt, f"没点名 {field}"
    for banned in ("没有犹豫", "不是碰巧", "已经掌握", "已经抓住主线"):
        assert banned in prompt, f"没点名禁止「{banned}」"


# ================================================================ 全文归因扫描器（第 2 条的执行侧）
#
# ⚠️ 真值不是我想出来的：下面每一条**引文**都来自 T4 开发集 72 份归档包里
#    **两名评审独立标出、并经复核确认**的 10 条违规原文（逐字）。
#    实测（脚本在 72 份上跑）：**召回 10/10、误报 1/62**，
#    而那 1 条（Q005「说明串联电流和分压他已经掌握」）恰是提示词第 9 条明令禁止的写法，
#    **很可能是评审漏标的真违规**。
#
# ⚠️ 测试**不读 `artifacts/`**（那是运行产物、已被 gitignore）——
#    所以把引文内联在这里。**改判据必须重跑那个脚本并更新这里的数字。**

_CONFIRMED_ATTRIBUTION_VIOLATIONS = [
    ("Q007", "encouragement", "不是靠「下落所以重力做正功」的直觉猜的"),
    ("Q008", "downstream", "凡是靠「重力做功」去列能量或动能关系的题，都会错"),
    ("Q023", "encouragement", "串联总电阻相加这一步没有犹豫"),
    ("Q026", "encouragement", "这个结果不是凑出来的"),
    ("Q034", "encouragement", "这说明他不是套结论套出来的"),
    ("Q036", "encouragement", "说明结果不是碰巧写出来的数字"),
    ("Q053", "encouragement", "这三个数不是碰出来的数"),
    ("Q056", "encouragement", "这个精确值不是随手写的近似值"),
    ("Q066", "encouragement", "说明串联电流相同这条主线你抓住了"),
    ("Q071", "encouragement", "说明 mgh 直接作为重力做功的这条线你用对了"),
]


@pytest.mark.parametrize("qid,where,quote", _CONFIRMED_ATTRIBUTION_VIOLATIONS)
def test_every_confirmed_attribution_violation_is_caught(qid, where, quote):
    """**召回回归** —— 复核确认的 10 条，逐条必须命中。"""
    assert unsupported_claims(quote), f"{qid}（{where}）的违规没被抓到：{quote!r}"


#: 实测**不该报**的形态。前两条来自真实的误报，后三条是同类边界。
_ATTRIBUTION_NEAR_MISSES = [
    # 真实误报①：模型在**否认**自己凑 —— 宾语被引号括起来是在给一类错误命名
    ("Q001", "所以这不是「凑对」，而是「前提错→结论跟着错」"),
    ("Q021", "所以这不是「凑对」型的自相矛盾，而是一处物理判断错误"),
    # 正当评价可见书写 —— 不是过程归因
    ("—", "这一步公式用对了，单位也写对了"),
    ("—", "他辅助线画对了，切线方向也标出来了"),
    # 否定/存疑表述 —— 方向正好相反
    ("—", "这不能说明他掌握了这个概念"),
    ("—", "仅凭结果不足以证明他理解了这条规律"),
]

#: ⚠️ **曾在这张表里、2026-10-09 被裁决移出的一条**（保留记录，别删）：
#:     ("Q047", "整段没有出现分步核对或改写痕迹")   ← 当时期望"不报"
#:
#: **为什么移出**：当时**同一形态在两处判定相反**（上一批 Q053 被标违规、Q047 没被标），
#: 我据此判定"形态有歧义 ⇒ 宁可漏不可误杀"。**那个判断在当时是对的**（判据未定）。
#:
#: 现在契约 ⑦-3（第三方裁决）把这条线定了：
#:     **「没有发生某事」一律违规** —— 静态快照无法证明过程事件不存在。
#: 依据锚在上一批**已确认**的 Q023「串联总电阻相加这一步没有犹豫」上。
#: ⇒ 它现在属于 `_CONFIRMED_...` 那一侧。**这不是反复，是判据定案后测试跟着定案。**
@pytest.mark.parametrize("qid,quote", [
    ("Q006", "没有出现半途改口或方向反复"),
    ("Q026", "没有中途换向"),
    ("Q011", "没有中途换过数值"),
    ("Q033", "没有出现回退或改写的痕迹"),
    ("Q058", "你把方向定成「向右为正」之后一直没换过"),
    ("Q047", "整段没有留下推导痕迹"),      # ⚠️ 变体实测补：「没有**留下**X」原版漏掉
    ("Q023", "串联总电阻相加这一步没有犹豫"),
    ("Q070", "说明开方后你没有默认取正根"),  # ⚠️ 变体实测补：第三种动词（评审 A′ 判对的那条）
    # ⚠️ 2026-10-10 第五轮**两评审共识**项就是这两个词（净增 6 处命中、全同族、0 误报）
    ("Q016", "三问是一条线走下来的，没有分步中断"),
    ("Q030", "三个量在同一轮里全部给出，没有再出现后续修改"),
    ("Q038", "两条结论都给出后没有再作改动"),   # 第六轮评审 A 标出的 (Q038)
    # ⚠️ 第六轮**裁决**判为 true、而闸门当时漏掉的（词表已按裁决补齐，先量后加）
    ("Q010", "两条电功率路径能对上就说明结果不是侥幸来的"),
    ("Q033", "三个小问按顺序做完，中间没有再回到前面的量"),
    # 上面这条裁决点名、但**两位评审都判了 false** ⇒ 不在分歧清单 ⇒ 裁决不到。
    # **"只裁决分歧"是这个流程的系统性盲区**，规则必须兜住它。
])
def test_the_no_such_thing_happened_family_is_caught(qid, quote):
    """契约 ⑦-3 第③条：**「没有发生某事」一律违规**。

    ⚠️ 这一族是"只裁决分歧"盲区的直接受益者 —— 两评审都漏就永远裁决不到。
    本批实测：两位评审漏了 4 条同族句，扫描器扩族后**全部抓到**，
    并且**在两批归档上没有产生任何一条误报**（扩族后额外命中的 12 条逐条查过，全是同族）。
    """
    assert unsupported_claims(quote), f"{qid} 的同族句没被抓到：{quote!r}"


#: ⚠️ **提示词里推荐给模型的替代写法** —— 扫描器**必须一条都不报**。
#:
#: 这一条是**变体测试里最要紧的**：如果模型**照提示词写的**反而被闸门打回，
#: 那就是 P0-B 那一类**自伤**（本轮开头刚吃过一次：服务端自己渲染的文本触发自己的判据）。
#: ⇒ 提示词与判据必须**互相对得上**，这条测试就是那个"对得上"的锚。
_PROMPT_RECOMMENDED_PHRASINGS = [
    ("材料描述", "材料里只保留了结论，中间代入没有留下"),
    ("材料描述", "你交上来的这一轮里，两个式子用的都是同一个 I"),
    ("引原文再评", "你写的 1×6=3v 把碰后总质量取成 3 kg，这一步是对的"),
    ("材料描述", "材料里没有出现这一步的中间式"),
    ("材料描述", "这一段材料在「v =」处截断，未看到最终数值"),
    ("材料描述", "原文只给了结果，没有展示代入过程"),
    ("标注重建", "核对时重建：合外力为 10 N − 4 N = 6 N 向右"),
]


@pytest.mark.parametrize("kind,quote", _PROMPT_RECOMMENDED_PHRASINGS)
def test_the_phrasings_the_prompt_recommends_are_not_flagged(kind, quote):
    """**不许自伤**：模型照提示词推荐的写法写，不能被判违规。

    上一轮出现过相反的事（服务端渲染的文本触发自己的判据 ⇒ 越听话越被打回），
    所以提示词每推荐一种写法，都要有一条测试保证判据不挡它。
    """
    assert unsupported_claims(quote) == [], f"提示词推荐的{kind}写法被误报：{quote!r}"


#: ⚠️ **加 ③族动词时的合法反例**（2026-10-10 加 `中断`/`修改` 时一并测的）。
#:
#: 分界线是**有没有 `没有 / 未见 / 不曾 …` 这个否定前缀**：
#: 断言"**过程未发生**"是违规；正面描述**可见的**中断/修改、或给建议，都不是。
@pytest.mark.parametrize("quote,why", [
    ("若中途被打断，重新上传即可，编号会保留。", "建议句，不是断言学生"),
    ("材料里有涂改痕迹，第二步被划掉后重写。", "对可见材料的正面描述"),
    ("他在第 2 轮修改了合力，把 14 N 改成了 6 N。", "正面描述可见的多轮变化"),
    ("本轮材料只在「损失为」处中断，后面没有内容。", "对可见材料的如实描述"),
])
def test_legitimate_uses_of_the_new_verbs_are_not_flagged(quote, why):
    """**加动词不能把正当用法一起扫进来** —— 这四条实测都不报。"""
    assert unsupported_claims(quote) == [], f"合法用法（{why}）被误报：{quote!r}"


#: ⚠️ **对比式否认 vs 表扬式断言** —— 这是同一个词（凑/碰巧）的两种相反用法。
#:
#: 实测依据：四批归档里，**已确认的 12 处「不是…」违规没有一处带「而是」结构**；
#: 而唯一那条「**不是凑对，而是一致地建立在同一个前提上**」（模型在**说明错误的性质**）
#: 被旧判据误报、导致整份复盘被拒。⇒ 加守卫：命中后紧跟「而是」⇒ 跳过。
@pytest.mark.parametrize("quote", [
    "这一问不是凑对，而是一致地建立在同一个前提上",
    "这不是套结论，而是把同一套关系用在了错的前提上",
])
def test_the_contrastive_denial_is_not_flagged(quote):
    """**说明错误性质**的对比句不报 —— 它和"表扬学生没蒙"是两件事。"""
    assert unsupported_claims(quote) == [], f"对比式否认被误报：{quote!r}"


@pytest.mark.parametrize("quote", [
    "这个结果不是凑出来的",
    "说明结果不是碰巧写出来的数字",
    "这说明他不是套结论套出来的",
])
def test_the_praise_style_denial_is_still_flagged(quote):
    """⚠️ **守卫不能把真违规一起放走** —— 表扬式的「不是凑/碰巧」照报。"""
    assert unsupported_claims(quote), f"表扬式断言漏报了：{quote!r}"





@pytest.mark.parametrize("qid,quote", _ATTRIBUTION_NEAR_MISSES)
def test_attribution_near_misses_are_not_flagged(qid, quote):
    """**精度回归** —— 实测过会误报的形态，逐条必须**不**命中。

    ⚠️ 这几条不是凑数的边界：前两条是**真跑出来的误报**（第一版判据命中它们），
    第三条是**同一形态在一处被评审标违规、在另一处没有** ⇒ 形态有歧义，宁可漏。
    """
    assert unsupported_claims(quote) == [], f"{qid} 被误报：{quote!r}"


def test_attribution_issues_flow_into_the_validation_and_the_category():
    """新判据必须**真的接进**结构校验，否则只是一段没人调用的代码。"""
    payload = {"timeline": "他做完了。", "sticking_points": [], "key_concept": "",
               "next_step": "", "encouragement": "这个结果不是碰巧写出来的数字",
               "answer_check": None, "findings": [], "evidence": [],
               "review_schema_version": "v3+prompt-v3"}
    kind = "unsupported_process_claim"
    assert [d["kind"] for d in structural_defects(payload)] == [kind]
    from app.services.review_builder import classify_defects

    assert classify_defects([{"kind": kind}]) == "无依据的过程/认知断言"
    assert "物理" not in classify_defects([{"kind": kind}])


def test_the_scanner_documents_what_it_cannot_reach():
    """⚠️ **它够不到的那一族，必须写在代码里** —— 别把它当"归因问题已堵住"。

    Q026 / Q071 还包含「把**未展示的**方法叙述成学生实际走过的路线」——
    那是**语义**问题（缺证据），**句式扫描器原理上抓不到**。
    """
    import inspect

    from app.services import review_builder as rb

    src = inspect.getsource(rb.unsupported_claims)
    assert "抓不到" in src or "够不到" in src, "没写明它够不到什么"
    head = inspect.getsource(rb)
    assert "语义" in head and "句式扫描器" in head




