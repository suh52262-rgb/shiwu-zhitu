"""Prompt 注册表与版本管理测试（P13）。

重点守住两件事：
1. **外置后内容逐字不变**——`tests/test_agent_graph.py` 与 `tests/test_step_matcher.py`
   有断言依赖 system prompt 文本（"Hint Policy"、"步骤匹配器"），内容漂移会直接挂；
2. **版本机制真的可用**——多版本共存、显式选择、环境变量覆盖、缺失时报错清晰。
"""
import pytest

from app.agent import prompt_registry as registry
from app.agent.prompt_registry import (
    DEFAULT_VERSIONS,
    PROMPT_ROLES,
    PromptNotFoundError,
    all_prompt_versions,
    get_prompt,
    prompt_version_fields,
    resolve_version,
)


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    """版本/目录切换后必须清缓存，否则会读到上一个用例的结果。"""
    registry.clear_cache()
    yield
    registry.clear_cache()


def test_tutor_prompt_bounds_praise_and_bans_unfounded_claims():
    """⚠️ **hint 越界句的来源就是"给一句简短肯定"没限定落点**（2026-10-10）。

    实测：第五轮评审在 `teaching[0].hint` 里抓到
    「思路正确，结果也一致，**说明串联关系掌握得很好**」——
    它正是"肯定落在**认知状态**上"的产物。闸门管不到 hint（它是教学主产物），
    所以提示词这一侧必须把边界写清。
    """
    from pathlib import Path

    prompt = (Path(__file__).resolve().parents[1] / "prompts/tutor/v1.md").read_text(encoding="utf-8")
    # 肯定必须落在"已写出的内容"上，不许落在"认知状态"上
    assert "掌握得很好" in prompt, "没点名那条真实的越界说法"
    assert "已写出的内容" in prompt and "认知状态" in prompt, "没写清肯定该落在哪"
    # 正反例都要有（只禁止不给写法 ⇒ 模型只能删掉整段）
    assert "请补写这一步的依据" in prompt, "没给面向行动的替代写法"
    assert "该做什么" in prompt and "你是怎么做的" in prompt, "缺那句一句话判据"


# ---------- 角色与文件 ----------

def test_all_registered_roles_have_default_version():
    assert set(PROMPT_ROLES) == set(DEFAULT_VERSIONS)
    for role in PROMPT_ROLES:
        body = get_prompt(role)
        assert body.strip(), f"{role} 提示词为空"
        assert len(body) > 100, f"{role} 提示词过短，疑似未正确加载"


def test_prompt_roles_cover_the_task_required_separation():
    """任务书 P13：不同职责的 Prompt 必须完全拆开。"""
    assert set(PROMPT_ROLES) >= {"step_matcher", "diagnosis", "tutor"}


def test_prompt_bodies_are_distinct():
    """各角色提示词不能是同一份内容。"""
    bodies = [get_prompt(role) for role in PROMPT_ROLES]
    assert len(set(bodies)) == len(bodies)


# ---------- 职责边界（任务书 P13） ----------

def test_step_matcher_prompt_does_not_generate_hints():
    body = get_prompt("step_matcher")
    assert "步骤匹配器" in body
    assert "hint_level" not in body          # 不涉及提示分级
    assert "提示" not in body.replace("未演示", "")  # 不负责教学提示


def test_tutor_prompt_does_not_rejudge_steps():
    body = get_prompt("tutor")
    assert "hint_level" in body
    assert "Hint Policy" in body             # 现有测试依赖该串
    assert "matched_nodes" not in body       # 不重新判断学生做到哪一步


def test_diagnosis_prompt_is_attribution_only():
    body = get_prompt("diagnosis")
    assert "issue_type" in body
    assert "hint_level" not in body
    assert "matched_nodes" not in body


# ---------- 版本机制 ----------

def test_resolve_version_precedence(monkeypatch):
    assert resolve_version("tutor") == DEFAULT_VERSIONS["tutor"]
    assert resolve_version("tutor", "v9") == "v9"              # 显式指定最优先
    monkeypatch.setenv("PROMPT_TUTOR_VERSION", "v7")
    assert resolve_version("tutor") == "v7"                     # 其次环境变量
    assert resolve_version("tutor", "v9") == "v9"               # 显式仍最优先


def test_unknown_role_raises_with_registered_roles():
    with pytest.raises(PromptNotFoundError, match="未知的提示词角色"):
        resolve_version("no_such_role")


def test_missing_version_lists_available_ones():
    with pytest.raises(PromptNotFoundError, match="已有版本"):
        get_prompt("tutor", "v99")


def test_prompt_version_fields_shape():
    fields = prompt_version_fields()
    assert fields["step_matcher_prompt_version"] == DEFAULT_VERSIONS["step_matcher"]
    assert fields["tutor_prompt_version"] == DEFAULT_VERSIONS["tutor"]
    assert set(fields) == {f"{role}_prompt_version" for role in PROMPT_ROLES}
    assert all(value.startswith("v") for value in fields.values())


def test_all_prompt_versions_matches_roles():
    assert set(all_prompt_versions()) == set(PROMPT_ROLES)


# ---------- 多版本共存（A/B 能力） ----------

def test_multiple_versions_can_coexist(tmp_path):
    """放一个 v2.md 就能被选中——这是 A/B 对比的基础。"""
    role_dir = tmp_path / "tutor"
    role_dir.mkdir()
    (role_dir / "v1.md").write_text("第一个版本", encoding="utf-8", newline="\n")
    (role_dir / "v2.md").write_text("第二个版本", encoding="utf-8", newline="\n")

    assert registry.available_versions("tutor", tmp_path) == ["v1", "v2"]
    assert get_prompt("tutor", "v1", directory=tmp_path) == "第一个版本"
    assert get_prompt("tutor", "v2", directory=tmp_path) == "第二个版本"


def test_available_versions_of_unknown_role_is_empty(tmp_path):
    assert registry.available_versions("ghost", tmp_path) == []


def test_prompt_content_is_used_verbatim(tmp_path):
    """注册表不做 strip / 拼接，文件内容即提示词原文。"""
    role_dir = tmp_path / "tutor"
    role_dir.mkdir()
    raw = "  前后都有空白\n\n第二行  \n"
    (role_dir / "v1.md").write_text(raw, encoding="utf-8", newline="\n")

    assert get_prompt("tutor", "v1", directory=tmp_path) == raw


def test_cache_avoids_repeated_disk_reads(tmp_path):
    role_dir = tmp_path / "tutor"
    role_dir.mkdir()
    target = role_dir / "v1.md"
    target.write_text("原始内容", encoding="utf-8", newline="\n")

    assert get_prompt("tutor", "v1", directory=tmp_path) == "原始内容"
    target.write_text("磁盘已被改写", encoding="utf-8", newline="\n")
    assert get_prompt("tutor", "v1", directory=tmp_path) == "原始内容"  # 命中缓存

    registry.clear_cache()
    assert get_prompt("tutor", "v1", directory=tmp_path) == "磁盘已被改写"


# ---------- 与旧常量的兼容 ----------

def test_legacy_constants_still_resolve_to_same_content():
    """agent/prompts.py 作为兼容 shim，内容须与注册表一致。"""
    from app.agent import prompts as legacy

    assert legacy.STEP_MATCH_SYSTEM == get_prompt("step_matcher")
    assert legacy.ANALYZE_STEP_SYSTEM == get_prompt("analyze_step")
    assert legacy.DIAGNOSE_SYSTEM == get_prompt("diagnosis")
    assert legacy.GENERATE_HINT_SYSTEM == get_prompt("tutor")
