"""Prompt 注册表与版本管理（P13，任务书第 P13 节）。

任务书要求：
- 不要让多个 Agent Node 共用一个巨大 Prompt，不同职责的 Prompt 完全拆开；
- 每个 Prompt 带版本号，每次 Trace 记录实际使用的版本，方便 A/B 对比与回归。

因此提示词正文不再写在 Python 里，而是放在项目根的 `prompts/<角色>/v<版本>.md`：

    prompts/
    ├── step_matcher/v1.md
    ├── analyze_step/v1.md
    ├── diagnosis/v1.md
    └── tutor/v1.md

加一个新版本 = 放一个 `v2.md`，再把 `DEFAULT_VERSIONS` 里的默认值改成 `v2`。
比较两版 = 跑 `scripts/run_eval.py` 与 `scripts/run_simulation.py --all`，
再从 `analysis_traces.prompt_versions` 确认本次用的是哪一版。

职责边界（任务书 P13 强制要求，改 prompt 时不要越界）：
- step_matcher：只判断输入对应哪些节点 / 是否覆盖多节点 / 是否正确 /
  是否缺关键推理 / 是否走了合理替代路线。**不生成教学提示**。
- analyze_step：只描述学生当前进展（步骤 / 方法 / 是否有明显错误 / 知识点 / 是否完成）。
- diagnosis：只做错误归因（类型 / 原因 / 知识点 / 严重程度 / 建议）。
- tutor：只按给定 hint_level 与诊断结果生成提示。**不重新判断学生做到了哪一步**。
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# 各角色的默认版本；A/B 时改这里，或用 PROMPT_<角色>_VERSION 环境变量临时覆盖
DEFAULT_VERSIONS: dict[str, str] = {
    "step_matcher": "v2",
    "analyze_step": "v3",        # 统一服务端小问归属和已写信息
    # ⚠️ **2026-10-11 切到 v3**（用户裁决，契约 §2.10）：
    #   v3 新增 `issue` 字段的**纸面措辞**要求 ——
    #   **只描述这份作答里"纸上写了什么"，不断言学生的理解状态、信念、动机或未展示的推导过程**。
    #   **切换依据**：v6/v14/v15 的隐藏历史结论复核里，
    #   `teaching[].issue` 的「误认为 / 未认识到」型措辞被判为**无依据的认知归因** ✓
    #   （v14 归档 2 条、v15 归档 4 条）⇒ 这是**已确认必须改**的业务项 ✓
    #   ⚠️ **v2 保留**，可 `PROMPT_DIAGNOSIS_VERSION=v2` 切回做 A/B。
    "diagnosis": "v4",
    "review": "v2",       # ⚠️ **v3 已实现并测过，但默认仍是 v2**
    # v3 要求模型**引用证据 ID**（`prompts/review/v3.md`）。
    # **为什么退回 v2**：契约（`docs/契约-媒体与复盘接口-冻结.md`）写着
    #   > **v3 通过开发集后再切换默认**。
    # ⚠️ **2026-10-09 更正理由**（独立审计 A5 指出）：此处原先写「**那个开发集还没建**」——
    #    **那是过期的**。开发集**已经建了，也跑过了**：
    #      · 第一轮 72 次（`artifacts/t4_development_20261009`）：门槛 **reject**，
    #        4 条严重物理矛盾 + 10 处无依据归因；
    #      · 修复后第二轮 72 次（`artifacts/t4_development_reviewfix_20261009`）：
    #        结构合规 70/72，但**质量门槛仍未通过**（要两名独立评审判定）。
    #   **结论没变（仍然不该切 v3），变的是依据** —— "还没建"与"建了但不通过"
    #   是两个不同的条件，后者才是现存的。**依据写错比结论写错更难被发现。**
    # **切回 v3 的条件**：开发集**通过**质量门槛（含两名独立评审），
    #   且留出集（0/36，尚未解锁）也通过。`PROMPT_REVIEW_VERSION=v3` 可临时启用。
                          # v1 保留，可 PROMPT_REVIEW_VERSION=v1 切回做 A/B           # v2：`concept` 必须取自本题知识点（供画像/推荐对齐）
    "tutor": "v4",
    "graph_generator": "v3",     # 必答小问与替代方法分开，逐节点来源与目标量覆盖
    "graph_validator": "v1",
    'figure_context':'v2',
    "page_reader": "v1",       # PDF 导入：整页读公式与插图位置     # P15：校验候选图的物理一致性
}

PROMPT_ROLES: tuple[str, ...] = tuple(DEFAULT_VERSIONS)


class PromptNotFoundError(RuntimeError):
    """提示词文件缺失（角色或版本不存在）。"""


def _env_override(role: str) -> str | None:
    """`PROMPT_STEP_MATCHER_VERSION=v2` 形式的临时覆盖，便于不改代码做 A/B。"""
    value = os.environ.get(f"PROMPT_{role.upper()}_VERSION", "").strip()
    return value or None


def available_versions(role: str, directory: Path | None = None) -> list[str]:
    """列出某角色已有的版本号（按文件名排序）。角色目录不存在时返回空列表。"""
    base = (directory or PROMPTS_DIR) / role
    if not base.is_dir():
        return []
    return sorted(path.stem for path in base.glob("v*.md"))


def resolve_version(role: str, version: str | None = None) -> str:
    """确定实际使用的版本：显式指定 > 环境变量 > 默认值。"""
    if version:
        return version
    override = _env_override(role)
    if override:
        return override
    if role not in DEFAULT_VERSIONS:
        raise PromptNotFoundError(
            f"未知的提示词角色 {role!r}；已注册角色：{'、'.join(PROMPT_ROLES)}"
        )
    return DEFAULT_VERSIONS[role]


@lru_cache(maxsize=64)
def _read(role: str, version: str, directory: str) -> str:
    path = Path(directory) / role / f"{version}.md"
    if not path.is_file():
        options = "、".join(available_versions(role, Path(directory))) or "（无）"
        raise PromptNotFoundError(
            f"未找到提示词 {path}。该角色已有版本：{options}"
        )
    # 文件内容即提示词原文，不做 strip / 拼接等任何加工，保证与旧常量逐字一致
    return path.read_text(encoding="utf-8")


def get_prompt(role: str, version: str | None = None, directory: Path | None = None) -> str:
    """读取提示词正文。结果按 (角色, 版本, 目录) 缓存，不会每次读盘。"""
    resolved = resolve_version(role, version)
    return _read(role, resolved, str(directory or PROMPTS_DIR))


def prompt_version_fields() -> dict[str, str]:
    """本次运行实际使用的版本，形如 {"step_matcher_prompt_version": "v1", ...}。

    任务书 P13 要求每次 Trace 都带上，用于事后对比与回归定位。
    """
    return {f"{role}_prompt_version": resolve_version(role) for role in PROMPT_ROLES}


def all_prompt_versions() -> dict[str, str]:
    """{角色: 版本}，供报告与调试展示。"""
    return {role: resolve_version(role) for role in PROMPT_ROLES}


def clear_cache() -> None:
    """清空读取缓存（测试切换目录或版本时使用）。"""
    _read.cache_clear()
