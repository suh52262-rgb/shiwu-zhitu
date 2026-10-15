"""Graph Validator 的输入/输出协议（P15，任务书第三阶段 P15 节）。

任务书的输出结构：

    {"valid": false, "confidence": 0.71,
     "issues": [{"type": "physics_inconsistency", "node": "S4", "description": "..."}],
     "recommended_action": "teacher_review"}

四类检查（任务书指定）：结构合法性 / 解题流程完整性 / 物理一致性 / 多路径一致性。

**`valid` 与 `needs_review` 是两件事**：校验器通过不等于人工确认过。
校验结论写进图 metadata 的 `validation` 字段，`needs_review` 仍然只由人清除。
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field

# ---- 问题类型（对应任务书的四类检查）----
ISSUE_STRUCTURE = "structure"                    # 结构合法性
ISSUE_FLOW = "flow_incompleteness"               # 解题流程完整性
ISSUE_PHYSICS = "physics_inconsistency"          # 物理一致性
ISSUE_PATH = "path_inconsistency"                # 多路径一致性
ISSUE_FALSE_BRANCH = "suspected_false_branch"     # 并列目标被建成可选支路（启发式）
ISSUE_BRANCH_MERGE = "suspected_branch_merge"     # 汇合适用性疑点（辅助复核）

ISSUE_TYPES = (ISSUE_STRUCTURE, ISSUE_FLOW, ISSUE_PHYSICS, ISSUE_PATH, ISSUE_FALSE_BRANCH, ISSUE_BRANCH_MERGE)

# ---- 严重程度 ----
#: error：图**不可用**（Agent 会走错 / 崩溃），必须修
SEVERITY_ERROR = "error"
#: warning：可疑或不够好，建议人工看一眼
SEVERITY_WARNING = "warning"

# ---- 建议动作 ----
ACTION_ACCEPT = "accept"                   # 可以直接用
ACTION_TEACHER_REVIEW = "teacher_review"   # 建议教师复核
ACTION_REGENERATE = "regenerate"           # 结构性问题，建议重新生成

# ---- 单项检查的状态（表示"该类检查有没有发现问题"，不含致命暗示）----
#: 该类检查未发现问题
CHECK_PASS = "pass"
#: 该类检查发现了问题；严重程度看 issues 里的 severity，
#: warning 级不影响 valid，error 级才会让图不可用
CHECK_ISSUES = "issues"
CHECK_SKIPPED = "skipped"                  # 例如没有题面时的物理一致性检查


class GraphIssue(BaseModel):
    """一条校验发现。"""

    type: str = ISSUE_STRUCTURE
    severity: str = SEVERITY_WARNING
    node: str = ""                         # 相关节点 id（结构类问题可能为空）
    description: str = ""
    source: str = "rule"                   # rule / llm —— 谁发现的


class GraphValidationReport(BaseModel):
    """校验报告（任务书 P15 的输出结构 + 少量执行元信息）。"""

    valid: bool = False
    #: 对"这张图能直接用于教学"的信心。**是分诊用的启发式，不是概率**，
    #: 计算方式见 graph_validator._confidence。
    confidence: float = 0.0
    issues: list[GraphIssue] = Field(default_factory=list)
    recommended_action: str = ACTION_TEACHER_REVIEW
    #: 每一项检查的状态：structure / flow / paths / physics / false_branches / branch_merges
    checks: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    metadata: dict = Field(default_factory=dict)

    # ---------- 辅助 ----------

    def errors(self) -> list[GraphIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    def warnings(self) -> list[GraphIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]

    def by_type(self, issue_type: str) -> list[GraphIssue]:
        return [i for i in self.issues if i.type == issue_type]

    def as_metadata(self) -> dict:
        """写进 `ProblemGraph.metadata["validation"]` 的紧凑形式。

        只留结论与摘要，不塞完整 issue 列表——metadata 会随图一起被
        反复读写，塞太多会让 graph.json 变得难读。
        """
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "checked_at": self.metadata.get("checked_at", ""),
            "checks": self.checks,
            "false_branches": {
                key: value for key, value in self.metadata.get("false_branches", {}).items()
                if key != "findings"
            },
            "branch_merges": {
                "merge_count": len(self.metadata.get("branch_merges", {}).get("merges", [])),
                "unknown_count": self.metadata.get("branch_merges", {}).get("unknown_count", 0),
                "reason": self.metadata.get("branch_merges", {}).get("reason", ""),
            },
            "issue_count": len(self.issues),
            "errors": [f"{i.type}/{i.node or '-'}: {i.description}" for i in self.errors()][:5],
            "warnings": [f"{i.type}/{i.node or '-'}: {i.description}" for i in self.warnings()][:5],
        }


def new_validation_metadata(*, model: str = "", provider: str = "") -> dict:
    return {
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "provider": provider,
    }
