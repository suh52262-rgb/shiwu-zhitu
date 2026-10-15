"""题目与 Problem Graph 的统一存取入口（P16，任务书第 P16 节）。

任务书要求：不能让整个项目只围绕弹簧题运行，所有 Agent Node 必须通过统一
Repository 获取题目与 Graph，**禁止在 Agent 内写死 `if problem_id == "spring"`**。

目录布局（任务书 P16 指定）：

    data/problems/
    ├── phy_001/
    │   └── problem.json
    └── phy_002/
        ├── problem.json
        ├── graph.json          # 可选；没有就是 graphless 模式
        └── figures/
            └── phy_002_fig1.png

新增一道题 = 新建一个目录 + `problem.json`（+ 可选 `graph.json` 与 `figures/`），
**不需要改任何 LangGraph 核心代码**。`tests/test_problem_repository.py` 里有一条
验收用例专门守住这一点。

设计说明：
- 不做跨请求缓存。题目 JSON 很小，而缓存会让 `save_graph()` 之后读不到新内容，
  也会让"新增题目立刻可用"变得不可预期；需要时在上层加缓存更安全。
- `get_*` 在文件损坏时**抛出** `ProblemDataError`（明确点名文件），
  而 `list_*` 跳过损坏项并告警——一个坏文件不应该让整个题库消失。
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from functools import wraps

from app.config import PROJECT_ROOT
from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph

logger = logging.getLogger(__name__)

DEFAULT_PROBLEMS_DIR = PROJECT_ROOT / "data" / "problems"

PROBLEM_FILE = "problem.json"
GRAPH_FILE = "graph.json"
FIGURES_DIR = "figures"


class ProblemDataError(RuntimeError):
    """题目 / Graph 数据文件存在但无法解析。"""


def _graph_locked(method):
    @wraps(method)
    def wrapped(self, problem_id, *args, **kwargs):
        from app.services.graph_versions import graph_lock
        with graph_lock(self.problem_dir(problem_id)):
            return method(self, problem_id, *args, **kwargs)
    return wrapped


def _release_errors(method):
    @wraps(method)
    def wrapped(self, problem_id):
        try:
            return method(self, problem_id)
        except ProblemDataError:
            raise
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ProblemDataError(f"{self.problem_dir(problem_id)} graph release invalid: {exc}") from exc
    return wrapped


def problems_dir() -> Path:
    """题库根目录，可用环境变量 `PROBLEMS_DIR` 覆盖（便于测试与多套题库切换）。"""
    override = os.environ.get("PROBLEMS_DIR", "").strip()
    return Path(override) if override else DEFAULT_PROBLEMS_DIR


class ProblemRepository:
    """题库仓储：题目、Problem Graph 与插图资源的唯一存取口径。

    不传 `root` 时跟随环境变量 `PROBLEMS_DIR`（每次访问实时解析），
    因此测试或多套题库可以在不改代码、不重建对象的情况下切换目录。
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._explicit_root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        return self._explicit_root if self._explicit_root is not None else problems_dir()

    # ---------------------------------------------------------- 路径

    def problem_dir(self, problem_id: str) -> Path:
        return self.root / problem_id

    def problem_path(self, problem_id: str) -> Path:
        return self.problem_dir(problem_id) / PROBLEM_FILE

    def graph_path(self, problem_id: str) -> Path:
        return self.problem_dir(problem_id) / GRAPH_FILE

    # ---------------------------------------------------------- 读取

    def get_problem(self, problem_id: str) -> Problem | None:
        """加载题目；文件不存在返回 None，内容损坏抛 ProblemDataError。"""
        path = self.problem_path(problem_id)
        if not path.is_file():
            return None
        return Problem(**_read_json(path))

    def get_problem_content(self, problem_id: str) -> str:
        """拼好标题的题干文本，供 LLM 节点使用；找不到题目返回空串。"""
        problem = self.get_problem(problem_id)
        return problem.full_text if problem is not None else ""

    def get_graph(self, problem_id: str) -> ProblemGraph | None:
        """加载 Problem-Solving Graph；没有 graph.json 返回 None（graphless 模式）。"""
        path = self.graph_path(problem_id)
        if not path.is_file():
            return None
        graph = ProblemGraph(**_read_json(path))
        if graph.problem_id and graph.problem_id != problem_id:
            logger.warning(
                "graph.json 里的 problem_id=%s 与目录名 %s 不一致，以目录名为准",
                graph.problem_id,
                problem_id,
            )
            graph.problem_id = problem_id
        return graph

    def has_graph(self, problem_id: str) -> bool:
        """题目是否配备 Problem Graph（P8：graph_mode / graphless_mode 的分流依据）。"""
        return self.graph_path(problem_id).is_file()

    @_release_errors
    @_graph_locked
    def graph_release(self, problem_id: str) -> dict:
        """Actual teaching selection; legacy compatibility never implies review."""
        from app.services.graph_versions import GraphVersions, graph_hash
        versions = GraphVersions(self.problem_dir(problem_id))
        manifest = versions.manifest()
        from app.config import get_settings
        policy = os.environ.get("GRAPH_RELEASE_POLICY", get_settings().graph_release_policy)
        if policy not in {"legacy-demo", "published-only"}:
            raise ProblemDataError("GRAPH_RELEASE_POLICY must be legacy-demo or published-only")
        candidate_hash = manifest.get("candidate", "")
        candidate_status = ("reviewed" if candidate_hash in manifest.get("reviews", {}) else "draft") if candidate_hash else "none"
        if manifest.get("published"):
            versions.validate_published(manifest)
            graph = versions.load(manifest["published"])
            return {"graph": graph, "graph_status": "published", "graph_version": manifest["published"], "graph_policy": policy, "candidate_status": candidate_status, "candidate_version": candidate_hash}
        graph = self.get_graph(problem_id)
        status = "graphless" if graph is None else ("reviewed" if graph_hash(graph) in manifest.get("reviews", {}) else "draft") if versions.manifest_path.exists() else "legacy_unreviewed"
        selected = graph if status == "legacy_unreviewed" and policy == "legacy-demo" else None
        return {"graph": selected, "graph_status": status, "graph_version": graph_hash(selected) if selected else "", "graph_policy": policy, "candidate_status": candidate_status, "candidate_version": candidate_hash}

    def get_teaching_graph(self, problem_id: str) -> ProblemGraph | None:
        return self.graph_release(problem_id)["graph"]

    @_graph_locked
    def publish_graph(self, problem_id: str, *, published_by: str) -> str:
        from app.services.graph_versions import GraphVersions
        graph = self.get_graph(problem_id)
        if graph is None:
            raise ProblemDataError("No candidate graph")
        return GraphVersions(self.problem_dir(problem_id)).publish(graph, published_by)

    def list_problem_ids(self) -> list[str]:
        """返回全部题目 id（按目录名排序）。"""
        if not self.root.is_dir():
            return []
        return sorted(
            path.parent.name
            for path in self.root.glob(f"*/{PROBLEM_FILE}")
            if path.is_file()
        )

    def list_problems(self) -> list[Problem]:
        """返回全部题目；单个文件损坏时跳过并告警，不影响其余题目。"""
        problems: list[Problem] = []
        for problem_id in self.list_problem_ids():
            try:
                problem = self.get_problem(problem_id)
            except ProblemDataError as exc:
                logger.warning("跳过损坏的题目 %s：%s", problem_id, exc)
                continue
            if problem is not None:
                problems.append(problem)
        return problems

    def figure_path(self, problem_id: str, relative: str) -> Path | None:
        """把 `figures/xxx.png` 这样的相对路径解析为磁盘路径。

        只允许指向该题目自己的 figures/ 目录，防止 `../` 越权读取其它文件。
        **problem_id 本身也必须校验**：否则 `phy_001/../phy_002` 会被解析到
        另一道题的目录，而"figures 根"也是由同一个未校验的 problem_id 推出来的，
        等于自己证明自己，越权读取照样成立（审计实测过）。
        """
        if not _is_safe_id(problem_id):
            return None
        candidate = (self.problem_dir(problem_id) / relative).resolve()
        figures_root = (self.problem_dir(problem_id) / FIGURES_DIR).resolve()
        if figures_root not in candidate.parents and candidate != figures_root:
            return None
        return candidate if candidate.is_file() else None

    # ---------------------------------------------------------- 写入

    @_graph_locked
    def save_graph(self, problem_id: str, graph: ProblemGraph) -> Path:
        """保存 Problem Graph（P14 Auto Graph Generator 的落地出口）。

        先写临时文件再原子替换，避免中途失败留下半个 JSON。
        """
        if graph.problem_id and graph.problem_id != problem_id:
            raise ProblemDataError(
                f"Graph 的 problem_id={graph.problem_id} 与目标题目 {problem_id} 不一致"
            )
        graph.problem_id = problem_id
        from app.services.graph_versions import GraphVersions
        GraphVersions(self.problem_dir(problem_id)).candidate(graph)
        target = self.graph_path(problem_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(graph.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        tmp.replace(target)
        logger.info("已保存 Problem Graph：%s", target)
        return target

    @_graph_locked
    def delete_graph(self, problem_id: str) -> bool:
        """删除 Graph（回到 graphless 模式）；不存在返回 False。"""
        path = self.graph_path(problem_id)
        if not path.is_file():
            return False
        path.unlink()
        from app.services.graph_versions import GraphVersions, atomic_json
        versions = GraphVersions(self.problem_dir(problem_id))
        if versions.manifest_path.exists():
            manifest = versions.manifest()
            manifest["candidate"] = ""
            manifest["published"] = ""
            atomic_json(versions.manifest_path, manifest)
        return True

    @_graph_locked
    def mark_reviewed(self, problem_id: str, *, reviewed_by: str = "", note: str = "") -> bool:
        """把图标记为"**人工**已确认"（P14 的 `needs_review` 只由人来清除）。

        为什么单独一个方法：校验器通过 ≠ 人看过。让机器结论自动冒充人工确认，
        正是 `needs_review` 这个字段要防的事。这里要求显式调用，
        并把是谁、什么时候确认的记进 metadata，便于回溯。
        """
        path = self.graph_path(problem_id)
        if not path.is_file():
            return False
        graph = self.get_graph(problem_id)
        from app.services.graph_versions import GraphVersions
        GraphVersions(self.problem_dir(problem_id)).review(graph, reviewed_by, note)
        graph.metadata["needs_review"] = False
        graph.metadata["reviewed_by"] = reviewed_by or "unknown"
        graph.metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if note:
            graph.metadata["review_note"] = note
        self.save_graph(problem_id, graph)
        logger.info("题目 %s 的图已标记为人工确认（by %s）", problem_id, reviewed_by or "unknown")
        return True

    def merge_knowledge_points(self, problem_id: str, points: Iterable[str]) -> list[str]:
        """把新知识点**并入** `problem.json` 的 `knowledge_points`，返回合并后的列表。

        为什么需要：P14 自动建图会产出整题级知识点，但那些只活在
        `graph.json` 的 metadata 里；教师端 / 前端按题目取知识点时看不到它们。

        三条约定：
        - **只增不删**：人工标注过的知识点不会被机器覆盖或重排（新点追加在后面）；
        - **原子写**：与 `save_graph` 一样先写临时文件再替换；
        - 题目不存在时返回 `[]` 且**不创建文件**——题面该由人来写，
          建图工具不该凭空造出一道题。
        """
        wanted = [p.strip() for p in points if p and p.strip()]
        if not wanted:
            return []
        path = self.problem_path(problem_id)
        if not path.is_file():
            logger.info("题目 %s 不存在，跳过知识点回写", problem_id)
            return []

        data = _read_json(path)
        existing = [k for k in data.get("knowledge_points", []) if isinstance(k, str)]
        merged = list(existing)
        added = [k for k in wanted if k not in merged]
        merged.extend(added)
        if not added:
            return merged

        data["knowledge_points"] = merged
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        tmp.replace(path)
        logger.info("已为 %s 补充 %d 个知识点：%s", problem_id, len(added), "、".join(added))
        return merged


def _is_safe_id(problem_id: str) -> bool:
    """problem_id 必须是单层目录名：不含路径分隔符、不是 . / ..、不是绝对路径。

    题号来自请求参数，不能直接当路径片段用。
    """
    if not problem_id or problem_id.strip() != problem_id:
        return False
    if problem_id in (".", ".."):
        return False
    return not any(sep in problem_id for sep in ("/", "\\", ":")) and not problem_id.startswith(".")


def _read_json(path: Path) -> dict:
    try:
        # utf-8-sig：有 BOM 就吃掉，没有则与 utf-8 完全一致。
        # 题库文件是给人手写的，而 Windows 记事本 / VS Code 保存 JSON 时常带 BOM——
        # 用纯 utf-8 读会抛出"Unexpected UTF-8 BOM"，用户完全不知道该怎么修。
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001 统一转成带路径的领域异常
        raise ProblemDataError(f"{path} 解析失败：{exc}") from exc


_REPOSITORY = ProblemRepository()


def get_problem_repository() -> ProblemRepository:
    """进程内默认仓储（路径由 `problems_dir()` 决定）。"""
    return _REPOSITORY
