"""Content-addressed graph snapshots and explicit human release records.

Teaching snapshots also live inside StudentState; reading a legacy graph never
writes to the question bank. Governance metadata is separate from graph content.
"""
import hashlib
import json
import os
import re
import threading
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.models.problem_graph import ProblemGraph

GOVERNANCE = {"needs_review", "reviewed_by", "reviewed_at", "review_note"}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def graph_lock(problem_dir: Path):
    """Single-process authoring lock; shared-filesystem multiworker needs external locking."""
    key = str(problem_dir.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with graph_lock(self.root.parent):
            return method(self, *args, **kwargs)
    return wrapped


def graph_content(graph: ProblemGraph | dict) -> dict:
    data = graph.model_dump(mode="json") if isinstance(graph, ProblemGraph) else json.loads(json.dumps(graph))
    data["metadata"] = {k: v for k, v in data.get("metadata", {}).items() if k not in GOVERNANCE}
    return data


def graph_hash(graph: ProblemGraph | dict) -> str:
    payload = json.dumps(graph_content(graph), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class GraphVersions:
    def __init__(self, problem_dir: Path):
        self.root = problem_dir / ".graph_versions"

    @property
    def manifest_path(self):
        return self.root / "manifest.json"

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"candidate": "", "published": "", "reviews": {}, "releases": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _valid_record(record: dict, version: str, identity: str) -> bool:
        if not isinstance(record, dict) or record.get("hash") != version:
            return False
        name = record.get(identity)
        if not isinstance(name, str) or not name.strip() or name.strip().lower() == "unknown":
            return False
        try:
            return datetime.fromisoformat(record.get("date", "")).tzinfo is not None
        except (ValueError, TypeError):
            return False

    def validate_published(self, manifest: dict) -> None:
        version = manifest.get("published", "")
        if not self._valid_record(manifest.get("reviews", {}).get(version), version, "reviewer"):
            raise ValueError("Published graph has no valid matching human review")
        if not any(self._valid_record(record, version, "publisher") for record in manifest.get("releases", [])):
            raise ValueError("Published graph has no valid matching release record")

    @synchronized
    def snapshot(self, graph: ProblemGraph) -> str:
        if graph.problem_id != self.root.parent.name:
            raise ValueError("Graph snapshot problem_id does not match its directory")
        content = graph_content(graph)
        version = graph_hash(content)
        path = self.root / (version + ".json")
        if path.exists():
            self.load(version)  # Never replace an existing immutable snapshot.
        else:
            atomic_json(path, content)
        return version

    def load(self, version: str) -> ProblemGraph:
        if not re.fullmatch(r"[0-9a-f]{64}", version):
            raise ValueError("Invalid graph hash")
        content = json.loads((self.root / (version + ".json")).read_text(encoding="utf-8"))
        if graph_hash(content) != version:
            raise ValueError("Graph snapshot hash mismatch")
        if content.get("problem_id") != self.root.parent.name:
            raise ValueError("Graph snapshot problem_id does not match its directory")
        return ProblemGraph(**content)

    @synchronized
    def candidate(self, graph: ProblemGraph) -> str:
        version = self.snapshot(graph)
        manifest = self.manifest()
        manifest["candidate"] = version
        atomic_json(self.manifest_path, manifest)
        return version

    @synchronized
    def review(self, graph: ProblemGraph, reviewer: str, note: str = "") -> str:
        if not reviewer.strip() or reviewer.strip().lower() == "unknown":
            raise ValueError("An explicit human reviewer is required")
        version = self.candidate(graph)
        manifest = self.manifest()
        manifest["reviews"][version] = {"hash": version, "reviewer": reviewer.strip(), "date": datetime.now(timezone.utc).isoformat(), "note": note}
        atomic_json(self.manifest_path, manifest)
        return version

    @synchronized
    def publish(self, graph: ProblemGraph, publisher: str) -> str:
        if not publisher.strip() or publisher.strip().lower() == "unknown":
            raise ValueError("An explicit publisher is required")
        version = graph_hash(graph)
        manifest = self.manifest()
        if not self._valid_record(manifest["reviews"].get(version), version, "reviewer"):
            raise ValueError("This exact graph content has not been reviewed")
        self.load(version)
        manifest["published"] = version
        manifest["releases"].append({"hash": version, "publisher": publisher.strip(), "date": datetime.now(timezone.utc).isoformat()})
        atomic_json(self.manifest_path, manifest)
        return version
