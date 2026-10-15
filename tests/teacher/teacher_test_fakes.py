"""Shared deterministic teacher DB-mode double; no real service connections."""
import teacher_data as TD

#: 替身数据源收到的调用（跨 rerun 累计）。测试据此断言 UI 传给数据层的参数。
RECORD: list[tuple] = []


class _Sentinel:
    """非 None 哨兵：让 `mode`/`filters_supported` 判为 DB 可用，但不连真库。"""
    pass


_SENTINEL = _Sentinel()


class FakeTeacherData(TD.TeacherData):
    """DB 模式替身：可控数据 + 记录数据层收到的筛选参数。

    继承真实 `TeacherData`（拿到 `mode`/`filters_supported`/`environment_consistency`
    /`account_readonly_status`/`_sql`/`class_source` 等），只覆盖 `__init__`
    （跳过真实探测）、`_db_class_probe`（强制回退 JSON 来源）与**数据返回方法**
    （返回可控数据、记录参数）。

    注意：`environment_consistency` / `account_readonly_status` 内部会走 `_sql`，
    而 `_engine` 是哨兵（无 `.connect()`）→ `_sql` 捕获异常返回空 →
    这两者安全返回 `unknown`，不会抛异常。
    """

    #: 测试在构造替身前设置的默认学生名单（类级注入，跨缓存实例共享）。
    default_roster: list[dict] = []

    def __init__(self, config=None):
        self.config = config
        if self.config is None:
            from teacher_config import load_config
            self.config = load_config()
        self.backend_url = "http://127.0.0.1:59998"   # 不可达 → 走 DB 分支
        self._engine = _SENTINEL
        self._engine_error = ""
        self._engine_note = "替身数据库（DB 模式）"
        self._db_error_kind = ""
        self._query_error = ""
        self._query_errors = {}
        #: 可控数据（从类级默认注入）
        self.roster = list(type(self).default_roster)

    # ---- 班级来源：强制回退 JSON（不让真 `class_source` 去 inspect 哨兵） ----

    def _db_class_probe(self):
        return {}, "", "替身：库中无班级归属表"

    # ---- 可插拔数据 ----

    def _record(self, method: str, **kw) -> None:
        RECORD.append((method, dict(kw)))

    def student_roster(self, *, start=None, end=None, student_ids=None):
        self._record("student_roster", start=start, end=end, student_ids=student_ids)
        return list(self.roster)

    def demo_student_ids(self):
        return []

    def range_bounds(self, *, student_ids=None):
        self._record("range_bounds", student_ids=student_ids)
        return ("", "")

    def review_store_status(self):
        return TD.REVIEW_STORE_UNAVAILABLE

    def weak_ranking(self, *, start=None, end=None, student_ids=None,
                     attempt=None, problem_id=None):
        self._record("weak_ranking", start=start, end=end,
                     student_ids=student_ids, attempt=attempt)
        return []

    def error_record_ranking(self, *, start=None, end=None, student_ids=None,
                             attempt=None, problem_id=None):
        self._record("error_record_ranking", start=start, end=end, student_ids=student_ids)
        return []

    def counts(self, *, start=None, end=None, student_ids=None, attempt=None):
        self._record("counts", start=start, end=end, student_ids=student_ids)
        if student_ids == []:
            return {"attempts": 0, "students": 0,
                    "problem_attempts": 0, "weak_flagged": 0}
        return {"attempts": len(self.roster), "students": len(self.roster),
                "problem_attempts": 0, "weak_flagged": 0}

    def hint_stats(self, *, start=None, end=None, student_ids=None,
                   attempt=None, problem_id=None):
        self._record("hint_stats", start=start, end=end, student_ids=student_ids)
        return {}

    def stuck_nodes(self, *, start=None, end=None, student_ids=None,
                    attempt=None, problem_id=None, limit=10):
        self._record("stuck_nodes", start=start, end=end, student_ids=student_ids)
        return []

    def problem_ranking(self, *, start=None, end=None, student_ids=None, attempt=None):
        self._record("problem_ranking", start=start, end=end, student_ids=student_ids)
        return []

    def issue_type_ranking(self, *, start=None, end=None, student_ids=None,
                           attempt=None, problem_id=None):
        self._record("issue_type_ranking", start=start, end=end, student_ids=student_ids)
        return []

    def wrong_attempts(self, *, start=None, end=None, student_ids=None,
                       attempt=None, problem_id=None, limit=300):
        self._record("wrong_attempts", start=start, end=end, student_ids=student_ids)
        return []

    def covered_students(self, *, start=None, end=None, student_ids=None,
                         attempt=None, problem_id=None):
        self._record("covered_students", start=start, end=end, student_ids=student_ids)
        return 0

    def scope_students(self, *, start=None, end=None, student_ids=None, attempt=None):
        self._record("scope_students", start=start, end=end, student_ids=student_ids)
        return 0

    def attempts(self, *, start=None, end=None, student_ids=None, limit=500):
        self._record("attempts", start=start, end=end, student_ids=student_ids)
        return []

    def problems(self):
        return {}

    def problem_figures(self, problem_id):
        return []

    def problem_stats(self, problem_id, *, start=None, end=None, student_ids=None):
        self._record("problem_stats", start=start, end=end, student_ids=student_ids)
        return {}

    def student_detail(self, student_id, *, start=None, end=None):
        self._record("student_detail", start=start, end=end, student_ids=[student_id])
        return {"student_id": student_id, "weak_concepts": [], "attempts": [],
                "wrong": [], "hints": {}, "attempt_count": 0}

    def attempt_evidence(self, attempt_id):
        return {}
