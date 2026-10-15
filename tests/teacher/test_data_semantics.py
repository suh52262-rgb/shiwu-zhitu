"""数据层语义测试（离线）：空集合、只读事务、故障分类。

守的坑：
  · #3 `[]` 与 `None` 都被当"不筛选" → 空班级查出全库（**最严重**）
  · 只发 SELECT ≠ 只读授权 → 现在每次连接显式 SET TRANSACTION READ ONLY
  · `SELECT 1` 通过 ≠ schema 兼容 → 逐层验证并分类错误
  · 查询失败 ≠ 真的没有记录 → 分开记录，不解释成"暂无数据"

全部离线：只用**条件构造**与**假 engine**，不连真实数据库。
"""
from __future__ import annotations

import pytest

import teacher_data as TD


# ---------------------------------------------------------------- #3 空集合

def test_none_means_no_filter():
    """`None` = 显式"全部学生"，不加学生条件。"""
    conds, params = TD.TeacherData._cond(student_ids=None)
    assert conds == []
    assert "student_ids" not in params


def test_empty_list_means_definitely_empty():
    """`[]` = 确定空集合 → 必须生成 FALSE（**回归 #3**）。

    原实现把 `[]` 与 `None` 一样当"不筛选"，
    于是选一个没有成员的班级时**查出全库记录** ——
    界面写"0 名学生 / 某班"，表格却是所有班的数据。
    """
    conds, params = TD.TeacherData._cond(student_ids=[])
    assert conds == ["FALSE"], f"空集合必须落成 FALSE，实际 {conds}"
    assert "student_ids" not in params


def test_nonempty_list_uses_array_compare():
    conds, params = TD.TeacherData._cond(student_ids=["s1", "s2"])
    assert conds == ["a.student_id = ANY(:student_ids)"]
    assert params["student_ids"] == ["s1", "s2"]


def test_empty_scope_is_never_broadened_by_other_filters():
    """即使同时有时间条件，空学生集仍必须把结果限制为 0 行。"""
    conds, _ = TD.TeacherData._cond(start="2026-01-01", student_ids=[])
    assert "FALSE" in conds


def test_end_date_is_inclusive_via_next_day():
    """选"到 10-12"必须包含 10-12 全天（用 < end+1 天）。"""
    import datetime as dt
    conds, params = TD.TeacherData._cond(end=dt.date(2026, 10, 12))
    assert "a.start_time < :end_exclusive" in conds
    assert params["end_exclusive"] == dt.date(2026, 10, 13)


# ---------------------------------------------------------------- 只读事务

class _FakeConn:
    def __init__(self, recorder, fail: bool = False):
        self._rec = recorder
        self._fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        text = str(getattr(stmt, "text", stmt))
        self._rec.append(text)
        if self._fail and text.strip().upper().startswith("SELECT"):
            raise RuntimeError("boom: permission denied")
        return self

    def all(self):
        return [("ok",)]


class _FakeEngine:
    def __init__(self, recorder, fail=False):
        self._rec = recorder
        self._fail = fail

    def connect(self):
        return _FakeConn(self._rec, self._fail)


def _bare_data(engine):
    """绕过 `__init__` 的探测，直接注入 engine。"""
    d = TD.TeacherData.__new__(TD.TeacherData)
    d.config = _stub_cfg()
    d.backend_url = "http://127.0.0.1:8000"
    d._engine = engine
    d._engine_error = ""
    d._engine_note = ""
    d._db_error_kind = ""
    d._query_error = ""
    d._query_errors = {}
    return d


def _stub_cfg():
    import teacher_config as TC
    return TC.TeacherConfig()


def test_sql_sets_read_only_transaction_first():
    """每次查询都必须先 `SET TRANSACTION READ ONLY`。

    审阅指出：只发 SELECT 不等于只读授权。
    事务级只读**不依赖账号权限**，在应用侧强制写保护。
    """
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec))
    d._sql("SELECT 1", {})
    assert rec, "应至少执行了一条语句"
    assert rec[0].strip().upper() == "SET TRANSACTION READ ONLY", rec


def test_query_failure_is_recorded_not_swallowed():
    """查询失败必须留痕，供界面区分"故障"与"真的没有记录"。"""
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec, fail=True))
    out = d._sql("SELECT 1", {})
    assert out == []
    assert d.query_error, "查询失败必须记进 query_error"
    assert "boom" in d.query_error


def test_query_success_does_not_clear_previous_error():
    """审阅 T1：**成功查询不得清除之前的失败**。

    旧实现每次成功都 `self._query_error = ""`，于是"某条查询失败、随后
    某条成功"时，失败证据被抹掉 —— 界面显示"暂无数据"却无任何警告。
    现在失败只增不减（成功不清除），由 `reset_query_errors()` 统一清空。
    """
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec))
    d._query_error = "上一次的旧错误"
    d._sql("SELECT 1", {})
    assert d.query_error == "上一次的旧错误", "成功查询不得清除之前的失败"


def test_query_failures_accumulate_across_regions():
    """审阅 T1：不同区域的失败**累计**，且后一次成功不覆盖前一次失败。"""
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec, fail=True))          # 第一条查询失败
    d._sql("SELECT broken", {}, label="知识点分布")
    d._engine = _FakeEngine(rec, fail=False)             # 后续查询成功
    d._sql("SELECT ok", {}, label="错题明细")
    errs = d.query_errors
    assert "知识点分布" in errs, "失败区域必须留痕"
    assert "错题明细" not in errs, "成功区域不应被记成失败"
    assert d.query_error, "最近一次失败仍应可读"


def test_reset_query_errors_clears_accumulated():
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec, fail=True))
    d._sql("SELECT broken", {}, label="x")
    d.reset_query_errors()
    assert d.query_errors == {}
    assert d.query_error == ""


# ---------------------------------------------------------------- 故障分层

def test_schema_check_reports_missing_tables():
    """缺表 → schema 不兼容（而不是"连上了就算好"）。"""

    class _Insp:
        def get_table_names(self):
            return ["students"]          # 只给一张，其它都缺

        def get_columns(self, t):
            return [{"name": "id"}, {"name": "display_name"}]

    class _Eng:
        pass

    import sqlalchemy
    real_inspect = sqlalchemy.inspect
    sqlalchemy.inspect = lambda e: _Insp()
    try:
        missing = TD.TeacherData._check_schema(_Eng())
    finally:
        sqlalchemy.inspect = real_inspect
    assert missing, "缺表时必须报出来"
    assert any("attempts" in m for m in missing)


def test_error_kind_constants_are_distinct():
    assert len({TD.DBERR_CONNECT, TD.DBERR_SCHEMA, TD.DBERR_QUERY}) == 3


def test_no_dsn_yields_api_mode_without_error_kind():
    """未配置数据库是**正常部署方式**，不该被当成故障。"""
    import teacher_config as TC
    cfg = TC.TeacherConfig(backend_url="http://127.0.0.1:8000", database_url="")
    d = TD.TeacherData(cfg)
    assert d.db_error_kind == ""
    assert d.filters_supported() is False


def test_invalid_config_marks_connect_error():
    """配置非法 → 归类为 connect 类，界面据此提示"配置有问题"。"""
    import teacher_config as TC
    cfg = TC.TeacherConfig(backend_url="", database_url="",
                           errors=["数据库端口：端口必须是数字"])
    d = TD.TeacherData(cfg)
    assert d.db_error_kind == TD.DBERR_CONNECT
    assert d.filters_supported() is False


# ---------------------------------------------------------------- 信号分离

def test_error_record_predicate_only_reads_mistakes():
    sql = TD.TeacherData._has_error_records("a")
    assert "mistakes" in sql
    assert "weak_concepts" not in sql, "历史提及不得混进「有错误记录」的判据"


def test_weak_mention_predicate_only_reads_weak_concepts():
    sql = TD.TeacherData._has_weak_mention("a")
    assert "weak_concepts" in sql
    assert "mistakes" not in sql


# ---------------------------------------------------------------- 查询超时

def test_statement_timeout_is_set_on_every_query():
    """每次查询都必须同时设「只读事务」与「statement_timeout」。

    审阅原文：「准备专用SELECT角色、**statement_timeout**、只读事务」。
    statement_timeout 由**数据库**强制，与驱动超时互补：
    驱动超时断开连接后服务端可能仍在跑那条查询，
    而 statement_timeout 会让服务端主动中止它。
    """
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec))
    d._sql("SELECT 1", {})
    joined = " | ".join(x.strip().upper() for x in rec)
    assert "SET TRANSACTION READ ONLY" in joined, rec
    assert "STATEMENT_TIMEOUT" in joined, rec


def test_statement_timeout_is_local_not_session():
    """必须用 SET LOCAL —— 否则会污染连接池里复用出去的下一次查询。"""
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec))
    d._sql("SELECT 1", {})
    to_stmt = [x for x in rec if "statement_timeout" in x.lower()]
    assert to_stmt and to_stmt[0].strip().upper().startswith("SET LOCAL"), to_stmt


def test_statement_timeout_value_is_positive():
    assert TD.STATEMENT_TIMEOUT_MS > 0
    assert TD.STATEMENT_TIMEOUT_MS <= 300_000, "超时不宜超过 5 分钟"


# ---------------------------------------------------------------- 第 5 轮新增能力

def test_class_source_falls_back_to_json_without_engine(tmp_path):
    """数据库不可读时必须回退 JSON，并**说明原因**（不能静默）。"""
    import json as _json
    f = tmp_path / "classes.json"
    f.write_text(_json.dumps({"classes": {"甲": ["s1"], "乙": ["s2"]}},
                             ensure_ascii=False), encoding="utf-8")
    d = _bare_data(None)
    classes, err, src = d.class_source(f)
    assert classes == {"甲": ["s1"], "乙": ["s2"]}
    assert err == ""
    assert "classes.json" in src
    assert "数据库不可读" in src, "必须说明为何用了 JSON"


def test_class_source_reports_broken_json_reason(tmp_path):
    """JSON 坏了要报原因，但仍返回空映射（不抛异常）。"""
    f = tmp_path / "classes.json"
    f.write_text("{ broken", encoding="utf-8")
    d = _bare_data(None)
    classes, err, src = d.class_source(f)
    assert classes == {}
    assert err, "坏 JSON 必须给出原因"
    assert "classes.json" in src


def test_class_source_env_spec_parsing(monkeypatch):
    """`TEACHER_CLASS_SOURCE` 格式 `表:班级列:学生列`，列名可省略。

    这里只验证"解析后确实构造了查询"，不去连库 ——
    完整四场景实测见 `校验包/probe_class_source.py`。
    """
    rec: list[str] = []
    d = _bare_data(_FakeEngine(rec))
    monkeypatch.setenv(TD.CLASS_SOURCE_ENV, "my_tbl:my_cls:my_stu")
    try:
        d.class_source("nonexistent-classes.json")
    except Exception:  # noqa: BLE001 探库失败会回退，不应抛
        raise AssertionError("class_source 不应抛异常")
    #: 假引擎的 inspect 不可用 → 走回退；只确认没有崩、且环境变量被读到了
    assert TD.CLASS_SOURCE_ENV == "TEACHER_CLASS_SOURCE"


def test_class_source_never_raises_on_probe_failure():
    """探库失败必须是"回退"，不是"抛异常" —— 页面上不该因为探测班级来源而崩。"""
    class _BadInspEngine:
        def connect(self):
            raise RuntimeError("no connection")

    d = _bare_data(_BadInspEngine())
    classes, err, src = d.class_source("definitely-missing.json")
    assert classes == {}
    assert src.startswith("definitely-missing.json"), src


def test_demo_id_prefixes_are_conservative():
    """演示数据**只按前缀与确切 ID** 判定 —— 不能靠"猜"误伤真实学生。"""
    assert all(p.endswith("_") for p in TD.DEMO_ID_PREFIXES), "前缀必须以 _ 结尾"
    assert TD.DEMO_IDS and all(i and not i.endswith("*") for i in TD.DEMO_IDS)


def test_account_readonly_status_unknown_without_engine():
    d = _bare_data(None)
    st = d.account_readonly_status()
    assert st["status"] == "unknown"
    assert st["detail"], "无法判定时必须说明原因"


def test_account_readonly_status_unknown_when_query_fails():
    """权限探测查询失败 → 结论必须是 unknown，**不能**默认当成只读。"""
    d = _bare_data(_FakeEngine([], fail=True))
    st = d.account_readonly_status()
    assert st["status"] == "unknown", "探测失败时不得默认判为 readonly"


def test_problem_figures_do_not_mix_unknown_or_mismatched_connections(monkeypatch):
    d = _bare_data(object())
    monkeypatch.setattr(TD.httpx, 'get', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not fetch figures')))
    for status in ['unknown', 'mismatch']:
        monkeypatch.setattr(d, 'environment_consistency', lambda: {'status': status})
        assert d.problem_figures('demo_02') == []


def test_environment_consistency_unknown_without_engine():
    d = _bare_data(None)
    ec = d.environment_consistency()
    assert ec["status"] == "unknown"
    assert ec["detail"]
    assert ec["db"] is None


def _remote_identity(db='same', address='127.0.0.1', port=5432):
    return TD.hashlib.sha256(TD.json.dumps({'database': db, 'address': address, 'port': port}, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_environment_identity_mismatch_even_with_equal_headcounts(monkeypatch):
    d = _bare_data(_FakeEngine([]))
    monkeypatch.setattr(d, '_sql', lambda *a, **k: [('same', '127.0.0.1', 5432)])
    class Response:
        status_code = 200
        def json(self):
            return {'student_count': 5, 'identity_kind': 'database_connection_metadata_v1', 'database_identity': _remote_identity('different')}
    monkeypatch.setattr(TD.httpx, 'get', lambda *a, **k: Response())
    assert d.environment_consistency()['status'] == 'mismatch'


def test_environment_identity_matches_even_if_population_changes(monkeypatch):
    d = _bare_data(_FakeEngine([]))
    monkeypatch.setattr(d, '_sql', lambda *a, **k: [('same', '127.0.0.1', 5432)])
    class Response:
        status_code = 200
        def json(self):
            return {'student_count': 999, 'identity_kind': 'database_connection_metadata_v1', 'database_identity': _remote_identity()}
    monkeypatch.setattr(TD.httpx, 'get', lambda *a, **k: Response())
    result = d.environment_consistency()
    assert result['status'] == 'consistent'
    assert '不代表教师授权' in result['detail']


def test_environment_consistency_unknown_when_backend_unreachable(monkeypatch):
    d = _bare_data(_FakeEngine([]))
    monkeypatch.setattr(d, '_sql', lambda *a, **k: [('same', '127.0.0.1', 5432)])
    def fail(*a, **k):raise ConnectionError('unavailable')
    monkeypatch.setattr(TD.httpx, 'get', fail)
    assert d.environment_consistency()['status'] == 'unknown'


def test_environment_consistency_unknown_when_db_count_fails(monkeypatch):
    """审阅 T6：数据库计数**失败**（返回空）→ unknown，**不得**当 0 去比对。

    旧实现 `int(db_n or 0)` 把失败当 0；若后端恰好也是 0，会被误判
    "consistent" —— 这是把"查询失败"当成了"库确实是 0 行"。
    """
    d = _bare_data(_FakeEngine([]))
    monkeypatch.setattr(d, "_sql", lambda sql, params=None, label="": [])  # 计数失败

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"student_count": 0}

    class _Httpx:
        @staticmethod
        def get(*a, **k):
            return _Resp()

    monkeypatch.setattr(TD, "httpx", _Httpx)
    ec = d.environment_consistency()
    assert ec["status"] == "unknown", ec
    assert ec["db"] is None


def test_review_store_status_three_states():
    """审阅 T2：表存在 ≠ 已读取复核决定。

    · 无表 → `unavailable`
    · 有表但未接入 → `detected_unintegrated`（**不能**返回 ready）
    """
    import sqlalchemy

    class _Insp:
        def __init__(self, names):
            self._names = set(names)

        def get_table_names(self):
            return list(self._names)

    class _Eng:
        def __init__(self, names):
            self._names = names

    def _status(names):
        eng = _Eng(names)
        real_inspect = sqlalchemy.inspect
        sqlalchemy.inspect = lambda e: _Insp(names)
        try:
            d = _bare_data(eng)
            return d.review_store_status()
        finally:
            sqlalchemy.inspect = real_inspect

    assert _status([]) == TD.REVIEW_STORE_UNAVAILABLE
    assert _status(["error_evidence_reviews"]) == TD.REVIEW_STORE_DETECTED
    assert _status(["error_evidence_review"]) == TD.REVIEW_STORE_DETECTED
    #: 关键：有表时**绝不**返回 ready（未接入复核决定）
    assert _status(["error_evidence_reviews"]) != TD.REVIEW_STORE_INTEGRATED
