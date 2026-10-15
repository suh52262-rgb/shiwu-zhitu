"""SQL 形态测试 —— 用**假引擎**捕获数据层生成的 SQL 并逐条体检。

## 为什么需要这个文件

本轮把 `teacher_data.py` 里就地硬写的 JSON 信号判据换成契约 helper 时，
**括号少了一个 / 函数名重复了一层**，生成的 SQL 直接语法错误。
后果非常隐蔽：

  `_sql()` 捕获异常 → 返回 **空列表** → 界面显示"暂无数据"，**一切看着正常**。
  实际上库里有 14 条带过程错误的作答，`wrong_attempts()` 静默返回 0 条。

`_sql()` 返回空是刻意的降级设计（一个查询坏掉不该让整页崩），
所以**不能靠"跑起来不报错"来发现这类问题**。必须直接检查 SQL 文本本身。

## 检查什么

1. **括号平衡**（剔除字符串字面量后计数）—— 少括号 / 多括号都会露馅
2. **函数调用形态**：`json_array_length(` / `json_array_elements(` /
   `json_array_elements_text(` 后面**必须紧跟 `COALESCE(`**。
   这一条专治"重构时留下的重复函数名"：
   `json_array_elements_text(   json_array_elements_text(COALESCE(...))`。
3. **不得残留就地硬写的信号列**（与契约测试互补：那条查源码，这条查生成结果）
4. **所有生成的 SQL 都以 SELECT 开头**（只读承诺的机械核对）
"""
from __future__ import annotations

import re

import pytest

import teacher_data as TD

#: 剔除单引号字符串字面量，避免把字面量里的括号/关键字算进来
_LIT = re.compile(r"'(?:[^']|'')*'")


def _strip(sql: str) -> str:
    return _LIT.sub("''", sql)


class _FakeResult:
    def all(self):
        return []


class _FakeConn:
    def __init__(self, sink: list[str]):
        self._sink = sink

    def execute(self, clause, params=None):
        #: `_sql` 会先发两条 SET，随后才是真正的查询 —— 只收 SELECT 那条
        s = str(clause)
        if s.lstrip().upper().startswith("SELECT"):
            self._sink.append(s)
        return _FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeEngine:
    def __init__(self):
        self.sink: list[str] = []

    def connect(self):
        return _FakeConn(self.sink)


@pytest.fixture()
def captured():
    """构造一个 TeacherData 并换上假引擎，返回 (td, 捕获到的 SQL 列表)。

    ⚠️ 内部属性名**必须与 `teacher_data.TeacherData.__init__` 一致**
    （`_engine` / `_engine_error` / `_engine_note` / `_db_error_kind` /
    `_query_error`）。写错名字不会立刻报错 —— 只有在访问对应 property
    时才会 `AttributeError`，所以属性名错了很容易"静默通过"。
    """
    td = TD.TeacherData.__new__(TD.TeacherData)   # 跳过 __init__（它要真库）
    eng = _FakeEngine()
    td._engine = eng
    td._engine_error = ""
    td._engine_note = ""
    td._query_error = ""
    td._query_errors = {}
    td._db_error_kind = ""
    td.backend_url = "http://127.0.0.1:8000"
    from teacher_config import TeacherConfig
    td.config = TeacherConfig()
    return td, eng.sink


#: 所有经由 `_sql` 生成 SQL 的公开方法（逐个调用以捕获其 SQL）
_QUERY_CALLS = [
    ("range_bounds", lambda td: td.range_bounds()),
    ("student_roster", lambda td: td.student_roster()),
    ("weak_ranking", lambda td: td.weak_ranking()),
    ("error_record_ranking", lambda td: td.error_record_ranking()),
    ("covered_students", lambda td: td.covered_students()),
    ("scope_students", lambda td: td.scope_students()),
    ("problem_ranking", lambda td: td.problem_ranking()),
    ("issue_type_ranking", lambda td: td.issue_type_ranking()),
    ("wrong_attempts", lambda td: td.wrong_attempts()),
    ("hint_stats", lambda td: td.hint_stats()),
    ("stuck_nodes", lambda td: td.stuck_nodes()),
    #: ⚠️ `review_store_status` **不在这里**：它走的是
    #: `inspect(engine).get_table_names()`（表存在性探测），
    #: 不生成任何 SELECT，所以不属于"SQL 形态"检查的范畴。
    ("counts", lambda td: td.counts()),
    ("attempts", lambda td: td.attempts()),
    ("attempt_evidence", lambda td: td.attempt_evidence("x")),
    ("demo_student_ids", lambda td: td.demo_student_ids()),
    ("account_readonly_status", lambda td: td.account_readonly_status()),
    ("problem_stats__db", lambda td: td.problem_stats("p", start="2026-01-01")),
    ("student_detail", lambda td: td.student_detail("s", start="2026-01-01")),
]


@pytest.mark.parametrize("name,call", _QUERY_CALLS, ids=[n for n, _ in _QUERY_CALLS])
def test_generated_sql_is_wellformed(captured, name, call):
    td, sink = captured
    call(td)
    assert sink, f"{name} 没有产生任何 SELECT（假引擎没捕获到）"

    for sql in sink:
        #: ⚠️ 括号计数要去掉字面量（`'[]'` 里的括号不该算），
        #: 但涉及**信号列写法**的断言必须用**原文** ——
        #: 去字面量会把 `COALESCE(a.mistakes,'[]'::json)` 变成
        #: `COALESCE(a.mistakes,''::json)`，断言就永远成立不了。
        s = _strip(sql)

        #: ① 括号平衡
        assert s.count("(") == s.count(")"), (
            f"{name} 生成的 SQL 括号不平衡："
            f"左 {s.count('(')} / 右 {s.count(')')}\n{sql}")

        #: ② JSON 函数后面必须紧跟 COALESCE —— 防"函数名重复一层"
        for m in re.finditer(r"(json_array_\w+)\s*\(\s*(\w+)?", s):
            fn, nxt = m.group(1), (m.group(2) or "")
            assert nxt.upper().startswith("COALESCE"), (
                f"{name} 生成的 SQL 里 `{fn}(` 后面不是 COALESCE"
                f"（疑似重构留下重复函数名）：…{s[max(0, m.start()-40):m.end()+60]}…")

        #: ③ 信号列若送进 JSON 函数，**必须**先经 `COALESCE(<col>,'[]'::json)`。
        #:
        #: ⚠️ 注意层次：**源码层**的"是否就地硬写"由
        #: `test_contract_pure.test_teacher_data_uses_contract_predicates` 守
        #: （生成结果里两者同形，无法区分）。
        #: 这里守的是**语义形态**：该列为 NULL 时必须先兜成 `[]`，
        #: 否则整条表达式变 NULL、判断静默为假（而 DB 里空数组可能存成 NULL）。
        #: 直接 `SELECT a.mistakes`（取原值、在 Python 里解析）是**合法的**，
        #: 所以只在"送进 json_array_* 函数"时才要求 COALESCE。
        bare = re.findall(
            rf"json_array_\w+\s*\(\s*a\.(mistakes|weak_concepts)\b", sql)
        assert not bare, (
            f"{name} 的 SQL 把信号列直接送进 JSON 函数、未经 COALESCE：{bare}")

        #: ④ 只读承诺：只发 SELECT
        assert sql.lstrip().upper().startswith("SELECT"), \
            f"{name} 发出了非 SELECT 语句：{sql[:80]}"


def test_signal_columns_only_via_contract(captured):
    """把所有查询的 SQL 拼起来，确认信号列的写法**只有一种形态**。"""
    td, sink = captured
    for _, call in _QUERY_CALLS:
        call(td)
    joined = "\n".join(sink)          #: 原文，保留字面量
    #: 契约 helper 生成的唯一形态
    assert "COALESCE(a.mistakes,'[]'::json)" in joined, "未发现契约 helper 生成的 mistakes 形态"
    assert "COALESCE(a.weak_concepts,'[]'::json)" in joined, "未发现契约 helper 生成的 weak 形态"
    #: 不允许出现省略 COALESCE 的裸写法
    assert not re.search(r"json_array_\w+\(\s*a\.(mistakes|weak_concepts)\b", joined)
