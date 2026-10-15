"""教师端数据访问：**只读**数据库优先，API 保底。

## 为什么要加一条「只读数据库」通道

需求里有几类**纯 API 做不到**的能力：

| 需求 | 仅用 API 为什么做不到 |
|---|---|
| 按时间段 / 最近 14 天 | `/teacher/*` 全部是**无时间参数的全局聚合** |
| 选某一次提交 | `/teacher/students/{id}` 的 attempts **不含时间戳**；`/students/{id}/profile` 含时间但**不含 weak_concepts**。两边拼不起来 |
| 选单个 / 多个学生 | 没有任何接口接受"学生集合" |
| **错题题号** | 后端没有"错题"这个概念，也没有按题目聚合的接口 |
| 按题目 / 类型检索 | 后端只有按知识点聚合那一条路 |

而数据**本来都在库里**：
  · `attempts.weak_concepts` / `.mistakes` —— 带 `start_time`，即"某时刻的薄弱点与错误"
  · `attempts.problem_id` —— 题号，是"错题"的分组键
  · `hint_records.node` / `.effective` —— 节点坐标与三态效果

因此本模块以**只读**方式直接读这些表做筛选与聚合，
历史聚合仅做只读查询；当前错误、既有复盘及原帧通过项目新增的只读教师接口取得，复用服务端证据规则，不自动建表或生成报告。

## 只读是硬约束

本模块**只发 SELECT**，绝不 INSERT / UPDATE / DELETE。
所有 SQL 都是本文件里的可读常量，评审时一眼能核对。

## 接入方式

连接目标由 `teacher_config.load_config()` 决定，支持
「环境变量 > teacher_config.json > 项目内自动探测」三级来源，
因此本包可以独立部署、接入任意的后端与数据库。

## 降级

数据库未配置或连不上时自动切到 API 模式：功能退化为"全局、无筛选"，
由 `filters_supported()` 告知界面 —— 界面必须**如实提示**，
不能假装筛选生效（那会让教师以为看到的是"最近 14 天"，实际是全量）。

## 降级是分层的（审阅要求）

`SELECT 1` 只证明"端口通"。本模块逐层验证并**分类记录原因**：

| 原因 | `db_error_kind` | 对用户意味着什么 |
|---|---|---|
| 连不上 | `connect` | 服务没起 / 地址口令端口错 |
| 结构不对 | `schema`  | 接错了库、或库版本不匹配 |
| 查询失败 | `query`   | 权限不足 / 超时 / SQL 错误 |
| 确实没记录 | —— | **不是故障**，是事实 |

⚠️ 最后两种必须分清：把"查询失败"显示成"暂无数据"，
教师会以为学生的作答丢了。`query_error` 保留最近一次查询失败的原因，
界面据此区分二者。

## 只读是双保险

除了"本文件只发 SELECT"，每次连接还显式
`SET TRANSACTION READ ONLY` —— 即使 SQL 被人改错，写操作也会被数据库拒绝。
这一点**不依赖账号权限**，在应用侧强制（审阅指出只发 SELECT 不等于只读授权）。
"""
from __future__ import annotations

import datetime as _dt
import os
import ssl as _ssl
import hashlib
import json
from pathlib import Path
from typing import Sequence

import httpx

from teacher_config import TeacherConfig, load_config, masked_database_url
#: 统一只读聚合契约 —— 口径与 SQL 谓词的**唯一来源**（见 teacher_contract.py）
from teacher_contract import (
    COL_MISTAKES,
    COL_WEAK_CONCEPTS,
    contract_fingerprint,
    sql_error_records,
    sql_json_elements,
    sql_json_elements_text,
    sql_json_len,
    sql_weak_mention,
)

MODE_DB = "db"
MODE_API = "api"
MODE_DOWN = "down"

#: 数据库不可读的具体原因分类。**必须分开**，因为它们对用户意味着不同的事：
#:   · connect  —— 连不上（服务没起 / 地址口令错 / 端口不对）
#:   · schema   —— 连上了但表或列不对（接错库 / 版本不匹配）
#:   · query    —— 连上、结构也对，但查询失败（权限 / 超时 / SQL 错误）
#:   · notfound —— 连上、结构对、查询成功，就是**真的没有记录**
#: 旧实现把后三种都当"没有数据"，教师会误以为班里的作答丢了。
DBERR_CONNECT = "connect"
DBERR_SCHEMA = "schema"
DBERR_QUERY = "query"

#: 错误证据复核库的三种状态（`TeacherData.review_store_status()` 返回值）。
#: 审阅 T2：**表存在 ≠ 已读取复核决定**。只看到表名就返回 "ready"，
#: 会让界面宣称"可区分 待复核/已确认/已撤销/已解决"，而排行其实
#: 仍读原始 mistakes、没按证据身份与最新决定处理撤销/解决 ——
#: 这是**假的 ready**。所以这里严格区分三档：
#:   · `unavailable`          —— 连复核表都没有，无法接触复核决定
#:   · `detected_unintegrated`—— 表在，但本包**尚未接入**最新复核决定
#:                              （仍按原始 mistakes 统计，未处理撤销/解决）
#:   · `integrated`           —— 已通过规范读接口接入最新复核决定（本包
#:                              当前未实现，保留语义供将来接入后使用）
REVIEW_STORE_UNAVAILABLE = "unavailable"
REVIEW_STORE_DETECTED = "detected_unintegrated"
REVIEW_STORE_INTEGRATED = "integrated"

#: 只读模式下必须存在的表与列。缺任一列即为 schema 不兼容——
#: 只探 `SELECT 1` 只能证明"能连上"，证明不了"这张库是教师端认识的那张"。
#: 审阅已指出这一点：`SELECT 1` 通过后 `_sql` 再出异常，原实现不捕获，
#: 于是 README 承诺的"持续降级"并不成立。
REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "students": ("id", "display_name"),
    "attempts": ("id", "student_id", "problem_id", "start_time", "weak_concepts", "mistakes"),
    "hint_records": ("attempt_id", "hint_level", "effective"),
    "student_profiles": ("student_id", "weak_concepts"),
}

#: 单条查询的服务端超时（毫秒）。
#: 审阅要求「准备专用 SELECT 角色、statement_timeout、只读事务」。
#: 教师端是只读统计，正常查询都在毫秒级；给 15 秒既够用，
#: 又能拦住"慢查询把界面卡死"这种情况。
#: 可通过 `TEACHER_STATEMENT_TIMEOUT_MS` 覆盖（运维按库规模调）。
STATEMENT_TIMEOUT_MS = int(os.environ.get("TEACHER_STATEMENT_TIMEOUT_MS", "15000") or 0) or 15000

#: 演示 / 测试数据的学生 ID 前缀与确切 ID。
#: 审阅第二批 #7 要求「演示 sim_* 与真实学生**明确分源、可筛除**，不默认混算」。
#: 界面据此提供「排除演示数据」开关。
DEMO_ID_PREFIXES = ("sim_",)
DEMO_IDS = ("tester",)

#: 班级归属的数据库来源（可插拔）。
#: 格式 `表名:班级列:学生列`，列名可省略（默认 `class_name` / `student_id`）。
#: 例：`TEACHER_CLASS_SOURCE=student_classes:class_name:student_id`
#: 设了就用它作**权威归属**；没设则自动探测常见表名，最后回退 `classes.json`。
#: 见 `TeacherData.class_source()`。
CLASS_SOURCE_ENV = "TEACHER_CLASS_SOURCE"

#: 预建的 SSL 上下文。`httpx` 每次新建 Client 都会调 `create_ssl_context()`
#: （实测约 300ms），教师端一屏要打多个接口，不预建就是每次交互白付近一秒。
_SSL_CTX = _ssl.create_default_context()


def _ssl_ctx():
    return _SSL_CTX


class TeacherData:
    """教师端数据源。DB 优先，失败降级 API。"""

    def __init__(self, config: TeacherConfig | None = None) -> None:
        self.config = config or load_config()
        self.backend_url = self.config.backend_url
        self._engine = None
        self._engine_error = ""
        self._engine_note = ""
        #: 不可读的**原因分类**（`DBERR_*`）；为空表示 DB 可用。
        self._db_error_kind = ""
        #: 最近一次查询失败的原因。**与"真的没有记录"是两件事**，
        #: 界面据此区分"暂无数据"和"查询失败"。
        self._query_error = ""
        #: **按区域累计的查询失败**（`label -> 失败原因`）。
        #:
        #: 审阅 T1：一次刷新会执行多条查询，`_sql` 若在"成功"时清空
        #: `_query_error`，那么前面某条查询失败、后面某条成功时，
        #: 失败就被抹掉了 —— 界面显示"暂无数据"却无任何警告。
        #: 所以失败**只增不减**（成功不清除），由 `reset_query_errors()`
        #: 在每次刷新开始时重置。
        self._query_errors: dict[str, str] = {}
        self._schema_info: dict = {}
        self._probe()

    # ------------------------------------------------------------ 初始化

    def _probe(self) -> None:
        """分层探测数据库。失败只记原因，不抛异常。

        ## 为什么分三层

        `SELECT 1` 只说明**端口通**；它证明不了：
          · 连的是不是**教师端认识的那张库**（schema 可能不匹配）
          · 账号**有没有 SELECT 权限**、查询会不会超时
        审阅实测：`SELECT 1` 通过后 `_sql` 仍可能异常，而原实现不捕获 →
        "已连接"的假象 + 查询静默变空。所以这里逐层验证。

        各层失败都**降级**（不抛异常）—— 降级是设计内的，
        但降级原因会**如实带到界面上**。
        """
        dsn = self.config.database_url
        if not dsn:
            #: 未配置不是错误 —— "只用后端接口"是一种正常部署方式。
            #: 但要说清楚，否则用户会以为筛选功能坏了。
            if self.config.errors:
                self._engine_note = "数据库配置非法（见连接配置）"
                self._db_error_kind = DBERR_CONNECT
            else:
                self._engine_note = "未配置数据库连接，已降级为后端接口模式（无时间/学生筛选）"
            return

        #: 连接参数交给 SQLAlchemy；`connect_timeout` 避免卡死。
        connect_args = {"connect_timeout": 5} if dsn.startswith("postgresql") else {}
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(dsn, pool_pre_ping=True, connect_args=connect_args)
        except Exception as exc:  # noqa: BLE001 DSN 本身不合法
            self._engine_error = f"{type(exc).__name__}: {exc}"
            self._engine_note = "数据库连接串无法解析，已降级为后端接口模式"
            self._db_error_kind = DBERR_CONNECT
            return

        # ---- 第 1 层：能连上吗 ----
        try:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 连不上是预期内的降级场景
            self._engine_error = f"{type(exc).__name__}: {exc}"
            self._engine_note = "数据库连接失败，已降级为后端接口模式"
            self._db_error_kind = DBERR_CONNECT
            return

        # ---- 第 2 层：schema 是不是教师端认识的那张 ----
        missing = self._check_schema(engine)
        if missing:
            self._engine_error = "缺少表/列：" + "；".join(missing)
            self._engine_note = "数据库结构不兼容（可能接错了库），已降级为后端接口模式"
            self._db_error_kind = DBERR_SCHEMA
            return

        self._engine = engine
        self._engine_note = "数据库连接正常（只读，schema 已核对）"

    @staticmethod
    def _check_schema(engine) -> list[str]:
        """核对必要表与列是否都在。返回缺失项（空列表 = 通过）。"""
        from sqlalchemy import inspect

        try:
            insp = inspect(engine)
            have_tables = set(insp.get_table_names())
        except Exception:  # noqa: BLE001 取不到表清单即视为不合格
            return ["无法读取表清单"]
        missing: list[str] = []
        for table, columns in REQUIRED_SCHEMA.items():
            if table not in have_tables:
                missing.append(f"{table}（整表缺失）")
                continue
            try:
                have_cols = {c["name"] for c in insp.get_columns(table)}
            except Exception:  # noqa: BLE001
                missing.append(f"{table}（列信息不可读）")
                continue
            lack = [c for c in columns if c not in have_cols]
            if lack:
                missing.append(f"{table}.{'/'.join(lack)}")
        return missing

    @property
    def mode(self) -> str:
        if self._engine is not None:
            return MODE_DB
        return MODE_API if self._api_ok() else MODE_DOWN

    @property
    def db_error(self) -> str:
        return self._engine_error

    @property
    def db_error_kind(self) -> str:
        """`DBERR_*` 之一；空串表示 DB 可用。"""
        return self._db_error_kind

    @property
    def db_note(self) -> str:
        return self._engine_note

    @property
    def query_error(self) -> str:
        """最近一次查询失败的原因；空串表示最近一次查询正常。"""
        return self._query_error

    @property
    def query_errors(self) -> dict[str, str]:
        """**按区域累计**的查询失败（`label -> 失败原因`）。

        与 `query_error`（只留最近一次）不同，这个字典**只增不减**，
        直到下一次 `reset_query_errors()`。界面据此能同时看到
        "知识点分布查询失败" + "错题明细查询成功"这样的**混合状态**，
        而不是被最后一条成功查询抹掉失败证据（审阅 T1）。
        """
        return dict(self._query_errors)

    def reset_query_errors(self) -> None:
        """清空累计的查询失败（每次页面刷新开始时调用）。"""
        self._query_error = ""
        self._query_errors = {}

    @property
    def database_display(self) -> str:
        """口令已隐藏的连接串，供界面回显。"""
        return masked_database_url(self.config)

    def filters_supported(self) -> bool:
        """时间 / 学生集合 / 单次提交筛选是否可用（仅 DB 模式支持）。

        ⚠️ **schema 不兼容时必须返回 False** —— 连接通了但表不对，
        筛选查询会全部失败；此时若声称"筛选可用"，
        用户会看到一屏空数据却以为"班里没作答"。
        """
        return self._engine is not None

    def _api_ok(self) -> bool:
        try:
            r = httpx.get(f"{self.backend_url}/health", timeout=5, verify=_SSL_CTX)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------ 查询底座

    @staticmethod
    def _cond(*, start=None, end=None, student_ids: Sequence[str] | None = None,
              attempt=None, problem_id=None) -> tuple[list[str], dict]:
        """构造条件列表与参数（表别名固定为 `a`，指 attempts）。

        `end` 用 `< end + 1 天`：教师选"到 10-12"时**应当包含 10-12 全天**，
        用 `a.start_time <= end` 会把当天 00:00 之后的记录全漏掉。

        ## `student_ids` 的三态语义（**这里曾经是个严重缺陷**）

        | 传入 | 含义 | 生成的条件 |
        |---|---|---|
        | `None`    | 显式"全部学生"（不筛选） | 不加条件 |
        | 非空列表  | 就这些学生             | `a.student_id = ANY(...)` |
        | **空列表** | **确定为空集合**       | **`FALSE`（必返回 0 行）** |

        ⚠️ 原实现把 `[]` 和 `None` 都当"不筛选"（`if student_ids:` 对空列表为假），
        于是**选了一个没有成员的班级时，反而查出全库记录** ——
        界面写着"0 名学生 / 某班"，数据却是所有人的。
        审阅已用条件生成实测到这一点，这是会直接误导教学决策的错误。

        所以现在**空列表必须落成 `FALSE`**。调用方要"全体"就传 `None`，
        不能传 `[]` —— 二者语义不同。
        """
        conds: list[str] = []
        params: dict = {}
        if start:
            conds.append("a.start_time >= :start")
            params["start"] = start
        if end:
            conds.append("a.start_time < :end_exclusive")
            params["end_exclusive"] = end + _dt.timedelta(days=1)
        if student_ids is not None:
            ids = list(student_ids)
            if ids:
                conds.append("a.student_id = ANY(:student_ids)")
                params["student_ids"] = ids
            else:
                #: 确定为空集合。写 `FALSE` 而不是 `= ANY(ARRAY[])`：
                #: 后者在部分方言/参数绑定下行为不一致，`FALSE` 无歧义。
                conds.append("FALSE")
        if attempt:
            conds.append("a.id = :attempt")
            params["attempt"] = attempt
        if problem_id:
            conds.append("a.problem_id = :problem_id")
            params["problem_id"] = problem_id
        return conds, params

    def _where(self, **kw) -> tuple[str, dict]:
        conds, params = self._cond(**kw)
        return (" AND ".join(conds) if conds else "TRUE"), params

    def _sql(self, sql: str, params: dict, *, label: str = "") -> list:
        """执行**只读**查询。

        ## 只读是双保险

        1. 每次连接显式 `SET TRANSACTION READ ONLY` —— 即使本文件的 SQL
           被人改错了，写操作也会被数据库直接拒绝（而不是悄悄改坏数据）。
        2. 本文件所有 SQL 都是常量且只发 SELECT，评审可逐条核对。

        审阅指出"只发 SELECT 不等于只读授权"：连接本身没设只读事务，
        且无法证明配置账号只有 SELECT 权限。事务级只读正好补上这一点 ——
        它**不依赖账号权限**，在应用侧强制。

        ## 查询超时（审阅点名的第三项）

        审阅要求「准备专用 SELECT 角色、`statement_timeout`、只读事务」。
        本方法补上 `statement_timeout`：**在连接级设置一个上限**，
        避免一条慢查询把界面卡死、把连接池占满。

        `statement_timeout` 由数据库强制，**不依赖驱动层超时** ——
        驱动超时断开连接后服务端可能仍在跑那条查询，而 statement_timeout
        会让服务端主动中止它。两者互补，都要有。

        ## 查询失败必须留痕，且**成功不得抹掉失败**（审阅 T1）

        失败时记进 `_query_errors[label]` 并返回空列表。返回空是"降级"，
        但**界面要能区分**"查询失败"与"真的没有记录" —— 前者是故障，
        后者是事实，混在一起会让教师以为数据丢了。

        ⚠️ **成功不清除失败**：一次刷新执行多条查询（知识点 / 提示 / 错题 /
        人数……），如果某条失败后、后续某条成功就把 `_query_error` 清空，
        那么"这一个统计失败、其他统计成功"的混合状态就会被掩盖 ——
        失败区域显示成"暂无数据"，却没有任何警告。所以失败**只增不减**，
        由 `reset_query_errors()` 在每次刷新开始统一清空。

        `label` 是**区域名**（如"知识点分布""错题明细"），用于让界面
        按区域分别提示哪些统计失败、哪些成功。
        """
        from sqlalchemy import text

        try:
            with self._engine.connect() as c:
                c.execute(text("SET TRANSACTION READ ONLY"))
                #: 超时只对**本次查询**生效（SET LOCAL 随事务结束即失效），
                #: 不会污染连接池里复用出去的下一次查询。
                c.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
                rows = list(c.execute(text(sql), params).all())
            #: 成功**不**清空 `_query_error` / `_query_errors` —— 见方法说明。
            return rows
        except Exception as exc:  # noqa: BLE001 查询失败不该把整页打崩
            msg = f"{type(exc).__name__}: {exc}"
            self._query_error = msg
            #: 防御：用 `__new__` 绕过 `__init__` 的测试替身可能没初始化
            #: `_query_errors` —— 就地补一个，避免 AttributeError。
            if not hasattr(self, "_query_errors"):
                self._query_errors = {}
            self._query_errors[label or "_default"] = msg
            return []

    @staticmethod
    def _has_error_records(col: str = "a") -> str:
        """「这次作答有过程错误记录」的判据（只看 `mistakes`）。

        ## 为什么不再把 `weak_concepts` 也算进来

        原实现（`_has_mistakes`）把 `mistakes 非空 或 weak_concepts 非空`
        都算作"有问题的作答"。审阅指出这是**把两种信号混算**：

        | 信号 | 含义 | 性质 |
        |---|---|---|
        | `mistakes` | 本次作答过程中模型记录的错误 | 过程证据 |
        | `weak_concepts` | 跨时间累计的知识点并集 | **历史提及**，只增不减 |

        混在一起的结果是：一个学生只要历史上被标记过某个薄弱点，
        **之后每一道题都会被算成"有问题"**，即使这次做得完全正确。
        所以两者必须分开计数、分开展示。

        ⚠️ 也**不能用 `final_correct` 判错**：实测演示数据 7 次作答
        `final_correct` 全为 true（学生最终都做完了），按它会得到"零错题"。

        ⚠️ 谓词本身**只在 `teacher_contract.py` 定义一处**（统一只读聚合契约），
        本方法只做转发 —— 避免"两处各写一遍、迟早不一致"。
        """
        return sql_error_records(col)

    @staticmethod
    def _has_weak_mention(col: str = "a") -> str:
        """「这次作答被标记过薄弱知识点」的判据（只看 `weak_concepts`）。

        与 `_has_error_records` 分开 —— 见那个方法的说明。
        谓词同样只来自契约模块。
        """
        return sql_weak_mention(col)

    # ------------------------------------------------------------ 学生

    def student_roster(self, *, start=None, end=None,
                       student_ids: Sequence[str] | None = None) -> list[dict]:
        """全部学生 + **该范围内**的作答 / 错题作答 / 弱项 / 提示计数。

        四个计数子查询共用同一份条件，因此 `params` 传一份即可。
        """
        if self._engine is None:
            return []
        where, params = self._where(start=start, end=end, student_ids=student_ids)
        has_m = self._has_error_records("a")
        rows = self._sql(
            "SELECT s.id, COALESCE(s.display_name, ''), "
            " (SELECT count(*) FROM attempts a "
            f"  WHERE a.student_id = s.id AND ({where})) AS n_att, "
            " (SELECT count(*) FROM attempts a "
            f"  WHERE a.student_id = s.id AND ({where}) AND {has_m}) AS n_bad, "
            " (SELECT COALESCE(sum(" + sql_json_len("a", COL_WEAK_CONCEPTS) + "),0) "
            f"  FROM attempts a WHERE a.student_id = s.id AND ({where})) AS n_weak, "
            " (SELECT count(*) FROM hint_records h JOIN attempts a ON h.attempt_id = a.id "
            f"  WHERE a.student_id = s.id AND ({where})) AS n_hint "
            "FROM students s ORDER BY s.id",
            params,
            label="学生名单",
        )
        return [
            {
                "student_id": r[0],
                "display_name": r[1] or r[0],
                "attempt_count": int(r[2] or 0),
                "problem_attempt_count": int(r[3] or 0),
                "weak_count": int(r[4] or 0),
                "hint_count": int(r[5] or 0),
            }
            for r in rows
        ]

    # ------------------------------------------------------------ 弱项排行

    def weak_ranking(self, *, start=None, end=None,
                     student_ids: Sequence[str] | None = None,
                     attempt=None, problem_id=None) -> list[dict]:
        """**历史画像提及**排行（`attempts.weak_concepts`）。

        ⚠️ 这不是"当前薄弱知识点排名"，也不是"已确认的错误"。
        它是**跨时间累计的历史画像**（并集只增不减）：
        学生后来掌握了也不会移除，被撤销/已解决的记录也还在里面。

        因此每行都带 `signal_type = "historical_mention"` 与
        `semantic_validated = False`，界面必须**如实标注**为"历史提及"，
        不得包装成"当前薄弱"或"反复出错"。真正的错误证据需要后端
        按生产证据规则聚合并带复核状态，本包（只读、不改后端）拿不到 ——
        见 `error_record_ranking` 与 `review_store_status`。

        DB 模式：支持时间 / 学生集合 / 单次提交 / 题目筛选。
        API 模式：退化为全局口径（`/teacher/knowledge-points/stats`），无筛选。
        """
        if self._engine is not None:
            where, params = self._where(
                start=start, end=end, student_ids=student_ids,
                attempt=attempt, problem_id=problem_id,
            )
            rows = self._sql(
                "SELECT kp, count(*) AS mentions, "
                "       count(DISTINCT a.student_id) AS students "
                "FROM attempts a, "
                f"     {sql_json_elements_text('a', COL_WEAK_CONCEPTS)} AS kp "
                f"WHERE {where} "
                "GROUP BY kp ORDER BY mentions DESC, students DESC, kp",
                params,
                label="知识点分布",
            )
            return [
                {"knowledge_point": r[0], "mention_count": int(r[1] or 0),
                 "student_count": int(r[2] or 0),
                 #: 口径标识 —— 界面据此选措辞，不靠调用方"记得"
                 "signal_type": "historical_mention",
                 "semantic_validated": False}
                for r in rows
            ]
        try:
            r = httpx.get(f"{self.backend_url}/teacher/knowledge-points/stats",
                          params={"limit": 200}, timeout=20, verify=_SSL_CTX)
            if r.status_code == 200:
                data = r.json() or {}
                #: 若后端是**生产版**（带错误证据聚合），字段会比演示版丰富。
                #: 有就用它 —— 那是按生产证据规则算出来的，口径更可信。
                # Keep historical profile mentions in this channel. Current error
                # evidence and review decisions use current_error_evidence().
                #: 演示版后端只有历史画像 —— 如实标注口径，不冒充"当前错误"
                return [
                    {**dict(x), "signal_type": "historical_mention",
                     "semantic_validated": False}
                    for x in (data.get("knowledge_points") or [])
                ]
        except Exception:  # noqa: BLE001
            pass
        return []

    def error_record_ranking(self, *, start=None, end=None,
                             student_ids: Sequence[str] | None = None,
                             attempt=None, problem_id=None) -> list[dict]:
        """**模型记录的过程错误**排行（`attempts.mistakes`）。

        与 `weak_ranking` 是**两种不同的信号**，刻意分开：
          · `weak_concepts` —— 跨时间累计的知识点并集（历史提及）
          · `mistakes`      —— 每次作答过程中模型记录的错误条目
            （含出错节点 / 错误性质 / 描述）

        后者带节点与性质，是"错在哪一步、错成什么性质"的直接证据；
        前者只是知识点名字的累计。

        ⚠️ `mistakes` 条目里**没有知识点字段**（只有 node / issue_type /
        issue / severity），所以按知识点聚合**在现有数据上做不到** ——
        这里按 `issue_type` + `node` 聚合，不硬凑一个知识点出来。
        """
        if self._engine is None:
            return []
        where, params = self._where(
            start=start, end=end, student_ids=student_ids,
            attempt=attempt, problem_id=problem_id,
        )
        rows = self._sql(
            "SELECT COALESCE(NULLIF(m->>'issue_type',''),'（未标注性质）') AS t, "
            "       COALESCE(NULLIF(m->>'node',''),'（未标注节点）') AS nd, "
            "       count(*) AS n, "
            "       count(DISTINCT a.student_id) AS students, "
            "       count(DISTINCT a.problem_id) AS problems "
            f"FROM attempts a, {sql_json_elements('a', COL_MISTAKES)} m "
            f"WHERE {where} "
            "GROUP BY t, nd ORDER BY n DESC, t, nd",
            params,
            label="错误记录排行",
        )
        return [
            {"issue_type": r[0], "node": r[1], "count": int(r[2] or 0),
             "student_count": int(r[3] or 0), "problem_count": int(r[4] or 0),
             "signal_type": "model_error_record"}
            for r in rows
        ]

    def review_store_status(self) -> str:
        """错误证据复核库的状态（三档，见模块级 `REVIEW_STORE_*` 常量）。

        现状：本演示环境**没有**复核表；即便将来接的库里**有**该表，
        本包也只读原始 `mistakes`、**没有**按证据身份与最新决定
        处理"已撤销 / 已解决" —— 所以此时必须返回 `detected_unintegrated`
        （"检测到表，尚未接入复核状态"），**不能**返回 `ready`。

        只有真正通过规范读接口读到了最新复核决定并用于统计时，
        才返回 `integrated`。本包当前不实现那条接口，因此不会返回它。
        """
        if getattr(self, '_review_scope', {}).get('review_store_status') == 'ready':
            return REVIEW_STORE_INTEGRATED
        if self._engine is None:
            return REVIEW_STORE_UNAVAILABLE
        try:
            from sqlalchemy import inspect
            names = set(inspect(self._engine).get_table_names())
        except Exception:  # noqa: BLE001
            return REVIEW_STORE_UNAVAILABLE
        if {"error_evidence_reviews", "error_evidence_review"} & names:
            #: 表在，但本包尚未接入复核决定 —— 如实标注，不冒充 ready。
            return REVIEW_STORE_DETECTED
        return REVIEW_STORE_UNAVAILABLE

    def _teacher_get(self, route, params=None):
        if self._engine is not None:
            identity = self.environment_consistency()
            if identity['status'] != 'consistent':
                raise ValueError('DB/API连接身份未匹配，已停止混合证据')
        response = httpx.get(self.backend_url + route, params=params, timeout=10, verify=_SSL_CTX)
        if response.status_code != 200:
            raise ValueError(f'后端证据接口返回{response.status_code}')
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError('证据响应格式异常')
        return payload

    def current_error_evidence(self, *, start=None, end=None, student_ids=None, attempt=None, problem_id=None):
        params = {}
        if start: params['start'] = start.isoformat()
        if end: params['end'] = end.isoformat()
        if student_ids is not None:
            if student_ids: params['student_ids'] = list(student_ids)
            else: params['empty_scope'] = 'true'
        if attempt: params['attempt_id'] = attempt
        if problem_id: params['problem_id'] = problem_id
        try:
            self._review_scope = self._teacher_get('/teacher/error-evidence', params)
        except Exception as exc:
            self._review_scope = {'review_store_status': 'unavailable', 'counts': None, 'evidence': [],
                                  'error': str(exc), 'legacy_attempt_count': None}
        return dict(self._review_scope)

    def covered_students(self, *, start=None, end=None,
                         student_ids: Sequence[str] | None = None,
                         attempt=None, problem_id=None) -> int:
        """**精确的**薄弱点覆盖人数：`COUNT(DISTINCT student_id)`。

        审阅实测的问题：原实现用"各知识点涉及学生数的**最大值**"当作覆盖人数。
        A 学生只涉及动量、B 学生只涉及电容，两个知识点各 1 人 →
        原实现算出"2 人中 1 人（50%）"，正确答案是 **2 人（100%）**。

        根本原因是**按知识点聚合后的行里已经没有学生名单**，
        取 max / sum 都无法还原并集 —— 只能**单独查一次**。
        这就必须用与其它指标**同一份筛选条件**，否则又会出现
        "两个数字来自不同范围"的口径矛盾。
        """
        if self._engine is None:
            return 0
        where, params = self._where(
            start=start, end=end, student_ids=student_ids,
            attempt=attempt, problem_id=problem_id,
        )
        rows = self._sql(
            "SELECT count(DISTINCT a.student_id) FROM attempts a "
            f"WHERE {where} AND {sql_weak_mention('a')}",
            params,
            label="覆盖面",
        )
        return int(rows[0][0] or 0) if rows else 0

    def scope_students(self, *, start=None, end=None,
                       student_ids: Sequence[str] | None = None,
                       attempt=None) -> int:
        """范围内**有过作答**的学生数（覆盖面的分母）。

        分母与分子必须同源：都用"该范围内 attempts 里出现过的学生"，
        不能一边用 student_profiles 的表规模、一边用 attempts 的去重学生数。
        """
        if self._engine is None:
            return 0
        where, params = self._where(
            start=start, end=end, student_ids=student_ids, attempt=attempt)
        rows = self._sql(
            f"SELECT count(DISTINCT a.student_id) FROM attempts a WHERE {where}",
            params,
            label="覆盖面",
        )
        return int(rows[0][0] or 0) if rows else 0

    # --------------------------------------------------- 平行检索键：题目 / 类型

    def problem_ranking(self, *, start=None, end=None,
                        student_ids: Sequence[str] | None = None,
                        attempt=None) -> list[dict]:
        """**按题目**聚合 —— 平行于"按知识点"的检索维度。

        每题给出：作答次数、**有过程错误记录的**作答次数、错误条数、涉及学生数。
        "有过程错误记录"的口径见 `_has_error_records`（只看 `mistakes`，
        **不含**历史 weak_concepts —— 后者会把"历史上弱过"的学生
        在每道题上都算成有问题）。
        """
        if self._engine is None:
            return []
        where, params = self._where(
            start=start, end=end, student_ids=student_ids, attempt=attempt)
        has_m = self._has_error_records("a")
        rows = self._sql(
            "SELECT a.problem_id, "
            " count(*) AS attempts, "
            f" sum(CASE WHEN {has_m} THEN 1 ELSE 0 END) AS flagged, "
            " sum(COALESCE(" + sql_json_len("a", COL_MISTAKES) + ",0)) AS mistakes, "
            " count(DISTINCT a.student_id) AS students "
            "FROM attempts a "
            f"WHERE {where} "
            "GROUP BY a.problem_id ORDER BY flagged DESC, mistakes DESC, a.problem_id",
            params,
            label="题目检索",
        )
        return [
            {
                "problem_id": r[0],
                "attempt_count": int(r[1] or 0),
                "flagged_count": int(r[2] or 0),
                "mistake_count": int(r[3] or 0),
                "student_count": int(r[4] or 0),
            }
            for r in rows
        ]

    def issue_type_ranking(self, *, start=None, end=None,
                          student_ids: Sequence[str] | None = None,
                          attempt=None, problem_id=None) -> list[dict]:
        """**按错误类型**聚合 —— 平行于"按知识点"的另一个检索维度。

        ⚠️ 统计的是 `mistakes[].issue_type`（`concept` / `method` 等
        **错误性质**），**不是知识点**。两者常被混淆，界面上会明确标注。
        """
        if self._engine is None:
            return []
        where, params = self._where(
            start=start, end=end, student_ids=student_ids,
            attempt=attempt, problem_id=problem_id,
        )
        rows = self._sql(
            "SELECT m->>'issue_type' AS t, count(*) AS n, "
            "       count(DISTINCT a.student_id) AS students, "
            "       count(DISTINCT a.problem_id) AS problems "
            f"FROM attempts a, {sql_json_elements('a', COL_MISTAKES)} m "
            f"WHERE COALESCE(m->>'issue_type','') <> '' AND ({where}) "
            "GROUP BY t ORDER BY n DESC, t",
            params,
            label="错误类型检索",
        )
        return [
            {"issue_type": r[0], "count": int(r[1] or 0),
             "student_count": int(r[2] or 0), "problem_count": int(r[3] or 0)}
            for r in rows
        ]

    # ------------------------------------------------------------ 错题

    def wrong_attempts(self, *, start=None, end=None,
                       student_ids: Sequence[str] | None = None,
                       attempt=None, problem_id=None, limit: int = 300) -> list[dict]:
        """**错题明细** —— 逐条列出「谁 · 哪道题 · 错在哪」。

        "展示错题题号"的数据源。每条含：题号、学生、时间、错误类型、
        出错节点、薄弱知识点、错误描述。
        """
        if self._engine is None:
            return []
        where, params = self._where(
            start=start, end=end, student_ids=student_ids,
            attempt=attempt, problem_id=problem_id,
        )
        params["lim"] = limit
        rows = self._sql(
            "SELECT a.id, a.student_id, a.problem_id, a.start_time, a.end_time, "
            " a.current_node, "
            " COALESCE((SELECT string_agg(DISTINCT m->>'issue_type', '、') "
            f"   FROM {sql_json_elements('a', COL_MISTAKES)} m "
            "   WHERE COALESCE(m->>'issue_type','') <> ''), '') AS types, "
            " COALESCE((SELECT string_agg(DISTINCT NULLIF(m->>'node',''), '、') "
            f"   FROM {sql_json_elements('a', COL_MISTAKES)} m), '') AS nodes, "
            " COALESCE((SELECT string_agg(DISTINCT NULLIF(m->>'issue',''), '；') "
            f"   FROM {sql_json_elements('a', COL_MISTAKES)} m), '') AS issues, "
            " COALESCE((SELECT string_agg(DISTINCT x, '、') "
            f"   FROM {sql_json_elements_text('a', COL_WEAK_CONCEPTS)} x), '') AS weak, "
            f" {sql_json_len('a', COL_MISTAKES)} AS n_mistakes "
            "FROM attempts a "
            f"WHERE ({where}) AND {self._has_error_records('a')} "
            "ORDER BY a.start_time DESC LIMIT :lim",
            params,
            label="错题明细",
        )
        out = []
        for r in rows:
            st, et = r[3], r[4]
            out.append({
                "attempt_id": r[0],
                "student_id": r[1],
                "problem_id": r[2],
                "start_display": st.strftime("%m-%d %H:%M") if st else "—",
                "duration_s": round((et - st).total_seconds(), 1) if (st and et) else None,
                "current_node": r[5] or "",
                "issue_types": r[6] or "",
                "nodes": r[7] or "",
                "issues": r[8] or "",
                "weak_text": r[9] or "",
                "mistake_count": int(r[10] or 0),
            })
        return out

    # ------------------------------------------------------------ 提示 / 卡点

    def hint_stats(self, *, start=None, end=None,
                   student_ids: Sequence[str] | None = None,
                   attempt=None, problem_id=None) -> dict:
        """提示三态统计。`effective` 为 NULL 时单列为 `undecided`。"""
        if self._engine is None:
            return {}
        where, params = self._where(
            start=start, end=end, student_ids=student_ids,
            attempt=attempt, problem_id=problem_id,
        )
        rows = self._sql(
            "SELECT count(*) AS total, "
            " sum(CASE WHEN h.effective THEN 1 ELSE 0 END) AS eff, "
            " sum(CASE WHEN h.effective IS FALSE THEN 1 ELSE 0 END) AS ineff, "
            " sum(CASE WHEN h.effective IS NULL THEN 1 ELSE 0 END) AS undec "
            "FROM hint_records h JOIN attempts a ON h.attempt_id = a.id "
            f"WHERE {where}",
            params,
            label="提示统计",
        )
        if not rows:
            return {}
        total, eff, ineff, undec = (int(x or 0) for x in rows[0])
        decided = eff + ineff
        return {
            "total": total,
            "effective": eff,
            "ineffective": ineff,
            "undecided": undec,
            #: 分母是**可判定**条数 —— 未判定不混入。
            #: 把未判定当无效会系统性地低估提示效果。
            "effectiveness": (eff / decided) if decided else None,
        }

    def stuck_nodes(self, *, start=None, end=None,
                    student_ids: Sequence[str] | None = None,
                    attempt=None, problem_id=None, limit: int = 10) -> list[dict]:
        """卡住节点排行。

        两个来源分开统计，**不混成一个数**：
          · `from_hint`    —— 在该节点下发过提示（口径：发提示 = 学生卡住）
          · `from_mistake` —— 在 mistakes 里被记为出错节点
            （有些错误没触发提示，但明确记了节点，同样重要）
        """
        if self._engine is None:
            return []
        where, params = self._where(
            start=start, end=end, student_ids=student_ids,
            attempt=attempt, problem_id=problem_id,
        )
        params["lim"] = limit
        rows = self._sql(
            "SELECT node, sum(from_hint) AS h, sum(from_mistake) AS m, "
            "       sum(from_hint + from_mistake) AS total FROM ( "
            "  SELECT h.node AS node, count(*) AS from_hint, 0 AS from_mistake "
            "  FROM hint_records h JOIN attempts a ON h.attempt_id = a.id "
            f"  WHERE h.node <> '' AND ({where}) GROUP BY h.node "
            "  UNION ALL "
            "  SELECT NULLIF(m->>'node','') AS node, 0, count(*) "
            f"  FROM attempts a, {sql_json_elements('a', COL_MISTAKES)} m "
            f"  WHERE COALESCE(m->>'node','') <> '' AND ({where}) "
            "  GROUP BY NULLIF(m->>'node','') "
            ") t WHERE node IS NOT NULL GROUP BY node "
            "ORDER BY total DESC, node LIMIT :lim",
            params,
            label="卡点排行",
        )
        return [
            {"node": r[0], "from_hint": int(r[1] or 0),
             "from_mistake": int(r[2] or 0), "count": int(r[3] or 0)}
            for r in rows
        ]

    # ------------------------------------------------------------ 作答

    def attempts(self, *, start=None, end=None,
                 student_ids: Sequence[str] | None = None, limit: int = 500) -> list[dict]:
        """作答清单（供"选某一次提交"）。含时间、题目、弱项、错误与提示数。"""
        if self._engine is None:
            return []
        where, params = self._where(start=start, end=end, student_ids=student_ids)
        params["lim"] = limit
        rows = self._sql(
            "SELECT a.id, a.student_id, a.problem_id, a.start_time, a.end_time, "
            " a.status, a.final_correct, a.hint_count, "
            " (SELECT count(*) FROM hint_records h WHERE h.attempt_id = a.id) AS n_hint_rec, "
            " COALESCE(array_length(ARRAY(SELECT "
            f"   {sql_json_elements_text('a', COL_WEAK_CONCEPTS)}),1),0) AS n_weak, "
            " COALESCE((SELECT string_agg(x,'、') FROM "
            f"   {sql_json_elements_text('a', COL_WEAK_CONCEPTS)} x),'') AS weak_text, "
            f" {sql_json_len('a', COL_MISTAKES)} AS n_mistakes "
            "FROM attempts a "
            f"WHERE {where} "
            "ORDER BY a.start_time DESC LIMIT :lim",
            params,
            label="作答清单",
        )
        out = []
        for r in rows:
            st, et = r[3], r[4]
            out.append({
                "attempt_id": r[0],
                "student_id": r[1],
                "problem_id": r[2],
                "start_time": st.isoformat() if st else "",
                "start_display": st.strftime("%m-%d %H:%M:%S") if st else "—",
                "duration_s": round((et - st).total_seconds(), 1) if (st and et) else None,
                "status": r[5] or "",
                "final_correct": r[6],
                "hint_count": int(r[7] or 0),
                "hint_records": int(r[8] or 0),
                "weak_count": int(r[9] or 0),
                "weak_text": r[10] or "",
                "mistake_count": int(r[11] or 0),
            })
        return out

    def attempt_evidence(self, attempt_id: str) -> dict:
        """**单次作答的证据下钻**：统计项 → 学生 → attempt_id → 当前版本证据。

        审阅要求"每个统计结果都能定位输入、版本和依据"。本方法给出一层
        可查看的原始证据：

          · `state`     —— 该次作答的状态投影（当前节点 / 已访问节点 / 弱项）
          · `mistakes`  —— 过程错误记录（节点 / 性质 / 描述）
          · `hints`     —— 每次提示的**原文 + 三态判定 + 判定依据**
            （`effective_evidence` 是后端写的可复核文本）
          · `traces`    —— 逐轮分析轨迹摘要（轮次 / 模型 / 耗时 / 是否降级）
          · `revisions` —— 从轨迹里还原的**状态修订序列**（用于核对"当时是什么状态"）

        ⚠️ 只读查看，**不新建复核真值** —— 复核状态由后端的复核服务管，
        本包拿不到就如实标 `unavailable`（见 `review_store_status`）。
        """
        if self._engine is None:
            return {}
        rows = self._sql(
            "SELECT a.id, a.student_id, a.problem_id, a.status, a.start_time, "
            " a.end_time, a.current_node, a.final_correct, a.weak_concepts, "
            " a.mistakes, a.missing_reasoning, a.state "
            "FROM attempts a WHERE a.id = :aid",
            {"aid": attempt_id},
            label="证据下钻",
        )
        if not rows:
            return {}
        r = rows[0]
        out: dict = {
            "attempt_id": r[0],
            "student_id": r[1],
            "problem_id": r[2],
            "status": r[3] or "",
            "start_display": r[4].strftime("%Y-%m-%d %H:%M:%S") if r[4] else "—",
            "end_display": r[5].strftime("%Y-%m-%d %H:%M:%S") if r[5] else "—",
            "current_node": r[6] or "",
            "final_correct": r[7],
            "weak_concepts": list(r[8] or []),
            "mistakes": list(r[9] or []),
            "missing_reasoning": list(r[10] or []),
            "state": r[11] or {},
        }

        hint_rows = self._sql(
            "SELECT h.hint_level, h.node, h.hint, h.issue_type, h.effective, "
            " h.effective_evidence, h.created_at "
            "FROM hint_records h WHERE h.attempt_id = :aid ORDER BY h.id",
            {"aid": attempt_id},
            label="证据下钻",
        )
        out["hints"] = [
            {
                "level": f"L{x[0]}",
                "node": x[1] or "",
                "hint": x[2] or "",
                "issue_type": x[3] or "",
                #: 三态**原样保留**：None 就是"未判定"，不能折叠成"无效"
                "effective": x[4],
                "evidence": x[5] or "",
                "time": x[6].strftime("%H:%M:%S") if x[6] else "",
            }
            for x in hint_rows
        ]

        trace_rows = self._sql(
            "SELECT round_index, provider, model, latency_ms, created_at, "
            "       before_state "
            "FROM analysis_traces WHERE attempt_id = :aid ORDER BY round_index",
            {"aid": attempt_id},
            label="证据下钻",
        )
        out["traces"] = [
            {
                "round": int(x[0] or 0),
                "model": f"{x[1] or ''}/{x[2] or ''}".strip("/"),
                "latency_ms": int(x[3] or 0),
                "time": x[4].strftime("%H:%M:%S") if x[4] else "",
                "before_state": x[5] or {},
            }
            for x in trace_rows
        ]
        #: 修订序列：每轮的"修改前状态"，用于核对当时的判断依据。
        out["revisions"] = [
            {
                "round": t["round"],
                "before_current_node": (t["before_state"] or {}).get("current_node", ""),
                "before_visited": list((t["before_state"] or {}).get("visited_nodes") or []),
                "before_weak": list((t["before_state"] or {}).get("weak_concepts") or []),
            }
            for t in out["traces"]
        ]
        out["review_store_status"] = self.review_store_status()

        # ================================================================
        # 下钻的**原始材料**（审阅第二批 #8 点名要求）
        #
        # 审阅原文：「教师需要看到**原题**、**原图/OCR**、作答修订、实际提示、
        #           模型判断出处、**当前与过期复盘**，以及错误复核状态」
        # 上一版只做了「实际提示 + 模型判断出处 + 复核状态」，
        # 本轮补齐：原题正文 / 原题配图 / 逐轮原始文本 / 逐轮复盘（含过期）
        # / 证据修订号。
        # ================================================================

        # ---- 原题正文 ----
        #: `state.problem_content` 是作答发起时**落库的原文快照** ——
        #: 比现查 `problems` 表更可靠（题目后来被改过也能看到当时那道题）。
        out["problem_content"] = str((r[11] or {}).get("problem_content") or "")
        out["problem_meta"] = self.problem_detail(ev_problem_id=out["problem_id"])
        out["problem_figures"] = self.problem_figures(out["problem_id"])

        # ---- 逐轮原始文本（学生每次提交的**原文**）----
        #: `payload.student_input` 是**模型当轮看到的输入**，即原始作答文本。
        #: 与 `attempts.mistakes` 的"结论"不同，这是**输入侧证据**。
        inp_rows = self._sql(
            "SELECT round_index, COALESCE(payload->>'student_input','') "
            "FROM analysis_traces WHERE attempt_id = :aid ORDER BY round_index",
            {"aid": attempt_id},
            label="证据下钻",
        )
        out["student_inputs"] = [
            {"round": int(x[0] or 0), "text": str(x[1] or "")}
            for x in inp_rows
        ]

        # ---- 逐轮诊断与提示（**不是复盘版本**）----
        #: 每一轮分析都会产生一次诊断与一次提示。审阅 T3 指出：这些字段
        #: 是**逐轮诊断**，不是"报告版本"；最后一轮**不**自动成为
        #: "当前有效复盘"，前面几轮也**不**因下一轮到来自动"过期" ——
        #: 不同步骤的前一轮仍可能是当时有效的结论。
        #:
        #: 真正的"复盘"（含撤销/已解决等复核决定）应由报告版本存储提供，
        #: 并以来源修订与指纹判断失效 —— 本包未接入该存储，故不伪造。
        diag_rows = self._sql(
            "SELECT round_index, "
            "       COALESCE(payload->'diagnosis'->>'diagnosed_issue',''), "
            "       COALESCE(payload->'diagnosis'->>'issue_type',''), "
            "       COALESCE(payload->'diagnosis'->>'severity',''), "
            "       COALESCE(payload->'diagnosis'->>'recommended_action',''), "
            "       COALESCE(payload->>'hint_level',''), "
            "       COALESCE(payload->>'tutor_output',''), "
            "       COALESCE(payload->>'source_revision',''), "
            "       created_at "
            "FROM analysis_traces WHERE attempt_id = :aid ORDER BY round_index",
            {"aid": attempt_id},
            label="证据下钻",
        )
        rounds = [
            {
                "round": int(x[0] or 0),
                "issue": str(x[1] or ""),
                "issue_type": str(x[2] or ""),
                "severity": str(x[3] or ""),
                "action": str(x[4] or ""),
                "hint_level": str(x[5] or ""),
                "tutor_output": str(x[6] or ""),
                "source_revision": str(x[7] or ""),
                "time": x[8].strftime("%H:%M:%S") if x[8] else "",
            }
            for x in diag_rows
        ]
        #: **逐轮诊断**（审阅 T3）：不区分"当前/过期"——那会误标。
        out["round_diagnoses"] = rounds

        # ---- 证据修订号 ----
        #: 审阅要求「使用当前错误证据标识、**修订号**及复核状态」。
        #: 从两处派生（优先媒体来源的显式修订号，退化到轮次序号）：
        #:   · `state.media_sources[].revision` —— 照片/视频经人工更正时的**显式版本号**
        #:   · `analysis_traces.round_index`      —— 文本作答的**修订序列号**
        media = list((r[11] or {}).get("media_sources") or [])
        out["media_sources"] = [
            {
                "source_id": str(m.get("source_id") or ""),
                "revision": int(m.get("revision") or 0),
                "original_text": str(m.get("original_text") or ""),
                "corrected_text": str(
                    ((m.get("revisions") or [{}])[-1] or {}).get("text") or ""),
                "revision_history": [
                    {
                        #: 审阅 T4：媒体历史里的修订字段名是 `revision`，
                        #: 不是 `base_revision`（旧实现读错字段 → 0/1/2 全显示成 0）。
                        "revision": int(rv.get("revision") or 0),
                        "text": str(rv.get("text") or ""),
                        "analysis": rv.get("analysis") or {},
                    }
                    for rv in (m.get("revisions") or [])
                ],
                "correction_closed": bool(m.get("correction_closed")),
            }
            for m in media
        ]
        #: 证据修订身份（审阅 T4）：**保持 source_id → revision 映射**，
        #: 不用"最大修订号"代替身份 —— 两个来源 2/1 与 2/2 不能都被折成 2，
        #: 那会丢失"哪个来源在哪个版本"的信息。
        out["source_revisions"] = {
            str(m["source_id"]): int(m["revision"])
            for m in out["media_sources"] if m["source_id"]
        }
        if out["source_revisions"]:
            _rev_parts = ",".join(
                f"{k}:{v}" for k, v in sorted(out["source_revisions"].items()))
            out["evidence_revision"] = _rev_parts
            out["evidence_revision_kind"] = "media_source_revision_map"
        else:
            _max_round = max((t["round"] for t in out["traces"]), default=0)
            out["evidence_revision"] = f"round:{_max_round}"
            out["evidence_revision_kind"] = "analysis_round_index"
        try:
            out['teacher_evidence'] = self._teacher_get(
                f"/teacher/attempts/{attempt_id}/evidence",
                {'student_id': out['student_id'], 'problem_id': out['problem_id']})
        except Exception as exc:
            out['teacher_evidence'] = {'status': 'unavailable', 'error': str(exc), 'images': [], 'reports': []}
        return out

    # ---------------------------------------------------- 原题 / 环境校验

    def problem_detail(self, ev_problem_id: str = "", *, problem_id: str = "") -> dict:
        """原题元信息：正文 / 答案 / 知识点 / 难度 / 章节。

        优先查 `problems` 表（后端首次作答时自动建档）；
        表里没有时返回空字典 —— 界面会退化为只显示 `state.problem_content`
        （那是作答发起时的原文快照，通常够用）。

        ⚠️ 参数名带 `ev_` 前缀是因为被 `attempt_evidence` 以关键字调用，
        避免与内部变量重名。
        """
        pid = problem_id or ev_problem_id
        if self._engine is None or not pid:
            return {}
        rows = self._sql(
            "SELECT id, title, content, answer, knowledge_points, difficulty, extra "
            "FROM problems WHERE id = :pid",
            {"pid": pid},
            label="原题详情",
        )
        if not rows:
            return {}
        x = rows[0]
        return {
            "problem_id": x[0],
            "title": str(x[1] or ""),
            "content": str(x[2] or ""),
            "answer": str(x[3] or ""),
            "knowledge_points": list(x[4] or []),
            "difficulty": int(x[5] or 0),
            "extra": x[6] or {},
        }

    def problem_figures(self, problem_id: str) -> list[str]:
        """原题配图 URL 列表。

        审阅 T5：旧实现读**本包内**的 `data/problems/<id>/problem.json`，
        但本包是独立交付、**不含题目资产目录**，路径必然不存在 ——
        于是有配图的题永远显示"没有配图"，图片展示形不成闭环。

        正确来源是**后端规范题目图片接口** `GET /problems/{id}`，
        它返回 `figure_urls`（相对路径，如 `/problem-assets/<id>/xxx.png`）。
        这里把它们拼成后端可访问的完整 URL。

        ⚠️ 后端不可达 / 该题无图 / 接口异常时返回空列表（不抛异常），
        界面据实显示"未取到配图"，不假装闭环已完成。
        """
        if not problem_id or not self.backend_url:
            return []
        if self._engine is not None and self.environment_consistency().get("status") != "consistent":
            return []
        try:
            r = httpx.get(f"{self.backend_url}/problems/{problem_id}",
                          timeout=15, verify=_SSL_CTX)
            if r.status_code != 200:
                return []
            data = r.json() or {}
        except Exception:  # noqa: BLE001 后端不可达不该让下钻整块失效
            return []
        raw = list(data.get("figure_urls") or data.get("figures") or [])
        out: list[str] = []
        for f in raw:
            s = str(f or "").strip()
            if not s:
                continue
            if s.startswith(("http://", "https://")):
                out.append(s)
            else:
                out.append(f"{self.backend_url.rstrip('/')}/{s.lstrip('/')}")
        return out

    def demo_student_ids(self) -> list[str]:
        """演示 / 测试数据的学生 ID。

        用途：界面提供「排除演示数据」开关 —— 审阅第二批 #7 要求
        「演示 sim_* 与真实学生**明确分源、可筛除**，不默认混算」。

        判定依据是 **ID 前缀**（`sim_`）与已知测试账号（`tester`），
        与 `classes.json` 里"（模拟数据）"的标注一致。

        ⚠️ 实现注意：SQL 用**逐个参数化的 OR 子句**，不用
        `LIKE ANY(ARRAY[:pats])` —— 实测后者在 SQLAlchemy `text()` 里
        参数不会按数组展开，查询会静默失败（`_sql` 捕获后返回空列表，
        看起来像"没有演示数据"，比报错更难查）。
        """
        if self._engine is None:
            return []
        conds: list[str] = []
        params: dict = {}
        for i, pre in enumerate(DEMO_ID_PREFIXES):
            conds.append(f"id LIKE :pre{i}")
            params[f"pre{i}"] = f"{pre}%"
        for i, exact in enumerate(DEMO_IDS):
            conds.append(f"id = :ex{i}")
            params[f"ex{i}"] = exact
        if not conds:
            return []
        rows = self._sql(
            f"SELECT id FROM students WHERE {' OR '.join(conds)} ORDER BY id",
            params,
            label="演示数据识别",
        )
        return [str(x[0]) for x in rows]

    def class_source(self, json_path) -> tuple[dict[str, list[str]], str, str]:
        """班级归属的**可插拔来源** —— 数据库优先、JSON 兜底。

        返回 `(classes, error, source_label)`。

        ## 为什么需要"可插拔"

        审阅原文：
        > 「正式公网前将数据库收回服务端，通过后端按教师归属授权；
        >    **本地 classes.json 只能作为演示映射**，不能充当访问控制。」

        也就是说 `classes.json` 是**过渡期的演示映射**，不是长期权威。
        但审阅也明确「`students` 表没有班级字段，加字段属于改后端 schema」
        —— 所以本包**不能**擅自改库。

        于是这里给一个**不改后端 schema** 的接入点：
        · 部署方的库里若**已有**班级归属来源（表或列，或教务系统同步表），
          用环境变量 `TEACHER_CLASS_SOURCE` 指过来 → 本包直接采用，
          此时它是**权威归属**；
        · 库里没有 → 回退到 `classes.json`，界面**明确标注**当前用的是哪一种。

        这样"接入真实教务数据"无需改一行代码 —— 只要库里出现该来源，
        或把环境变量指过去即可（详见 `_db_class_probe` 的候选清单）。

        ## 为什么不自动建表

        建表属于改后端 schema（审阅明确排除），且演示库里也没有对应数据，
        自动建表只会造出一个空壳，反而让人以为"已经接了教务数据"。
        **宁可如实回退，也不伪造权威来源。**
        """
        #: 1) 先尝试数据库来源
        db_classes, db_label, db_err = self._db_class_probe()
        if db_classes:
            return db_classes, "", f"数据库（{db_label}）"
        #: 2) 回退到 classes.json（含"配置坏/重复成员"等原因）
        from teacher_logic import load_classes
        classes, err = load_classes(json_path)
        label = f"{Path(json_path).name}{'（数据库来源不可用：' + db_err + '）' if db_err else ''}"
        return classes, err, label

    def _db_class_probe(self) -> tuple[dict[str, list[str]], str, str]:
        """探测库里的班级归属来源。返回 `(classes, 命中描述, 失败原因)`。

        ## 候选（按优先级）

        1. 环境变量 `TEACHER_CLASS_SOURCE` = `表名:班级列:学生列`
           （列名省略时用默认 `class_name` / `student_id`）—— **显式指定最可靠**
        2. 自动探测常见表名 × 常见列名组合：
           · 表 `student_classes` / `classes` / `class_members`
           · 班级列 `class_name` / `name` / `class` / `class_id`
           · 学生列 `student_id` / `student` / `id`

        ⚠️ **探测失败不是错误**，是"当前部署没有这个来源"。
        返回的 `db_err` 只作说明用，界面据此告知"为何用了 JSON"。
        """
        if self._engine is None:
            return {}, "", "数据库不可读"
        from sqlalchemy import inspect as _inspect, text as _text

        #: 候选表 → (班级列候选, 学生列候选)
        _CLASS_COLS = ("class_name", "name", "class", "class_id")
        _STUDENT_COLS = ("student_id", "student", "id")
        try:
            insp = _inspect(self._engine)
            tables = set(insp.get_table_names())
        except Exception as exc:  # noqa: BLE001
            return {}, "", f"无法列出表（{type(exc).__name__}）"

        candidates: list[tuple[str, str, str]] = []
        #: ① 显式指定优先
        spec = (os.environ.get(CLASS_SOURCE_ENV) or "").strip()
        if spec:
            parts = [p.strip() for p in spec.split(":")]
            tbl = parts[0]
            ccol = parts[1] if len(parts) > 1 and parts[1] else "class_name"
            scol = parts[2] if len(parts) > 2 and parts[2] else "student_id"
            candidates.append((tbl, ccol, scol))
        #: ② 自动探测
        for tbl in ("student_classes", "class_members", "classes"):
            if tbl in tables:
                candidates.append((tbl, "class_name", "student_id"))
                candidates.append((tbl, "name", "student_id"))

        for tbl, ccol, scol in candidates:
            if tbl not in tables:
                continue
            try:
                cols = {c["name"] for c in insp.get_columns(tbl)}
            except Exception:  # noqa: BLE001
                continue
            if ccol not in cols or scol not in cols:
                continue
            rows = self._sql(
                f"SELECT {ccol}, {scol} FROM {tbl} "
                f"WHERE {scol} IS NOT NULL AND {ccol} IS NOT NULL "
                f"ORDER BY {ccol}, {scol}",
                {},
                label="班级来源探测",
            )
            if not rows:
                #: 表在、结构对、但没有数据 —— 不是权威来源，继续回退。
                continue
            out: dict[str, list[str]] = {}
            for cls, sid in rows:
                out.setdefault(str(cls), []).append(str(sid))
            return out, f"{tbl}.{ccol} / {tbl}.{scol}", ""
        return {}, "", "库中未发现班级归属表（已按候选清单探测）"

    def account_readonly_status(self) -> dict:
        """判定**当前数据库账号在库侧是否只读** —— 把"未证明"变成"已证明"。

        审阅原文：
        > 「当前 SQL 代码只发 SELECT，但连接未设置只读事务，也**未证明
        >   配置账户具有只读授权**。」

        应用侧的两道（只发 SELECT / `SET TRANSACTION READ ONLY`）已经做了；
        这里回答的是**第三问**：*这个账号在库里到底能不能写？*

        ## 判定依据（全部由数据库自己回答，不是本包的推断）

        1. `has_schema_privilege(current_user,'public','CREATE')`
           —— 能否在 public 建对象（建表 = 写能力）
        2. 对 public 下每张普通表问 `has_table_privilege(...,'INSERT'/'UPDATE'/'DELETE')`
           —— 用 `pg_class` 枚举，`has_table_privilege` **会计入角色继承**，
           比查 `role_table_grants`（只看直接授权）可靠
        3. `current_setting('default_transaction_read_only')`
           —— 角色级只读是否已开

        ## 三态（不夸大）

        · `readonly` —— 三条都表明不能写 ⇒ 可以明确地说"该账号只读"
        · `writable` —— 任一可写 ⇒ 明确告警"建议改用专用只读角色"
        · `unknown`  —— 查询失败（权限/view 限制）⇒ **不下结论**，
          只说"无法判定"，并指向 `deployment/create_readonly_role.py`
        """
        res: dict = {"status": "unknown", "user": "", "detail": "", "ro_guc": None}
        if self._engine is None:
            res["detail"] = "数据库不可读，无法判定账号权限"
            return res
        sql = """
            SELECT current_user AS who,
                   current_setting('default_transaction_read_only') AS ro,
                   has_schema_privilege(current_user, 'public', 'CREATE') AS can_create,
                   EXISTS (
                       SELECT 1 FROM pg_class c
                       WHERE c.relkind = 'r'
                         AND c.relnamespace = 'public'::regnamespace
                         AND (has_table_privilege(current_user, c.oid, 'INSERT')
                           OR has_table_privilege(current_user, c.oid, 'UPDATE')
                           OR has_table_privilege(current_user, c.oid, 'DELETE'))
                   ) AS can_write
        """
        rows = self._sql(sql, {}, label="只读授权探测")
        if not rows:
            res["detail"] = "权限探测查询失败（可能被 view 限制）"
            return res
        who, ro, can_create, can_write = rows[0]
        res["user"] = str(who)
        res["ro_guc"] = str(ro)
        writable = bool(can_create) or bool(can_write)
        if not writable:
            res["status"] = "readonly"
            res["detail"] = (f"账号 `{who}` 在库侧**只读**："
                             f"无 INSERT/UPDATE/DELETE、无 CREATE；"
                             f"`default_transaction_read_only={ro}`。"
                             "（应用侧 + 库侧双重只读均已生效）")
        else:
            _why = []
            if can_create:
                _why.append("可在 public 建对象")
            if can_write:
                _why.append("对表有 INSERT/UPDATE/DELETE 权限")
            res["status"] = "writable"
            res["detail"] = (f"⚠️ 账号 `{who}` 在库侧**可写**（{'；'.join(_why)}）。"
                             "教师端只发 SELECT，功能上不受影响；"
                             "但审阅要求「避免复用可写演示账户」——"
                             "建议运行 `deployment/create_readonly_role.py` 建专用只读角色，"
                             "并把它填进 `teacher_config.json`。")
        return res

    def environment_consistency(self) -> dict:
        """Compare connection identities, never infer identity from headcounts."""
        result = {'status': 'unknown', 'detail': '', 'db': None, 'api': None}
        if self._engine is None:
            result['detail'] = '数据库不可读，连接身份无法核对'
            return result
        rows = self._sql('SELECT current_database(), inet_server_addr()::text, inet_server_port()', {}, label='环境身份')
        if not rows or len(rows[0]) != 3:
            result['detail'] = '数据库连接身份未取到；不把失败当作零人数'
            return result
        row = rows[0]
        identity = {'database': str(row[0]), 'address': str(row[1] or ''), 'port': row[2]}
        fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        try:
            response = httpx.get(self.backend_url + '/teacher/environment', timeout=5, verify=_SSL_CTX)
            remote = response.json() if response.status_code == 200 else {}
            if remote.get('identity_kind') != 'database_connection_metadata_v1' or not remote.get('database_identity'):
                raise ValueError('后端未提供连接身份')
        except Exception:
            result['detail'] = '后端连接身份暂不可核对；人数相同或不同均不能替代身份'
            return result
        result['status'] = 'consistent' if remote['database_identity'] == fingerprint else 'mismatch'
        result['detail'] = ('数据库连接身份匹配（不代表教师授权已验证）' if result['status'] == 'consistent'
                            else '数据库连接身份不匹配，已停止混合原题与作答证据；请核对连接配置')
        return result

    def range_bounds(self, *, student_ids: Sequence[str] | None = None) -> tuple[str, str]:
        """实际数据的起止日期（给日期选择器一个合理默认值）。"""
        if self._engine is None:
            return "", ""
        where, params = self._where(student_ids=student_ids)
        rows = self._sql(
            f"SELECT min(a.start_time), max(a.start_time) FROM attempts a WHERE {where}",
            params,
            label="时间范围",
        )
        if not rows or not rows[0][0]:
            return "", ""
        lo, hi = rows[0]
        return lo.date().isoformat(), hi.date().isoformat()

    # ------------------------------------------------------------ 单题 / 学生

    def problems(self) -> dict[str, dict]:
        """题目列表 —— 从 `/problems`（文件题库）取。

        **不查 `problems` 表**：它靠首次作答时才自动建档，初始为空，
        直接查会得到空列表、界面上题目选择器就没有选项。
        """
        try:
            r = httpx.get(f"{self.backend_url}/problems", timeout=15, verify=_SSL_CTX)
            if r.status_code != 200:
                return {}
            items = (r.json() or {}).get("items") or []
            return {str(i["problem_id"]): i for i in items if i.get("problem_id")}
        except Exception:  # noqa: BLE001
            return {}

    def problem_stats(self, problem_id: str, *, start=None, end=None,
                      student_ids: Sequence[str] | None = None) -> dict:
        """单题卡点统计（含错题明细与错误类型）。

        有筛选时必须走 DB（后端接口没有时间 / 学生参数）；无筛选时走后端，
        保证与既有口径一致。
        """
        if self._engine is not None and (start or end or student_ids is not None):
            where, params = self._where(
                start=start, end=end, student_ids=student_ids, problem_id=problem_id)
            rows = self._sql(f"SELECT count(*) FROM attempts a WHERE {where}", params,
                             label="单题统计")
            return {
                "problem_id": problem_id,
                "attempt_count": int(rows[0][0] or 0) if rows else 0,
                "frequent_stuck_nodes": self.stuck_nodes(
                    start=start, end=end, student_ids=student_ids, problem_id=problem_id),
                "frequent_issues": self.issue_type_ranking(
                    start=start, end=end, student_ids=student_ids, problem_id=problem_id),
                "hint_effect": self.hint_stats(
                    start=start, end=end, student_ids=student_ids, problem_id=problem_id),
                "wrong_attempts": self.wrong_attempts(
                    start=start, end=end, student_ids=student_ids, problem_id=problem_id),
            }
        if self._engine is not None:
            where, params = self._where(problem_id=problem_id)
            rows = self._sql(f"SELECT count(*) FROM attempts a WHERE {where}", params,
                             label="单题统计")
            return {
                "problem_id": problem_id,
                "attempt_count": int(rows[0][0] or 0) if rows else 0,
                "frequent_stuck_nodes": self.stuck_nodes(problem_id=problem_id),
                "frequent_issues": self.issue_type_ranking(problem_id=problem_id),
                "hint_effect": self.hint_stats(problem_id=problem_id),
                "wrong_attempts": self.wrong_attempts(problem_id=problem_id),
            }
        try:
            r = httpx.get(f"{self.backend_url}/teacher/problems/{problem_id}/stats",
                          timeout=20, verify=_SSL_CTX)
            if r.status_code == 200:
                data = r.json()
                data["wrong_attempts"] = []
                return data
        except Exception:  # noqa: BLE001
            pass
        return {}

    def student_detail(self, student_id: str, *, start=None, end=None) -> dict:
        """学生画像 + 范围内作答 + 提示统计 + 错题清单。

        ⚠️ 两处口径差别，界面必须注明（否则会被误读）：
          · **画像 `weak_concepts` 是跨时间累计的**（并集只增不减），
            不随筛选变化 —— 这是 `persist_profile_weak_concepts` 的定义
          · **作答 / 提示 / 错题按范围过滤**，会随筛选变化
        """
        out: dict = {
            "student_id": student_id,
            "weak_concepts": [],
            "attempts": [],
            "wrong": [],
            "hints": {},
            "attempt_count": 0,
        }
        if self._engine is None:
            return out

        prof = self._sql(
            "SELECT weak_concepts FROM student_profiles WHERE student_id = :sid",
            {"sid": student_id},
            label="学生明细",
        )
        if prof and prof[0][0]:
            out["weak_concepts"] = list(prof[0][0])

        ids = [student_id]
        out["attempts"] = self.attempts(start=start, end=end, student_ids=ids)
        out["wrong"] = self.wrong_attempts(start=start, end=end, student_ids=ids)
        out["attempt_count"] = len(out["attempts"])

        hints = self.hint_stats(start=start, end=end, student_ids=ids)
        where, params = self._where(start=start, end=end, student_ids=ids)
        rows = self._sql(
            "SELECT 'L' || h.hint_level AS lv, count(*) AS n "
            "FROM hint_records h JOIN attempts a ON h.attempt_id = a.id "
            f"WHERE {where} GROUP BY lv ORDER BY lv",
            params,
            label="学生明细",
        )
        hints["by_level"] = {r[0]: int(r[1] or 0) for r in rows}
        out["hints"] = hints
        return out

    def counts(self, *, start=None, end=None,
               student_ids: Sequence[str] | None = None, attempt=None) -> dict:
        """范围内的规模：作答数 / 学生数 / 有过程错误记录的作答数 / 被标记过弱项的作答数。

        ⚠️ 后两项**刻意分开**（审阅要求）：
          · `problem_attempts` —— 有 `mistakes` 的作答（过程错误证据）
          · `weak_flagged`      —— 有 `weak_concepts` 的作答（历史提及）
        原实现把两者并成一个"有问题的作答"，会让"历史上弱过"的学生
        在每道题上都被算成有问题。
        """
        if self._engine is None:
            return {"attempts": 0, "students": 0, "problem_attempts": 0, "weak_flagged": 0}
        where, params = self._where(
            start=start, end=end, student_ids=student_ids, attempt=attempt)
        rows = self._sql(
            "SELECT count(*), count(DISTINCT a.student_id), "
            f" sum(CASE WHEN {self._has_error_records('a')} THEN 1 ELSE 0 END), "
            f" sum(CASE WHEN {self._has_weak_mention('a')} THEN 1 ELSE 0 END) "
            f"FROM attempts a WHERE {where}",
            params,
            label="规模统计",
        )
        if not rows:
            return {"attempts": 0, "students": 0, "problem_attempts": 0, "weak_flagged": 0}
        return {
            "attempts": int(rows[0][0] or 0),
            "students": int(rows[0][1] or 0),
            "problem_attempts": int(rows[0][2] or 0),
            "weak_flagged": int(rows[0][3] or 0),
        }
