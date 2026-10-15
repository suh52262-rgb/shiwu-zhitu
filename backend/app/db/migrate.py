"""轻量 schema 迁移。

背景：`Base.metadata.create_all()` 只建**不存在的表**，对已存在的表**不会 ADD COLUMN**。
于是给 ORM 加列后，本地已经跑过的数据库会停留在旧结构，报出难以理解的
"column does not exist"。项目当前没有 Alembic（开发文档第十三节要求"不过度工程化"），
这里提供一个最小可用的迁移：

    读取 ORM 元数据 → 与数据库实际列比对 → 用 ADD COLUMN IF NOT EXISTS 补齐缺列

特点：
- **不依赖手写列清单**，直接由 `db/tables.py` 的 ORM 定义推导，不会与实际定义漂移；
- 幂等，可重复执行；
- 只做"加列"这一类向后兼容的变更。**改类型 / 删列 / 加约束不在范围内**，
  真到那一步应引入 Alembic（届时本模块可直接删除）。

局限：新加的列不会带上 NOT NULL / DEFAULT 约束（我们的列都有 Python 侧默认值，
可接受）。若将来需要严格约束，同样应改用 Alembic。
"""
import logging

from sqlalchemy import inspect, text

from app.db.postgres import get_engine
from app.db.tables import Base

logger = logging.getLogger(__name__)


def pending_columns(engine=None) -> dict[str, list[str]]:
    """返回 {表名: [数据库中缺失的列名]}，不修改任何东西（供预检/测试使用）。"""
    engine = engine or get_engine()
    inspector = inspect(engine)
    missing: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {col["name"] for col in inspector.get_columns(table.name)}
        absent = [col.name for col in table.columns if col.name not in existing]
        if absent:
            missing[table.name] = absent
    return missing


def pending_indexes(engine=None) -> dict[str, list[str]]:
    """返回 {表名: [数据库中缺失的索引名]}，不修改任何东西。

    为什么必须单独处理索引：`ADD COLUMN` 只加列，**不会**把 ORM 里
    `index=True` 声明的索引一并建出来。结果是"新建的库有索引、迁移过的库没有"，
    两个环境悄悄跑在不同结构上（P19 的 `analysis_traces.problem_id` 就踩过这个坑，
    而它正是 P24 教师端按题聚合要用的列）。
    """
    engine = engine or get_engine()
    inspector = inspect(engine)
    missing: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        declared = {index.name for index in table.indexes if index.name}
        actual = {index["name"] for index in inspector.get_indexes(table.name)}
        absent = sorted(name for name in declared if name not in actual)
        if absent:
            missing[table.name] = absent
    return missing


def _index_ddl(table_name: str, index) -> str:
    columns = ", ".join(f'"{column.name}"' for column in index.columns)
    unique = "UNIQUE " if index.unique else ""
    return f'CREATE {unique}INDEX IF NOT EXISTS "{index.name}" ON "{table_name}" ({columns})'


def apply_schema_migrations(engine=None) -> list[str]:
    """补齐所有缺失的列与索引，返回已执行的变更列表（空列表表示无需迁移）。

    只做"加列 / 加索引"这类向后兼容的变更，幂等可重复执行。
    """
    engine = engine or get_engine()
    missing_columns = pending_columns(engine)
    missing_indexes = pending_indexes(engine)
    applied: list[str] = []
    if not missing_columns and not missing_indexes:
        return applied

    with engine.begin() as conn:
        for table_name, columns in missing_columns.items():
            table = Base.metadata.tables[table_name]
            for column_name in columns:
                column = table.columns[column_name]
                ddl_type = column.type.compile(engine.dialect)
                # **必须带上 server default**：不带的话，已存在的行会是 NULL，
                # 而 Python 侧 `bool(None)` 是 False——一个语义上默认 true 的
                # 布尔列会静默变成"全部 false"（踩过：22976 块全被标成不可索引，
                # 重建索引写了 0 块，同时按范围删掉了全部向量）。
                default = ""
                if column.server_default is not None:
                    default = f" DEFAULT {column.server_default.arg}"
                conn.execute(
                    text(
                        f'ALTER TABLE "{table_name}" '
                        f'ADD COLUMN IF NOT EXISTS "{column_name}" {ddl_type}{default}'
                    )
                )
                # 已有行若是 NULL，补成默认值
                if column.server_default is not None:
                    conn.execute(
                        text(
                            f'UPDATE "{table_name}" SET "{column_name}" = '
                            f'{column.server_default.arg} '
                            f'WHERE "{column_name}" IS NULL'
                        )
                    )
                applied.append(f"{table_name}.{column_name}")
        for table_name, index_names in missing_indexes.items():
            table = Base.metadata.tables[table_name]
            by_name = {index.name: index for index in table.indexes}
            for index_name in index_names:
                conn.execute(text(_index_ddl(table_name, by_name[index_name])))
                applied.append(f"{table_name}#{index_name}")

    if applied:
        logger.info("schema 迁移：应用 %d 项变更", len(applied))
    return applied
