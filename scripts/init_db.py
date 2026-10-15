"""初始化数据库：创建开发文档第八节定义的全部表，并补齐 ORM 新增的列与索引。

运行（PostgreSQL 需已启动，如 docker compose up -d）：
    .venv/Scripts/python scripts/init_db.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.db.migrate import apply_schema_migrations  # noqa: E402
from app.db.postgres import get_engine  # noqa: E402
from app.db.tables import BASE_TABLES, Base  # noqa: E402
from app.services import error_evidence_reviews  # noqa: E402,F401


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("已确保数据表存在：")
    for table in BASE_TABLES:
        print(f"  - {table}")

    # create_all 不会给已存在的表加列，也不会补 ORM 声明的索引，这里一并补齐（幂等）
    applied = apply_schema_migrations(engine)
    if applied:
        print(f"已应用 {len(applied)} 项结构变更（新增列 / 索引）：")
        for item in applied:
            print(f"  ~ {item}")
    else:
        print("表结构已是最新，无需迁移")

    engine.dispose()
    print("INIT_DB_OK")


if __name__ == "__main__":
    main()
