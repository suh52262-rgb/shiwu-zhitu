"""构建 RAG 知识库：data/textbooks/*.pdf → 分块 → Embedding → 向量库。

运行：.venv/Scripts/python scripts/build_rag.py
注意：Qdrant 本地模式（未装 Docker 时）同一时间只允许一个进程访问，
重建索引前请先停掉 uvicorn 后端。
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import PROJECT_ROOT  # noqa: E402
from app.rag.ingest import ingest_directory  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    textbook_dir = PROJECT_ROOT / "data" / "textbooks"
    pdfs = sorted(textbook_dir.glob("*.pdf"))
    if not pdfs:
        print(f"未找到 PDF。请把教材放入 {textbook_dir} 后重试。")
        print("也可先运行 scripts/make_demo_textbook.py 生成演示教材。")
        sys.exit(1)

    results = ingest_directory(textbook_dir)
    for name, count in results.items():
        print(f"  {name}: {count} chunks 入库")
    from app.db.qdrant import get_qdrant

    get_qdrant().close()  # 显式关闭，避免本地模式退出时的清理噪音
    print("BUILD_OK")


if __name__ == "__main__":
    main()
