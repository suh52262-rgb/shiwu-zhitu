"""教材 PDF 文本提取与入库（Phase 4）。

流程：data/textbooks/*.pdf → PyMuPDF 逐页提取 → chunk → embedding → Qdrant。
每个 chunk 携带开放 metadata（source / chapter / page / type），
未来扩展期末题、教师资料等只需加 type 字段，不改结构。
"""
import logging
import re
import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.db.qdrant import get_qdrant
from app.rag.chunk import chunk_text
from app.rag.embedding import get_embedding_client

logger = logging.getLogger(__name__)

# 章节标题识别（如 "第4章 机械能守恒定律" / "第四章 动量"）
_CHAPTER_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百]+\s*章\s*\S+")


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """逐页提取文本，并跟踪页面上出现的章节标题。

    返回：[{"page": 1, "text": "...", "chapter": "第4章 机械能守恒定律"}, ...]
    """
    import fitz  # PyMuPDF

    pages: list[dict] = []
    current_chapter = ""
    with fitz.open(str(pdf_path)) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            match = _CHAPTER_RE.search(text)
            if match:
                current_chapter = match.group(0).strip()
            if text:
                pages.append({"page": index, "text": text, "chapter": current_chapter})
    return pages


def _point_id(source: str, page: int, chunk_index: int) -> str:
    """确定性 ID：同一文件重复入库时覆盖旧向量，保证幂等。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"shiwu:{source}:{page}:{chunk_index}"))


def _ensure_collection(client, collection: str, dim: int) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def ingest_pdf(pdf_path: str | Path, batch_size: int = 16) -> int:
    """解析单个 PDF 并写入向量库，返回入库 chunk 数。重复执行幂等覆盖。"""
    pdf_path = Path(pdf_path)
    settings = get_settings()
    client = get_qdrant()
    embedder = get_embedding_client()
    source = pdf_path.name

    records: list[dict] = []
    for page in extract_pages(pdf_path):
        for i, chunk in enumerate(chunk_text(page["text"])):
            records.append(
                {
                    "id": _point_id(source, page["page"], i),
                    "text": chunk,
                    "payload": {
                        "text": chunk,
                        "source": source,
                        "chapter": page["chapter"],
                        "page": page["page"],
                        "type": "textbook",
                    },
                }
            )
    if not records:
        logger.warning("%s 未提取到文本（可能是扫描版 PDF）", source)
        return 0

    # 先编码一批以确定向量维度，再建集合（维度跟随 embedding 模型）
    first = records[:batch_size]
    first_vectors = embedder.embed([r["text"] for r in first])
    _ensure_collection(client, settings.qdrant_collection, dim=len(first_vectors[0]))

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(id=r["id"], vector=v, payload=r["payload"])
            for r, v in zip(first, first_vectors)
        ],
    )

    total = len(first)
    for start in range(batch_size, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embedder.embed([r["text"] for r in batch])
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                PointStruct(id=r["id"], vector=v, payload=r["payload"])
                for r, v in zip(batch, vectors)
            ],
        )
        total += len(batch)

    logger.info("入库完成：%s → %s chunks", source, total)
    return total


def ingest_directory(dir_path: str | Path) -> dict[str, int]:
    """批量入库目录下所有 PDF。"""
    results = {}
    for pdf in sorted(Path(dir_path).glob("*.pdf")):
        results[pdf.name] = ingest_pdf(pdf)
    return results
