"""知识库检索接口（P7：对齐 RAG Tool 统一签名）。"""
from fastapi import APIRouter, HTTPException

from app.agent.tools import retrieve_knowledge
from app.db.qdrant import QdrantUnavailable

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/search")
def search_knowledge(query: str, top_k: int = 5, knowledge_points: str = "") -> dict:
    """从教材知识库检索相关内容。

    knowledge_points：逗号分隔的可选知识点约束（对齐 retrieve_knowledge Tool 签名）。
    返回格式：[{"text", "source", "page", "chapter", "score"}, ...]

    向量库不可用时返回 503（与持久化层"查询抛领域异常 → 503"的口径一致）：
    这是一个"明确要检索知识库"的请求，静默返回空列表会让人以为"教材里没有"。
    Agent 工作流内的 RAG 节点则相反——它按"本轮无教材依据"降级继续，不受影响。
    """
    kps = [k.strip() for k in knowledge_points.split(",") if k.strip()] or None
    try:
        results = retrieve_knowledge(query, knowledge_points=kps, top_k=top_k)
    except QdrantUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"向量库不可用：{exc}") from exc
    return {"query": query, "knowledge_points": kps or [], "items": results}
