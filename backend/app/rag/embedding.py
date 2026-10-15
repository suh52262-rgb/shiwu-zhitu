"""Embedding 抽象接口与双实现（Phase 4）。

设计约束：接口可替换，当前支持：
- bge  ：本地 sentence-transformers 加载 BGE 中文模型，离线、免费、无需 Key（默认）
- qwen ：DashScope OpenAI 兼容 /embeddings 接口，需 QWEN_API_KEY

上层代码只依赖 EmbeddingClient 抽象（见 get_embedding_client）。
"""
import os
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

# 约束：HF_ENDPOINT 必须在 huggingface_hub 导入前设置（国内网络下载模型用镜像）
_settings = get_settings()
if _settings.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = _settings.hf_endpoint


class EmbeddingClient(Protocol):
    """所有 embedding 实现需满足的最小接口。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量（已归一化，配合余弦距离）。"""
        ...


class LocalBgeEmbedding:
    """本地 BGE 模型（BAAI/bge-small-zh-v1.5，512 维）。

    首次使用会从 HF 镜像下载模型（约 100MB）到本地缓存，之后离线可用。
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，加快启动

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]


class QwenApiEmbedding:
    """Qwen（DashScope 兼容模式）Embedding API。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]


@lru_cache(maxsize=4)
def get_embedding_client() -> EmbeddingClient:
    """根据 EMBEDDING_PROVIDER 配置返回 embedding 客户端（进程内缓存）。

    必须缓存：`LocalBgeEmbedding` 的构造会**加载整个 BGE 模型**（数百 ms~数秒、
    上百 MB 内存抖动），而 `retriever.retrieve()` 每次检索都会调用本函数。
    不缓存就等于每轮 RAG 都重新加载一遍模型。
    与 `get_engine` / `get_qdrant` 的懒加载单例做法保持一致。
    """
    settings = get_settings()
    provider = settings.embedding_provider.strip().lower()
    if provider == "bge":
        return LocalBgeEmbedding(settings.embedding_model)
    if provider == "qwen":
        if not settings.qwen_api_key:
            raise ValueError("EMBEDDING_PROVIDER=qwen 需要在 .env 中设置 QWEN_API_KEY")
        return QwenApiEmbedding(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.embedding_model,
        )
    raise ValueError(f"未知 EMBEDDING_PROVIDER={provider!r}，支持: bge / qwen")


def clear_cache() -> None:
    """清空客户端缓存（切换配置或测试隔离时使用）。"""
    get_embedding_client.cache_clear()
