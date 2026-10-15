"""全局配置：所有密钥与连接信息从环境变量 / .env 读取，禁止硬编码。

.env 统一放在项目根目录（与 docker-compose.yml 同级），
此处按文件位置向上解析，保证无论从哪个目录启动都能读到。
"""
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM（DeepSeek / Qwen / 任意 OpenAI 兼容网关 / Anthropic）----
    llm_provider: str = "deepseek"          # deepseek / qwen / openai_compatible / openai / anthropic
    llm_model: str = ""                     # 留空则用 provider 默认模型
    llm_temperature: float = 0.2
    llm_timeout: float = 30.0               # 单次请求超时（秒）；显式设置，避免 SDK 默认长超时挂死
    llm_max_retries: int = 2                # 仅重试超时 / 限流 / 网络 / 5xx
    llm_role_options: dict[str, dict] = {}  # Explicit measured opt-ins; empty preserves provider defaults.
    photo_analysis_budget_s: float = 60.0  # Cooperative execution budget, excluding queue wait.
    photo_role_options: dict[str, dict] = {
        'rerank': {'thinking': 'disabled', 'max_tokens': 256},
        'tutor': {'thinking': 'disabled', 'max_tokens': 1600},
    }

    # 通用 OpenAI 兼容入口（LLM_PROVIDER=openai_compatible 时使用）
    llm_api_key: str = ""
    llm_base_url: str = ""
    openai_api_key: str = ""                # LLM_PROVIDER=openai 时的备用 Key

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #: 视频帧识别用的视觉模型（`app/vision/service.py` 的 `resolve_model`）。
    #:
    #: ⚠️ **默认选 `qwen3-vl-plus` 而不是 `qwen3-vl-flash`**：
    #: 实测记录 `qwen3-vl-flash` **2026-10-10 下架**。默认值指向一个
    #: 会消失的模型，等于埋一个"到时候才发现"的故障。
    #: 但 `plus` **未经本项目实测**——换模型前请先用真实视频跑一次。
    qwen_vl_model: str = "qwen3-vl-plus"
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""            # 留空用官方端点

    # ---- 按节点指定模型（P17）：留空则回落到 LLM_MODEL ----
    step_matcher_model: str = ""
    analyze_step_model: str = ""
    diagnosis_model: str = ""
    review_model: str = ""                 # 空值兼容沿用诊断模型
    review_timeout: float = 120.0          # 长报告独立超时，不延长即时辅导
    review_max_retries: int = 0            # 避免长超时在页面300秒内重复堆叠
    tutor_model: str = ""
    rerank_model: str = ""                 # Empty inherits tutor model for compatibility.
    graph_generator_model: str = ""         # P14 预留
    graph_validator_model: str = ""         # 独立物理复核角色覆盖
    graph_generator_timeout: float = 120.0  # 建图结构化输出较长；不改学生即时提示超时
    graph_release_policy: str = "legacy-demo"  # legacy-demo / published-only

    rag_material_policy: str = "hit_first_total_v1"  # legacy remains available for comparison

    # ---- Embedding（接口可替换）----
    embedding_provider: str = "bge"         # bge（本地）/ qwen（DashScope API）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_warmup_enabled: bool = False  # Local launchers opt in; tests/API stay lightweight.
    hf_endpoint: str = "https://hf-mirror.com"   # 国内下载 HF 模型镜像，留空则直连

    # ---- PostgreSQL（Phase 6 使用）----
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "shiwu"
    postgres_password: str = ""
    postgres_db: str = "shiwu_zhitu"
    # 连接超时（秒）。libpq 默认无限等待：PostgreSQL 未启动时 engine.connect()
    # 会长时间阻塞（实测 >120s），既拖死"数据库是否可用"的探测，也会在 pytest
    # 收集阶段把整个测试套件卡住。显式设小值以保证快速失败、优雅降级。
    postgres_connect_timeout: int = 3

    # ---- Qdrant（服务不可达时自动降级为 data/qdrant_local 本地模式）----
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "textbook_chunks"

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy 连接串（psycopg3 驱动）。

        口令必须做 URL 转义：只要出现 `@`（如 `p@ssw0rd`），未转义时 make_url
        会把 `@` 当成用户信息与主机的分隔符，解析出 `host=ssw0rd@localhost`
        这种结果，连接失败时报错完全指不到病根。
        """
        return (
            f"postgresql+psycopg://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password)}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
