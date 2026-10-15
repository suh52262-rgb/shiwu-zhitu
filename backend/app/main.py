"""视物知途后端入口：FastAPI 服务。

当前状态：
- /health 健康检查（不依赖外部服务）
- /problems 题库列表 / 单题详情 / Problem Graph（P16 Repository 驱动）
- /attempt/analyze：LangGraph 教学工作流 + PostgreSQL 持久化
- /attempt/{id}：作答记录与 Learning Timeline 回查
- /knowledge/search：教材 RAG 检索
- /students/{id}/profile：学生画像与作答历史
- /problem-assets：题目插图静态资源（Task 书 4.1 节的 figures 字段落地）
"""
# ⚠️ **必须在导入任何 `app.*` 之前加载 `.env`。**
#
# ## 为什么
# `config.py` 走 pydantic-settings，**自己会读 `.env`** —— 所以主 LLM 一直能用。
# 但**视觉层是直接读 `os.environ` 的**（`vision/service.py` 找
# `DEEPSEEK_API_KEY` / `QWEN_API_KEY` / `ZHIPU_API_KEY`），
# **而进程环境里没有这些键**（启动脚本也没传）。
#
# ⇒ 结果：文字作答一切正常，**一传视频就
#   `没有可用的视觉模型：请至少配置 DEEPSEEK_API_KEY / ...`**。
#
# **这是被"视频作答"这条新路才暴露出来的** —— 在此之前视觉层只从
# CLI 脚本调用，而**那些脚本每个都自己 `load_dotenv()`**，
# 所以这个缺口一直没露头。
#
# 放在这里（而不是只给视觉层打补丁）是因为**任何直接读 env 的模块都受益**，
# 且与仓库里所有脚本的做法一致。
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

_load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from app.api import attempts, chat, demo, knowledge, problems, students, teacher, recommend, video
from app.observability import RequestIdMiddleware
from app.services.problem_repository import get_problem_repository
from app.api import video_jobs
from app.api import problem_imports
from app.api import graph_authoring
from app.api import figure_context_drafts
from app.api import error_evidence_reviews
from app.api import teacher_evidence
from app.api import photo_jobs

@asynccontextmanager
async def lifespan(app):
    from app.config import get_settings
    from app.services.startup_warmup import warm_local_embedding
    await run_in_threadpool(warm_local_embedding, get_settings())
    yield


app = FastAPI(
    lifespan=lifespan,
    title="视物知途 · 物理过程感知智能教学系统",
    version="1.4.0",
    description="本地演示版：过程辅导、教学证据与资源工作台",
)

# P19：为每个请求分配 request_id，回显在 X-Request-ID 响应头，
# 并写入每轮 analysis_traces，便于把"一次请求"与"一条 trace"对上。
app.add_middleware(RequestIdMiddleware)


@app.get("/health", tags=["system"])
def health() -> dict:
    """健康检查。

    刻意不访问数据库 / Qdrant / LLM，保证基础框架在
    未安装 Docker、未配置密钥时也能独立启动。
    """
    return {"status": "ok", "app": "shiwu-zhitu", "version": app.version}


# 题目插图：URL 形如 /problem-assets/{problem_id}/figures/xxx.png
@app.get('/runtime/readiness', tags=['system'])
def runtime_readiness() -> dict:
    from app.services.startup_warmup import status
    return status()


@app.get('/runtime/configuration', tags=['system'])
def runtime_configuration() -> dict:
    from app.services.runtime_configuration import public_runtime_configuration
    return public_runtime_configuration()


# check_dir=False：题库目录尚未创建时也不应阻止服务启动（P16）
# Serve registered figures only: mounting the bank exposed reference-answer JSON.

app.include_router(problems.router)
app.include_router(problems.assets_router)
app.include_router(problem_imports.router)
app.include_router(graph_authoring.router)
app.include_router(figure_context_drafts.router)
app.include_router(error_evidence_reviews.router)
app.include_router(attempts.router)
app.include_router(knowledge.router)
app.include_router(students.router)
app.include_router(chat.router)
app.include_router(teacher.router)
app.include_router(teacher_evidence.router)
app.include_router(demo.router)
app.include_router(recommend.router)
app.include_router(video.router)
app.include_router(video_jobs.router)
app.include_router(photo_jobs.router)
