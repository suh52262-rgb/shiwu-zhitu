"""配置解析 —— 让教师端能**独立接入**任意后端与数据库。

## 为什么需要这一层

原实现直接 `from app.config import get_settings`，即**必须**跟部署包放在一起
（要能找到 `backend/app/config.py`）。要把它作为独立交付物接入别的环境，
就必须把"连哪里"变成可配置项。

## 配置优先级（高 → 低）

    1. 环境变量      TEACHER_BACKEND_URL / TEACHER_DATABASE_URL / TEACHER_CLASSES_FILE
    2. 配置文件      <包根>/teacher_config.json
    3. 自动探测      <包根>/../backend/app/config.py（在项目内直接运行时）

排列的理由：**环境变量优先级最高**，便于在 CI / 容器 / 临时排障时
不动物理文件就切换目标；配置文件是日常改的对象。

## 只读约束

本模块只负责"算出连哪里"，**不发起任何连接**，也不写任何文件。
连接与查询一律在 `teacher_data.py`，且那里只发 SELECT。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.parse import quote, urlparse, urlunparse

#: 包根 = 本文件的上一级（frontend/ 的父目录）
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: 配置文件默认位置
DEFAULT_CONFIG_FILE = PACKAGE_ROOT / "teacher_config.json"

#: 环境变量名。集中在这里，避免各处硬编码字符串（拼错不会报错，只会静默失效）。
ENV_BACKEND_URL = "TEACHER_BACKEND_URL"
ENV_DATABASE_URL = "TEACHER_DATABASE_URL"
ENV_CLASSES_FILE = "TEACHER_CLASSES_FILE"
ENV_CONFIG_FILE = "TEACHER_CONFIG_FILE"

#: 后端默认地址（与学生端/后端默认一致）
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

#: 数据库分项写法时的默认值（与部署包 .env 一致）
DB_DEFAULTS = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "shiwu",
    "password": "shiwu_demo_local",
    "dbname": "shiwu_zhitu_demo",
}


@dataclass
class TeacherConfig:
    """算好的连接配置。"""

    backend_url: str = DEFAULT_BACKEND_URL
    database_url: str = ""
    classes_file: Path = field(default_factory=lambda: PACKAGE_ROOT / "frontend" / "teacher_classes.json")
    #: 每一项的来源说明，用于界面回显与排障（"这个地址是哪来的"）
    sources: dict[str, str] = field(default_factory=dict)
    #: 配置错误（人可读）。**非空时必须显示给用户**，见 `load_config` 的说明。
    errors: list[str] = field(default_factory=list)

    @property
    def db_configured(self) -> bool:
        return bool(self.database_url)

    @property
    def ok(self) -> bool:
        """配置是否完全可用（无错误）。"""
        return not self.errors


def _normalize_pg_url(url: str) -> str:
    """把常见写法规范成 SQLAlchemy + psycopg 驱动形式。

    SQLAlchemy 需要显式驱动名，而用户手写的串通常没有：
      · `postgres://...`            → `postgresql+psycopg://...`
      · `postgresql://...`          → `postgresql+psycopg://...`
      · `postgresql+psycopg://...`  → 原样
      · `sqlite:///...`             → 原样（便于用本地文件库做演示）

    `postgres://` 这个别名是历史遗留（早于 SQLAlchemy 2.0），
    直接喂给它会被解析成未知方言而报错，所以必须转。
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://"):]
    return raw


#: 端口合法范围。**必须校验**：审阅实测 `database.port = "not-a-port"` 时
#: 原实现直接 `int(...)` 抛 ValueError，而 `load_config` 声称"任何异常都不抛"——
#: 配置写错就把整个页面崩掉，用户看不到任何提示、也不知道该改哪。
MIN_PORT, MAX_PORT = 1, 65535

#: `postgresql+psycopg` 是 SQLAlchemy 的方言名；`sqlite` 便于用本地文件库做演示。
DB_URL_SCHEMES = ("postgresql", "postgres", "sqlite", "postgresql+psycopg")


def validate_port(value: object, where: str) -> tuple[int | None, str]:
    """校验端口。返回 `(端口, 错误说明)`，错误说明非空即表示校验失败。

    刻意**不复用 `int()`**：`int("not-a-port")` 会抛异常，
    而调用方要的是"一条能显示给人看的错误"，不是异常。
    """
    if isinstance(value, bool):
        #: `bool` 是 `int` 的子类，不先挡掉的话 `True` 会被当成端口 1。
        return None, f"{where}：端口不能是布尔值（{value!r}）"
    if isinstance(value, int):
        port = value
    else:
        text = str(value if value is not None else "").strip()
        if not text:
            return None, f"{where}：端口为空"
        if not text.lstrip("+-").isdigit():
            return None, f"{where}：端口必须是数字，当前是 {value!r}"
        port = int(text)
    if not (MIN_PORT <= port <= MAX_PORT):
        return None, f"{where}：端口越界，应在 {MIN_PORT}~{MAX_PORT} 之间，当前 {port}"
    return port, ""


def validate_url(url: str, *, schemes: Sequence[str], where: str) -> str:
    """校验 URL 的协议与主机。返回错误说明，空串表示通过。"""
    text = (url or "").strip()
    if not text:
        return f"{where}：为空"
    try:
        p = urlparse(text)
    except Exception as exc:  # noqa: BLE001 URL 畸形时 urlparse 可能抛
        return f"{where}：无法解析（{type(exc).__name__}）"
    if not p.scheme:
        return f"{where}：缺少协议（应形如 http://… 或 postgresql://…）"
    if p.scheme.lower() not in schemes:
        return f"{where}：不支持的协议 {p.scheme!r}（支持 {'、'.join(schemes)}）"
    if p.scheme.lower() != "sqlite" and not p.hostname:
        return f"{where}：缺少主机名"
    if p.scheme.lower() != "sqlite":
        _, err = validate_port(p.port or 0, where)
        if err:
            return err
    return ""


def build_dsn_from_parts(parts: dict) -> tuple[str, str]:
    """由分项字段拼出 DSN。返回 `(dsn, 错误说明)`。

    **口令必须做 URL 编码**：口令里出现 `@` / `:` / `/` / `#`
    （随机口令里很常见）会让 URL 解析错位 —— `p@ss` 会被当成
    `p` 用户 + `ss` 主机。这类错误表现为"连不上"，很难从报错看出来。

    ⚠️ 这里**不再抛异常**（原实现 `int(g("port"))` 会在端口非法时抛
    ValueError，把 `load_config` 的"配置坏了要能提示"承诺破坏掉）。
    端口非法时返回空 DSN + 错误说明，由调用方记进 `cfg.errors`。
    """
    def g(key):
        v = parts.get(key)
        return DB_DEFAULTS[key] if v in (None, "") else v

    port, port_err = validate_port(g("port"), "数据库端口")
    if port_err:
        return "", port_err

    user = quote(str(g("user")), safe="")
    password = quote(str(g("password")), safe="")
    host = str(g("host")).strip()
    dbname = str(g("dbname")).strip()
    if not host:
        return "", "数据库主机为空"
    if not dbname:
        return "", "数据库名为空"
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}", ""


def _mask(url: str) -> str:
    """隐藏口令，用于回显。界面和日志里**不能出现明文口令**。"""
    if not url:
        return "（未配置）"
    try:
        p = urlparse(url)
        if p.password:
            netloc = f"{p.username}:***@{p.hostname}"
            if p.port:
                netloc += f":{p.port}"
            p = p._replace(netloc=netloc)
        return urlunparse(p)
    except Exception:  # noqa: BLE001 只是个显示函数，解析失败就直接返回
        return url


def masked_database_url(cfg: TeacherConfig) -> str:
    if cfg.database_url:
        return _mask(cfg.database_url)
    #: 区分"压根没配"与"配了但非法" —— 两者该给用户的提示完全不同。
    return "（配置非法，已停用）" if [e for e in cfg.errors if "数据库" in e] else "（未配置）"


def load_config(config_file: Path | None = None) -> TeacherConfig:
    """按优先级算出最终配置。**任何异常都不抛**，最差返回纯默认值 + 错误说明。

    配置坏掉时应当**能启动、能提示**，而不是直接崩 ——
    否则用户看不到任何界面，也不知道该改哪。

    ## 校验与"禁止悄悄回退"

    ⚠️ 配置里的值**非法时不再默默换成默认值**。
    例如 `database.port = "not-a-port"`：
      · 旧行为：`int()` 抛 ValueError，整页崩（审阅已复现）
      · 中间方案：悄悄退回 5432 —— 更糟，用户以为连上了对的库，
        实际连的是另一个实例，统计结果来源不明
      · 现在：记进 `cfg.errors`、`database_url` 置空并在界面上明确报错

    同理，某项配置**写坏了就不再往下走"自动探测"那条路** ——
    否则"我明明配了库，它却连了别的地方"这种问题极难排查。
    """
    cfg = TeacherConfig()
    notes: list[str] = []

    # ---- 第 2 层：配置文件 ----
    path = Path(os.environ.get(ENV_CONFIG_FILE) or (config_file or DEFAULT_CONFIG_FILE))
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                cfg.errors.append(f"配置文件内容不是对象：{path.name}")
                data = {}
        except Exception as exc:  # noqa: BLE001 配置坏了要能提示，不是崩
            cfg.errors.append(f"配置文件解析失败（{path.name}）：{type(exc).__name__}: {exc}")
            data = {}
    else:
        notes.append(f"未找到配置文件 {path.name}，使用默认值")

    #: 顶层可能写成 {"database": {...}} 分项，也可能直接给 database_url
    db_url = str(data.get("database_url") or "").strip()
    db_parts = data.get("database") if isinstance(data.get("database"), dict) else None
    if db_parts is not None and not isinstance(data.get("database"), dict):
        cfg.errors.append("配置文件 database 字段应为对象（分项写法）")
    backend_url = str(data.get("backend_url") or "").strip()
    classes_rel = str(data.get("classes_file") or "").strip()

    #: **是否"显式配了数据库"**。显式配了（哪怕写坏）就不再自动探测，
    #: 避免"配了却连到别处"的静默错源。
    db_configured_explicitly = bool(db_url) or (db_parts is not None)

    # ---- 第 3 层：自动探测项目内的 backend 配置 ----
    if not db_configured_explicitly and not os.environ.get(ENV_DATABASE_URL, "").strip():
        probed = _probe_project_backend()
        if probed:
            db_url = probed
            notes.append("数据库来自项目内 backend/app/config.py（自动探测）")

    # ---- 第 1 层：环境变量（最高优先级，最后覆盖）----
    env_db = os.environ.get(ENV_DATABASE_URL, "").strip()
    if env_db:
        db_url = env_db
        db_parts = None          #: 环境变量给的是完整串，分项写法作废
        notes.append(f"数据库来自环境变量 {ENV_DATABASE_URL}")
    env_backend = os.environ.get(ENV_BACKEND_URL, "").strip()
    if env_backend:
        backend_url = env_backend
        notes.append(f"后端地址来自环境变量 {ENV_BACKEND_URL}")
    env_classes = os.environ.get(ENV_CLASSES_FILE, "").strip()
    if env_classes:
        classes_rel = env_classes
        notes.append(f"班级配置来自环境变量 {ENV_CLASSES_FILE}")

    # ---- 校验：后端地址 ----
    backend_url = (backend_url or DEFAULT_BACKEND_URL).strip()
    err = validate_url(backend_url, schemes=("http", "https"), where="后端地址")
    if err:
        cfg.errors.append(err)
        #: 置空而不是留一个坏值 —— 界面据"空"就该知道后端不可用。
        #: 不退回默认地址：那会让人以为连的是 127.0.0.1:8000，实际配的是别的。
        cfg.backend_url = ""
    else:
        cfg.backend_url = backend_url.rstrip("/")

    # ---- 校验：数据库连接 ----
    if db_url:
        normalized = _normalize_pg_url(db_url)
        err = validate_url(
            normalized,
            schemes=("postgresql+psycopg", "postgresql", "postgres", "sqlite"),
            where="数据库连接串",
        )
        if err:
            cfg.errors.append(err)
            cfg.database_url = ""
        else:
            cfg.database_url = normalized
    elif db_parts is not None:
        dsn, err = build_dsn_from_parts(db_parts)
        if err:
            cfg.errors.append(err)
            cfg.database_url = ""
        else:
            cfg.database_url = dsn
            notes.append("数据库来自配置文件的 database 分项字段")
    else:
        notes.append("未配置数据库连接（仅使用后端接口模式）")

    # ---- 校验：班级文件 ----
    if classes_rel:
        p = Path(classes_rel)
        cfg.classes_file = p if p.is_absolute() else (PACKAGE_ROOT / p)
    #: 班级文件不存在**不算 cfg.errors**（不是连接配置错误）：
    #: 它由 `load_classes` 报给界面，那里有专门的空状态处理。

    cfg.sources = {
        "backend_url": cfg.backend_url or "（配置非法，已停用）",
        "database_url": masked_database_url(cfg),
        "classes_file": str(cfg.classes_file),
        "notes": "；".join(notes) if notes else "（无）",
    }
    return cfg


def _probe_project_backend() -> str:
    """尝试从项目内的 backend/app/config.py 读出 DSN。

    用于"把 frontend/ 放在部署包里直接跑"这个场景 —— 那条路径下
    连接参数本来就写在项目的 .env 里，没必要让用户再抄一份。

    找不到就返回空串（不是错误：独立部署时本来就没有 backend/）。
    """
    for base in (PACKAGE_ROOT, PACKAGE_ROOT.parent):
        backend = base / "backend"
        if not (backend / "app" / "config.py").is_file():
            continue
        try:
            sp = str(backend)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            from app.config import get_settings  # type: ignore

            return str(get_settings().postgres_dsn)
        except Exception:  # noqa: BLE001 探测失败属正常，继续尝试下一个
            continue
    return ""


def describe(cfg: TeacherConfig) -> str:
    """给人看的一行摘要（口令已隐藏）。配置有错时把错误一并带出。"""
    db = masked_database_url(cfg)
    line = f"后端 {cfg.backend_url or '（配置非法，已停用）'} ｜ 数据库 {db}"
    if cfg.errors:
        line += " ｜ ⚠️ " + "；".join(cfg.errors)
    return line
