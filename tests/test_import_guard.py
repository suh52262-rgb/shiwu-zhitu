"""导入护栏：doc_id 安全、并发互斥、图片引用、文件摘要。

## 这一组测的是「别写到仓库外」与「别让两个导入互相踩」

`doc_id` 会被直接拼进 `data/documents/{doc_id}/figures/`。
不校验的话 `--doc-id ../../..` 能写到仓库外——`ProblemRepository`
早就有 `_is_safe_id` 处理同类问题，导入这条路当时漏了。
"""
import pytest

from app.rag.import_guard import (
    ImportGuardError,
    file_sha256,
    safe_join,
    validate_doc_id,
)


# ---------------------------------------------------------------- doc_id 校验

@pytest.mark.parametrize("doc_id", [
    "jiaocai-3-lixue", "demo_exam", "abc123", "a.b.c", "A" * 100, "x_1-2.3",
])
def test_valid_doc_ids_pass(doc_id):
    assert validate_doc_id(doc_id) == doc_id


@pytest.mark.parametrize("doc_id", [
    "../etc",              # 目录穿越
    "..",                  # 就是父目录
    ".hidden",             # 隐藏目录
    "a/b",                 # 路径分隔符
    "a\\b",                # Windows 分隔符
    "C:evil",              # 盘符 / ADS
    "",                    # 空
    "x" * 200,             # 超长
    "-leading-dash",       # 以非字母数字开头
])
def test_unsafe_doc_ids_are_rejected(doc_id):
    with pytest.raises(ImportGuardError):
        validate_doc_id(doc_id)


def test_safe_join_blocks_traversal_even_if_validation_is_bypassed(monkeypatch):
    """**双重保险**：即使 `validate_doc_id` 将来被放宽或绕过，
    `safe_join` 的 resolve 后检查仍要挡住。

    （防护不该只有一层——校验规则会被改，路径检查不会。）
    """
    import app.rag.import_guard as guard

    monkeypatch.setattr(guard, "validate_doc_id", lambda d: d)   # 假装校验被绕过
    with pytest.raises(ImportGuardError):
        safe_join("data/documents", "../../evil")


def test_safe_join_keeps_normal_id_inside_root(tmp_path):
    got = safe_join(tmp_path, "my-doc")
    assert got.parent == tmp_path.resolve()
    assert got.name == "my-doc"


# ---------------------------------------------------------------- 文件摘要

def test_file_sha256_distinguishes_same_name_different_content(tmp_path):
    """**这正是摘要的用途**：区分"同一文件重试"与"同名但内容不同的文件"。

    只看文件名的话，后者会被当成前者，用旧结果糊弄过去——
    导入出来的东西和 PDF 对不上，而且看不出来。
    """
    a = tmp_path / "same-name.pdf"
    a.write_bytes(b"version one")
    h1 = file_sha256(a)
    a.write_bytes(b"version two")
    h2 = file_sha256(a)
    assert h1 != h2

    b = tmp_path / "other" / "same-name.pdf"
    b.parent.mkdir()
    b.write_bytes(b"version one")
    assert file_sha256(b) == h1, "内容相同则摘要必须相同（重试判定依赖它）"


def test_file_sha256_is_stable_across_calls(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"x" * 100_000)
    assert file_sha256(f) == file_sha256(f)


# ---------------------------------------------------------------- 并发互斥

@pytest.mark.real_db
def test_import_lock_is_exclusive_within_a_process():
    """同一 doc_id 同时只允许一个导入。

    用 PostgreSQL advisory lock（session 级）而不是 threading.Lock：
    后者只在单进程内有效，而导入常常是多进程/多终端跑的。
    """
    from app.rag.import_guard import import_lock

    try:
        with import_lock("lock-test-doc"):
            with pytest.raises(ImportGuardError, match="已有另一个导入"):
                with import_lock("lock-test-doc"):
                    pass
    except ImportGuardError as exc:
        pytest.skip(f"数据库不可用，跳过：{exc}")


@pytest.mark.real_db
def test_import_lock_released_after_exit():
    """退出后锁必须释放，否则第二次导入会被自己挡住。"""
    from app.rag.import_guard import import_lock

    try:
        with import_lock("lock-release-doc"):
            pass
        with import_lock("lock-release-doc"):
            pass          # 不该抛
    except ImportGuardError as exc:
        if "已有另一个导入" in str(exc):
            pytest.fail("锁没有在退出时释放")
        pytest.skip(f"数据库不可用，跳过：{exc}")


@pytest.mark.real_db
def test_import_lock_released_on_exception():
    """导入中途抛异常时锁也要释放——否则一次失败会让文档永久锁死。"""
    from app.rag.import_guard import import_lock

    try:
        with pytest.raises(ValueError):
            with import_lock("lock-exc-doc"):
                raise ValueError("模拟中途失败")
        with import_lock("lock-exc-doc"):
            pass
    except ImportGuardError as exc:
        if "已有另一个导入" in str(exc):
            pytest.fail("异常路径没有释放锁")
        pytest.skip(f"数据库不可用，跳过：{exc}")


# ---------------------------------------------------------------- 图片引用

def test_verify_figure_detects_missing_file(tmp_path):
    from app.rag.import_guard import verify_figure

    rel, problem = verify_figure("figures/nope.png", tmp_path)
    assert rel == "figures/nope.png"
    assert problem, "文件不存在时必须报出问题，否则前端会拿到死链"


def test_verify_figure_accepts_existing_file(tmp_path):
    from app.rag.import_guard import verify_figure

    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "ok.png").write_bytes(b"png")
    rel, problem = verify_figure("figures/ok.png", tmp_path)
    assert problem == ""
    assert rel == "figures/ok.png"


def test_verify_figure_flags_empty_file(tmp_path):
    """空文件（裁切失败留下的）也要报——它比"不存在"更难发现。"""
    from app.rag.import_guard import verify_figure

    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "empty.png").write_bytes(b"")
    _, problem = verify_figure("figures/empty.png", tmp_path)
    assert problem


def test_verify_figure_normalizes_backslashes(tmp_path):
    """存库的路径统一正斜杠，反斜杠要能容错。"""
    from app.rag.import_guard import verify_figure

    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "x.png").write_bytes(b"png")
    rel, problem = verify_figure("figures\\x.png", tmp_path)
    assert problem == ""
    assert rel == "figures/x.png"


def test_verify_figure_handles_empty_input(tmp_path):
    from app.rag.import_guard import verify_figure

    assert verify_figure("", tmp_path) == ("", "")
