"""配置层测试：非法值必须**报错**，且**不得静默回退到另一个数据源**。

守的坑（审阅第三批）：
  · `database.port` 为非数值时原实现 `int()` 抛 ValueError → 整页崩
  · 中间方案（悄悄退回 5432）更糟 → 用户以为连了对的库，实际连了别的实例
"""
from __future__ import annotations

import json

import pytest

import teacher_config as TC


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """环境变量优先级最高，测试前必须清干净，否则会串味。"""
    for name in (TC.ENV_DATABASE_URL, TC.ENV_BACKEND_URL,
                 TC.ENV_CLASSES_FILE, TC.ENV_CONFIG_FILE):
        monkeypatch.delenv(name, raising=False)


def _write(tmp_path, payload: dict):
    f = tmp_path / "teacher_config.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return f


# ---------------------------------------------------------------- 端口校验

@pytest.mark.parametrize("bad", ["not-a-port", "", None, "5432.5", "abc", "1e3"])
def test_validate_port_rejects_non_numeric(bad):
    port, err = TC.validate_port(bad, "端口")
    assert port is None and err, f"{bad!r} 应被拒绝"


@pytest.mark.parametrize("bad", [0, -1, 65536, 99999])
def test_validate_port_rejects_out_of_range(bad):
    port, err = TC.validate_port(bad, "端口")
    assert port is None and err


def test_validate_port_accepts_valid():
    assert TC.validate_port("5432", "端口") == (5432, "")
    assert TC.validate_port(5432, "端口") == (5432, "")


def test_validate_port_rejects_bool():
    """`bool` 是 `int` 子类，不先挡掉的话 `True` 会被当成端口 1。"""
    port, err = TC.validate_port(True, "端口")
    assert port is None and err


# ---------------------------------------------------------------- 非法配置

def test_invalid_port_raises_no_exception_but_records_error(tmp_path):
    """核心回归：非法端口**不抛异常**，而是记进 cfg.errors。"""
    f = _write(tmp_path, {
        "backend_url": "http://127.0.0.1:8000",
        "database": {"host": "127.0.0.1", "port": "not-a-port",
                     "user": "u", "password": "p", "dbname": "d"},
    })
    cfg = TC.load_config(f)          # ← 不得抛异常
    assert cfg.errors, "应记录配置错误"
    assert any("端口" in e for e in cfg.errors)


def test_invalid_port_does_not_silently_fall_back(tmp_path):
    """非法配置时 database_url 必须为空 —— 不能让统计来源不明。"""
    f = _write(tmp_path, {
        "database": {"host": "h", "port": "nope",
                     "user": "u", "password": "p", "dbname": "d"},
    })
    cfg = TC.load_config(f)
    assert cfg.database_url == ""
    assert "非法" in TC.masked_database_url(cfg)


def test_build_dsn_returns_error_instead_of_raising():
    dsn, err = TC.build_dsn_from_parts({"port": "oops"})
    assert dsn == "" and err


def test_valid_dsn_encodes_special_chars_in_password():
    """口令含 @ : / # 时必须做 URL 编码，否则会被当成主机分隔符。"""
    dsn, err = TC.build_dsn_from_parts({
        "host": "127.0.0.1", "port": 5432,
        "user": "u", "password": "p@ss:w/rd#1", "dbname": "d",
    })
    assert err == "" and dsn
    assert "p%40ss%3Aw%2Frd%231" in dsn, dsn


def test_invalid_backend_url_records_error_and_disables(tmp_path):
    f = _write(tmp_path, {"backend_url": "not a url"})
    cfg = TC.load_config(f)
    assert any("后端" in e for e in cfg.errors)
    assert cfg.backend_url == "", "非法后端地址应停用，而不是留个坏值"


def test_backend_url_scheme_validated(tmp_path):
    f = _write(tmp_path, {"backend_url": "ftp://example.com"})
    cfg = TC.load_config(f)
    assert any("协议" in e for e in cfg.errors)


def test_missing_config_file_is_not_an_error(tmp_path):
    """没配置文件不是错误 —— 用默认值即可启动。"""
    cfg = TC.load_config(tmp_path / "nope.json")
    assert cfg.errors == []
    assert cfg.backend_url == TC.DEFAULT_BACKEND_URL


def test_broken_json_records_error_not_crash(tmp_path):
    f = tmp_path / "teacher_config.json"
    f.write_text("{ broken", encoding="utf-8")
    cfg = TC.load_config(f)
    assert cfg.errors and "解析失败" in cfg.errors[0]


def test_env_var_overrides_config_file(tmp_path, monkeypatch):
    f = _write(tmp_path, {"backend_url": "http://from-file:8000"})
    monkeypatch.setenv(TC.ENV_BACKEND_URL, "http://from-env:9000")
    cfg = TC.load_config(f)
    assert cfg.backend_url == "http://from-env:9000"


def test_describe_masks_password(tmp_path):
    f = _write(tmp_path, {
        "database_url": "postgresql://u:supersecret@127.0.0.1:5432/d"})
    cfg = TC.load_config(f)
    out = TC.describe(cfg)
    assert "supersecret" not in out
    assert "***" in out


def test_ok_property_reflects_errors(tmp_path):
    good = TC.load_config(_write(tmp_path, {"backend_url": "http://127.0.0.1:8000"}))
    bad = TC.load_config(_write(tmp_path, {"backend_url": "nonsense"}))
    assert good.ok is True
    assert bad.ok is False
