"""conftest：把 `frontend/` 放进 sys.path，让测试直接 import 三个模块。

本包不是标准 Python 包（`frontend/` 下是脚本而非包），
所以需要显式加路径 —— 与 `teacher_app.py` 顶部 `sys.path.insert` 的做法一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = PKG_ROOT / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest


@pytest.fixture
def teacher_db_fake(monkeypatch):
    """Opt-in DB-mode UI double, shared by both teacher interaction suites."""
    import streamlit as st
    import teacher_data as TD
    from teacher_test_fakes import FakeTeacherData, RECORD

    RECORD.clear()
    FakeTeacherData.default_roster = []
    monkeypatch.setattr(TD, 'TeacherData', FakeTeacherData)
    monkeypatch.setenv('TEACHER_DATABASE_URL', 'postgresql://u:p@127.0.0.1:59876/nodb')
    for name in ('TEACHER_BACKEND_URL', 'TEACHER_CLASSES_FILE', 'TEACHER_CONFIG_FILE'):
        monkeypatch.delenv(name, raising=False)
    st.cache_resource.clear()
    try:
        yield FakeTeacherData, RECORD
    finally:
        st.cache_resource.clear()
        FakeTeacherData.default_roster = []
        RECORD.clear()
