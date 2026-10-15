"""Shared presentation only. No request, state mutation or teaching policy."""
from html import escape
import os
from pathlib import Path

import streamlit as st

TOKENS = {
    'primary': '#2458B5', 'ink': '#172B46', 'muted': '#52647A',
    'background': '#F3F6FA', 'surface': '#FFFFFF', 'accent': '#087F86',
}
WORKSPACES = {
    'student': ('学生学习', 8501), 'teacher': ('教师分析', 8502),
    'problems': ('题库与导入', 8503), 'materials': ('教材管理', 8504),
}


def workspace_url(key):
    """Use the same local port contract for the overview and workspace links."""
    port = 8500 if key == 'overview' else WORKSPACES[key][1]
    try:
        value = int(os.getenv(f'WORKSPACE_{key.upper()}_PORT', str(port)))
        if not 1 <= value <= 65535:
            raise ValueError('invalid port')
    except ValueError:
        value = port
    return f'http://127.0.0.1:{value}'


def navigation_markup(active):
    """Fixed local navigation; config ports, never interpolate student content."""
    links = []
    for key, (label, port) in WORKSPACES.items():
        current = ' aria-current="page"' if key == active else ''
        links.append(f'<a href="{workspace_url(key)}" target="_blank" rel="noopener noreferrer"{current}>{escape(label)}</a>')
    return '<nav class="sz-nav" aria-label="切换工作区">' + ''.join(links) + '</nav>'


def apply_workspace_style(active):
    st.markdown('<style>' + Path(__file__).with_name('workspace.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('''<div class="sz-brand"><svg aria-hidden="true" viewBox="0 0 40 40">
        <path d="M8 29V11M8 29H32M11 26L20 13L29 20" fill="none" stroke="currentColor" stroke-width="2.5"/>
        <circle cx="20" cy="13" r="3" fill="currentColor"/><circle cx="29" cy="20" r="3" fill="currentColor"/>
        </svg><div><strong>视物知途</strong><span>大学物理过程辅导</span></div></div>''', unsafe_allow_html=True)
        st.markdown(f'<a class="sz-overview-link" href="{workspace_url("overview")}" target="_blank" rel="noopener noreferrer">项目导览</a>', unsafe_allow_html=True)
        st.markdown(navigation_markup(active), unsafe_allow_html=True)
        st.caption('各入口在新标签页打开，保留当前页面。')


def page_intro(title, subtitle, role):
    st.markdown(f'<div class="sz-role">{escape(role)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def section_note(title, body):
    st.markdown(f'<div class="sz-note"><strong>{escape(title)}</strong><p>{escape(body)}</p></div>', unsafe_allow_html=True)
