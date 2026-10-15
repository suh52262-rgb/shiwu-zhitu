"""Project overview: static introduction and links, no API or database calls."""
from html import escape
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workspace_style import workspace_url

st.set_page_config(page_title='视物知途 · 项目导览', page_icon='🔬', layout='wide')
st.markdown('<style>'+Path(__file__).with_name('portal.css').read_text(encoding='utf-8')+'</style>', unsafe_allow_html=True)

ENTRIES = [
    ('student','学生学习','从完整原题出发，提交作答、获取分级提示，核对照片与视频过程。',
     '<path d="M5 5h14v14H5zM8 9h8M8 12h5M8 15h7"/>'),
    ('teacher','教师分析','查看作答记录与书面证据，核对系统判断，安排下一次讲评。',
     '<path d="M4 19h16M7 16V9M12 16V5M17 16v-4"/>'),
    ('problems','题库与导入','检索已发布题目，对照来源页完善题面、配图和参考解。',
     '<path d="M4 6h6l2 2h8v11H4zM8 12h8M8 15h5"/>'),
    ('materials','教材管理','浏览教材原文与导入批次，让参考材料有名称、有来源。',
     '<path d="M12 6c-3-2-6-2-9-1v13c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1zM12 6v13"/>'),
]


def entry_cards():
    cards=[]
    for key,title,description,icon in ENTRIES:
        cards.append(f'''<a class="portal-card portal-{key}" href="{workspace_url(key)}" target="_blank" rel="noopener noreferrer">
        <span class="portal-icon"><svg viewBox="0 0 24 24" aria-hidden="true">{icon}</svg></span>
        <div><span class="portal-card-title" role="heading" aria-level="3">{escape(title)}</span><p>{escape(description)}</p></div>
        <span class="portal-open" aria-hidden="true">↗</span></a>''')
    return ''.join(cards)


portal_markup = '''<div class="portal-shell">
<header class="portal-header"><div class="portal-brand"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M8 29V11M8 29H32M11 26L20 13L29 20"/><circle cx="20" cy="13" r="3"/><circle cx="29" cy="20" r="3"/></svg><strong>视物知途</strong></div><div class="portal-meta"><span>大学物理 · 过程辅导</span><label class="portal-motion-control"><input id="portal-static" type="checkbox">静态模式</label></div></header>
<section class="portal-hero"><div class="portal-intro"><div class="portal-title" role="heading" aria-level="1">看见解题过程，<br>找到下一步。</div><p>面向大学物理的过程辅导助手。结合照片识别、作答记录与教材证据，提供分级提示和可核对的过程复盘。</p></div>
<figure class="portal-diagram"><svg viewBox="0 0 380 170" role="img" aria-label="过程辅导示意：原题、书写、核对、下一步，不代表具体学生的解题轨迹">
<path class="portal-guide" d="M50 38h90q22 0 22 22v28q0 22 22 22h120"/>
<path class="portal-guide portal-secondary" d="M50 38v68q0 22 22 22h60"/>
<path class="portal-flow" d="M50 38h90q22 0 22 22v28q0 22 22 22h120" aria-hidden="true"/>
<path class="portal-flow portal-evidence-flow" d="M50 38v68q0 22 22 22h60" aria-hidden="true"/>
<circle class="portal-pulse" cx="310" cy="110" r="13" aria-hidden="true"/>
<circle cx="50" cy="38" r="7"/><circle cx="162" cy="75" r="7"/><circle cx="220" cy="110" r="7"/><circle cx="310" cy="110" r="7"/>
<text x="35" y="18">原题</text><text x="183" y="65">书写</text><text x="200" y="148">核对</text><text x="283" y="148">下一步</text>
<text class="portal-diagram-note" x="87" y="156">教材证据</text></svg><figcaption class="portal-diagram-caption">过程辅导示意</figcaption></figure></section>
<section class="portal-workspaces" aria-labelledby="portal-choose"><div class="portal-section-head"><span class="portal-section-title" id="portal-choose" role="heading" aria-level="2">选择你的工作区</span><span>学习、教学与资源，在这里连接。</span></div>
<nav class="portal-grid" aria-label="四个工作区">'''+entry_cards()+'''</nav></section>
<footer class="portal-footer"><span>提示与复盘可结合原稿核对。</span><span>入口在新标签页打开，当前页面会保留。</span></footer></div>'''
st.markdown(portal_markup, unsafe_allow_html=True)
