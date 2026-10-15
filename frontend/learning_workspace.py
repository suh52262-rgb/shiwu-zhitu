"""Presentation for learning and evidence; never invokes teaching models."""
import json
from html import escape
from urllib.parse import urlencode, quote

import streamlit as st

HINT_STEPS = ['方向提示', '概念提示', '方法提示', '关键步骤', '完整讲解']


def render_hint_ladder(tutor):
    value = tutor.get('hint_level', 0)
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(6):
        st.caption('当前提示等级暂不可确认。')
        return
    items = []
    for level, label in enumerate(HINT_STEPS, 1):
        selected = ' aria-current="step"' if level == value else ''
        items.append(f'<li{selected}><span>L{level}</span>{label}</li>')
    st.markdown('<ol class="sz-hint-ladder" aria-label="五阶提示">'+''.join(items)+'</ol>', unsafe_allow_html=True)
    st.caption('当前：'+(f'L{value} · {HINT_STEPS[value-1]}' if value else 'L0 · 本轮未下发提示'))


def progress_dot(snapshot, tutor):
    """Only reference topology + neutral node status. No equations or solutions."""
    nodes = snapshot.get('nodes') or []
    ids = [str(n['id']) for n in nodes if isinstance(n, dict) and n.get('id')]
    visited = {str(i) for i in tutor.get('visited_nodes') or []}
    current = str(tutor.get('current_node') or '')
    lines = ['digraph progress { rankdir=LR; graph [bgcolor="transparent"]; node [shape=box, style="rounded,filled", fontname="Microsoft YaHei", fontsize=11];']
    for nid in ids:
        status = '当前核对' if nid == current else ('已有匹配记录' if nid in visited else '待开放环节')
        label = f'{nid}\n{status}'
        color = '#DDE9FC' if nid == current else ('#E0F1EF' if nid in visited else '#F0F3F7')
        lines.append(f'{json.dumps(nid,ensure_ascii=False)} [label={json.dumps(label,ensure_ascii=False)}, fillcolor="{color}", color="#A6B6CB", fontcolor="#172B46"];')
    for edge in snapshot.get('edges') or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 2 and all(str(e) in ids for e in edge):
            lines.append(f'{json.dumps(str(edge[0]))} -> {json.dumps(str(edge[1]))} [color="#A6B6CB"];')
    return '\n'.join(lines + ['}'])


def render_progress(get, base, sid, pid, aid, tutor):
    st.subheader('解题进度图')
    if not aid or not tutor.get('graph_mode'):
        st.caption('开始作答后显示系统匹配记录；无参考图时保留逐轮过程。')
        return
    try:
        response = get(f'{base}/attempt/{quote(aid, safe="")}', timeout=5)
        response.raise_for_status()
        record = response.json()
        if record.get('student_id') != sid or record.get('problem_id') != pid:
            raise ValueError('context mismatch')
        state = record.get('state') or {}
        snapshot = state.get('graph_snapshot') or {}
        if not snapshot or (tutor.get('graph_version') and state.get('graph_version') != tutor['graph_version']):
            raise ValueError('bound graph unavailable')
        st.graphviz_chart(progress_dot(snapshot, tutor), width='stretch')
        st.caption('使用本次作答绑定的参考图。匹配记录不等于已掌握；箭头不代表学生实际计算顺序。未开放的方程与答案不显示。')
    except Exception:
        st.caption('本次作答的绑定图暂不可读；不使用当前题库图替代历史版本。')


def target_round(rounds, source_id, revision):
    return next((i for i, r in enumerate(rounds) if r.get('source_id') == source_id
                 and r.get('source_revision', 0) == revision), None)


def set_evidence_target(run_id, source_id, revision):
    st.session_state['review_evidence_target'] = {'run_id':run_id, 'source_id':source_id, 'revision':revision}


def render_evidence_browser(result, get, base, sid, pid, aid, *, stacked=False):
    """Service IDs only. The selected OCR round is not an invented image citation."""
    if not result or not aid:
        return
    run_id = result.get('run_id') or ''
    rounds = result.get('rounds') or []
    st.subheader('原画面与识别核对')
    material_key = run_id or (rounds[-1].get('source_id') if rounds else '') or aid
    source_key = f'evidence_round_{material_key}'
    target = st.session_state.get('review_evidence_target') or {}
    if target.get('run_id') == run_id:
        index = target_round(rounds, target.get('source_id'), target.get('revision'))
        if index is not None:
            st.session_state[source_key] = index
        else:
            st.warning('引用对应的来源或修订版本不在当前材料中，不能定位到另一份原文。')
        st.session_state.pop('review_evidence_target', None)
    if not rounds:
        st.info('尚无可核对的教学快照；处理进度与已有结果会保留。')
        return
    chosen = st.selectbox('选择教学快照', list(range(len(rounds))),
        format_func=lambda i:f"第 {rounds[i].get('round', i+1)} 轮 · {rounds[i].get('ts', rounds[i].get('time_sec', '时刻未记录'))} 秒", key=source_key)
    item = rounds[chosen]
    a, b = (st.container(), st.container()) if stacked else st.columns([1.05, 1])
    with a:
        try:
            params = urlencode({'student_id':sid, 'problem_id':pid})
            response = get(f'{base}/teacher/attempts/{quote(aid,safe="")}/evidence?{params}', timeout=8)
            response.raise_for_status()
            images = response.json().get('images') or []
            if result.get('media_type') == 'image':
                images = [e for e in images if e.get('source_id') == item.get('source_id')]
            else:
                images = [e for e in images if run_id and e.get('run_id') == run_id]
            if images:
                preferred = 0
                timestamp = item.get('ts', item.get('time_sec'))
                if isinstance(timestamp, (float, int)):
                    preferred = min(range(len(images)), key=lambda i:abs((images[i].get('timestamp_sec') or 0)-timestamp))
                image = st.selectbox('保留画面', images, index=preferred,
                    format_func=lambda e:f"{e.get('timestamp_sec', '照片')} 秒 · {e['evidence_id'][:8]}", key=f'frame_{material_key}_{chosen}')
                st.image(f"{base}/teacher/attempts/{quote(aid,safe='')}/images/{quote(image['evidence_id'],safe='')}?{params}", width='stretch')
                st.caption('原始保留画面。视频按近邻时刻初选，请核对图文是否对应；不把近邻画面自动当作报告引文。')
            else:
                st.info('本次来源没有可读取的保留原图；已有识别文本仍可核对。')
        except Exception:
            st.warning('原图读取暂未成功；不等于材料没有原图。')
    with b:
        st.markdown('**当前识别文本**')
        st.code(item.get('answer') or item.get('recognized_text') or item.get('student_input') or '本轮未识别到作答', language=None)
        st.caption(f"来源：{item.get('source_id','未记录')} · 修订：{item.get('source_revision','未记录')}")
        st.caption('只核对这份来源的文字；最近一步更正仍使用下方原有更正入口。')
        if item.get('error'):
            st.warning('本轮识别或教学未成功，不能作为完整批改结果。')


def render_authoring_steps(active=0):
    labels = ['选择来源', '提取题目', '核对题面与参考解', '保存复核', '发布题目', '审核Graph']
    st.markdown('<ol class="sz-authoring-steps">'+''.join(
        f'<li'+(' aria-current="step"' if i == active else '')+f'><span>{i+1}</span>{escape(label)}</li>' for i,label in enumerate(labels))+'</ol>', unsafe_allow_html=True)
    st.caption('这是操作阶段指引，不是自动质量认证；题目发布与Graph发布保持独立。')
