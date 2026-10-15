"""AI figure drafts remain visibly provisional until an operator checks them."""
from uuid import uuid4
import streamlit as st


def render_figure_context(pid,request):
    detail=request('GET',f'/problems/{pid}')
    figures=(detail or {}).get('figures',[])
    if not figures:return '',False
    if len(figures)>1:st.info('每次只读取所选一张图；请补充并核对其他配图信息。不会自动验证多图是否完整。')
    path=f'/problems/{pid}/figure-context-drafts'
    with st.expander('AI读取配图，生成待核对的图示说明'):
        st.caption('调用视觉模型，仅提取可见对象、连接、方向、数值和不确定项；不自动建图或发布。')
        index=st.selectbox('选择已登记配图',range(len(figures)),format_func=lambda i:f'配图{i+1}',key=f'figure_index_{pid}')
        key=f'figure_request_{pid}_{index}'
        if st.button('生成图示草稿（调用一次视觉模型）',key=f'figure_read_{pid}'):
            rid=st.session_state.setdefault(key,str(uuid4()))
            draft=request('POST',path,json={'client_request_id':rid,'figure_index':index})
            if draft:st.rerun()
        if st.button('新开图示提取请求（另一次模型调用）',key=f'figure_again_{pid}'):
            st.session_state[key]=str(uuid4())
            st.caption('已分配新编号；点击生成后才调用模型。')
        listed=request('GET',path)
        drafts=(listed or {}).get('items',[])
        if not drafts:return '',False
        draft=next((d for d in drafts if d['request_id']==st.session_state.get(key)),drafts[0])
        st.write('状态：'+draft['status'])
        if draft.get('error'):st.warning(draft['error'])
        if draft.get('stale'):st.warning('题面或配图已变化，不能将此旧草稿直接用于建图。')
        if draft['status']=='ready' and not draft.get('stale'):
            st.write(draft['description'])
            for uncertain in draft.get('uncertainties',[]):st.warning('需核对：'+uncertain)
            if st.button('复制到图示说明输入框（仍需核对）',key=f'figure_copy_{pid}'):
                st.session_state[f'graph_context_{pid}']=draft['description']
                st.session_state[f'graph_context_draft_{pid}']=draft['request_id']
                st.session_state[f'graph_context_confirmed_{pid}']=False
                st.rerun()
    rid=st.session_state.get(f'graph_context_draft_{pid}','')
    checked=st.checkbox('我已对照原图核对AI图示草稿，并修正模糊或错误内容',key=f'graph_context_confirmed_{pid}') if rid else False
    return rid,checked
