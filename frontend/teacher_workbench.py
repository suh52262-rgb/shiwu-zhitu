"""Task-oriented teacher UI with explicit, idempotent local review actions."""
import json
from uuid import uuid4
from urllib.parse import quote
import httpx
import streamlit as st


def review_request_id(state, sid, eid, body):
    signature = json.dumps([sid, eid, body], sort_keys=True, ensure_ascii=False)
    return state.setdefault('review_request_ids', {}).setdefault(signature, str(uuid4()))


def render_teacher_tasks(data, errors, base, render_evidence, *, start=None, end=None, student_ids=None, attempt=None):
    st.header('待办与证据')
    ready = errors.get('review_store_status') == 'ready'
    if not ready:
        st.warning('当前错误证据或复核状态读取失败，待办数量暂不可得。不能解释成没有待办。')
    else:
        counts = errors.get('counts') or {}
        cols = st.columns(3)
        cols[0].metric('待核对书面记录', counts.get('unreviewed', 0))
        cols[1].metric('本地已采纳记录', counts.get('confirmed', 0))
        cols[2].metric('已处理记录', counts.get('resolved', 0))
        st.caption(f"另有{errors.get('legacy_attempt_count',0)}次旧口径作答不具备当前证据结构；待办数不包含它们，不代表已全部核对。")
    items = errors.get('evidence') or [] if ready else []
    queue, detail = st.columns([.8, 1.3], gap='large')
    with queue:
        st.subheader('书面记录队列')
        selection = st.radio('处理范围', ['待核查','本地确认','已撤销或解决'], key='teacher_task_filter')
        statuses = {'待核查':['unreviewed'],'本地确认':['confirmed'],'已撤销或解决':['dismissed','resolved']}[selection]
        filtered = [e for e in items if e.get('review_status') in statuses]
        if filtered:
            eid = st.selectbox('选择一条证据', [e['evidence_id'] for e in filtered],
                format_func=lambda value: next(f"{e['student_id']} · {e['problem_id']} · {e['concept']}" for e in filtered if e['evidence_id']==value))
        else:
            eid = None
            st.info('本轮读取范围内没有这类可展示记录；请结合历史作答覆盖核对。' if ready else '等待恢复证据读取后再核对。')
    with detail:
        st.subheader('核对原文，再处理判断')
        if eid:
            item = next(e for e in filtered if e['evidence_id'] == eid)
            st.code(item['student_text'], language=None)
            st.write('系统书面判断：'+item['issue'])
            st.caption(f"来源 {item.get('source_id','未记录')} · 修订 {item.get('source_revision','未记录')}")
            with st.expander('完整作答、原画面与修订历史'):
                render_evidence(item['attempt_id'])
            with st.form('review_task_'+eid):
                actor = st.text_input('本地核对者标识（AI核查须写明AI）')
                decision = st.selectbox('核对决定', ['confirmed','dismissed','resolved'],
                    format_func=lambda k:{'confirmed':'采纳这条书面判断','dismissed':'撤销这条系统判断','resolved':'这条记录已处理'}[k])
                note = st.text_area('核对依据与说明')
                confirmed = st.checkbox('我已核对这份来源及修订版本，决定只针对这条记录')
                save = st.form_submit_button('保存核对决定')
            if save:
                if not actor.strip() or not confirmed:
                    st.warning('请填写核对者标识，并确认已经核对来源与修订版本。')
                else:
                    body = {'attempt_id':item['attempt_id'],'decision':decision,'actor_label':actor,'note':note}
                    body['client_request_id'] = review_request_id(st.session_state,item['student_id'],eid,body)
                    try:
                        response = httpx.post(f"{base}/teacher/students/{quote(item['student_id'],safe='')}/error-evidence/{quote(eid,safe='')}/decisions",json=body,timeout=15)
                        response.raise_for_status()
                    except Exception:
                        st.error('保存结果未确认；请刷新复核状态后再决定是否重试。请求编号已保留，不自动重复提交。')
                    else:
                        st.success('决定已保存，原作答保留。')
                        st.rerun()
            st.caption('本地操作者自报身份，不代表正式教师认证；采纳或处理一条错误记录不证明概念掌握。')
        else:
            st.caption('从左侧选择证据。旧口径记录可从下方最近作答进入核对。')
    st.divider()
    st.subheader('最近作答入口')
    rows = data.attempts(start=start,end=end,student_ids=student_ids,limit=20)
    if attempt:
        ids = [attempt]
        labels = {attempt:'当前筛选的作答'}
    else:
        ids = [r['attempt_id'] for r in rows]
        labels = {r['attempt_id']:f"{r['student_id']} · {r['problem_id']} · {r.get('start_display','')}" for r in rows}
    st.caption('最多展示当前范围最近20次作答；这里只定位记录，不把旧标签迁移成已确认错误。')
    if ids:
        chosen = st.selectbox('打开一份作答',ids,format_func=lambda i:labels[i],key='teacher_recent_attempt')
        with st.expander('查看该次作答证据',expanded=False):
            render_evidence(chosen)
    else:
        st.info('本轮未取得作答记录。查询失败时请先恢复读取，再判断是否为空。')
