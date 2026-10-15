"""Local evidence review. Self-reported operators are not authenticated teachers."""
import json,os
from uuid import uuid4
import httpx
import streamlit as st

BASE=os.getenv('BACKEND_URL','http://127.0.0.1:8000')
st.set_page_config(page_title='书面错误复核',layout='wide')
st.title('书面错误证据复核')
st.caption('本地操作者页面，未验证教师身份。采纳或已处理只针对这一条书面记录，不证明概念掌握。')
sid=st.text_input('学生编号',value='stu_demo')
actor=st.text_input('本地操作者标识（AI核查请明确写AI）')

def request(method,path,**kwargs):
    try:
        response=httpx.request(method,BASE+path,timeout=15,trust_env=False,**kwargs)
        response.raise_for_status();return response.json()
    except httpx.HTTPStatusError as exc:st.error(str(exc.response.json().get('detail','复核接口错误')))
    except httpx.RequestError:st.error('请求结果未知，请刷新历史记录后再操作。')
    return None

path=f'/teacher/students/{sid}/error-evidence'
data=request('GET',path) if sid else None
if not data:st.stop()
if data['review_store_status']!='ready':st.warning('复核状态未同步，当前推荐不使用未确认的错误标签。')
if not data['items']:st.info('暂无带原文的当前书面错误证据；历史标签不直接作为当前错误。')
LABELS={'unreviewed':'待复核','confirmed':'本地采纳','dismissed':'撤销系统判断','resolved':'本记录已处理','unavailable':'复核状态暂未同步'}
for item in data['items']:
    eid=item['evidence_id']
    with st.expander(f"{item['concept']} · {LABELS[item['review_status']]}"):
        st.code(item['student_text'])
        st.write('系统判断（不是认知事实）：'+item['issue'])
        st.caption(f"作答{item['attempt_id']} · 来源{item.get('source_id','未记录')} · 修订{item.get('source_revision','未知')}")
        if item.get('review_decision'):st.json(item['review_decision'])
        decision=st.selectbox('处理方式',['confirmed','dismissed','resolved'],format_func=lambda k:LABELS[k],key='decision_'+eid)
        note=st.text_area('核对说明（例如参照哪份新书写）',key='note_'+eid)
        if st.button('保存这条复核决定',key='save_'+eid,disabled=not actor.strip()):
            body={'attempt_id':item['attempt_id'],'decision':decision,'actor_label':actor,'note':note}
            signature=json.dumps([sid,eid,body],sort_keys=True,ensure_ascii=False)
            ids=st.session_state.setdefault('review_request_ids',{})
            body['client_request_id']=ids.setdefault(signature,str(uuid4()))
            if request('POST',path+'/'+eid+'/decisions',json=body):st.rerun()
with st.expander('复核历史（不覆盖原始作答与诊断）'):st.json(data['history'])
