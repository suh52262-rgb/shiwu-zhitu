"""Generation, inspection and version-bound review in the question import page."""
import json
from uuid import uuid4
import streamlit as st

ACTIVE={'queued','running'}
LABELS={'queued':'排队','running':'生成/校验中','needs_review':'候选图待复核','needs_changes':'需修改候选图',
        'failed':'失败','interrupted':'服务重启中断','stale':'输入或候选版本已变'}

def render_graph_authoring(pid,request):
    st.subheader('自动生成解题Graph')
    st.caption('题目已发布不等于Graph已审定。模型只读题干和图示文字说明，未发布候选不用于教学。')
    from figure_context_ui import render_figure_context
    context_draft,context_confirmed=render_figure_context(pid,request)
    path=f'/problems/{pid}/graph-jobs'
    with st.form(f'graph_generate_{pid}'):
        context=st.text_area('已核对的图示说明（带图题必填：对象、连接、方向、图中数值）',key=f'graph_context_{pid}')
        quantities=st.text_input('要求解的目标量（用中文逗号分隔；不同小问都要覆盖）')
        physics=st.checkbox('增加模型物理一致性检查（额外模型请求；不等于人工审定）')
        repair=st.checkbox('校验发现可修问题时自动修订一次（最多额外一次生成；物理检查开启时还会复查）',value=True)
        generate=st.form_submit_button('生成候选Graph（调用模型）')
    if generate:
        signature=json.dumps([pid,context,quantities,physics,repair,context_draft,context_confirmed],ensure_ascii=False)
        requests=st.session_state.setdefault('graph_request_ids',{})
        rid=requests.setdefault(signature,str(uuid4()))
        result=request('POST',path,json={'client_request_id':rid,'figure_context':context,
            'quantities_asked':[q.strip() for q in quantities.replace(',','，').split('，') if q.strip()],
            'check_physics':physics,'repair_once':repair,
            'figure_context_draft_id':context_draft,'figure_context_confirmed':context_confirmed})
        if result:
            st.session_state[f'graph_next_{pid}']=result['job_id']
            st.rerun()
    listed=request('GET',path)
    if not listed or not listed['items']:
        st.info('还没有建图任务；题目可以先以graphless模式作答。')
        return
    st.caption(f"当前教学图：{listed.get('teaching_version') or '尚未发布（graphless）'}")
    jobs={j['job_id']:j for j in listed['items']}
    next_job=st.session_state.pop(f'graph_next_{pid}',None)
    if next_job in jobs:st.session_state[f'graph_job_{pid}']=next_job
    jid=st.selectbox('找回建图任务',list(jobs),key=f'graph_job_{pid}',
        format_func=lambda j:f"{j[:8]} · {LABELS.get(jobs[j]['status'],jobs[j]['status'])}")
    job=jobs[jid]
    st.code(f"任务：{jid}\n输入：{job['input_sha256']}\n候选版本：{job['graph_version'] or '尚未生成'}")
    if job['status'] in ACTIVE:
        @st.fragment(run_every='3s')
        def poll():
            latest=request('GET',path+'/'+jid)
            if latest:
                st.info(f"{LABELS.get(latest['status'],latest['status'])} · {latest['phase']}")
                if latest['status'] not in ACTIVE:st.rerun()
        poll()
        return
    if job['error']:st.error(job['error'])
    if st.button('新建一次生成请求（再次调用模型；不会覆盖已发布图）',key=f'graph_retry_{jid}'):
        options=job['input']['options']
        result=request('POST',path,json={'client_request_id':str(uuid4()),**options})
        if result:
            st.session_state[f'graph_next_{pid}']=result['job_id']
            st.rerun()
    if not job.get('graph'):
        if job.get('candidate'):
            with st.expander('失败前保留的候选输出'):st.json(job['candidate'])
        return
    graph=job['graph']
    dot=['digraph solution {','rankdir=TB;']
    for node in graph['nodes']:
        label=f"{node['id']} · {node['name']}"
        dot.append(f'{json.dumps(node["id"])} [label={json.dumps(label,ensure_ascii=False)}];')
    for source,target in graph['edges']:
        dot.append(f'{json.dumps(source)} -> {json.dumps(target)};')
    st.graphviz_chart('\n'.join(dot+['}']))
    for node in graph['nodes']:
        with st.expander(f"{node['id']} · {node['name']}"):
            st.write(node.get('description',''))
            st.write({'类型':node['type'],'知识点':node.get('knowledge_points',[]),'典型错误':node.get('common_errors',[])})
            catalogue={e['id']:e for e in graph['metadata'].get('source_evidence',[])}
            for eid in graph['metadata'].get('node_evidence',{}).get(node['id'],[]):
                if eid in catalogue:st.code(f"{eid} · {catalogue[eid]['source']}\n{catalogue[eid]['text']}")
    st.write('小问目标覆盖：',graph['metadata'].get('quantity_coverage',{}))
    if job.get('quality') and job['quality'].get('coverage_status')=='not_checked':
        st.info('未填写目标量，本轮没有自动验证每个小问是否覆盖；请对照原题核对。')
    if job.get('quality',{}).get('issues'):
        st.warning('仍有需核对问题，请优先处理下面的具体发现。')
        st.json(job['quality']['issues'])
    st.write('有效路径：',graph['valid_paths'])
    st.warning('规则校验和模型物理检查均不能证明解法正确；请核对原题与参考答案。')
    if job.get('vocabulary') and not job['vocabulary']['ok']:
        st.error(f"知识点含表外名称：{job['vocabulary']['unknown'] + job['vocabulary']['dropped']}。请编辑节点知识点为受控词表名称再复核发布。")
    with st.expander('可选的受控知识点名称（未知概念不自动并入词表）'):
        st.write(listed.get('knowledge_vocabulary',[]))
    with st.expander('校验详情、归一化记录、复核历史及耗时'):
        st.json({'validation':job['validation'],'normalizations':graph['metadata'].get('normalizations',[]),
                 'reviews':job['reviews'],'timings':job['timings'],'input':job['input'],
                 'model_calls':job.get('model_calls',{}),'vocabulary':job.get('vocabulary')})
        st.json({'quality':job.get('quality'),'automatic_repairs':job.get('automatic_repairs',[])})
    if job['publication']:
        if listed.get('teaching_version')==job['publication']['version']:
            st.success('此版本已发布且当前用于新作答；既有作答的绑定版本不变。')
        else:
            st.info('此版本曾发布；当前教学图已是另一版本，请以上方当前教学版本为准。')
        return
    if job['status'] not in {'needs_review','needs_changes'}:return
    with st.form(f'graph_edit_{jid}_{job["revision"]}'):
        editable=st.text_area('编辑节点、边与完整路径（JSON；来源与审核元数据由服务端维护）',
                              json.dumps(graph,ensure_ascii=False,indent=2),height=300)
        save=st.form_submit_button('保存修改并重新规则校验')
    if save:
        try:payload=json.loads(editable)
        except ValueError:st.error('Graph JSON格式有误')
        else:
            if request('PUT',path+'/'+jid,json={'expected_version':job['graph_version'],'graph':payload}):st.rerun()
    with st.form(f'graph_review_{jid}_{job["revision"]}'):
        kind=st.radio('复核身份',['人工复核','AI意见（不获得发布资格）'])
        reviewer=st.text_input('复核者标识')
        note=st.text_area('复核说明：条件、所有小问、方向符号、支路与汇合适用性')
        confirmed=st.checkbox('已核对当前版本的条件、所有小问、物理关系与每条完整路径；身份如实填写')
        reviewed=st.form_submit_button('记录当前版本复核')
    if reviewed:
        if request('POST',path+'/'+jid+'/review',json={'expected_version':job['graph_version'],'reviewer':reviewer,
            'note':note,'kind':'human' if kind=='人工复核' else 'ai','confirmed':confirmed}):st.rerun()
    publisher=st.text_input('发布者标识',key=f'publisher_{jid}')
    st.caption('发布的是已保存并复核的版本；编辑框内未保存的修改不生效。')
    if st.button('发布已人工复核的Graph',disabled=not job['review'],key=f'publish_graph_{jid}'):
        if request('POST',path+'/'+jid+'/publish',json={'expected_version':job['graph_version'],'publisher':publisher}):st.rerun()
