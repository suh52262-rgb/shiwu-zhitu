"""Standalone local authoring page."""
import json
import os
import sys
from pathlib import Path
import httpx
import streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parent))
from graph_import_ui import render_graph_authoring

BASE=os.getenv('BACKEND_URL','http://127.0.0.1:8000')
st.set_page_config(page_title='题目导入与复核',layout='wide')
from workspace_style import apply_workspace_style, page_intro, section_note
from learning_workspace import render_authoring_steps
apply_workspace_style('problems')
with st.sidebar:
    workspace_mode=st.radio('题库工作区',['导入与复核','已发布题库'],key='authoring_workspace')
page_intro('题库资源中心 · 导入与复核' if workspace_mode=='导入与复核' else '题库浏览',
           '对照原页核对题干与配图，再发布到学生题库。' if workspace_mode=='导入与复核' else '查找已发布题目，核对完整题面与配图。', '题库与导入')

def request(method,path,**kwargs):
    try:
        response=httpx.request(method,BASE+path,timeout=600,trust_env=False,**kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(str(exc.response.json().get('detail',exc.response.text)))
    except httpx.RequestError:
        st.error('连接失败或请求结果未知。先刷新已有草稿，核对是否已创建/发布，不要直接重复提交。')
    return None

if workspace_mode=='已发布题库':
    st.header('已发布题库')
    catalogue=request('GET','/problems')
    if catalogue is not None:
        entries=catalogue.get('items') or []
        st.caption(f'当前题库 {len(entries)} 道题；本页只读浏览，不会覆盖题目或生成Graph。')
        query=st.text_input('按标题或题号查找')
        entries=[p for p in entries if query.casefold() in (str(p.get('title',''))+' '+str(p['problem_id'])).casefold()]
        if entries:
            chosen=st.selectbox('选择已发布题目',entries,format_func=lambda p:f"{p.get('title',p['problem_id'])} · {p['problem_id']}")
            detail=request('GET',f"/problems/{chosen['problem_id']}")
            if detail:
                st.subheader(detail.get('title',chosen['problem_id']))
                st.markdown(detail.get('statement') or detail.get('content') or '题干暂不可读')
                st.caption('知识点：'+'、'.join(detail.get('knowledge_points') or []))
                for fig in detail.get('figure_urls') or []:
                    st.image(fig if fig.startswith(('http://','https://')) else BASE+fig,width='stretch')
                labels={'published':'已发布','reviewed':'已复核，尚未发布','draft':'草稿，尚未发布','legacy_unreviewed':'演示图，尚待复核'}
                st.caption('解题流程图：'+labels.get(detail.get('graph_status'),'暂未提供可用状态'))
        else:
            st.info('未找到匹配题目，可修改关键词或切换到导入与复核。')
    st.stop()

_stepper=st.empty()
with _stepper.container():render_authoring_steps(0)
section_note('原页 → 核对草稿 → 发布题目', '草稿与学生题库分开；参考答案不向学生展示，解题Graph另行审核。')
st.header('导入来源文件')
upload=st.file_uploader('上传题目图片或试卷PDF（≤30MB、PDF≤30页）',type=['pdf','png','jpg','jpeg','webp'],max_upload_size=30)
vision=st.checkbox('调用视觉模型转写（会产生API费用；扫描件可选，不勾选也能手工录入）')
if st.button('创建导入草稿',disabled=upload is None):
    with st.spinner('提取与转写中，请等待；不承诺固定耗时'):
        record=request('POST','/problem-imports',files={'file':(upload.name,upload.getvalue(),upload.type)},data={'use_vision':str(vision).lower()})
    if record:
        st.session_state.import_id=record['import_id']
        st.rerun()

st.header('恢复与核对草稿')
st.button('刷新已有草稿')
listing=request('GET','/problem-imports')
if not listing or not listing['items']:
    st.info('暂无草稿。扫描件未启用视觉转写时，请对照原图手工录入。')
    st.stop()
records={r['import_id']:r for r in listing['items']}
ids=list(records)
selected=st.selectbox('恢复已有导入',ids,index=ids.index(st.session_state.get('import_id')) if st.session_state.get('import_id') in ids else 0,
                      format_func=lambda i:f"{records[i]['filename']} · {i[:8]} · {records[i]['status']}")
record=request('GET',f'/problem-imports/{selected}')
if not record:st.stop()
st.code(f"导入编号：{selected}\n原件SHA256：{record['source_sha256']}")
for warning in record['warnings']:st.warning(warning)
if record.get('error'):st.error(record['error'])
if not record['pages']:st.stop()
left,right=st.columns([1,1])
selected_crop=None
with left:
    page=st.selectbox('核对来源页（文件页序号，不是印刷页码）',[p['page'] for p in record['pages']])
    source=next(p for p in record['pages'] if p['page']==page)
    st.image(f'{BASE}/problem-imports/{selected}/pages/{page}',width='stretch')
    st.caption(f"{source['method']} · {source['status']}；原图尺寸{source.get('preview_width','?')}×{source.get('preview_height','?')}像素，裁切坐标以原图为准")
    if source.get('vision_error'):st.error(source['vision_error'])
    with st.expander('未经修改的提取/识别原文'):
        st.code(source['original_text'])
    with st.expander('拖拽框选配图（只加入草稿，需另行核对保存）'):
        try:
            from crop_selector import select_crop, validated_box
            preview=httpx.get(f'{BASE}/problem-imports/{selected}/pages/{page}',timeout=5,trust_env=False)
            preview.raise_for_status()
            selection=select_crop(preview.content,preview.headers.get('content-type','image/png'),key=f'crop_{selected}_{page}')
            selected_crop=validated_box(selection,source['preview_width'],source['preview_height'])
            if selected_crop:st.caption('原图坐标：'+str(selected_crop))
        except Exception:
            st.caption('拖拽预览暂不可用；可继续使用右侧像素坐标JSON。')
with right:
    if st.button('新增一道题（未切出、合并错误时可手工重建）'):
        if request('POST',f'/problem-imports/{selected}/items'):st.rerun()
    if not record['items']:st.stop()
    items={i['item_id']:i for i in record['items']}
    from authoring_labels import item_label
    item_ids=list(items)
    requested=st.session_state.get('requested_item_id')
    item_id=st.selectbox('选择草稿题目',item_ids,
        index=item_ids.index(requested) if requested in item_ids else 0,
        format_func=lambda i:item_label(items[i],item_ids.index(i)))
    item=items[item_id]
    with _stepper.container():render_authoring_steps(4 if item['publication'] else (3 if item['review'] else 2))
    for index in range(len(item['crops'])):
        st.image(f'{BASE}/problem-imports/{selected}/items/{item_id}/crops/{index}?revision={item["revision"]}',caption=f'已保存配图 {index+1}（核对是否完整、不含答案或别题）')
    if item['publication']:
        st.success(f"已发布：{item['publication']['problem_id']}。学生端刷新题库可使用。Graph审核独立进行。")
    else:
        crop_key=f'crop_text_{selected}_{item_id}_{item["revision"]}'
        confirm_key=f'confirm_{selected}_{item_id}_{item["revision"]}'
        if crop_key not in st.session_state:st.session_state[crop_key]=json.dumps(item['crops'],ensure_ascii=False)
        if st.button('将拖拽区域加入当前题目的配图草稿',disabled=selected_crop is None):
            try:
                draft_crops=json.loads(st.session_state[crop_key])
                candidate={'page':page,'bbox':selected_crop}
                if not isinstance(draft_crops,list):raise ValueError()
                if candidate not in draft_crops:draft_crops.append(candidate)
                st.session_state[crop_key]=json.dumps(draft_crops,ensure_ascii=False)
                st.session_state[confirm_key]=False
                st.rerun()
            except ValueError:st.error('当前配图JSON不合法，请先修正再加入区域。')
        with st.form(f'edit_{selected}_{item_id}_{item["revision"]}'):
            _question_tab,_answer_tab,_review_tab=st.tabs(['题面与配图','参考解','核对与保存'])
            with _question_tab:
                pid=st.text_input('题目编号（已有编号拒绝覆盖）',item['problem_id'],disabled=bool(item.get('parent_problem_id')))
                title=st.text_input('标题',item['title'])
                statement=st.text_area('完整题干、小问与选项（核对公式和单位）',item['statement'],height=240)
                pages=st.multiselect('实际来源页',[p['page'] for p in record['pages']],default=item['source_pages'])
                crops=st.text_area('确认的配图裁切JSON；无图填[]。不要把答案区域作为配图',key=crop_key,help='如 [{"page":1,"bbox":[50,80,350,260]}]；左、上、右、下')
            with _answer_tab:
                answer=st.text_area('参考答案（没有则留空，不向学生展示）',item['answer'],height=180)
            with _review_tab:
                reviewer=st.text_input('复核者标识（如本人姓名或角色；不得冒充教师审定）')
                confirmed=st.checkbox('我已对照来源页核对题干、答案、配图归属，配图不含答案或别题',key=confirm_key)
                saved=st.form_submit_button('保存核对结果')
        if saved:
            try:crop_values=json.loads(crops)
            except ValueError:st.error('配图JSON格式不正确')
            else:
                result=request('PUT',f'/problem-imports/{selected}/items/{item_id}',json={'base_revision':item['revision'],
                    'values':{'problem_id':pid,'title':title,'statement':statement,'answer':answer,'source_pages':pages,'crops':crop_values},
                    'reviewer':reviewer,'confirmed':confirmed})
                if result:st.rerun()
        if st.button('发布已核对的保存版本',disabled=not item['review']):
            if request('POST',f'/problem-imports/{selected}/items/{item_id}/publish',json={'base_revision':item['revision']}):st.rerun()
    with st.expander('识别原文、修订历史与复核记录'):
        st.json({k:item[k] for k in ('original_statement','original_answer','revision','history','review','publication')})

if item.get('publication'):
    st.caption('修订会建立关联的新题目版本；旧作答、题面和Graph保留。新版本须重新核对并生成Graph。')
    if st.button('建立题目修订版', key=f'revise_{item_id}'):
        revised=request('POST',f'/problem-imports/{selected}/items/{item_id}/revise',
                        json={'base_revision':item['revision']})
        if revised:
            st.session_state['requested_item_id']=revised['item_id']
            st.rerun()
    render_graph_authoring(item['publication']['problem_id'],request)
