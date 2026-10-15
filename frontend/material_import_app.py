"""Local textbook catalogue and import guide. Does not execute imports."""
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'frontend'))
load_dotenv(ROOT / '.env')
from library_data import catalogue, material_preview
from resource_labels import document_title, document_label, document_matches
from workspace_style import apply_workspace_style, page_intro, section_note

st.set_page_config(page_title='视物知途 · 教材管理', layout='wide')
apply_workspace_style('materials')
page_intro('教材资源中心', '从来源文档进入材料与批次记录，再安排导入核对。', '资源中心 / 教材')
with st.sidebar:
    st.header('管理范围')
    st.caption('本页只读展示数据库材料清单，不上传文件、不调用模型、不重建索引。')
    st.button('刷新教材清单')

section_note('教材证据与题目作答分开管理', '教材入库为检索提供来源材料；导入成功不代表OCR、公式或检索质量已验证。')
try:
    documents = catalogue()
except Exception:
    documents = None
    st.error('教材清单读取失败。请检查数据库服务与项目配置；这不表示教材不存在。')

if documents is not None:
    a, b, c = st.columns(3)
    a.metric('已入库文档', len(documents))
    b.metric('已入库页数', sum(d['pages'] for d in documents))
    c.metric('材料块数', sum(d['blocks'] for d in documents))
    st.caption('数字来自PostgreSQL材料记录，不代表原PDF完整页数或Qdrant索引健康状态。')
    st.header('教材清单')
    query = st.text_input('按教材名称或编号查找', placeholder='如：电磁学、大学物理学、jiaocai-5')
    selected = [d for d in documents if document_matches(d['doc_id'],query)]
    if selected:
        st.dataframe([{'教材名称':document_title(d['doc_id']), '来源编号': d['doc_id'], '类型': d['doc_type'], '已入库页数': d['pages'], '材料块数': d['blocks']} for d in selected], hide_index=True, width='stretch')
        chosen=st.selectbox('核对一份来源文档',[d['doc_id'] for d in selected],format_func=document_label)
        st.caption('教材名称按本地源文件规范化显示；来源编号用于引用定位与导入，保持不变。')
        with st.expander('材料预览与最近导入批次'):
            st.caption('只读展示该文档最早入库页的前12块。页序号不等于印刷页码；片段可能有OCR错误，不代表该文档整体质量。')
            try:
                preview=material_preview(chosen)
            except Exception:
                st.warning('材料或批次预览读取失败；清单仍保留。不能解释成没有材料。')
            else:
                for block in preview['blocks']:
                    st.caption(f"第{block['page']}文件页 · 块{block['block_index']} · {block['type']} · "+('机器生成材料' if block['is_generated'] else '提取材料')+(' · 不参与索引' if not block['indexable'] else ''))
                    raw=block['text'] or ''
                    st.code(raw[:1200],language=None)
                    if len(raw)>1200:st.caption('预览截断；其余原文未在此展示。')
                st.markdown('**最近导入批次**')
                if preview['runs']:
                    st.dataframe(preview['runs'],hide_index=True,width='stretch')
                    st.caption('批次状态是工程记录，不证明教材内容已核对或检索质量达标。')
                else:st.caption('未读到历史批次记录；旧材料可能没有这项元数据。')
    else:
        st.info('没有匹配的材料记录。可修改关键词，或按下方指引检查与导入。')

st.header('沿用现有导入工具')
st.caption('在项目根目录的终端执行。示例路径与编号需要替换；本页不会自动执行这些操作。')
list_tab, import_tab, verify_tab = st.tabs(['查看清单', '导入教材', '核对批次'])
with list_tab:
    st.code(r'.\.venv\Scripts\python.exe scripts/import_pdf.py --list', language='powershell')
    st.write('数据库不可读时，先恢复服务，再核对清单；不要据此重复导入。')
with import_tab:
    st.code(r'.\.venv\Scripts\python.exe scripts/import_pdf.py "D:\教材\教材.pdf" --doc-id my-textbook --pages 1-3', language='powershell')
    st.warning('导入可能调用视觉模型并产生费用。建议先处理少量页，对照原PDF核对公式、文字与配图。')
    st.caption('同一文档编号的局部导入可能替换指定页，执行前请确认范围。--no-vl仅跳过视觉转写，并不保证没有嵌入模型请求。')
with verify_tab:
    st.code(r'.\.venv\Scripts\python.exe scripts/import_pdf.py --runs my-textbook', language='powershell')
    st.write('查看已有导入批次，确认处理范围与失败记录；页序号不能直接当作印刷页码。')
    st.caption('本页不提供删除、索引修复或自动重试按钮。需要检索质量判断时，应另行运行冻结评测。')
