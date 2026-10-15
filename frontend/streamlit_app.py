"""Task-oriented student workspaces: home, learning, video review, history.
Original problem and current work share a learning context. Presentation uses
bound Graph topology and owned evidence IDs without changing hint policy.
"""
import json
import os
import ssl
import sys
import time
import uuid
from pathlib import Path

import httpx
import streamlit as st

#: 纯逻辑模块（不依赖 streamlit、可单测）。
#: `sys.path` 里没有 frontend/ 时（比如从仓库根跑 streamlit）也要能导入。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_logic import (  # noqa: E402
    MODE_OK,
    apply_ocr_correction, correction_unavailable,
    apply_video_review, video_review_request, can_generate_review, review_categories,
    invalidate_video_job, video_job_context, retain_video_job, current_video_job, apply_video_job_result,
    rollback_video_job_start,
    clear_media, current_attempt_id, media_notices, record_photo_failure,
    reset_session, restore_attempt, switch_session,
    RECOVER_ABSENT, RECOVER_FOUND, RECOVER_UNKNOWN, remember_request_id,
    forget_request_id,
    connection_error_text,
    health_view,
    problems_from_payload,
    profile_view,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
STUDENT_ID = "stu_demo"  # Demo 单用户；后续接登录体系

st.set_page_config(page_title="视物知途 · 物理智能教学", page_icon="🔬", layout="wide")
from workspace_style import apply_workspace_style, page_intro
apply_workspace_style('student')
from problem_browser import matching_ids, problem_tags, tag_choices, selection_options
from resource_labels import document_title
from photo_job_state import matches as photo_matches, busy as photo_busy, submission_rejected
from learning_workspace import render_hint_ladder, render_progress, render_evidence_browser, target_round, set_evidence_target
def route_workspace_media():
    st.session_state['media_mode']='📹 传一段视频' if st.session_state.get('student_workspace')=='视频复盘' else '📷 拍一张照片'

if 'student_workspace' not in st.session_state and (st.query_params.get('video_job') or st.query_params.get('video_request')):
    st.session_state['student_workspace']='视频复盘'
with st.sidebar:
    workspace = st.radio('学习工作区', ['学习首页','学习作答','视频复盘','学习记录'], index=1, key='student_workspace',on_change=route_workspace_media)
_titles = {'学习首页':'我的物理学习','学习作答':'物理学习工作台','视频复盘':'视频过程复盘','学习记录':'作答与提示记录'}
page_intro(_titles[workspace], '原题、作答、提示与证据，在同一上下文里核对。', '学生学习')

def open_student_workspace(value):
    st.session_state['student_workspace'] = value
    st.session_state['media_mode'] = '📹 传一段视频' if value=='视频复盘' else '📷 拍一张照片'

def route_media_workspace():
    st.session_state['student_workspace'] = '视频复盘' if st.session_state.get('media_mode','').startswith('📹') else '学习作答'



#: **预先建好的** SSL 上下文 —— 只为绕开 httpx 每次建 Client 的开销，见 `_get`。
#:
#: 每次重跑建一次要 15ms；不建它、让 httpx 自己建，是 **305ms/次**。
_VERIFY = ssl.create_default_context()


def _get(url, **kwargs):
    """所有 **GET** 都走这里 —— 唯一的作用是**补上预先建好的 SSL 上下文**。

    ## 为什么（实测，2026-10-09）
    ```
    httpx.get(url)                 : 373ms 334ms 321ms 327ms   ← 用户说的「卡顿」
    本函数（补上 verify=_VERIFY）   :   2ms  27ms   2ms   2ms   ← 快两个数量级
    httpx.Client(verify=False)     :   0ms                    ← 与 TLS 无关就更便宜
    httpx.create_ssl_context()     : 305ms   ← 成本在这
    ssl.create_default_context()   :  15ms   ← 标准库同功能只要这么点
    ```
    `httpx.get()` 每次都新建 Client，而新建 Client 会调 `httpx.create_ssl_context()`
    （比标准库多花约 290ms，多半是加载证书包）。
    一次脚本重跑要打 3 个端点（题库 / 健康 / 画像）⇒ **每次交互白付近一秒**。
    Streamlit 的模型是"每次交互重跑整个脚本"，改不了；**能改的是别每次重造那 300ms。**

    ⚠️ **它照常调用 `httpx.get`** —— 测试与故障注入打桩的那个接缝**原样有效**。
    我第一版想"检测到被打桩就走打桩"，那是**错的**：
    `_ORIGINAL_GET = httpx.get` 在模块执行时求值，而测试**先**打桩
    ⇒ "原版"变成了桩，判断整个反过来，**34 条测试挂掉**。
    ⇒ **加一层转发就够，别去猜别人有没有打桩。**
    """
    kwargs.setdefault("verify", _VERIFY)
    return httpx.get(url, **kwargs)


def _scan_local_problems() -> dict[str, dict]:
    """后端不可用时的退路：扫描 data/problems/{id}/problem.json（P16 布局）。"""
    problems = {}
    for path in sorted((PROJECT_ROOT / "data" / "problems").glob("*/problem.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            problems[data["problem_id"]] = data
        except Exception:  # noqa: BLE001 单个题目文件损坏不拖垮页面
            continue
    return problems


def load_problems() -> dict[str, dict]:
    """题目列表：优先走后端 /problems（P16 起的唯一口径），失败则扫本地。

    多题管理交给后端 ProblemRepository，前端不再自己解析题库目录结构。

    **判断逻辑在 `ui_logic.problems_from_payload`** —— 那个模块不依赖
    streamlit、可单测。这里只负责拿响应和回退。
    """
    try:
        resp = _get(f"{BACKEND_URL}/problems", timeout=3)
        got = problems_from_payload(resp.status_code, resp.json())
        if got:
            return got
    except Exception:  # noqa: BLE001 后端未启动属正常情况，静默回退
        pass
    return _scan_local_problems()


PROBLEMS = load_problems()

# ---------------- 侧边栏：后端状态 + 学生画像；选题在主区 ----------------
with st.sidebar:
    st.header("系统状态")
    backend_ok = False
    try:
        resp = _get(f"{BACKEND_URL}/health", timeout=3)
        # ⚠️ 原来这里是 `resp.json()['status']` 直接取——
        # **非 200 时 body 可能是 `{"detail": ...}`，会 KeyError**
        # 然后被 `except` 说成"后端未连接"，而其实后端连着、只是返回异常。
        hv = health_view(resp.status_code, resp.json())
        if hv['ok']:
            ready_response = _get(f'{BACKEND_URL}/runtime/readiness', timeout=3)
            if ready_response.status_code == 200 and not ready_response.json().get('ready', True):
                hv = {'ok': False, 'text': ready_response.json().get('message') or '本地教材模型尚未就绪，请稍后刷新'}
        if hv["ok"]:
            st.success(hv["text"])
            backend_ok = True
        else:
            st.warning(hv["text"])
    except Exception as exc:  # noqa: BLE001 连接失败只提示，不中断页面
        st.error(connection_error_text(BACKEND_URL, exc))

    if backend_ok:
        with st.expander('运行配置与验收范围'):
            try:
                runtime_response = _get(f'{BACKEND_URL}/runtime/configuration', timeout=3)
                runtime_response.raise_for_status()
                runtime = runtime_response.json()
                st.write('教学模型：' + runtime['models']['tutor'])
                st.write('提示版本：' + runtime['prompts']['tutor']['version'])
                st.write('复盘版本：' + runtime['prompts']['review']['version'])
                st.caption('配置显示不代表当前源码已完成教学质量验收。')
            except Exception:
                st.caption('运行配置暂不可读；不据此推断验收状态。')

    st.divider()
    st.subheader("学生画像")
    if backend_ok:
        try:
            resp = _get(f"{BACKEND_URL}/students/{STUDENT_ID}/profile", timeout=5)
            # ⚠️ **判断逻辑在 `ui_logic.profile_view`** —— 可单测。
            # 原来这里是 `.json()` 直接接下去，于是 404（新学生，
            # **最常见的情况**）的 body `{"detail": "学生 xxx 不存在"}`
            # 让下一行 `profile["attempts"]` 抛 KeyError，被最外层
            # `except Exception` 吞掉，界面显示"画像暂不可用（**数据库未连接**）"
            # —— 数据库好得很，只是这个学生还没有记录。
            #
            # 而且 `except httpx.HTTPStatusError` 那条**从来没生效过**：
            # 没调 `raise_for_status()`，httpx 根本不会抛这个异常。
            view = profile_view(resp.status_code, resp.json())
            if view["mode"] == MODE_OK:
                s = view["stats"]
                weak = s["weak"]
                st.markdown(
                    f"- 作答次数：**{s['attempt_count']}**\n"
                    f"- 完成题目：**{s['done']}**\n"
                    f"- 书面错误涉及知识点（系统判断）：{'、'.join(weak) if weak else '暂无'}"
                )
            else:
                st.caption(view["caption"])
        except Exception:  # noqa: BLE001 后端未启动等情况
            st.caption("画像暂不可用（后端未连接）")

if not PROBLEMS:
    st.error("题库为空：请检查 data/problems/{problem_id}/problem.json，或后端 /problems 接口")
    st.stop()
_all_problem_ids = list(PROBLEMS)
_restore_pid = st.query_params.get('video_job_problem') or st.query_params.get('photo_problem')
_current_pid = st.session_state.get('problem_select')
if _current_pid not in PROBLEMS:
    _current_pid = (_restore_pid if (st.query_params.get('video_job_student') or st.query_params.get('photo_student')) == STUDENT_ID
                    and _restore_pid in PROBLEMS else _all_problem_ids[0])
st.caption(f"当前选题：{PROBLEMS[_current_pid].get('title', _current_pid)} · {_current_pid}")
with st.expander('选择题目 · 搜索与标签筛选', expanded=workspace=='学习首页'):
    _available_tags = tag_choices(PROBLEMS)
    if 'problem_tags' in st.session_state:
        st.session_state['problem_tags'] = [t for t in st.session_state['problem_tags'] if t in _available_tags]
    _search_row = st.columns(2, gap='large')
    with _search_row[0]:
        _query = st.text_input('搜索题目', placeholder='标题、题号或知识点', key='problem_query')
    with _search_row[1]:
        _tags = st.multiselect('按知识点标签筛选', _available_tags, key='problem_tags',
                              disabled=not _available_tags, placeholder='选择知识点标签，可多选',
                              help='标签来自题目已有知识点，不通过模型临时生成。')
    _actions = st.columns([3,1])
    with _actions[0]:
        _require_all = st.checkbox('同时包含所有所选标签', key='problem_tags_all', disabled=len(_tags)<2)
    def clear_problem_filters():
        st.session_state['problem_query']=''
        st.session_state['problem_tags']=[]
        st.session_state['problem_tags_all']=False
    with _actions[1]:
        st.button('清空选题筛选', on_click=clear_problem_filters, key='clear_problem_filters',
                  disabled=not (_query or _tags), use_container_width=True)
    _matches = matching_ids(PROBLEMS, _query, _tags, require_all=_require_all)
    st.caption(f'匹配 {len(_matches)} / {len(PROBLEMS)} 道题；多标签默认满足任意一个。')
    if not _matches:
        st.info('没有匹配题目。当前题目和作答保留，可清空筛选继续查找。')
    elif _current_pid not in _matches:
        st.caption('当前题不在筛选结果中，仍保留；选择匹配题目后才切换。')
    _problem_ids = selection_options(_matches, _current_pid)
    def problem_option_label(pid):
        title=f"{PROBLEMS[pid].get('title',pid)}（{pid}）"
        if pid not in _matches:
            return title+' · 当前题（不在筛选结果）'
        tags=problem_tags(PROBLEMS[pid])
        return title+' · '+(' / '.join(tags[:2])+('等' if len(tags)>2 else '') if tags else '未标注标签')
    problem_id = st.selectbox(
        "题目", _problem_ids, index=_problem_ids.index(_current_pid),
        format_func=problem_option_label, key="problem_select", disabled=not _matches,
    )
    with st.expander('当前题目标签'):
        st.write('、'.join(problem_tags(PROBLEMS[problem_id])) or '此题暂未标注知识点；仍可按标题或题号查找。')
    st.caption("切换题目会保存当前进度，返回时恢复未完成作答；重新开始才重置")


problem = PROBLEMS[problem_id]
problem_detail_error = None
if backend_ok:
    try:
        detail_response = _get(f"{BACKEND_URL}/problems/{problem_id}", timeout=5)
        detail_response.raise_for_status()
        detail = detail_response.json()
        if detail.get("problem_id") != problem_id:
            raise ValueError("题目详情编号不匹配")
        if not (detail.get("statement") or detail.get("content") or "").strip():
            raise ValueError("题目详情缺少题干")
        problem = {**problem, **detail}
    except Exception as exc:
        # 列表仅用于选题，不能把摘要或上一题的材料当作完整原题。
        problem_detail_error = f"完整原题读取失败，请刷新后重试：{exc}"

# ---------------- 会话状态初始化 ----------------
def _reset_local() -> None:
    st.session_state['photo_epoch'] = st.session_state.get('photo_epoch', 0) + 1
    st.session_state.pop('photo_pending', None)
    for photo_key in ('photo_job','photo_request','photo_problem','photo_student','photo_base_attempt'):
        st.query_params.pop(photo_key, None)
    invalidate_video_job(st.session_state)
    st.query_params.pop("video_job", None)
    st.query_params.pop("video_job_problem", None)
    st.query_params.pop("video_job_student", None)
    reset_session(st.session_state, time.time())
    st.session_state.pop("_answer_draft", None)
    st.session_state["_clear_answer_on_rerun"]=True


_drafts=st.session_state.setdefault('workbench_drafts', {})
_previous=st.session_state.get('active_session')
if _previous and 'answer_input' in st.session_state:
    _drafts[tuple(_previous)]=st.session_state['answer_input']
if _previous != (STUDENT_ID, problem_id):
    st.session_state['answer_input']=_drafts.get((STUDENT_ID,problem_id),'')
    st.session_state['_answer_draft']=st.session_state['answer_input']
if st.session_state.pop('_clear_answer_on_rerun', False):
    st.session_state['answer_input']=''
    st.session_state['_answer_draft']=''
    _drafts[(STUDENT_ID,problem_id)]=''
_changed = switch_session(st.session_state, STUDENT_ID, problem_id, time.time())
if _changed:
    st.session_state.pop("_answer_draft", None)
if _changed and current_attempt_id(st.session_state) and backend_ok:
    try:
        _response = _get(
            f"{BACKEND_URL}/attempt/{st.session_state.attempt_id}", timeout=10)
        _response.raise_for_status()
        if not restore_attempt(st.session_state, _response.json(), STUDENT_ID, problem_id):
            # A completed/abandoned record must not be sent as a continuation.
            st.session_state.attempt_id = None
            st.warning("原作答已结束或归属不匹配，下次提交将开始新的作答。")
    except Exception as exc:
        st.warning(f"暂时无法刷新历史记录，已恢复本地进度：{exc}")

# ---------------- 会话工具函数 ----------------


def add_event(event_type: str, content: str) -> None:
    st.session_state.timeline.append(
        {"timestamp": round(time.time() - st.session_state.start_ts, 1),
         "type": event_type, "content": content}
    )


def call_analyze(answer: str, request_hint: bool) -> None:
    """调用后端分析接口，更新 Tutor 面板与本地 Timeline。"""
    if photo_busy(st.session_state, STUDENT_ID, problem_id):
        st.warning('照片提示仍在生成，请等本次结果后继续作答或请求帮助。')
        return
    idle_time = round(time.time() - st.session_state.last_action_ts, 1)
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/attempt/analyze",
            json={
                "student_id": STUDENT_ID,
                "problem_id": problem_id,
                "answer": answer,
                "attempt_id": current_attempt_id(st.session_state) or "",
                "idle_time": idle_time,
                "request_hint": request_hint,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        st.session_state.tutor = {"hint": f"⚠️ 调用后端失败：{exc}"}
        return

    invalidate_video_job(st.session_state)
    if answer.strip():
        clear_media(st.session_state)
    st.session_state.tutor = data
    st.session_state.attempt_id = data.get("attempt_id")
    st.session_state.attempt_status = "completed" if data.get("finished") else "in_progress"
    st.session_state.db_record = None
    st.session_state.last_action_ts = time.time()
    add_event("hint_request" if request_hint else "submit",
              answer if not request_hint else "请求提示")
    if data.get("finished"):
        st.balloons()


def call_image_analyze(uploaded) -> None:
    """**拍一张照片 → 一次批改 → 下一步提示。**

    ## 为什么加这条
    视频的成本主要是**解码**（855 帧），而**卡住时的求助根本不需要过程** ——
    学生举着纸拍一张就够了。这不是"视频的简化版"，是**另一条更贴切的入口**：
    他要的不是"回放我怎么做题的"，是"我卡在这了，你看一眼"。

    ## 结果怎么进面板
    与视频那条**共用同一套渲染**（`video_result` + `rounds`）——
    后端返回的形状是一致的，所以这里只负责存进去。

    ## ⚠️ 不承诺秒数（2026-10-09 决定）
    这里以前写着**一个固定的预计秒数**，而那是**一次实测的数字**，
    不是对网络模型的保证 —— 它的来源（原本含复盘）后来也变了。
    ⇒ 处理中只说在做什么，**不报预计秒数**；失败时给**原因 + 重试入口**。
    （回归测试按「约 N 秒」这个**模式**断言，不是只挡某一个数字。）
    """
    # 重试要能**原样重发**同一张照片：学生不该为了重试再选一次文件。
    # 只留最小的三件（名字/字节/类型），不把 Streamlit 的 UploadedFile 存进 session。
    name = getattr(uploaded, "name", "photo.jpg")
    mime = getattr(uploaded, "type", "") or "image/jpeg"
    data = uploaded.getvalue()
    clear_media(st.session_state)
    try:
        files = {"file": (name, data, mime)}
        # 契约 §2.1「照片即时辅导」：**显式不要复盘**。
        #
        # 学生卡住时举着一张纸，要的是**下一步提示**，不是一整段过程复盘；
        # 而复盘会把它拖慢数秒（实测这段约 4.9 秒），他此刻正在等。
        # ⚠️ `request_hint=true` 是**主动求助**的语义（走分级提示策略），
        # 不是「让他看见答案」—— 级别仍由 Hint Policy 决定。
        data_form = {"problem_id": problem_id, "student_id": STUDENT_ID,
                     "request_hint": "true", "include_review": "false"}
        if current_attempt_id(st.session_state):
            data_form["attempt_id"] = current_attempt_id(st.session_state)
        with st.spinner("正在识别作答并生成下一步提示……"):
            resp = httpx.post(f"{BACKEND_URL}/video/analyze_image",
                              files=files, data=data_form, timeout=300)
        if resp.status_code != 200:
            # ⚠️ **后端 `detail` 不能丢** —— 否则又是一个光秃秃的状态码。
            _detail = ""
            try:
                _detail = (resp.json() or {}).get("detail") or ""
            except Exception:  # noqa: BLE001
                _detail = (resp.text or "")[:300]
            record_photo_failure(st.session_state,
                                 f"⚠️ 照片处理失败（HTTP {resp.status_code}）：{_detail}",
                                 name, data, mime)
            return
        result = resp.json()
    except Exception as exc:  # noqa: BLE001
        record_photo_failure(st.session_state,
                             f"⚠️ 照片处理失败：{type(exc).__name__}: {exc}", name, data, mime)
        return

    rounds = result.get("rounds") or []
    if not rounds:
        record_photo_failure(
            st.session_state,
            "⚠️ 这张照片里没有识别到作答内容。"
            "确认字迹清晰、画面里确实有解题过程，再拍一张。", name, data, mime)
        return
    invalidate_video_job(st.session_state)
    st.session_state.tutor = dict(rounds[0])
    st.session_state.attempt_id = result.get("attempt_id")
    st.session_state.attempt_status = "completed" if rounds[0].get("finished") else "in_progress"
    st.session_state.db_record = None
    # 与视频那条保持一致。
    st.session_state.last_action_ts = time.time()
    st.session_state.video_result = result
    # 成功了就没有要重试的东西。
    st.session_state.pop("photo_error", None)
    st.session_state.pop("photo_retry", None)
    add_event("submit", "（照片作答）")


def _photo_url(pending):
    for key, value in {'photo_problem': pending['problem_id'], 'photo_student': pending['student_id'],
                       'photo_base_attempt': pending.get('base_attempt_id', ''),
                       'photo_request': pending['request_id']}.items():
        st.query_params[key] = value
    if pending.get('job_id'):
        st.query_params['photo_job'] = pending['job_id']


def _submit_photo_job(pending):
    name, raw, mime = pending['upload']
    form = {'student_id': pending['student_id'], 'problem_id': pending['problem_id'],
            'attempt_id': pending.get('request_attempt_id', pending['base_attempt_id']), 'client_request_id': pending['request_id'],
            'request_hint': 'true', 'include_review': 'false'}
    try:
        response = httpx.post(f'{BACKEND_URL}/photo/jobs', data=form,
                              files={'file': (name, raw, mime)}, timeout=30)
        response.raise_for_status()
        job = response.json()
        pending.update(job_id=job['job_id'], status=job['status'])
        if job.get('attempt_id'):
            st.session_state.attempt_id = job['attempt_id']
            pending['base_attempt_id'] = job['attempt_id']
        pending.pop('lookup_error', None)
    except Exception as exc:
        # Unknown POST outcome: GET the same UUID; never invent a second request.
        pending['lookup_error'] = f'提交结果待确认：{type(exc).__name__}。正在按原编号查询。'
        rejection = submission_rejected(exc)
        if rejection is not None:
            pending['status'] = 'failed'
            pending['lookup_error'] = f'照片提交被拒绝（{rejection}），本次没有启动分析。请检查题目、作答状态或队列容量。'
    _photo_url(pending)


def start_photo_job(uploaded):
    old = st.session_state.get('photo_pending') or {}
    if old and old.get('status') not in {'succeeded', 'failed', 'interrupted'}:
        st.warning('已有照片任务正在处理或待找回，请先查看该任务。')
        return
    pending = {'student_id': STUDENT_ID, 'problem_id': problem_id,
               'base_attempt_id': current_attempt_id(st.session_state) or '',
               'request_attempt_id': current_attempt_id(st.session_state) or '',
               'epoch': st.session_state.get('photo_epoch', 0),
               'request_id': str(uuid.uuid4()), 'status': 'submitting',
               'upload': (getattr(uploaded, 'name', 'photo.jpg'), uploaded.getvalue(),
                          getattr(uploaded, 'type', '') or 'image/jpeg')}
    st.session_state['photo_pending'] = pending
    clear_media(st.session_state)
    _photo_url(pending)
    _submit_photo_job(pending)


def render_photo_job():
    pending = st.session_state.get('photo_pending')
    if not pending and st.query_params.get('photo_student') == STUDENT_ID and st.query_params.get('photo_problem') == problem_id:
        pending = {'student_id': STUDENT_ID, 'problem_id': problem_id,
                   'base_attempt_id': st.query_params.get('photo_base_attempt', ''),
                   'job_id': st.query_params.get('photo_job'),
                   'request_id': st.query_params.get('photo_request', ''), 'status': 'unknown',
                   'epoch': st.session_state.get('photo_epoch', 0), 'restored': True}
        st.session_state['photo_pending'] = pending
    if not pending or not photo_matches(st.session_state, pending, STUDENT_ID, problem_id):
        return
    if pending.get('status') == 'failed' and not pending.get('job_id'):
        st.error(pending.get('lookup_error', '照片提交未成功。'))
        return
    active = pending.get('status') not in {'succeeded', 'failed', 'interrupted'}

    @st.fragment(run_every=3 if active else None)
    def poll():
        if not photo_matches(st.session_state, pending, STUDENT_ID, problem_id):
            return
        st.caption('照片任务：' + (pending.get('job_id') or pending['request_id']))
        try:
            route = ('/photo/jobs/' + pending['job_id'] if pending.get('job_id') else
                     '/photo/jobs/by-request/' + pending['request_id'])
            response = _get(BACKEND_URL + route, params={'student_id': STUDENT_ID, 'problem_id': problem_id}, timeout=10)
            if response.status_code == 404 and not pending.get('job_id'):
                st.warning('原编号尚未查到任务；不会自动重新上传。可继续查询。')
                if pending.get('upload') and st.button('按原编号重新提交（不重复分析）', key='photo_same_uuid_retry'):
                    _submit_photo_job(pending)
                    st.rerun()
                return
            response.raise_for_status()
            job = response.json()
            pending.update(job_id=job['job_id'], status=job['status'])
            if job.get('attempt_id'):
                st.session_state.attempt_id = job['attempt_id']
                pending['base_attempt_id'] = job['attempt_id']
            _photo_url(pending)
        except Exception as exc:
            st.warning(f'照片进度暂不可读：{type(exc).__name__}。查询失败不会重新上传。')
            st.button('手动查询照片结果', key='photo_manual_lookup')
            return
        if job.get('recognized_text'):
            st.markdown('**已识别到的原文（请核对，尚不代表批改结论）**')
            st.code(job['recognized_text'], language=None)
        if job['status'] in {'queued', 'running'}:
            st.info('已识别原文，正在生成下一步提示。' if job.get('recognized_text') else '照片已接收，等待或正在识别。')
            st.caption('每3秒只读查询；刷新页面会找回同一任务。')
            return
        if job['status'] != 'succeeded':
            if active:
                st.rerun()  # rebuild fragment without its periodic timer
            st.error((job.get('error') or {}).get('message', '照片任务未完成，请先核对已保存作答。'))
            st.caption('任务已停止自动查询，不会自动重新调用模型；中断时可能存在部分作答。')
        elif not pending.get('applied'):
            result = job.get('result') or {}
            rounds = result.get('rounds') or []
            if not rounds:
                st.error('没有可展示的分析结果，请核对原稿。')
                return
            # Apply only once, to the creating attempt/problem. GET refreshes do not add submissions.
            st.session_state.tutor = dict(rounds[-1])
            st.session_state.attempt_id = result.get('attempt_id')
            st.session_state.attempt_status = 'completed' if rounds[-1].get('finished') else 'in_progress'
            st.session_state.video_result = result
            st.session_state.db_record = None
            st.session_state.last_action_ts = time.time()
            invalidate_video_job(st.session_state)
            pending['applied'] = True
            pending.pop('upload', None)
            add_event('submit', '（照片作答）')
            st.rerun()
        else:
            st.caption('照片任务已完成；自动查询已停止。')
    poll()


class _RetryUpload:
    """把 `session_state` 里那份最小的上传三元组还原成 `call_image_analyze` 要的形状。

    只实现它真正用到的三件：`name` / `type` / `getvalue()`。
    """

    def __init__(self, payload: dict) -> None:
        self.name = payload.get("name") or "photo.jpg"
        self.type = payload.get("mime") or "image/jpeg"
        self._data = payload.get("data") or b""

    def getvalue(self) -> bytes:
        return self._data


def stuck_technical_detail(tutor: dict, rounds: list | None = None) -> None:
    """系统卡顿分：**收进技术详情，并标明它是启发式指标**（T3，2026-10-09）。

    ## 为什么把它从学生主文案里撤下来
    `stuck_score` 是**规则型**打分：空闲时长 / 错误次数 / 请求提示次数。
    它**没有校准过**与「学生真的卡住」的关系 —— 文档里反复写过这件事，
    无图题（`phy_001`）上还实测偏低 +0.35（那是口径问题，不是 bug）。

    ⇒ 把一个未校准的分数摆在学生面前、还配上「建议回顾薄弱知识点」，
      就是在**替他下认知结论**。这个数字要留着（排障和教师端要用），
      **但得放在它该在的位置，并说清它是什么、不是什么。**
    """
    if not tutor:
        return
    score = tutor.get("stuck_score")
    if score is None:
        return
    with st.expander("🔧 技术详情：系统卡顿分（**启发式指标**，不代表你的状态）"):
        st.caption(
            "**这不是对你「卡没卡住」的判断。** 它由固定规则算出："
            "离开多久、连续错几次、点了几次「请求提示」。"
            "**它与真实卡顿的关系没有校准过**，无图题上实测还偏低。"
            "提示级别也不只看它 —— 还看是否关键节点、是否刚换节点、是否反复求助。"
        )
        st.caption(f"本轮卡顿分：**{float(score):.2f}**（0~1）")
        if tutor.get("stuck_reasons"):
            st.caption("触发信号：" + "；".join(tutor["stuck_reasons"]))
        if tutor.get("hint_reasons"):
            st.caption("提示级别依据：" + "；".join(tutor["hint_reasons"]))
        series = [(r.get("round"), r.get("stuck_score")) for r in (rounds or [])]
        series = [(n, s) for n, s in series if s is not None]
        if series:
            st.caption("逐轮：" + "　".join(f"第 {n} 轮 {float(s):.2f}" for n, s in series))


def call_ocr_correction(source: dict, corrected_text: str) -> bool:
    """Recompute from retained OCR text without uploading the media again."""
    source_id = source["source_id"]
    error_key = f"ocr_error_{source_id}"
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/video/correct_ocr",
            json={"student_id": STUDENT_ID, "problem_id": problem_id,
                  "attempt_id": current_attempt_id(st.session_state), "source_id": source_id,
                  "base_revision": source.get("source_revision", 0),
                  "corrected_text": corrected_text}, timeout=120)
        if resp.status_code != 200:
            detail = resp.json().get("detail") or f"HTTP {resp.status_code}"
            raise ValueError(str(detail))
        apply_ocr_correction(st.session_state, resp.json(), source_id)
        invalidate_video_job(st.session_state)
    except Exception as exc:  # noqa: BLE001
        st.session_state[error_key] = f"更正未生效，原进度与编辑内容已保留：{exc}"
        return False
    st.session_state.pop(error_key, None)
    st.session_state.last_action_ts = time.time()
    return True


def recover_video_job_by_request(request_id: str) -> str:
    """创建请求**结果未知**时，先按上传编号找回（契约 §2.2）。

    ⚠️ **超时不等于没建成。** 服务端可能已经收下并开始处理了，
    这时让学生重传，就是**第二次付费**、还可能产生第二条作答记录。
    ⇒ 契约把动作定死了：先按编号查询，**不得直接重传**。

    ⇒ 所以它**必须区分「确认没有」与「没能确认」**：
    只有 **404** 才说明"确实没有这个编号的任务"。
    连不上、5xx、响应读不出来 —— 一律 `unknown`，**不能说可以重传**。
    """
    context = video_job_context(st.session_state, STUDENT_ID, problem_id)
    st.session_state.pop("video_job_request_absent_confirmed", None)
    try:
        response = _get(f"{BACKEND_URL}/video/jobs/by-request/{request_id}",
                             params={"student_id": STUDENT_ID, "problem_id": problem_id},
                             timeout=15)
    except Exception:  # noqa: BLE001
        return RECOVER_UNKNOWN
    if response.status_code == 404:
        # ⚠️ 404 是**唯一的**"确认没有"信号（归属不符也用 404，不泄露存在性）。
        # A 404 is point-in-time: an earlier POST can still be spooling.
        # Keep its id for explicit retry so late admission cannot duplicate work.
        remember_request_id(st.session_state, request_id, st.query_params)
        st.session_state.video_job_request_absent_confirmed = True
        return RECOVER_ABSENT
    if response.status_code != 200:
        return RECOVER_UNKNOWN
    try:
        payload = response.json()
        job_id = payload["job_id"]
        retain_video_job(st.session_state, context, payload)
    except Exception:  # noqa: BLE001 拿到 200 但用不了 ⇒ 仍属未知
        return RECOVER_UNKNOWN
    st.query_params["video_job"] = job_id
    st.query_params["video_job_problem"] = problem_id
    st.query_params["video_job_student"] = STUDENT_ID
    # 编号的使命结束：任务已经接管到手。
    st.session_state.pop("video_job_request_id", None)
    st.session_state.pop("video_job_request_absent_confirmed", None)
    st.query_params.pop("video_request", None)
    return RECOVER_FOUND


def call_video_analyze(uploaded) -> None:
    """Enqueue once; later reruns only query the existing task."""
    invalidate_video_job(st.session_state)
    context = video_job_context(st.session_state, STUDENT_ID, problem_id)
    # ⚠️ 编号在**一次上传动作开始时**生成（契约 §2.2）——
    # 它不是"任务的编号"，而是"**这次上传**的编号"：
    # 重传会拿到一个新编号，而**同一次上传的重试**必须带同一个编号。
    request_id = (st.session_state.get("video_job_request_id")
                  or st.query_params.get("video_request") or str(uuid.uuid4()))
    remember_request_id(st.session_state, request_id, st.query_params)
    st.session_state.pop("video_job_request_absent_confirmed", None)
    # 新的一次上传 ⇒ 自动查询重新开始（上一次可能因为查询失败被暂停过）。
    st.session_state.pop("video_job_poll_paused", None)
    admitted = False
    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "video/mp4")}
        data_form = {"problem_id": problem_id, "student_id": STUDENT_ID, "every_sec": "20",
                     "client_request_id": request_id}
        if current_attempt_id(st.session_state):
            data_form["attempt_id"] = current_attempt_id(st.session_state)
        with st.spinner("正在上传并创建视频任务……"):
            response = httpx.post(f"{BACKEND_URL}/video/jobs", files=files, data=data_form, timeout=120)
        response.raise_for_status()
        payload = response.json()
        retain_video_job(st.session_state, context, payload)
        admitted = True
        st.query_params["video_job"] = payload["job_id"]
        st.query_params["video_job_problem"] = problem_id
        st.query_params["video_job_student"] = STUDENT_ID
        st.session_state.pop("video_job_error", None)
        st.session_state.pop("video_job_request_id", None)
        st.query_params.pop("video_request", None)
    except Exception as exc:
        # ⚠️ **409 要单独处理**（契约 T0 不变式⑤）：同一个编号 + **不同内容/参数**
        #    ⇒ 服务端按内容判冲突。**这正是"想试第二个视频"会撞上的那一步。**
        #    旧实现落到下面的"按编号查询"分支，而那次请求**根本没建成**
        #    ⇒ 提示「可以用**同一编号**重试」⇒ 再传还是 409（**死循环**，刷新也甩不掉，
        #      因为编号写在 URL 里）。
        #    ⇒ 这里直接给出出口，不再建议"用同一编号重试"。
        if getattr(getattr(exc, "response", None), "status_code", None) == 409:
            rollback_video_job_start(st.session_state, context)
            st.session_state.video_job_error = (
                f"⚠️ 这次上传的内容与编号 `{request_id}` 记录的那一次**不一样**，"
                "服务端按内容判为冲突（409）。编号复用**只对「同一次上传」成立**。"
                "⇒ 想上传新视频：点下面的「**放弃这次上传，开始新的一次**」；"
                "想找上一次的结果：点「按编号查询那次上传」。")
            return
        # 结果未知 ⇒ **先按编号找回**；三种结果的说法完全不同。
        outcome = recover_video_job_by_request(request_id)
        if outcome == RECOVER_FOUND:
            admitted = True
            # ⚠️ handler 自己**显式清一次**，不依赖查找函数的副作用 ——
            # 状态机不该建立在"被调用的那个函数顺手做了件事"之上。
            st.session_state.pop("video_job_request_id", None)
            st.query_params.pop("video_request", None)
            st.session_state.video_job_error = (
                f"⚠️ 创建请求没收到回执（{exc}），但**按编号查到了任务**，已接管。"
                "不会重复上传。")
            return
        rollback_video_job_start(st.session_state, context)
        if outcome == RECOVER_ABSENT:
            st.session_state.video_job_error = (
                f"⚠️ 视频任务创建未确认，原进度已保留：{exc}。"
                f"已按编号 `{request_id}` 查询，服务端当前未找到任务；原请求仍可能稍后完成。"
                "可以用**同一编号**重试，系统会检查内容和参数，避免重复创建。")
        else:
            # ⚠️ **这一条最要紧**：查不通 ≠ 没建成。
            # 旧版在这里说"没有找到任务，可以重新上传" —— **那就是重复付费的入口**。
            st.session_state.video_job_error = (
                f"⚠️ 视频任务创建未确认：{exc}。"
                f"**按编号查询也没能确认**（网络或服务端问题）—— "
                "**结果仍然未知，请不要直接重新上传**：那次上传可能已经建成并开始处理。"
                f"编号已保留：`{request_id}`，可稍后用下面的「按编号查询」再试。")


def refresh_video_job() -> bool:
    """查一次任务进度（**只发 GET**）。返回是否查成功。

    ⚠️ 这里**永远不会重新上传、也不会再次调用模型** ——
    契约 T3 的原话是「查询失败不重新上传、不再次调用模型」。
    """
    job = current_video_job(st.session_state, STUDENT_ID, problem_id)
    if not job:
        return False
    context = video_job_context(st.session_state, STUDENT_ID, problem_id)
    try:
        response = _get(f"{BACKEND_URL}/video/jobs/{job['job_id']}",
            params={"student_id": STUDENT_ID, "problem_id": problem_id}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if payload.get("job_id") != job["job_id"]:
            raise ValueError("任务编号不匹配")
        applied = apply_video_job_result(st.session_state, context, payload, time.time())
        st.session_state.pop("video_job_error", None)
        if applied and (st.session_state.get("tutor") or {}).get("finished"):
            st.balloons()
        return True
    except Exception as exc:  # noqa: BLE001
        st.session_state.video_job_error = f"进度刷新失败，已有作答和任务编号已保留，可重试查询：{exc}"
        return False


#: 自动查询间隔（秒）—— 契约 T3 定的是 **3 秒**。
#:
#: 设为 **0 或负数 = 关闭自动查询**（只留手动刷新）。留这个开关是因为
#: 运维上可能想关掉轮询（比如后端在维护），测试里也要能覆盖"关闭"这条分支。
#:
#: ⚠️ **它不是为了测试提速**。我曾经量到"开启后 17 个 UI 用例从 8 秒变成 110 秒"，
#: 并据此写了"AppTest 会给每个片段等一个间隔"——**那个结论是错的**。
#: 真实原因见 `tests/conftest.py` 的 `clean_import_runs`：
#: 它是逐用例的，且每次都碰数据库；**Docker 没开时每个用例要付两次连接超时**
#: （约 3 秒 × 2）⇒ 17 个用例正好约 110 秒。
#: Docker 开着时实测：**定时器开 6.68 秒 / 关 7.02 秒 / 未改动基线 7.82 秒，三者相同。**
#: ⇒ 自动查询本身不拖慢测试。**别再拿这个开关去"优化测试"。**
VIDEO_JOB_POLL_SECONDS = float(os.getenv("VIDEO_JOB_POLL_SECONDS", "3") or 0)

#: 任务的**终态**。到这几个状态就**停止自动查询**（契约 T3）。
_JOB_TERMINAL = {"succeeded", "failed", "interrupted"}

_JOB_LABELS = {"queued": "排队中", "running": "处理中", "succeeded": "处理完成",
               "failed": "处理失败", "interrupted": "处理已中断"}
_JOB_PHASES = {"queued": "排队", "upload": "上传", "perception": "画面识别",
               "extracting": "解码与抽帧", "recognizing": "识别帧中文字",
               "exporting": "整理识别结果", "review": "生成复盘",
               "snapshots": "整理作答快照", "teaching": "逐轮教学分析", "completed": "完成"}


def render_video_job_progress() -> None:
    """任务状态与进度（**静态渲染**，不带定时器）。

    自动查询与手动刷新**共用这一段** —— 两处各画一遍，迟早会长得不一样。
    """
    job = current_video_job(st.session_state, STUDENT_ID, problem_id)
    if not job:
        return
    status = job.get("status")
    st.caption(f"视频任务编号：{job['job_id']}")
    st.caption(f"{_JOB_LABELS.get(status, status)} · "
               f"{_JOB_PHASES.get(job.get('phase'), '等待查询当前阶段')}")
    progress = job.get("progress") or {}
    if progress.get("total"):
        st.caption(f"当前阶段进度：{progress.get('completed', 0)} / {progress['total']}"
                   "（不代表整体完成百分比）")
    # 说清**自动查询现在是开着还是停着** —— 否则学生不知道还要不要等。
    if status in _JOB_TERMINAL:
        st.caption("任务已到终态，**自动查询已停止**。")
    elif st.session_state.get("video_job_poll_paused"):
        st.caption("**自动查询已暂停**（上次查询没成功）；可点下面手动刷新，不会重新上传。")
    elif VIDEO_JOB_POLL_SECONDS <= 0:
        st.caption("**自动查询已关闭**（`VIDEO_JOB_POLL_SECONDS=0`）；可手动刷新。"
                   "查询只读：不会重新上传，也不会重跑模型。")
    else:
        st.caption(f"每 {VIDEO_JOB_POLL_SECONDS:g} 秒自动查询一次。"
                   "查询只读：**不会重新上传，也不会重新跑一遍模型**。")
    if status in {"failed", "interrupted"}:
        st.warning((job.get("error") or {}).get("message") or "服务重启或任务失败导致处理中断。")
        st.caption("系统不会自动重新处理或产生新的模型请求；请核查原因后决定是否重新上传。")
    # 手动刷新**保留**：终态之后、自动查询暂停之后，它是唯一的查询入口。
    if not job.get("applied") and st.button("刷新处理进度", key="refresh_video_job",
                                            disabled=not backend_ok):
        if refresh_video_job():
            st.session_state.pop("video_job_poll_paused", None)   # 手动成功 ⇒ 恢复自动
        st.rerun()


@st.fragment(run_every=VIDEO_JOB_POLL_SECONDS or None)
def poll_video_job_progress() -> None:
    """运行中的任务：**每 3 秒自动查一次**（契约 T3）。

    ⚠️ **只发 GET。** 这条路径上没有任何"重新提交"的可能 ——
    自动重传会让一次卡顿变成两次付费，而且会多出一条作答记录。
    """
    job = current_video_job(st.session_state, STUDENT_ID, problem_id)
    if job and job.get("status") not in _JOB_TERMINAL:
        if not refresh_video_job():
            # 查询失败 ⇒ **暂停自动查询**，改回手动。
            # 不这么做，后端不通时这里会每 3 秒敲一次，把日志刷满也把浏览器拖住。
            # **暂停 ≠ 重传**：任务编号和已有作答都还在。
            st.session_state.video_job_poll_paused = True
            st.rerun()          # 整页重跑 ⇒ 外层不再注册定时器（= 停表）
    job = current_video_job(st.session_state, STUDENT_ID, problem_id)
    if job and job.get("status") in _JOB_TERMINAL:
        st.rerun()              # **到终态就停表**：重跑一次，外层改走静态渲染
    render_video_job_progress()


def call_video_review() -> bool:
    request = video_review_request(st.session_state, STUDENT_ID, problem_id)
    if request is None:
        return False
    try:
        with st.spinner("正在生成这段视频的复盘，已有作答记录会保留……"):
            response = httpx.post(f"{BACKEND_URL}/video/review", json={key: value for key, value in request.items() if not key.startswith("_ui_")}, timeout=300)
        if response.status_code != 200:
            try:
                detail = response.json().get("detail") or f"HTTP {response.status_code}"
            except Exception:
                detail = (response.text or "")[:300]
            raise ValueError(str(detail))
        apply_video_review(st.session_state, request, response.json())
    except Exception as exc:
        st.session_state.video_review_error = f"复盘生成失败，作答记录已保留，可重试：{exc}"
        return False
    return True


if workspace == '学习首页':
    st.subheader('从当前任务继续')
    st.write(problem.get('title', problem_id))
    st.caption('当前作答：'+('已结束，需要明确重新开始' if (st.session_state.get('tutor') or {}).get('finished') else ('有进行中的记录' if current_attempt_id(st.session_state) else '尚未开始')))
    home_cols=st.columns(3)
    home_cols[0].button('继续当前题',on_click=open_student_workspace,args=('学习作答',),use_container_width=True)
    home_cols[1].button('选择题目后开始',on_click=open_student_workspace,args=('学习作答',),use_container_width=True)
    home_cols[2].button('复盘一段作答视频',on_click=open_student_workspace,args=('视频复盘',),use_container_width=True)

_learning_columns=st.columns([.95,1.35], gap='large') if workspace=='学习作答' else None
_question_area=_learning_columns[0] if _learning_columns else st.expander('本次完整原题',expanded=workspace=='学习首页')
_response_slot=None
with _question_area:
    # ---------------- 1. 当前题目 ----------------
    with st.container(border=True, key='question_surface'):
        st.header("① 当前题目")
        st.markdown(f"**{problem.get('title', problem_id)}**　`{problem_id}`")
        if problem_detail_error:
            st.error(problem_detail_error)
        else:
            statement = problem.get("statement") or problem.get("content", "")
            if statement.strip():
                st.markdown(statement)
            else:
                st.warning("当前题目未提供题干，请补全题目材料。")
            for index, figure in enumerate(problem.get("figure_urls") or problem.get("figures") or []):
                if backend_ok:
                    image = (figure if figure.startswith(("http://", "https://")) else
                             f"{BACKEND_URL.rstrip('/')}/{figure.lstrip('/')}")
                else:
                    # 本地离线预览只允许读取本题目录内的配图。
                    root = (PROJECT_ROOT / "data" / "problems" / problem_id).resolve()
                    image = (root / figure).resolve()
                    if not image.is_relative_to(root) or not image.is_file():
                        st.warning(f"原题配图 {index + 1} 暂不可用。")
                        continue
                    image = str(image)
                st.image(image, caption=f"原题配图 {index + 1}", width="stretch")
        st.caption("知识点：" + "、".join(problem.get("knowledge_points", [])))
        if problem.get("graph_status") == "legacy_unreviewed":
            if problem.get("has_graph"):
                st.warning("本题使用尚未完成复核的演示解题图，步骤判断可能有误。")
            else:
                st.caption("本题解题图尚未发布，当前按无图模式分析。")
        elif problem.get("graph_status") == "draft":
            st.caption("本题解题图为待复核草稿，当前按无图模式分析。")
        elif problem.get("graph_status") == "reviewed":
            st.caption("本题解题图已复核但尚未发布，当前按无图模式分析。")

if 'answer_input' in st.session_state:
    st.session_state['_answer_draft']=st.session_state['answer_input']
else:
    st.session_state['answer_input']=st.session_state.get('_answer_draft','')
answer=st.session_state.get('answer_input','')
submit_clicked=hint_clicked=reset_clicked=False
if workspace=='学习作答':
    with _learning_columns[1]:
        # ---------------- 2. 学生答题区域 ----------------
        st.header("② 学生答题区域")
        answer = st.text_area(
            "在此输入你的解题步骤（建议一次提交一步，根据提示逐步修正）",
            height=120,
            key="answer_input",
        )

        col_submit, col_hint, col_reset = st.columns(3)
        submit_clicked = hint_clicked = reset_clicked = False
        with col_submit:
            submit_clicked = st.button("提交当前步骤", type="primary", use_container_width=True,
                                       disabled=not backend_ok)
        with col_hint:
            hint_clicked = st.button("请求提示", use_container_width=True,
                                     disabled=not backend_ok,
                                     help="还没被启发？点这里。\n\n"
                                          "**每点一次升一级**：L1 点拨 → L2 → L3 给方法 → "
                                          "L4 → L5 完整讲解。\n"
                                          "连着提交几次没进展也会自动升级。")
        with col_reset:
            reset_clicked = st.button("重新开始本题", use_container_width=True,
                                      disabled=not backend_ok or photo_busy(st.session_state, STUDENT_ID, problem_id))

if _learning_columns:
    _response_slot=_learning_columns[1].container(border=True,key='coach_surface')

# ---- 或者：上传一段录好的作答视频 ----
#
# 学生端的视频链路原来是断的 —— 感知层、bridge、驱动脚本都做完了，
# 但 `st.text_area` 是唯一的输入方式，**学生用不了**。
# 这里把起点接上：**录完再上传**（"边写边拍"要实时，感知层 12 秒/帧，做不到）。
if workspace in ('学习作答','视频复盘'):
    if 'media_mode' not in st.session_state:
        st.session_state['media_mode']='📹 传一段视频' if workspace=='视频复盘' else '📷 拍一张照片'
    _media_area=_learning_columns[1] if _learning_columns else st.container()
    with _media_area:
        with st.expander("照片求助" if workspace == "学习作答" else "上传与处理进度", expanded=True):
            # **两条路，默认走图片那条。**
            #
            # 照片适合当前步骤求助，视频用于分析整段作答过程。
            _mode = st.radio(
                "怎么给", ["📷 拍一张照片", "📹 传一段视频"],
                key="media_mode", horizontal=True, disabled=not backend_ok,
                label_visibility="collapsed", on_change=route_media_workspace)

            if _mode.startswith("📷"):
                st.caption(
                    "**卡在哪就拍哪** —— 把当前写到的地方拍清楚就行。\n\n"
                    "建议：光线足、纸放平、**字迹清楚**；只拍你写的那部分，别拍整张桌子。\n\n"
                    "会先给你**识别到的原文**和**下一步提示**。"
                    "可保留原图供核对；更正识别不会改写原图。"
                )
                render_photo_job()
                _img = st.file_uploader(
                    "选择照片", type=["jpg", "jpeg", "png", "webp", "bmp"],
                    key=f"image_upload_{st.session_state.upload_version}", disabled=not backend_ok,
                    help="手机直接拍的就行。")
                if _img is not None and st.button("开始批改这张照片", type="primary",
                                                  key="image_go",
                                                  disabled=not backend_ok):
                    start_photo_job(_img)
                    st.rerun()
                # ---- 失败：原因 + **重试入口**（2026-10-09 决定）----
                #
                # 学生已经拍过一次了，让他为了重试再选一次文件是白折腾。
                # 上传内容就留在 session 里（见 `call_image_analyze` 开头那三行）。
                if st.session_state.get("photo_error"):
                    st.error(st.session_state.photo_error)
                    _retry_payload = st.session_state.get("photo_retry") or {}
                    if (_retry_payload.get("data")
                            and st.button("重试这张照片", key="photo_retry_go", disabled=not backend_ok)):
                        call_image_analyze(_RetryUpload(_retry_payload))
                        st.rerun()
            else:
                st.caption(
                    "把**整段解题过程**录下来传上来，系统会抽取关键画面，分析作答并给提示。\n\n"
                    "建议：镜头对准纸面、字写清楚、**每一步停留几秒**。\n\n"
                    "处理可能需要数分钟。上传后会生成任务编号，"
                    "**页面每 3 秒自动查询一次进度**，也可以手动刷新；"
                    "自动查询**只读**，不会重新上传、也不会重跑模型。"
                    "原视频会尝试在处理后删除；识别文本、批改记录等分析产物可能保留，实际删除状态见结果。"
                )
                _up = st.file_uploader(
                    "选择视频文件", type=["mp4", "mov", "avi", "mkv", "webm"],
                    key=f"video_upload_{st.session_state.upload_version}", disabled=not backend_ok,
                    help="手机录的即可。时长几分钟都可以，处理时间随视频长度增加。")
                if _up is not None and st.button("开始处理这段视频", type="primary",
                                                 key="video_go",
                                                 disabled=(not backend_ok or bool(
                                                     st.session_state.get("video_job_request_id")
                                                     or st.query_params.get("video_request"))
                                                     and not st.session_state.get("video_job_request_absent_confirmed"))):
                    call_video_analyze(_up)
                    st.rerun()

            # ---- 处理完了要**在这里说一声** ----
            #
            # ⚠️ 轨迹表格渲染在 ④ AI Tutor（右栏、Tutor 面板下面）—— 那是**反馈该在的地方**。
            # 但学生在 ② 上传，结果在下面一屏，**他很可能以为没反应**。
            # 而处理本来就要一两分钟，等的过程中最怕的就是"到底成没成"。
            # ⇒ **在上传处回一句，并说清去哪看。**
            _vres = st.session_state.get("video_result")
            if _vres:
                _notices = media_notices(_vres)
                for _notice in _notices["warnings"]:
                    st.warning(_notice)
                st.caption(_notices["storage"])
                _n = _vres.get("submitted", 0)
                # 把分阶段耗时也给学生看一眼 —— 他等了几分钟，有权知道花在哪。
                # （也是排查"怎么这么慢"的第一手数据。）
                _tm = _vres.get("timings") or {}
                if _tm:
                    st.caption("⏱ 各阶段耗时：" + "　".join(
                        f"{k} {v:g}s" for k, v in _tm.items()))
                if _vres.get("media_type") == "image":
                    # ---- 照片主线：**先把「我看到了什么 + 你下一步做什么」摆出来** ----
                    #
                    # 2026-10-09 决定：照片这条是**即时辅导**。
                    # 学生拍完就在等这两件东西，而不是在等一个"已批改 1 步"的计数。
                    #
                    # ⚠️ 识别原文**必须**先给：手写数字实测被读错过（`49 N` → `4`），
                    # 而复盘/提示全都建立在识别文本上 —— **只有学生能发现读错了**。
                    _rec = (_vres.get("recognized_text") or "").strip()
                    if _rec:
                        st.markdown("**我识别到的作答**　"
                                    "（手写容易被误读；若与你写的不符，**以你的原稿为准**）")
                        st.code(_rec, language=None)
                    _hint_now = ((st.session_state.get("tutor") or {}).get("hint") or "").strip()
                    if _hint_now:
                        st.markdown("**下一步提示**")
                        st.info(_hint_now)
                    else:
                        # 空 hint 是 L0 的**设计语义**（本轮不提示），不是出错。
                        # ⇒ 说清这件事，别让学生以为系统没反应。
                        st.caption("本轮不提示（先观察你自己能不能推进）；需要的话点「请求提示」。")
                    st.caption("👉 完整逐轮记录在本页辅导区。"
                               "照片这条**不提供延后生成复盘**的入口。")
                else:
                    _off = _vres.get("off_graph_rounds") or 0
                    st.success(f"✅ 已批改 {_n} 步 —— **结果在本页辅导与复盘区**"
                               + (f"（其中 {_off} 步与本题无关）" if _off else ""))
                    if _vres.get("weak_concepts"):
                        st.caption("⚑ 这段材料中错误涉及的知识点（系统判断）："
                                   + "、".join(_vres["weak_concepts"]))
                    st.caption("👉 往下滚到 **当前反馈与核对** 查看逐轮记录，也可按需生成视频复盘。")


def render_rag_block(tutor: dict) -> None:
    """如实展示**本轮有没有用上教材、卡在哪一步**，以及**实际投递的片段**。

    ## 🔴 为什么措辞要这么小心（RAG 核查第 2 项）
    ```
    后端原先**两条路都返回空列表** ⇒ 页面若一律写"教材里没有相关内容"，
    就把**系统故障**说成了**知识库的正常结论** —— 学生会以为"书上确实没讲"。
    ⇒ 四种状态必须**分别措辞**，且**失败态绝不许出现"没有"式说法**。
    ```
    ⚠️ 来源展示用的是**后端算好的"实际投递片段"**（`rag_sources`）——
    与 Tutor 收到的是同一份；**不许把模型没拿到的内容显示成它的依据**。
    """
    status = (tutor or {}).get("rag_status") or {}
    state = status.get("state") or ""
    if not state:
        return                                   # 老响应没有这个字段 ⇒ 不显示，不猜
    _t = status.get("timings") or {}
    _cost = "　".join(f"{k.replace('_sec', '')} {v}s" for k, v in _t.items() if v is not None)
    _tail = f"　｜　耗时 {_cost}" if _cost else ""

    if state == "not_triggered":
        # ⚠️ **不是"查了没有"** —— 本轮压根没触发检索。
        st.caption("📚 本轮**没有触发**教材检索（没有诊断出错误或薄弱点）。")
        return
    if state == "failed":
        # 🔴 **失败不能说成"没有"** —— 这正是本项要修的语义。
        st.warning("📚 教材检索**这一轮没成功**（这不等于「教材里没有」）。"
                   f"已按无教材参考继续。原因：{status.get('reason', '未提供')}{_tail}")
        return
    if state == "no_hit":
        # ⚠️ **不能说"教材里没有"**（用户口径 2026-10-10）：
        #    「无相关材料」**只代表本轮候选的判断**，不代表那本书里没有讲。
        #    而且要区分"为什么没有"：向量检索没命中 vs **重排判断候选里没有相关材料**。
        if status.get("rerank") == "none":
            st.caption("📚 检索找到了候选，但**重排判断这些候选与本轮问题无关** ⇒ 本轮不提供教材参考。"
                       "（这是**本轮候选**的判断，不代表书上没有讲。）" + _tail)
        else:
            st.caption("📚 本轮**没有找到相关教材材料**。"
                       "（这只是**本轮检索**的结论，不代表书上没有讲。）" + _tail)
        return
    sources = (tutor or {}).get("rag_sources") or []
    _rk = status.get("rerank")
    _rk_txt = {"ok": "重排已生效", "fallback": "**重排超时，已回退原序**",
               "timeout": "**重排超时**", "failed": "**重排失败**"}.get(
        _rk or "", "重排状态未知" if _rk == "unknown" else "")
    with st.expander(f"📚 本轮参考了教材 {len(sources)} 段（点击看原文与来源页）"):
        st.caption(f"检索到 {status.get('returned', '?')} 条，"
                   f"实际投递给模型 **{len(sources)} 段**"
                   + (f"　｜　{_rk_txt}" if _rk_txt else "") + _tail)
        for s in sources:
            cut = "（**此处被截断**，后面的内容模型没看到）" if s.get("truncated") else ""
            _source=s.get('source', '未知来源')
            st.markdown(f"**{document_title(_source)} 第 {s.get('page', '?')} 页**{cut}")
            if document_title(_source) != _source:
                st.caption(f'来源编号：{_source}（页码为文件页序）')
            st.caption(s.get("text") or "")

# ---- 当前提示等级：**让学生知道自己在第几级、还能往上升** ----
# ---- 视频任务进度：**放在展开区外面** ----
#
# ⚠️ 为什么不在 `with st.expander(...)` 里（实测确认过两件事）：
# ```
# · 折叠**不影响执行** —— 展开区里的代码每次都照跑（探针验证过），
#   所以自动查询的定时器在折叠时也会注册。**不是"会静默停表"。**
# · 但折叠**影响可见性** —— 刷新页面后展开区默认收起，
#   学生只看到一行「📷 卡住了？」，**看不到任务在跑，也看不到到哪一步了**。
#   ⇒ 进度必须在外面，它不该藏在"上传媒体"那个折叠面板里。
# ```
# 顺带：URL 里的任务编号要在这里恢复（刷新后重新接管，**不重新上传**）。
_job = current_video_job(st.session_state, STUDENT_ID, problem_id)
_url_job = st.query_params.get("video_job")
if (not _job and _url_job and st.query_params.get("video_job_problem") == problem_id
        and st.query_params.get("video_job_student") == STUDENT_ID
        and not st.session_state.get("attempt_id") and not st.session_state.get("timeline")
        and video_job_context(st.session_state, STUDENT_ID, problem_id)["generation"] == 0):
    retain_video_job(st.session_state, video_job_context(st.session_state, STUDENT_ID, problem_id),
        {"job_id": _url_job, "student_id": STUDENT_ID, "problem_id": problem_id,
         "status": "queued", "phase": "unknown", "progress": {}})
    _job = current_video_job(st.session_state, STUDENT_ID, problem_id)
# **上传编号也随 URL 保留** —— 刷新页面后仍能按编号查那次"结果未知"的上传。
if not st.session_state.get("video_job_request_id"):
    _url_request = st.query_params.get("video_request")
    if _url_request:
        st.session_state.video_job_request_id = _url_request
if _job:
    # ⚠️ **只有"运行中"才注册定时器**（`st.fragment(run_every=…)`）；
    # 终态、暂停、后端不可用、自动查询被关掉，都走静态渲染
    # ⇒ 这就是「终态停止」的落点。
    if (_job.get("status") in _JOB_TERMINAL or not backend_ok
            or VIDEO_JOB_POLL_SECONDS <= 0
            or st.session_state.get("video_job_poll_paused")):
        render_video_job_progress()
    else:
        poll_video_job_progress()
elif st.session_state.get("video_job_request_id"):
    # **有编号、没任务** ⇒ 那次上传的结果**仍然未知**。
    # 这里给一个手动再查的入口；**不提供"重新上传"按钮** ——
    # 结果未知时重传就是重复付费（契约 §2.2）。
    _pending_request = st.session_state.video_job_request_id
    st.caption(f"有一次上传的结果还没确认，编号：`{_pending_request}`")
    _rc1, _rc2 = st.columns(2)
    with _rc1:
        _query_clicked = st.button("按编号查询那次上传", key="recover_video_job",
                                   disabled=not backend_ok, use_container_width=True)
    with _rc2:
        # ⚠️ **契约 ⑤ 要求的出口**：没有它，换了视频就永远 409（死循环）。
        #    但放弃 = **承认那次结果不再找回**，所以措辞必须说清代价，
        #    而不是轻描淡写地叫"重新上传"。
        _abandon_clicked = st.button("放弃这次上传，开始新的一次", key="abandon_video_request",
                                     use_container_width=True)
    if _abandon_clicked:
        forget_request_id(st.session_state, st.query_params)
        st.session_state.pop("video_job_error", None)
        # ⚠️ 复用既有的通知键（`video_job_absent_notice` 是**已经有人渲染**的那个）；
        #    新造一个没人渲染的键 = 提示静默消失。
        st.session_state.video_job_absent_notice = (
            "已放弃这一次上传的编号。**下一次上传会生成新编号**，不会再撞 409。"
            "⚠️ 若那一次其实已在服务端建成并跑了模型，它的结果**不会**再被找回。")
        st.rerun()
    if _query_clicked:
        _outcome = recover_video_job_by_request(_pending_request)
        if _outcome == RECOVER_FOUND:
            st.rerun()
        elif _outcome == RECOVER_ABSENT:
            # ⚠️ **这里不清编号**：404 只说明"此刻查不到"，与"慢创建"之间存在竞态，
            #    保留编号正是防重复付费的那道闸（契约 §2.2）。
            #    要换视频的人走上面的「放弃这次上传」显式出口（契约 ⑤）。
            st.session_state.video_job_absent_notice = (
                "服务端当前未找到这个编号；原请求仍可能稍后完成。"
                "可以用**同一编号**重试，系统会检查内容和参数，避免重复创建；"
                "若你确定要改传别的视频，用上面的「放弃这次上传，开始新的一次」。")
            st.rerun()
        else:
            st.warning("还是**没能确认**（网络或服务端问题）。编号已保留，可稍后再试；"
                       "**在此之前请不要重新上传**。"
                       "若你确定要放弃这一次、改传别的视频，用上面的「放弃这次上传」。")
if st.session_state.get("video_job_error"):
    st.warning(st.session_state.video_job_error)
if st.session_state.get("video_job_absent_notice"):
    st.success(st.session_state.pop("video_job_absent_notice"))


if reset_clicked:
    try:
        _reset_response = httpx.post(
            f"{BACKEND_URL}/attempt/reset",
            json={"student_id": STUDENT_ID, "problem_id": problem_id}, timeout=10)
        _reset_response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 保留进度，避免下一次提交悄悄续做旧记录
        st.error(f"重新开始失败，当前进度已保留：{exc}")
    else:
        _reset_local()
        st.rerun()

if submit_clicked and backend_ok:
    if answer.strip():
        call_analyze(answer.strip(), request_hint=False)
    else:
        st.warning("请先输入解题步骤再提交。")

if hint_clicked and backend_ok:
    # 空输入表示继续询问已有作答，由服务端从本次 attempt 取原文。
    # 不能拿“还没有作答”的占位文案替换刚上传的照片书写。
    if st.session_state.get("attempt_id") and not current_attempt_id(st.session_state):
        st.warning("这次作答已结束，已有提示会保留。请点击「重新开始本题」明确开启新的作答。")
    else:
        call_analyze(answer.strip(), request_hint=True)

if _response_slot is None:
    if workspace=='视频复盘':
        _frame_column,_report_column=st.columns([.9,1.2],gap='large')
        with _frame_column:
            render_evidence_browser(st.session_state.get('video_result'), _get, BACKEND_URL, STUDENT_ID, problem_id, st.session_state.get('attempt_id'), stacked=True)
        _response_slot=_report_column.container(key='review_surface')
    else:
        _response_slot=st.expander('当前辅导与材料核对',expanded=False)
with _response_slot:
    render_hint_ladder(st.session_state.get('tutor') or {})
    _t = st.session_state.get("tutor") or {}
    if (_t.get('subquestion_context') or {}).get('status') == 'conflict':
        st.warning('本轮小问编号与作答条件有冲突，请核对识别原文。这不是作答错误判定。')
    if _t.get("hint_level") is not None:
        _lv = int(_t.get("hint_level") or 0)
        _names = {0: "L0 本轮未下发提示", 1: "L1 方向提示", 2: "L2 概念提示",
                  3: "L3 方法提示", 4: "L4 关键步骤提示", 5: "L5 完整讲解"}
        _line = f"提示等级 **{_names.get(_lv, _lv)}**"
        # ⚠️ 卡顿分**不在这里显示**（T3，2026-10-09）——
        # 它是启发式指标，摆在"提示等级"旁边等于把它当成了对学生状态的判断。
        # 完整的技术详情（分数 / 信号 / 依据）见 ⑤ 的折叠区。
        # 判对与判错**共用一套升级阶梯**（产品决定 2026-10-07）——
        # 做对的学生点请求提示，意思是"我想多懂一点"，不是"我卡住了"。
        # 所以这里不再分情形讲，统一说"再点会升一级"。
        if 0 < _lv < 5 and current_attempt_id(st.session_state):
            _line += "　｜　**还没启发到你？分析可用时，再点「请求提示」会升一级**"
        st.caption(_line)

    # ---- 教材依据：**本轮到底用没用上教材、卡在哪一步**（RAG 核查第 1、2 项）----
    # ⚠️ 放在提示等级下面：学生看完"我该做什么"，紧接着就能看"这是不是有书上依据"。
    render_rag_block(st.session_state.get("tutor") or {})

# ---------------- 3/4. Learning Timeline + AI Tutor ----------------
left = st.expander("作答过程时间线", expanded=workspace=="学习记录")
right = _response_slot

with left:
    st.header("③ Learning Timeline")
    if st.session_state.timeline:
        for event in reversed(st.session_state.timeline):
            if event["type"] == "ocr_correction":
                st.markdown(f"`t={event['timestamp']}s` **识别更正** — 已根据更正文本重新分析，替代原分析。")
            else:
                st.markdown(f"`t={event['timestamp']}s` **{event['type']}** — {event['content']}")
    else:
        st.info("暂无事件。开始作答后，这里会记录你的学习过程。")

with right:
    st.header("最新作答反馈与复盘" if workspace == "视频复盘" else "当前反馈与核对")
    if workspace == '视频复盘':
        _feedback_rounds=(st.session_state.get('video_result') or {}).get('rounds') or []
        if _feedback_rounds:
            st.caption(f"当前提示对应最新第 {_feedback_rounds[-1].get('round',len(_feedback_rounds))} 轮；左侧可核对其他时刻的材料，切换画面不重算提示。")
    tutor = st.session_state.tutor
    if tutor:
        if tutor.get("graph_release_status") == "legacy_unreviewed":
            st.warning("本次作答沿用未复核的演示解题图，请核对步骤判断。")
        if tutor.get("graph_binding_status") == "unknown":
            st.caption("旧作答未记录解题图版本，本轮按无图模式分析；重新开始可使用当前可用图。")
        if tutor.get("finished"):
            st.success("🎉 本题完成！" + tutor.get("hint", ""))
        elif tutor.get("hint"):
            st.markdown(tutor["hint"])
        else:
            st.info("（L0：本轮不提示，继续你的思路）")

        # P2-P6 的过程感知信息
        #
        # ⚠️ 卡顿分**已从这里撤走**（T3，2026-10-09）：它属于技术详情，
        # 见 ⑤ 的「技术详情：系统卡顿分」。这行留给"这一轮系统认为你在哪"。
        mode = "Graph 模式 🧭" if tutor.get("graph_mode") else "Graphless 模式 🗺️"
        line1 = (
            f"`{mode}` ｜ 步骤：{tutor.get('current_step', '—')} ｜ "
            f"问题：{tutor.get('issue') or '无'} ｜ 提示：L{tutor.get('hint_level', 0)}"
        )
        if not tutor.get("graph_mode"):
            line1 += f" ｜ 阶段 / 方法：{tutor.get('current_method') or tutor.get('current_step') or '—'}"
        if tutor.get("graph_mode"):
            line1 += f" ｜ 当前节点：**{tutor.get('current_node') or '—'}**"
        st.caption(line1)

        # ---- 解题路径 ----
        #
        # ⚠️ `visited_nodes` 的定义是"**成功匹配过**的节点"，所以**答错时它是空的**
        # （`state_tracker` 只在 `not off_graph` 时累积，而错答的 `matched_nodes`
        # 本来就是空）。实测：
        #   答对 → visited_nodes = ['S1']
        #   答错 → visited_nodes = []      但 current_node = 'S2a'
        #
        # **后果：一个一直在错的学生，看不到"我走到哪了"** ——
        # 而那正是他最需要看到的时候。
        #
        # 补法：`visited_nodes` 为空时退而用 `step_match.attempted_nodes`。
        #
        # ⚠️ **措辞必须中性。** 我第一版写的是"（尝试过，**尚未走通**）"，
        # 但实测发现：喂一句**正确**的作答，`matched_nodes` 也可能是空
        # （匹配器没匹配到，不等于学生没做对）。
        # **`visited_nodes` 为空 ≠ 没走通** —— 它只说明"没有节点被确认匹配"。
        # 替匹配器下这个结论就是我编造。
        _visited = tutor.get("visited_nodes") or []
        _match = tutor.get("step_match") or {}
        _attempted = _match.get("attempted_nodes") or []
        if _visited:
            st.caption("参考图匹配记录：" + "、".join(_visited))
        elif _attempted:
            st.caption("相关节点（**匹配器识别到的**）：" + " → ".join(_attempted))
        match = tutor.get("step_match") or {}
        if match.get("missing_reasoning"):
            st.caption("材料中未展示的推理（系统判断）：" + "；".join(match["missing_reasoning"]))
        if tutor.get("weak_concepts"):
            st.caption("书面错误涉及知识点（系统判断）：" + "、".join(tutor["weak_concepts"]))

        # ---- 视频作答：把整段轨迹给学生看 ----
        #
        # ⚠️ 这是一个**跨轮**的信息，tutor 面板只显示最后一轮。
        # 而"他是怎么走到这一步的"恰恰是学生自己最该看到的 ——
        # 提示级别什么时候升上去、什么时候降回来，比单点状态有用得多。
        with st.expander('查看解题进度图', expanded=workspace=='学习记录'):
            render_progress(_get, BACKEND_URL, STUDENT_ID, problem_id, st.session_state.get('attempt_id'), tutor)
        if workspace!='视频复盘':
            render_evidence_browser(st.session_state.get('video_result'), _get, BACKEND_URL, STUDENT_ID, problem_id, st.session_state.get('attempt_id'))
        _vr = st.session_state.get("video_result")
        if _vr and (_vr.get("rounds") or []):
            _ok = [r for r in _vr["rounds"] if "error" not in r]
            # ---- 复盘放在**最前面** ----
            #
            # 学生要的是结论（我错在哪、为什么、下一步），
            # 逐轮那 15 行是**过程记录**，留给想深究的人。
            # 顺序反了，学生就得自己从 15 行里找规律 —— 那正是原形态的问题。
            _rev = _vr.get("review")
            if _rev:
                # ⚠️ **阅读约定：由服务端署名，不由模型写**（契约 §2.6 ⑫，2026-10-10）。
                #
                # 为什么要有这一句：复盘里**混着三类内容** ——
                #   ① 学生原文（引号内 / 「你写的」栏）② **系统自己的复算** ③ 系统的判断与建议。
                # 模型在自由文本里引用与复算**混着写**，而"要不要记得加标记"（实测）
                # **靠不住**：泄题裁决查出 70 条里只有 16 条的 next_step 自带标记。
                # ⇒ 既然标签不能指望模型，就**在呈现层把约定说清**，让读者自己分得清。
                st.caption(
                    "**阅读约定**：引号内与「你写的」栏是**材料原文**；"
                    "标「**系统核对时重建**」的是**系统自己的复算**，"
                    "**不是学生写过的步骤**；其余是系统的判断与建议。"
                    "复盘基于识别文本与 AI 推断；停顿原因需核查，请结合原稿判断。")
            # 按钮只认 `can_generate_review`（= 契约 §2.1 的 `review_available`）。
            #
            # ⚠️ 它与 `review_status` **正交**，也**不看** `review_refresh_required`：
            # 「旧报告需要更新」和「现在能不能生成」是两件事 ——
            # **照片更正后正是「需要更新、却无处可生成」⇒ 只提示，不给按钮。**
            _can_review = can_generate_review(st.session_state, STUDENT_ID, problem_id)
            if _can_review and not _rev:
                st.caption("将使用更正后的识别文本生成新版复盘。" if _vr.get("ocr_corrected")
                           else "逐轮分析已完成；可按需生成这段视频的复盘。")
                if st.button("根据更正内容生成新版复盘" if _vr.get("ocr_corrected") else "生成复盘",
                             key="generate_video_review", disabled=not backend_ok):
                    if call_video_review():
                        st.rerun()
            if st.session_state.get("video_review_error"):
                st.warning(st.session_state.video_review_error)
            if _vr.get("media_type") == "video":
                with st.expander("处理耗时"):
                    _times = _vr.get("timings") or {}
                    for _label, _key in (("感知", "② 感知层"), ("教学分析", "④ 驱动教学链路")):
                        if _key in _times:
                            st.caption(f"{_label}：{_times[_key]} 秒")
                    if _vr.get("review_elapsed_sec") is not None:
                        st.caption(f"复盘：{_vr['review_elapsed_sec']} 秒（独立请求）")
                    if _vr.get("perception_timings"):
                        st.write({"感知细分耗时（秒）": _vr["perception_timings"]})
            if _vr.get("ocr_corrected"):
                # 旧报告**不会**因为更正而消失（它保留在原版本里，见下面的历史复盘），
                # 但**不能作为当前结论展示** —— `review` 此刻是空的。
                # 这里只说清「现在手上有没有一份当前材料对应的报告」。
                st.info("识别更正已重算，原始识别文本仍保留。" + (
                    "当前复盘已基于更正内容重新生成。" if _rev else (
                        "原复盘已失效，可生成新版复盘。" if _can_review
                        else "原复盘已失效；**本次不提供重新生成入口**，"
                             "下面「历史复盘」里保留原版本仅供核对。")))
            if _vr.get("review_history"):
                with st.expander("历史复盘（**已过期**，仅供核对，不是当前结论）"):
                    for _old in _vr["review_history"]:
                        # `superseded` 是那个**独立字段** —— 过期与否不从 `review_status` 读。
                        _mark = "（已过期）" if _old.get("superseded") else ""
                        st.caption(f"材料版本 {_old.get('input_revision', 0)}{_mark}，"
                                   "以下为更正前的复盘。")
                        st.write(_old.get("review") or {})
            if _rev:
                st.markdown("#### 📋 复盘")
                if _vr.get("review_scope") == "video_run_snapshot":
                    st.caption("此复盘基于本次视频分析记录。")
                # **学生唯一能发现 OCR 读错的机会。**
                # 复盘的所有判断都建立在"识别到的文本"上，
                # 而它自己无法察觉识别错误 —— 只有学生能。
                st.caption(
                    "⚠️ 下面的判断基于**系统识别到的作答原文**（见下方表格第一列）。"
                    "**手写数字容易被误读**（实测出现过 `49` 读成 `4`）。"
                    "若和你实际写的不符，以你的原稿为准。"
                )
                if _rev.get("timeline"):
                    st.markdown(_rev["timeline"])

                # ---- v3：按契约的三类分开展示（T3，2026-10-09）----
                #
                # ⚠️ **不要把三类都写成「做错了」。**
                # `observed_error` / `uncertain` / `suggestion` 要学生做**不同的事**：
                # 前者去核原文，中者去核识别（我可能看错了），后者根本不是错。
                # 混成一句「你错了」，就是错误归因。
                _v3 = review_categories(_rev)
                _v3_observed = next((g["items"] for g in _v3["groups"]
                                     if g["category"] == "observed_error"), [])
                if _v3["is_v3"]:
                    _ICONS = {"observed_error": "🔎", "uncertain": "❓", "suggestion": "✏️"}
                    for _group in _v3["groups"]:
                        if not _group["items"]:
                            continue
                        st.markdown(f"**{_ICONS[_group['category']]} {_group['label']}**")
                        st.caption(_group["hint"])
                        for _item_index, _item in enumerate(_group["items"]):
                            st.markdown(f"- **{_item['where'] or _group['label']}**")
                            if _item["what"]:
                                # 刻意**不加**「做错了」前缀 —— 类别已经说明了性质。
                                st.markdown(f"    - {_item['what']}")
                            if _item["why"]:
                                st.markdown(f"    - 理由：{_item['why']}")
                            if _item["downstream"]:
                                st.markdown(f"    - 往下的影响：{_item['downstream']}")
                            for _ev in _item["evidence"]:
                                if not _ev["found"]:
                                    # 契约⑦说这种报告不该被缓存；但前端**不能假设**
                                    # 后端永远拦得住 —— 看不见的坏引用比看得见的更危险。
                                    st.caption(f"    ⚠️ 引用了不存在的证据 `{_ev['id']}`")
                                    continue
                                _rev_note = (f"，修订 {_ev['revision']}"
                                             if _ev["revision"] is not None else "")
                                st.caption(f"    证据 `{_ev['id']}`｜第 {_ev['round']} 轮"
                                           f"{_rev_note}：{_ev['text']}")
                                _raw_ev = next((e for e in _rev.get('evidence',[]) if e.get('id') == _ev['id']), {})
                                _source = _raw_ev.get('source_id')
                                _revision = _raw_ev.get('source_revision', _ev.get('revision'))
                                if _source and target_round(_vr.get('rounds') or [], _source, _revision) is not None:
                                    st.button('定位这份来源', key=f"locate_{_vr.get('run_id')}_{_group['category']}_{_item_index}_{_ev['id']}",
                                              on_click=set_evidence_target,args=(_vr.get('run_id') or '',_source,_revision))
                    # 契约⑥：信息不足 ≠ 学生做错。**空列表是合法结果**，如实说。
                    if not any(g["items"] for g in _v3["groups"]) and not _v3["unknown"]:
                        st.info("**没有发现需要指出的问题。** 这可能是你做对了，"
                                "也可能是材料里看不出来 —— **看不出来不等于做错**。")
                    if _v3["unknown"]:
                        # 契约说 category 固定三选一。真出现第四种值说明有一边错了，
                        # **要让人看见**，不能静默塞进三类里。
                        st.warning("这份复盘里有 **无法归类** 的条目（类别不在契约的三类内），"
                                   "已原样列出，请以原文为准：")
                        for _odd in _v3["unknown"]:
                            st.caption(f"· [{_odd.get('category')}] "
                                       f"{_odd['where']}：{_odd['what']}")
                else:
                    for _i, _sp in enumerate(_rev.get("sticking_points") or [], 1):
                        st.markdown(f"**卡点 {_i}｜{_sp.get('where', '')}**")
                        if _sp.get("what"):
                            st.markdown(f"- **做错了**：{_sp['what']}")
                        if _sp.get("why"):
                            st.markdown(f"- **为什么**：{_sp['why']}")
                        if _sp.get("downstream"):
                            st.markdown(f"- **往下的影响**：{_sp['downstream']}")
                # ---- ⭐ 结论核对（v2 新增）----
                #
                # 两件事都要给学生看：
                # ① **结论和他的推导是否自洽** —— 直接指出「凑对」
                #    （答案对、理由错，是自动批改天生抓不到的一类错误）
                # ② **识别可疑之处** —— 复盘自己说"这里我可能看错了"
                #    **这条比什么都重要**：它把"系统自信地判断错了"
                #    变成"系统告诉你它在哪不确定"
                #
                # ⚠️ 2026-10-09（T3）：右栏文案由「用你自己的方程推出来」改为
                # 「**系统核对时重建**」。旧措辞把**系统的补算**说成**学生写过的步骤** ——
                # 学生看到会以为"我明明没写这一步"，是错误归因。
                _ac = _rev.get("answer_check") or {}
                if _ac:
                    st.markdown("**🔍 结论核对**")
                    _cols = st.columns(2)
                    with _cols[0]:
                        st.caption("你写的（**逐字**）："
                                   + (_ac.get("student_answer") or "—"))
                    with _cols[1]:
                        st.caption("**系统核对时重建**："
                                   + (_ac.get("derived_from_work") or "—"))
                    _cons = (_ac.get("consistent") or "").strip()
                    if "不一致" in _cons:
                        if _v3["is_v3"] and _v3_observed:
                            # v3 里同一条已作为「观察到的问题」出现，**且带证据引用** ——
                            # 这里只留一句指路，不把同一段话印两遍。
                            st.caption("⚠️ 结论与推导对不上 —— 见上方「观察到的问题」。")
                        else:
                            # 非 v3，或 v3 却没给出对应条目 ⇒ **这条不能丢**：
                            # 它是"凑对"唯一的出口。
                            st.warning("**结论和推导对不上** —— "
                                       + (_ac.get("note") or ""))
                    elif "无法判断" in _cons:
                        st.caption("（这一步无法核对：" + (_ac.get("note") or "信息不足") + "）")
                    elif _cons:
                        st.caption("✅ " + (_ac.get("note") or "结论与推导一致"))

                # ⚠️ **识别可疑要说在最显眼的地方。**
                # 复盘是基于"识别到的文本"判断的 —— 它自己无法察觉识别错误，
                # **只有学生能**。所以这一条不能藏在折叠区里。
                #
                # v3 起它已经是「需要核查」类别里的正式条目（还带证据），
                # 就不再单独出一条横幅 —— 那会把同一句话印两遍。
                if (not _v3["is_v3"]) and (_rev.get("answer_check") or {}).get("ocr_caution"):
                    st.error("⚠️ **这里我可能看错了**："
                             + _rev["answer_check"]["ocr_caution"]
                             + "\n\n如果和你实际写的不符，**以你的原稿为准**，"
                               "下面的判断也请一并打折扣。")

                if _v3["is_v3"]:
                    # `next_step` / `encouragement` 已被映射进「可选练习」，这里不重复；
                    # 只有 `key_concept` **没有**被映射进去。
                    if _rev.get("key_concept"):
                        st.caption("⚑ 最该补的知识点：**" + _rev["key_concept"] + "**")
                    _val = _v3["validation"]
                    if _val:
                        _vissues = _v3["validation_issues"]
                        if _vissues:
                            st.warning("这份复盘的**结构与引用校验有问题** —— "
                                       + "；".join(i["text"] for i in _vissues))
                        with st.expander("结构与引用校验（技术详情）"):
                            # 契约 §2.3⑧：**不得**显示成「物理语义已验证」。
                            # `scope` 那句就是为这件事写在数据里的。
                            st.caption("校验范围：" + (_val.get("scope") or "结构与引用"))
                            st.caption(f"证据 {_val.get('evidence_count', 0)} 条"
                                       f"；校验器 {_val.get('validator_version') or '—'}")
                            # 报告版本**绑了实际跑过的提示词版本**（`v3+prompt-v2`）——
                            # v2 提示词**不要求模型引用证据**，读的人有权知道这件事。
                            if _rev.get("review_schema_version"):
                                st.caption("报告版本：" + str(_rev["review_schema_version"]))
                            # 有多少条判断**指不回具体原文** —— 如实报出来，
                            # 而不是让它们悄悄混进「观察到的问题」。
                            if _val.get("downgraded_claims"):
                                st.caption(f"有 {_val['downgraded_claims']} 条判断因为"
                                           "**找不到对应的原文引用**，已归入「需要核查」。")
                            if _vissues:
                                for _vi in _vissues:
                                    st.caption("· " + _vi["text"])
                            else:
                                st.caption("结构与引用检查通过 —— **通过 ≠ 物理正确**。")
                else:
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        if _rev.get("key_concept"):
                            st.caption("⚑ 最该补的知识点：**" + _rev["key_concept"] + "**")
                    with _c2:
                        if _rev.get("next_step"):
                            st.caption("→ 下一步：" + _rev["next_step"])
                    if _rev.get("encouragement"):
                        st.info(_rev["encouragement"])
            else:
                st.caption("（这次没生成复盘，下面是逐轮记录）")

            with st.expander(
                f"📹 逐轮记录（识别到 {_vr.get('submitted', 0)} 步）",
                # ⚠️ **默认展开** —— 它是学生核对"识别对不对"的唯一入口，
                # 折叠起来等于把校验机会藏了。（原来是 `not _rev`，有复盘就收起。）
                expanded=True,
            ):
                if _vr.get("off_graph_rounds"):
                    st.caption(f"⚠️ 其中 {_vr['off_graph_rounds']} 步与本题无关"
                               f"（可能是镜头外的内容或空白）")
                st.dataframe(
                    [{"步": r.get("round"), "秒": r.get("ts"),
                      # ⚠️ **这一列是学生唯一能发现 OCR 读错的地方。**
                      #
                      # 实测踩到：学生写的是 `F = 49 N`（正确答案），
                      # **OCR 读成 `F = 4`**（用户看原始视频确认），
                      # 而复盘据此说成"得到 F = 4 这种没有物理意义的结果" ——
                      # **把"碰巧对了"说成了"错得离谱"。**
                      #
                      # **复盘判断的依据和它看到的是同一份文本，OCR 读错它无法发现。**
                      # 所以必须把识别原文摆到学生眼前，让他能说"这不是我写的"。
                      "**识别到的作答**": (r.get("answer") or "（空）"),
                      "节点 / 阶段 / 方法": r.get("current_node") or r.get("current_step") or r.get("current_method") or "—",
                      "提示级别": f"L{r.get('hint_level', 0)}",
                      # ⚠️ 卡顿分**已从这张表里撤走**（T3，2026-10-09）。
                      # 这张表的用途是**让学生核对识别对不对**，
                      # 而卡顿分是启发式指标 —— 混在这里会显得它和"识别原文"一样是事实。
                      # 逐轮分数仍在 ⑤ 的技术详情里可查。
                      # ⚠️ **不要 fallback 到 `issue`。**
                      #
                      # `issue` 是**面向教师的错误归因**（见 prompts/diagnosis），
                      # 而 `hint` 为空**恰恰是 L0 的设计语义**："本轮不提示"。
                      # 两者一混，L0 那一行的"提示"列里就出现了
                      # 「学生仅凭直觉断言加速度为0，未对三个小问分别进行受力分析…」
                      # —— **学生看到会当成提示读，而它是"你错在哪"的答案线索。**
                      #
                      # 实测就是从这个 CSV 里发现的（用户导出 st.dataframe
                      # 自带的下拉下载，所以这张表和导出的 CSV 是同一份）。
                      "提示": (r.get("hint") or "")[:60]}
                     for r in _ok],
                    use_container_width=True, hide_index=True)
                if _vr.get("weak_concepts"):
                    st.caption("⚑ 这段材料中错误涉及的知识点（系统判断）："
                               + "、".join(_vr["weak_concepts"]))
                for _round in _ok:
                    with st.expander(f"第 {_round.get('round', '—')} 步 · 完整识别原文"):
                        st.text(_round.get("original_answer", _round.get("answer")) or "（空）")
                        _reason = correction_unavailable(st.session_state, _vr, _round)
                        if _reason:
                            st.caption(_reason)
                        else:
                            _sid = _round["source_id"]
                            st.caption("只修正识别错误；重算会替代这一步的分析，不增加一次作答。")
                            _edited = st.text_area("更正识别文本", value=_round.get("answer") or "",
                                                   key=f"ocr_edit_{_sid}", height=120)
                            if st.button("保存更正并重新分析", key=f"ocr_submit_{_sid}",
                                         disabled=not _edited.strip() or _edited.strip() == (_round.get("answer") or "").strip()):
                                with st.spinner("正在根据更正文本重新分析……"):
                                    _success = call_ocr_correction(_round, _edited)
                                if _success:
                                    st.rerun()
                            if st.session_state.get(f"ocr_error_{_sid}"):
                                st.error(st.session_state[f"ocr_error_{_sid}"])
                st.caption(media_notices(_vr)["storage"])
    else:
        st.info("提交作答后，AI Tutor 会在这里给出分级提示（不会直接给答案）。")

with st.expander('本次作答概览', expanded=workspace=='学习记录'):
    # ---------------- 5. 简单学习分析 ----------------
    st.header("⑤ 简单学习分析")
    tutor = st.session_state.tutor
    if tutor:
        elapsed = round(time.time() - st.session_state.start_ts)
        submit_count = sum(1 for e in st.session_state.timeline if e["type"] == "submit")
        lines = [
            f"- **本题状态**：{'已完成 ✅' if tutor.get('finished') else '进行中'}，"
            f"共提交 **{submit_count}** 次，用时约 **{elapsed} 秒**",
            f"- **诊断**：{tutor.get('issue') or '无明显问题'}（{tutor.get('issue_type') or '—'}）",
        ]
        if tutor.get("weak_concepts"):
            lines.append(f"- **建议复习**：{'、'.join(tutor['weak_concepts'])}")
        st.markdown("\n".join(lines))
        # ⚠️ 卡顿分从上面那几条里撤下来了（T3，2026-10-09）——
        # 它不再冒充"学习分析"的结论，改在这里作为**技术详情**出现。
        # 自取 `rounds` 而不是复用外面的 `_vres`：那是在上传区的 `with` 块里绑的，
        # 隔这么远依赖它，改动一处顺序就会炸。
        stuck_technical_detail(tutor, (st.session_state.get("video_result") or {}).get("rounds"))
    else:
        st.caption("开始作答后，这里会给出本次作答的过程性总结。")

with st.expander('已保存的过程与提示', expanded=workspace=='学习记录'):
    # ---------------- 6. 学习记录（数据库回放） ----------------
    st.header("⑥ 学习记录（PostgreSQL 回放）")
    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        refresh = st.button("从数据库刷新当前记录",
                            disabled=not (backend_ok and st.session_state.attempt_id))
    with col_b:
        st.caption(f"attempt_id：`{st.session_state.attempt_id or '—'}`")

    if refresh and st.session_state.attempt_id:
        try:
            st.session_state.db_record = _get(
                f"{BACKEND_URL}/attempt/{st.session_state.attempt_id}", timeout=10
            ).json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"读取失败：{exc}")

    record = st.session_state.db_record
    if record:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Learning Events（数据库）**")
            for event in record.get("events", []):
                st.markdown(f"`t={event['timestamp']}s` **{event['event_type']}** — {event['content'][:60]}")
        with a2:
            st.markdown("**提示下发历史**")
            for i, hint in enumerate(record.get("hints", []), 1):
                st.markdown(f"{i}. `L{hint['hint_level']}`（{hint['issue_type'] or '—'}）{hint['hint'][:60]}")
            st.caption(
                f"final_correct={record.get('final_correct')} ｜ "
                f"hint_count={record.get('hint_count')} ｜ stuck_count={record.get('stuck_count')}"
            )
    elif st.session_state.attempt_id:
        st.caption("点击「从数据库刷新当前记录」查看已落库的完整过程数据。")
    else:
        st.caption("尚未开始作答。")


# ================= 做完了？推荐下一题 =================
#
# 依据是**两样东西合起来**：
#   · 学生画像（跨题累积的薄弱知识点）
#   · 本次作答（这一遍暴露出来的薄弱点）
# 按"与题库各题知识点的重合度"排序，同分时先易后难。
#
# **冷启动不会给空** —— 学生还没暴露薄弱点时，退化为"没做过且最简单的几道"。
st.divider()
st.subheader("下一步做哪道")

if not backend_ok:
    st.caption("后端未连接，推荐暂不可用。")
else:
    _done = st.session_state.get("tutor", {}).get("finished")
    _btn_label = "看看推荐下一题" + ("（本题已完成）" if _done else "")
    if st.button(_btn_label, use_container_width=False):
        try:
            _r = _get(
                f"{BACKEND_URL}/students/{STUDENT_ID}/recommendations",
                params={"exclude": problem_id, "limit": 3}, timeout=10)
            st.session_state.recommendations = (
                _r.json() if _r.status_code == 200 else
                {"error": f"后端返回 {_r.status_code}"})
        except Exception as exc:  # noqa: BLE001
            st.session_state.recommendations = {"error": f"读取失败：{exc}"}

    _rec = st.session_state.get("recommendations")
    if _rec:
        if _rec.get("error"):
            st.warning(_rec["error"])
        else:
            if _rec.get("cold_start"):
                st.caption("暂无可追溯的书面错误记录，先按难度从易到难推荐：")
            else:
                _w = _rec.get("weak_concepts") or []
                if _w:
                    st.caption("依据你的书面错误涉及知识点（系统判断）：" + "、".join(_w[:6]))
            for _i, _x in enumerate(_rec.get("recommendations") or [], 1):
                st.markdown(
                    f"{_i}. **{_x['title']}**　`{_x['problem_id']}`　"
                    f"难度 {_x['difficulty']}　—　{_x.get('why', '')}")
            st.caption("在上面的「选择题目」里切过去即可开始新的一题。")
