"""学生端「上传视频」那条路的静态测试。

## 为什么要有这个文件
`streamlit_app.py` 那 400+ 行渲染代码**没有测试**，而本轮出过两次问题
（"解题路径"措辞过头、无图字段）**都在渲染层** —— 纯函数测试挡不住。

## 测什么 / 不测什么
**测接线**：控件在不在、字段传没传、**有没有"写了没接"**、失败有没有人话。
**不测渲染效果** —— 那要人工过目。静态断言只能挡住"根本不存在"。

## 三个我踩过的坑（都写在这儿，免得下次再踩）
1. **不剥注释** -> 断言 `"step_match" in code` 匹配到我自己写的解释性注释，
   **测试假通过**。
2. **用 tokenize 逐 token 再拼回去** -> token 之间被插入换行，
   `def call_video_analyze(` 被拆成三行，**断言永远匹配不上**。
3. **用 `Set-Content -Encoding utf8` 写文件** -> 加了 BOM，
   `ast.parse` 直接 `SyntaxError: invalid non-printable character U+FEFF`。

=> **最终做法：按整行过滤注释（保持代码原样），文件用 Python 写（无 BOM）。**
   行级过滤足够挡住宿 1 那种假通过 —— 那条假通过来自**整行注释**。

## 为什么不用正则配对括号
`st.file_uploader(...)` 的参数里就有 `type=["mp4", ...]`，
`re.search(r"\\((.*?)\\)")` 会在第一个 `)` 提前截断。改用窗口取法。
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

UI = Path(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py'))


@pytest.fixture(scope="module")
def code() -> str:
    """源码，**按整行去掉注释**。

    行级过滤保留了代码的原样（缩进、相邻 token 都在），
    又能挡住"断言匹配到整行注释"那种假通过。
    """
    out = []
    for line in UI.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def window(code: str, anchor: str, size: int = 700) -> str:
    """从 anchor 往后取一段 —— **不试图配对括号**。"""
    i = code.find(anchor)
    assert i >= 0, f"找不到 {anchor!r}"
    return code[i:i + size]


def func_body(code: str, name: str) -> str:
    """取一个顶层函数的函数体（到下一个顶层 def 或文件尾）。"""
    i = code.find(f"def {name}(")
    assert i >= 0, f"找不到函数 {name}"
    j = code.find("\ndef ", i + 1)
    return code[i:j if j > 0 else len(code)]


# ================================================================ 控件

def test_review_states_who_wrote_what(code):
    """**阅读约定必须由服务端署名**（契约 §2.6 ⑫，2026-10-10）。

    为什么：复盘里混着三类内容 —— 学生原文、**系统自己的复算**、系统的判断。
    模型在自由文本里**混着写**，而"要不要记得加标记"实测**靠不住**
    （泄题裁决：70 条里只有 16 条的 `next_step` 自带标记）。
    ⇒ 标签不能指望模型，就在**呈现层**把约定说清，让读者分得清。
    """
    i = code.find("阅读约定")
    assert i >= 0, "复盘没有署名阅读约定 ⇒ 学生分不清哪句是系统复算"
    block = code[i:i + 400]
    assert "材料原文" in block, "没说清引号内是学生原文"
    assert "系统自己的复算" in block, "没说清「核对时重建」是系统复算而非学生步骤"
    assert "不是学生写过的步骤" in block, "没把最容易误读的那一点说破"


def test_rag_block_never_shows_a_failure_as_no_content(code):
    """🔴 **不能把故障显示成"教材里没有"**（RAG 核查第 2 项的验收点）。

    ```
    后端原先两条路都返回空列表 ⇒ 页面若一律写"教材里没有相关内容"，
    就把**系统故障**说成了**知识库的正常结论** —— 学生会以为"书上确实没讲"。
    ```
    ⚠️ 后端给了状态**不等于**缺口修好：**前端措辞不改，等于没修**。
    ⇒ 这条断言钉的是**措辞**，不是字段。
    """
    i = code.find("def render_rag_block")
    assert i > 0, "没有 render_rag_block"
    block = code[i:code.find("\n# ---- 当前提示等级", i)]
    # ① 失败态**必须明说"没成功"，且必须与"没有"划清界限**
    fail = block[block.find('state == "failed"'):block.find('state == "no_hit"')]
    assert "没成功" in fail, "失败态没说清是失败"
    assert "不等于" in fail and "教材里没有" in fail, "失败态没与「教材里没有」划清界限"
    # ② 四种状态必须**分别措辞**（合并成一句就等于没区分）
    for s in ('"not_triggered"', '"failed"', '"no_hit"'):
        assert f"state == {s}" in block, f"{s} 没有独立分支"
    # ③ 来源展示要标**每片自己的页码**与**截断**
    assert "get('page'" in block or 'get("page"' in block, "来源没标页码"
    assert "truncated" in block, "没标「模型没看到被截断的部分」"
    # ④ **"无相关材料"不许说成"教材里没有"**（用户口径 2026-10-10）
    nohit = block[block.find('state == "no_hit"'):]
    assert "不代表书上没有讲" in nohit, "无命中态没与「教材里没有」划清界限"
    assert "重排判断" in nohit, "没区分「向量检索没命中」与「重排判断候选无关」"


def test_has_a_video_uploader(code):
    """学生必须能上传视频 —— 原来 st.text_area 是唯一入口。"""
    assert "file_uploader" in code, "UI 里没有上传控件"


def test_uploader_accepts_common_phone_formats(code):
    # ⚠️ 现在**有两个** `st.file_uploader`（图片 + 视频），
    # 所以不能再用"第一处 `st.file_uploader(`"定位 —— 那会取到图片那个。
    # 按 `type=[...]` 里的格式列表精确定位。
    assert '"mp4"' in code and '"mov"' in code, "视频上传不接受 mp4/mov"


def test_upload_is_disabled_when_backend_is_down(code):
    """**后端不通时不该让学生白等一两分钟。**"""
    assert "backend_ok" in window(code, "st.file_uploader(")


# ================================================================ 调用

def test_calls_the_video_endpoint(code):
    assert "/video/jobs" in code


def test_sends_all_required_fields(code):
    """端点要求 problem_id / student_id，漏了会 422。"""
    win = window(code, "data_form")
    for field in ("problem_id", "student_id"):
        assert field in win, f"少传了 {field}"


def test_timeout_is_long_enough(code):
    """**实测要几分钟，默认超时会让它必然失败。**

    两条路各有各的超时：视频约 4 分钟、图片约 10 秒。
    """
    # ⚠️ 正则必须精确到 `/video/analyze"`（带引号）——
    # 否则 `.*?` 会**非贪婪地匹配到 `analyze_image`**，取到图片那条的超时。
    m = re.search(r'/video/jobs"\s*,(.*?)timeout=(\d+)', code, re.S)
    assert m, "视频请求没显式设超时"
    assert int(m.group(2)) >= 120, f"视频超时 {m.group(2)}s 太短"
    m2 = re.search(r'/video/analyze_image"\s*,(.*?)timeout=(\d+)', code, re.S)
    assert m2, "图片请求没显式设超时"
    assert int(m2.group(2)) >= 120, f"图片超时 {m2.group(2)}s 太短"


# ================================================================ 结果要有人读

def test_video_result_is_rendered(code):
    """**防「写了没接」**。

    存进 session_state.video_result 却没人读，学生就白等一两分钟什么都看不到
    —— 而静态看代码不报错。**我第一版就是这样，靠这条检查才发现。**
    """
    assert "session_state.video_result =" in code, "没有把结果存进 session_state"
    assert ('get("video_result")' in code
            or "get('video_result')" in code),\
        "**存了但从来没读 —— 结果被丢掉了**"


def test_shows_progress_hint(code):
    """处理要一两分钟，**必须告诉学生等着**。"""
    assert "spinner" in code
    assert "分钟" in code, "没说大概要多久"


def test_tells_the_student_the_video_is_deleted(code):
    """隐私承诺要**对学生可见** —— 只在后端删、不告诉学生，等于没承诺。"""
    assert "删除" in code or "不会留档" in code


# ================================================================ 失败路径

def test_failure_paths_give_readable_messages(code):
    body = func_body(code, "call_video_analyze")
    assert "except" in body, "没有异常处理"
    assert "⚠️" in body, "失败提示不像给人看的"


def test_empty_result_is_handled(code):
    """**「视频里没内容」是最可能的失败**，要有专门的话。"""
    body = Path(str(Path(__file__).resolve().parents[1] / 'frontend/ui_logic.py')).read_text(encoding="utf-8")
    assert "rounds" in body, "没检查返回的轮次"
    assert "没有可用" in body, "没有专门处理空结果"

# ================================================================ L0 不能泄露诊断

def test_hint_column_does_not_fall_back_to_issue(code):
    """**「提示」列不许拿 `issue` 顶上。**

    实测踩到：某轮 `hint` 为空（**那正是 L0 的设计语义："本轮不提示"**），
    代码却 fallback 到 `issue` —— 于是表格的"提示"列里出现了
    「学生仅凭直觉断言加速度为0，未对三个小问分别进行受力分析…」。

    **`issue` 是面向教师的错误归因（见 prompts/diagnosis）。**
    学生看到会当成提示读，**而它是"你错在哪"的答案线索** ——
    等于把 L0 变成了 L1 以上。

    这条是用户**导出那张 CSV 之后**才发现的 ——
    `st.dataframe` 自带下载按钮，所以表格和 CSV 是同一份。
    """
    assert 'r.get("hint") or r.get("issue")' not in code,\
        "「提示」列又 fallback 到 issue 了 —— L0 会泄露诊断"


def test_issue_is_still_shown_where_it_is_labelled(code):
    """**别把孩子跟洗澡水一起倒掉** —— `issue` 该显示的地方还要显示。

    它只是不该**冒充提示**（前面那条），但明确标注为"问题"/"诊断"时是给学生看的。
    """
    assert "问题：" in code or "问题:" in code, "Tutor 面板不再显示问题了"

# ================================================================ OCR 校验入口

def test_table_shows_the_recognized_answer(code):
    """**这是学生唯一能发现 OCR 读错的地方。**

    实测踩到：学生写的是 `F = 49 N`（正确答案），**OCR 读成 `F = 4`**
    （用户看原始视频确认），而复盘据此说成"得到 F = 4 这种没有物理意义的结果"
    —— **把"碰巧对了"说成了"错得离谱"。**

    **复盘判断的依据和它看到的是同一份文本，OCR 读错它无法发现。**
    ⇒ 必须把识别原文摆到学生眼前。
    """
    assert "识别到的作答" in code, "表格里没有「识别到的作答」这一列"
    assert 'r.get("answer")' in code, "那一列没取 answer 字段"


def test_review_warns_that_ocr_may_be_wrong(code):
    """复盘上要有提醒 —— 否则学生读完结论就走了，不会去核对原文。"""
    assert "以你的原稿为准" in code or "原稿" in code, "复盘没提醒学生核对原文"
    assert "误读" in code, "没说明手写数字会被误读"


def test_round_table_is_expanded_by_default(code):
    """**不许默认折叠** —— 折叠起来等于把校验机会藏了。

    （原来是 `expanded=not _rev`：有复盘就收起表格 —— 那正好把
    学生唯一能核对的地方藏在了复盘后面。）
    """
    # 逐轮表格现在叫「逐轮记录」，位置在复盘之后。
    i = code.find("st.dataframe(")
    assert i >= 0, "找不到轨迹表格"
    win = code[max(0, i - 1200):i]
    assert "expanded=True" in win, "逐轮表格不是默认展开的"


def test_endpoint_records_the_answer_field(code=None):
    """端点必须把学生作答记进 `rounds` —— 表格那一列才有东西可显示。"""
    import inspect

    from app.api import video as V

    src = inspect.getsource(V.analyze_video)
    assert '"answer"' in src, "rounds 里没记学生作答（表格那一列会是空的）"

# ================================================================ 图片那条路

def test_has_an_image_branch(code):
    """**卡住时的求助不需要过程** —— 拍一张就够。

    视频那条 254 秒（成本主要是解码 855 帧），图片这条约 10 秒（实测）。
    """
    assert "analyze_image" in code, "UI 没接图片端点"
    assert "call_image_analyze" in code, "没有图片的处理函数"


def test_image_branch_offers_the_common_formats(code):
    for fmt in ("jpg", "png"):
        assert f'"{fmt}"' in code, f"不接受 {fmt}"


def test_image_branch_is_available_alongside_video(code):
    """**两条都要在** —— 不是替换关系，是给学生的两个选择。"""
    assert "media_mode" in code, "没有模式选择控件"
    assert "video_upload" in code and "image_upload" in code


def test_image_branch_tells_the_student_the_photo_is_not_stored(code):
    """隐私承诺要**对学生可见**（视频那条也这么写）。"""
    assert "可保留原图" in code, "必须如实说明新照片可保留供核对，不能承诺从不落盘"


def test_image_branch_handles_the_backend_detail(code):
    """失败时要把后端的 `detail` 显示出来 —— 否则又是一个光秃秃的状态码。"""
    body = func_body(code, "call_image_analyze")
    assert "detail" in body, "没读后端 detail"
    assert "except" in body and "⚠️" in body


def test_no_bogus_session_keys(code):
    """**防我自己写错字段名。**

    第一版我写了 `st.session_state.last_action_times = None` ——
    而全项目只有那一处叫这个名字（正确的是 `last_action_ts`）。
    Streamlit 允许任意键，**所以它不报错、只是静默地没用**。
    """
    keys = set(re.findall(r"session_state\.([a-z_]+)", code))
    assert "last_action_times" not in keys, "又写了那个不存在的字段名"
    assert "last_action_ts" in keys, "正确的字段名不见了"


# ================================================================ 照片：即时辅导（2026-10-09 决定）

def test_photo_path_promises_no_seconds(code):
    """**决定 2**：本轮不显示未经测量支持的预计秒数。

    这里以前写着「约 10 秒」—— 那是**一次实测的数字**，不是对模型的保证，
    而它的来源（原本含复盘）后来也变了。
    ⇒ 按**模式**断言，不是只挡那一个数字：换个数、加个星号都算回归。
    """
    leftover = re.findall(r"约\s*\*{0,2}\s*\d+\s*秒", code)
    assert not leftover, f"又出现了预计秒数：{leftover}"
    assert "正在识别作答并生成下一步提示" in code, "处理中没说在做什么"
    # ⚠️ 这个正则扫的是**整个文件** ⇒ **注释里也不能写「约 N 秒」**。
    # 实测踩过：一条记录性能实测数据的注释被它拦下了（那是陈述，不是承诺）。
    # 故意不缩小范围 —— 缩成"只看界面字符串"要靠静态分析、容易漏；
    # 全文件一刀切更好维护，代价只是措辞绕开一下。


def test_photo_failure_offers_reason_and_retry(code):
    """**决定 2**：失败时给**原因**与**重试入口**。

    ⚠️ 原因**不能塞进 `tutor["hint"]`** —— `hint` 是分级提示的载体
    （契约 §1.1 不变式①），学生会把「处理失败」当成一条教学提示读。
    """
    body = func_body(code, "call_image_analyze")
    assert "photo_error" in body, "失败原因没有独立字段"
    assert '"hint": f"⚠️' not in body and "'hint': f'⚠️" not in body,\
        "失败信息又写回 hint 了"
    assert "重试这张照片" in code, "没有重试入口"
    assert "photo_retry" in code, "重试没有留下可重发的上传内容"


def test_photo_success_shows_recognition_and_hint_first(code):
    """**决定 2**：成功后**先**展示识别文本和提示。

    识别原文必须最先给：手写数字实测被读错过（`49 N` → `4`），
    而提示与复盘全都建立在识别文本上 —— **只有学生能发现读错了**。
    """
    assert 'get("recognized_text")' in code, "没有先展示识别原文"
    assert "我识别到的作答" in code, "识别原文没有标题"
    assert "下一步提示" in code, "没有先展示提示"


def test_photo_branch_does_not_advertise_a_later_review(code):
    """**决定 1 / 3**：照片没有延后生成入口 ⇒ **不能**说有。

    视频那句「也可按需生成视频复盘」只属于视频分支。
    """
    video_line = "也可按需生成视频复盘"
    assert video_line in code, "视频那条的提示不见了"
    photo_notice = "照片这条**不提供延后生成复盘**的入口"
    assert photo_notice in code, "照片分支没有说清没有延后复盘入口"
