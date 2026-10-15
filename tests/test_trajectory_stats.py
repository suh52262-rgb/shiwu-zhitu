"""服务端轨迹统计的测试 —— 契约硬约束④「轮次/次数/范围由服务端统计」。

## 为什么要有这一段
审计发现一处**真实的计数错**：`ocr_caution` 说某数值出现在「四轮片段」，
**实际是五轮**（第 4、5、6、7、8 轮）。

**这不是"模型不小心"，是任务本身不该给它** ——
让模型从自由文本里数"第几轮出现了几次"它做得不稳，
**而服务端手里有全部轮次。** ⇒ 把"数数"从模型手里拿走，改成喂给它。

## 两个我自己在冒烟时测出来的缺陷（都写成了测试）
```
① 归一化把 `^` 删了 ⇒ `1.98×10^4` 显示成 `1.98×104` —— **那是在改数值**
② 抓进了噪声 ⇒ 从 `(1) … (2) …1.65e5` 里抓出 `1`/`2`/`5`
   （题号与指数残片）—— **列进去会误导模型**
```
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.services.review_builder import (  # noqa: E402
    _looks_like_a_quantity,
    build_review_user,
    compute_trajectory_stats,
    fill_stats_placeholders,
    find_stray_counts,
    format_stats_block,
    report_text_fields,
    structural_defects,
)

#: **审计那次的真实情形** —— 1.98×10^4 出现在五轮里
AUDIT_ROUNDS = [
    {"round": 1, "ts": 0.0, "answer": "I = ."},
    {"round": 2, "ts": 20.6, "answer": "I = mΔv ="},
    {"round": 3, "ts": 41.2, "answer": "I = 1.98×10^4 N·s"},
    {"round": 4, "ts": 62.9, "answer": "(1) I=1.98*10^4 N·s (2) F=I/Δt="},
    {"round": 5, "ts": 83.7, "answer": "F = I/Δt = 1.65×10^5 N"},
    {"round": 6, "ts": 104.5, "answer": "1.98×10^4，方向与Δv一致"},
]


# ================================================================ 治那次计数错

def test_counts_occurrences_across_rounds():
    """**这就是治那次计数错的那条。** 服务端数，不靠模型。"""
    s = compute_trajectory_stats(AUDIT_ROUNDS)
    # ⚠️ 形状在复核后改过：从 `[轮号…]` 改成 `{rounds, occurrences}`
    assert s["value_rounds"]["1.98×10^4"]["rounds"] == [3, 4, 6]
    assert s["value_rounds"]["1.98×10^4"]["occurrences"] == 3
    assert s["round_count"] == 6
    assert s["count_basis"].startswith("服务端统计")


def test_block_tells_the_model_not_to_count_itself():
    """**必须明说「不要自己数」** —— 否则模型还是会自己数一遍。"""
    blk = format_stats_block(compute_trajectory_stats(AUDIT_ROUNDS))
    assert "不要自己数" in blk
    assert "1.98×10^4" in blk and "第 3、4、6 轮" in blk


def test_block_refuses_to_guess_about_unlisted_values():
    """没列出的量，**不许它声称出现了几次**。"""
    blk = format_stats_block(compute_trajectory_stats(AUDIT_ROUNDS))
    assert "不要声称它出现了几次" in blk
    assert "无法从材料确认" in blk


# ================================================================ 缺陷①：`^` 不能被删

def test_preserves_the_caret():
    """**冒烟时测出来的**：第一版归一化把 `^` 删了，
    `1.98×10^4` 显示成 `1.98×104` —— **那是在改数值，不是归一化写法。**"""
    s = compute_trajectory_stats([{"round": 1, "answer": "x = 1.98×10^4"}])
    assert "1.98×10^4" in s["value_rounds"], s["value_rounds"]
    assert "1.98×104" not in s["value_rounds"]


def test_normalizes_only_the_multiplication_sign():
    """只归一化写法：`*` / `x` 与 `×` 等价，数字本身不动。"""
    s = compute_trajectory_stats([
        {"round": 1, "answer": "a = 1.98*10^4"},
        {"round": 2, "answer": "b = 1.98×10^4"},
    ])
    assert s["value_rounds"]["1.98×10^4"]["rounds"] == [1, 2]


# ================================================================ 缺陷②：噪声 token

def test_drops_single_and_double_digit_integers():
    """**冒烟时测出来的**：从 `(1) … (2) …1.65e5` 里抓出 `1`/`2`/`5` ——
    题号与指数残片。**列进去会误导模型。**"""
    s = compute_trajectory_stats(
        [{"round": 1, "answer": "(1) I=1.98*10^4 (2) F=1.65*10^5"}])
    for noise in ("1", "2", "5"):
        assert noise not in s["value_rounds"], f"噪声 {noise} 混进来了"


def test_keeps_real_quantities():
    """但**真量要留住**：带小数点的、带科学计数法的、≥3 位的整数。"""
    assert _looks_like_a_quantity("1.98")
    assert _looks_like_a_quantity("1.98×10^4")
    assert _looks_like_a_quantity("1200")
    assert not _looks_like_a_quantity("1")
    assert not _looks_like_a_quantity("22")


def test_documents_what_it_loses():
    """**口径的代价要写下来**：丢单/双位整数会漏掉 `m₂ = 2 kg` 这类真量。
    但漏了模型只会说"无法确认"，而混进噪声它会**拿编号当物理量引用**。
    ⇒ 宁可漏。"""
    import inspect

    from app.services import review_builder as rb

    src = inspect.getsource(rb._looks_like_a_quantity)
    assert "会漏掉" in src and "宁可漏" in src


# ================================================================ 结构与如实

def test_reports_empty_rounds_honestly():
    """空轮**如实给出**，不替模型猜。"""
    s = compute_trajectory_stats([{"round": 1, "answer": ""},
                                  {"round": 2, "answer": "F=ma"}])
    assert s["empty_rounds"] == [1]


def test_reports_non_contiguous_round_indices():
    """`round_indices` **可能不连续**，服务端如实给出。"""
    s = compute_trajectory_stats([{"round": 2, "answer": "a"}, {"round": 5, "answer": "b"}])
    assert s["round_indices"] == [2, 5]


def test_no_timestamp_means_no_fabricated_time():
    """没有时间戳时**不许编造时刻**。"""
    s = compute_trajectory_stats([{"round": 1, "answer": "a"}])
    assert s["time_range"] is None
    assert "不得编造时刻" in format_stats_block(s)


def test_handles_empty_input():
    s = compute_trajectory_stats([])
    assert s["round_count"] == 0 and s["value_rounds"] == {}


# ================================================================ 接线

def test_stats_are_fed_into_the_prompt():
    """**防「写了没接」** —— 统计必须在交给模型的 user 消息里。"""
    u = build_review_user("题干", ["冲量"], AUDIT_ROUNDS)
    assert "【服务端统计】" in u, "统计没进提示词"
    assert "不要自己数" in u
    assert "1.98×10^4" in u


def test_prompt_still_carries_the_students_text():
    """**别把已有的挤掉** —— 学生原文仍是判「凑对」的唯一依据。"""
    u = build_review_user("题干", [], AUDIT_ROUNDS)
    assert "1.98×10^4 N·s" in u
    assert "可用识别轨迹片段" in u

# ================================================================ 复核指出的四条缺陷

def test_same_value_twice_in_one_round_is_not_two_rounds():
    """**复核指出**：第 7 轮写两次 `19.6`，旧版给 `[7, 7]`，渲染成「共 2 轮」。

    ⇒ 现在分列 `rounds`（去重轮号）与 `occurrences`（总次数）。
    **那是两个不同的量，混在一起就是在说假话。**
    """
    s = compute_trajectory_stats([{"round": 7, "answer": "x=19.6 N；y=19.6 N"}])
    slot = s["value_rounds"]["19.6"]
    assert slot["rounds"] == [7], "同一轮不该重复计数"
    assert slot["occurrences"] == 2, "但出现次数要如实记"
    blk = format_stats_block(s)
    assert "出现 2 次" in blk and "涉及 1 轮" in blk


def test_sign_is_not_lost():
    """**复核指出**：`-19.6` 与 `+19.6` 被归成一个 —— **方向信息在统计里丢了**，
    而复盘恰恰要判方向。"""
    s = compute_trajectory_stats([{"round": 1, "answer": "a = -19.6"},
                                  {"round": 2, "answer": "b = +19.6"}])
    assert "-19.6" in s["value_rounds"] and "+19.6" in s["value_rounds"]
    assert "19.6" not in s["value_rounds"], "不该还有一个不带符号的裸值"


def test_round_numbering_matches_the_trajectory_text():
    """**复核指出**：统计用真实 `round`，正文按顺序重编 ⇒ **两套号互相矛盾**，
    **比原来那个计数错更糟**（模型看到两个都对的号，只能猜）。

    ⇒ 现在两处一律用真实 `round`。
    """
    rounds = [{"round": 2, "ts": 0.0, "answer": "I = 1.98×10^4"},
              {"round": 5, "ts": 20.0, "answer": "F = 1.65×10^5"}]
    u = build_review_user("题干", [], rounds)
    assert "轮次编号：[2, 5]" in u
    assert "第2轮" in u and "第5轮" in u, "正文要用真实轮号"
    assert "第1轮" not in u, "不许出现顺序重编的号"


def test_occurrence_count_and_round_count_are_both_reported():
    """两个量都要给 —— 只给一个就会误导。"""
    s = compute_trajectory_stats([
        {"round": 1, "answer": "1.98×10^4"},
        {"round": 2, "answer": "1.98×10^4"},
        {"round": 2, "answer": "1.98×10^4"},
    ])
    slot = s["value_rounds"]["1.98×10^4"]
    assert slot["occurrences"] == 3
    assert slot["rounds"] == [1, 2]

# ================================================================ 服务端填充（硬约束④后半）

def test_placeholder_is_filled_by_the_server():
    """**把数字从模型手里拿走** —— 它只写占位符，真实轮次由服务端填。"""
    from app.services.review_builder import fill_stats_placeholders

    rounds = [{"round": 4, "answer": "1.98×10^4"},
              {"round": 5, "answer": "1.98*10^4"},
              {"round": 8, "answer": "1.98×10^4"}]
    st = compute_trajectory_stats(rounds)
    filled, missed = fill_stats_placeholders("出现在 [[value:1.98×10^4]]。", st)
    assert "第 4、5、8 轮" in filled
    assert "出现 3 次" in filled and "涉及 3 轮" in filled
    assert missed == []
    assert "[[" not in filled, "占位符不该残留给学生看"


def test_unresolved_placeholder_is_reported_not_silently_left():
    """**未命中要报出来** —— 不能静默留一个占位符在给学生看的文本里。"""
    from app.services.review_builder import fill_stats_placeholders

    st = compute_trajectory_stats([{"round": 1, "answer": "1.98×10^4"}])
    filled, missed = fill_stats_placeholders("看 [[value:9.99×10^9]]", st)
    assert missed == ["9.99×10^9"]
    assert "[[" not in filled
    assert "不得据此推断次数" in filled


def test_stray_counts_are_detected_including_chinese_numerals():
    """**那次真实计数错用的是中文数字**（「四轮」）—— 所以中文也要认。"""
    from app.services.review_builder import find_stray_counts

    # ⚠️ 判据收紧后，捕获的是**带前缀的短语**（更有用 —— 一眼看得出是哪种说法）
    assert find_stray_counts("出现在四轮片段") == ["出现在四轮"]
    assert find_stray_counts("共 5 轮出现") == ["共 5 轮"]


def test_detector_distinguishes_counting_from_locating():
    """⭐ **真机跑 v3 时逼出来的判据修正。**

    第一版判据太宽，把合法内容也报了。真机剩下的 3 处是：
    ```
    「第7轮出现 a = 1.6，第9轮起改为 a = 1.49」
    「第14轮 F = 49 N 中的 49 识别清晰」
    ```
    **那不是「出现了几次」，那是「在某一轮里」** ——
    而**定位正是可追溯性要的东西**，不该被报成问题。

    ⇒ 计数要报，定位不报。
    """
    from app.services.review_builder import find_stray_counts as f

    # 定位：不该报
    assert f("第7轮出现 a = 1.6") == []
    assert f("第14轮 F = 49 N 中的 49 识别清晰") == []
    assert f("第9轮起改为 a = 1.49") == []

    # 计数：要报
    assert f("该值出现在四轮片段") != []          # **那次真实缺陷**
    assert f("第4、5、6、7、8轮都有") != []
    assert f("共五轮") != []
    assert f("出现了 3 次") != []


def test_tightening_also_removed_the_old_false_positive():
    """收紧判据**顺带修掉了原来那个误报**：

        「两轮车里」→ 原来抓 `两轮`（那不是计数）⇒ **现在没有了**

    ⇒ 已把代码注释里「误报仍在」的话改成事实。
    """
    from app.services.review_builder import find_stray_counts

    assert find_stray_counts("两轮车里") == []


def test_detector_documents_what_it_still_misses():
    """仍知的洞**如实留着** —— 不假装它完备。"""
    import inspect

    from app.services import review_builder as rb

    src = inspect.getsource(rb.find_stray_counts)
    assert "漏报" in src, "仍知的洞要写下来"


def test_trajectory_wiring_fills_report_fields():
    """**防「写了没接」** —— 填充必须真接进 `review_trajectory`。"""
    import inspect

    from app.services import review_builder as rb

    src = inspect.getsource(rb.review_trajectory)
    assert "fill_stats_placeholders" in src, "没接填充"
    assert "find_stray_counts" in src, "没接兜底检测"
    assert "只报不改" in src or "未命中的统计占位符" in src


# ================================================================ 兜底检测的判据边界
#
# ⚠️ 这一组的由来：函数文档字符串里写着**两个"没修的洞"**
# （漏报「第4、5、6、7、8轮」/ 误报「两轮车里」），而正则那段注释说已经修掉。
# **两处互相矛盾** ⇒ 实测 13 条样本，结论：**两个都早已修好，是那段注释过时了**。
# ⇒ 把实测结果钉成测试，并写明残留的那一个。
#
# **为什么方向要紧**：它现在被用作**阻断判据**。
#   漏报 = 一份坏报告放行（等同现状）；误报 = 一份好报告被打掉、还要重试。
#   所以判据**刻意偏保守：宁可漏，不误杀**。

@pytest.mark.parametrize("text,must_report", [
    # —— 该报：计数 ——
    ("第4、5、6、7、8轮都有", True),      # 那个"漏报的洞"：实测**已修**
    ("第4、5、6轮都写了这一步", True),
    ("共五轮", True),
    ("出现在 5 轮里", True),
    ("一共改了 3 次", True),
    # —— 不该报：定位（单轮 + 「第」） ——
    ("第7轮写的是 a = 1.6", False),
    ("第14轮 F = 49 N 中的 49 识别清晰", False),
    ("第9轮起改为 a = 1.49", False),
    # —— 不该报：不是计数 ——
    ("两轮车里坐着两个人", False),         # 那个"误报的洞"：实测**已修**
    ("两轮车与四轮车的区别", False),
    ("这题共 2 问", False),
])
def test_stray_count_detector_battery(text, must_report):
    """判据边界**逐条钉住** —— 文档说"修了/没修"都不算，跑出来才算。"""
    got = find_stray_counts(text)
    assert bool(got) is must_report, f"{text!r} → {got}（期望 must_report={must_report}）"


def test_the_detector_still_misses_one_form_and_we_say_so():
    """**残留的洞，如实钉住**：**裸量词 + 轮** 一律抓不到。

    ⚠️ 我第一版把这条写成"只是里/中的形态漏"，**实测打脸**：
    「这五轮都稳」同样抓不到 —— 盲区是**整个裸量词形态**，不是某个后缀。
    方向是**漏报**（少拦），与"宁可漏，不误杀"一致。

    ⚠️ **所以它不是第 3 条修复的完成标志**：真正的堵法是让模型没机会写计数
    （占位符 + 服务端填充），这个检测只兜住"它还是不听话"的一部分形态。
    真要收紧，**先补样本再改正则**，别为了让这一条变绿就把误报面放大。
    """
    for missed in ("五轮里都有", "这五轮都稳", "前四轮都对"):
        assert find_stray_counts(missed) == [], f"{missed!r} 居然被抓住了 —— 判据变宽了？"
    # 同一意思换一种**判据覆盖**的写法是抓得到的
    assert find_stray_counts("共五轮") == ["共五轮"]
    assert find_stray_counts("出现在 5 轮里") == ["出现在 5 轮"]


# ================================================================ P0（独立复核发现，均已修）
#
# ⚠️ 这一组的存在本身是教训：**我给新写的归因扫描器量了召回与误报，
#    却对那个"早就存在"的计数检测器只用手写的 13 条电池试了试就信了它的注释。**
#    结果它第一次碰到真实输出（冒烟）就是误报 —— 而我把那次误报读成了"模型不服从"。

@pytest.mark.parametrize("text", [
    "整份材料只有一次快照",
    "本轮学生一次性给出了完整解答链条",
    "y 是 x 的一次函数",
    "他把电压与电流的关系当成一次函数处理",
    "做一次核对即可",
    "一次侧电压为 4 V",
])
def test_the_ordinary_adverb_yici_is_not_a_count(text):
    """**P0-A**：「一次」是中文最普通的副词/量词，**不是**轮次计数。

    旧判据的裸 `N 次` 分支会命中它 ⇒ 在 fail-closed 下**误杀内容正确的报告**。
    复核重放归档实测：**72 份里 42 份（58%）的唯一命中就是「一次」**。
    """
    assert find_stray_counts(text) == [], f"{text!r} 被误判成计数"


@pytest.mark.parametrize("text,expected", [
    ("出现 2 次", ["出现 2 次"]),
    ("一共改了 3 次", ["一共改了 3 次"]),
    ("重复 4 次", ["重复 4 次"]),
    ("共五轮", ["共五轮"]),
    ("第4、5、6轮都写了", ["4、5、6轮"]),
    ("出现在 5 轮里", ["出现在 5 轮"]),
])
def test_the_real_counts_are_still_caught(text, expected):
    """收紧之后**该报的还是要报** —— 否则这一条修复就变成了"把闸门关掉"。"""
    assert find_stray_counts(text) == expected


# ---------------------------------------------------------------- P0-B：服务端不能打自己

def test_the_server_render_format_is_coupled_to_its_stripper():
    """**P0-B 的耦合测试**：`render_value_occurrence` 的产物必须能被 `_SERVER_RENDER_RE` 摘掉。

    ⚠️ 这两处一旦漂开，就会恢复成"**服务端自己渲染的统计文本触发自己的阻断判据**" ——
    实测过：每一次成功的占位符填充都必然产出一条 `model_written_count`，
    **模型越听话越必然被打回**。改了渲染格式而没改正则，这条测试会红。
    """
    from app.services import review_builder as rb

    rounds = [{"round": 4, "ts": 62.9, "answer": "I = 1.98*10^4 N·s"},
              {"round": 5, "ts": 83.7, "answer": ""},
              {"round": 8, "ts": 144.2, "answer": "1.98×10^4，方向与Δv一致"}]
    stats = compute_trajectory_stats(rounds)
    rendered = rb.render_value_occurrence(stats, "1.98*10^4")
    assert rendered, "渲染结果为空，测试失去意义"
    # ① 渲染结果**确实**带计数形态（所以必须摘）
    assert find_stray_counts(rendered), "渲染格式变了？那 P0-B 的前提也要重新看"
    # ② 摘掉之后必须干净
    assert find_stray_counts(rb.strip_server_renders(rendered)) == []


def test_a_placeholder_fill_does_not_convict_the_model():
    """**P0-B 的端到端**：模型乖乖用占位符 ⇒ 填充后**不得**被判违规。

    这是"模型越听话越必然被打回"那个死循环的回归测试。
    """
    rounds = [{"round": 4, "ts": 62.9, "answer": "I = 1.98*10^4 N·s"},
              {"round": 5, "ts": 83.7, "answer": ""},
              {"round": 8, "ts": 144.2, "answer": "1.98×10^4，方向与Δv一致"}]
    stats = compute_trajectory_stats(rounds)
    filled, missed = fill_stats_placeholders("出现在 [[value:1.98*10^4]]，见 [E4.1]。", stats)
    assert missed == [], "夹具本身就不合规，测试失去意义"
    payload = {"timeline": filled, "sticking_points": [], "key_concept": "",
               "next_step": "", "encouragement": "", "answer_check": None,
               "findings": [], "evidence": [], "review_schema_version": "v3+prompt-v3"}
    assert structural_defects(payload) == [],\
        "服务端自己填的数字把模型判成了违规（P0-B 回归）"


def test_the_prompt_no_longer_pushes_the_model_onto_that_mine():
    """提示词里教的那句 ✅ 示例，正是踩 P0-B 的形态 —— 现在必须写明边界。"""
    prompt = (Path(str(Path(__file__).resolve().parents[1] / 'prompts/review/v3.md'))).read_text(encoding="utf-8")
    assert "别自己数" in prompt
    assert "单轮定位" in prompt

#
# ⚠️ 这两条防的是一个**很容易犯的错**：
#   `answer_check.student_answer` 是**逐字引用学生**，
#   `evidence[*].text` 是**服务端逐字取自轨迹** ——
#   把它们算进"模型自写次数"的扫描范围，等于**拿学生自己的话去判模型违规**，
#   而"两轮车"这种词恰恰最可能出现在学生的作答里。
#   ⇒ 误杀的代价是"好报告被打掉 + 学生重试付费"，而判据本来只想管模型。

def test_the_count_scan_never_blames_the_model_for_the_students_words():
    payload = {
        "timeline": "他把每一步都写清楚了",
        "answer_check": {"student_answer": "两轮车里坐着两个人",   # 逐字引用学生
                         "derived_from_work": "沿用他的写法"},
        "evidence": [{"text": "第4、5、6轮都写了"}],               # 服务端逐字取自轨迹
        "findings": [], "sticking_points": [],
    }
    paths = [p for p, _ in report_text_fields(payload)]
    assert "answer_check.student_answer" not in paths, "把学生原文算进了模型文本范围"
    assert not any("evidence" in p for p in paths), "把服务端证据原文算进了模型文本范围"
    assert [d for d in structural_defects(payload) if d["kind"] == "model_written_count"] == []


def test_the_model_own_text_is_still_scanned():
    """反过来：**模型自己写的**计数（判据覆盖的形态）必须抓到，否则闸门形同虚设。

    ⚠️ 这里**只挑判据覆盖的写法**（逗号列表 / 共N轮 / 出现在N轮 / N次）。
    裸量词形态是**已知漏报**，见上一条测试 —— 拿它来断言会把测试写成假的。
    """
    for field, value in (("timeline", "他一共写了 3 次"),
                         ("encouragement", "共五轮都稳"),
                         ("next_step", "把第4、5、6轮再核对一遍")):
        payload = {"timeline": "", "encouragement": "", "next_step": "",
                   "findings": [], "sticking_points": [], "answer_check": None,
                   field: value}
        kinds = [d["kind"] for d in structural_defects(payload)]
        assert "model_written_count" in kinds, f"{field} 里的自写计数没被抓到"

    # 反过来钉一句：裸量词**确实漏**，别以为它被管住了
    payload = {"timeline": "这五轮都稳", "encouragement": "", "next_step": "",
               "findings": [], "sticking_points": [], "answer_check": None}
    assert [d for d in structural_defects(payload)
            if d["kind"] == "model_written_count"] == [], "裸量词居然被管住了？判据变宽了"

