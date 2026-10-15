"""稳定期：只在内容**写完并稳定**之后才产出事件。

当前测试使用项目自编的合成记录，不读取学生视频或历史OCR档案。
以下实测描述仅保留为历史背景，不由本测试复验，也不作为当前识别质量声明。

## 为什么需要它
学生逐笔书写时，关键帧抓到的常是**画到一半的字**，而 VL 不会说"没写完"，
它会**给一个看起来合理的符号**。实测 60 帧：

```
'由题知'           60/60 帧   尾部连续 60   ← 写完并留住了
'⊙' '⊕' 'Γ' '⊥' '√' 'Σ'  ≤6/60 帧   尾部连续 0   ← 闪一下就没
'F' '=' '_' '二'    17~39/60 帧  尾部连续 0~18  ← **真的，但正在写**
```

**第三行是关键**：`F`、`=` 是学生正在写的真实公式，读数**同样不稳定**。
⇒ 这个信号测的**不是"真 vs 幻觉"，而是"写完了 vs 还没写完"**。

**所以不能用"滤掉低留存的"** —— 那会把答案本身滤掉。
正确用法反着来：**只对已稳定的内容发事件**。正在写的内容是**推迟**，不是丢弃；
幻觉永远稳定不下来，于是**自动**被排除（副产品，不是专门过滤）。

## 历史观察记录（final54，非本测试复验）
| | 幻觉符号 | 末尾读数 |
|---|---|---|
| 关闭 | 20 个 | `由题知 F_k = -` |
| window=3 | **1 个** | `由题知 F_k = -`（保留）|
| window=5 | **0 个** | `由题知 F_k =` |

## ⚠️ 已知不够
**事件流没有改善**（仍 22 条，且震荡：`F` → `F =` → `F`）。
因为稳定前缀序列**自身仍会摇摆**，而 `frames_to_learning_events`
把每次变化都当事件。**事件层还需要自己的稳定判据**（见文件尾的说明）。
"""
import inspect
from tests.fixture_assets import FIXTURE_ROOT

import pytest

from app.vision import bridge as br


def test_settle_constants_exist():
    assert br.SETTLE_WINDOW >= 2
    assert br.SETTLE_MIN_CHARS >= 1


def test_settled_prefix_requires_full_window():
    """窗口没凑满就不算稳定——否则开头两帧会立刻"稳定"。"""
    assert br.settled_prefix(["由题知"], 3) == ""
    assert br.settled_prefix(["由题知", ""], 3) == ""
    assert br.settled_prefix(["由题知", "由题知", "由题知"], 3) == "由题知"


def test_settled_prefix_takes_common_part():
    assert br.settled_prefix(["由题知 F", "由题知 F =", "由题知 F_"], 3) == "由题知 F"
    assert br.settled_prefix(["由题知", "abc", "xyz"], 3) == ""


def test_flicker_never_survives():
    """**核心断言**：一闪而过的读数进不了稳定前缀。

    这正是 `⊙ ⊕ Γ` 被排除的机制——不是"被过滤"，是**活不过窗口**。
    """
    seq = ["由题知", "由题知 ⊙", "由题知", "由题知", "由题知"]
    out = br.settle_snapshots([{"ts": i, "text": t} for i, t in enumerate(seq)],
                              window=3)
    assert all("⊙" not in s["text"] for s in out),\
        f"闪烁符号穿过了稳定期：{[s['text'] for s in out]}"


def test_real_content_is_delayed_not_dropped():
    """**正在写的内容是推迟，不是丢弃。**

    这是与"滤掉低留存"最本质的区别——那种做法会把答案本身滤掉。
    """
    seq = ["由题知"] * 3 + ["由题知 F"] * 3 + ["由题知 F ="] * 3
    out = br.settle_snapshots([{"ts": i, "text": t} for i, t in enumerate(seq)],
                              window=3)
    texts = [s["text"] for s in out]
    assert texts[-1] == "由题知 F =", f"末尾真内容丢了：{texts[-1]!r}"
    assert texts[0] == "", "开头没等够窗口就报内容了"


def test_short_prefix_is_not_content():
    """稳定但太短（如一个标点）不算内容——避免为 `。` 产出事件。"""
    out = br.settle_snapshots([{"ts": i, "text": "。"} for i in range(5)],
                              window=3)
    assert all(s["text"] == "" for s in out)


def test_settle_window_zero_disables():
    """`window<=0` = **关闭稳定期**，原样返回。

    ⚠️ 我第一版这条写的是 `assert ... or True` —— **永远成立，等于没测**，
    而它掩盖了一个真 bug：`window=0` 会让 `texts[-0:]` 取到整个列表，
    再取 `[0]` 越界崩掉。断言里加 `or True` 是自欺。
    """
    seq = [{"ts": 0, "text": "a"}, {"ts": 1, "text": "b"}]
    assert br.settle_snapshots(seq, window=0) == seq
    assert br.settle_snapshots(seq, window=-1) == seq


def test_settled_prefix_handles_zero_k():
    """`k<=0` 必须安全返回空串，而不是越界。"""
    assert br.settled_prefix(["a", "b"], 0) == ""
    assert br.settled_prefix([], 0) == ""
    assert br.settled_prefix([], 3) == ""


def test_default_is_none_not_a_constant():
    """**默认值必须是 None，不能直接写常量。**

    默认参数在**函数定义时**求值，常量位置就成了硬约束——
    我因此连撞两次 `NameError`。用 None 占位、运行时解析，顺序依赖消失。
    """
    sig = inspect.signature(br.records_to_learning_events)
    assert sig.parameters["settle_window"].default is None
    src = inspect.getsource(br.records_to_learning_events)
    assert "SETTLE_WINDOW if settle_window is None" in src


def test_settle_can_be_disabled():
    """`settle_window=0` 关掉稳定期，退回原始行为。"""
    import json
    from pathlib import Path

    p = FIXTURE_ROOT / 'video_records.json'
    assert p.is_file(), "独立合成稳定期夹具必须随源码提供"
    recs = json.loads(p.read_text(encoding="utf-8"))["frames"]
    a = br.records_to_learning_events(recs, settle_window=0)
    b = br.records_to_learning_events(recs, settle_window=3)
    assert a and b, "两条路都该产出事件"


def test_synthetic_flicker_count_drops_and_content_is_retained():
    """**合成序列回归**：指定闪烁字符数必须显著下降，且稳定内容保留。

    本断言不报告真实OCR幻觉率；原final54计数只保留为历史背景。
    """
    import json
    from pathlib import Path

    p = FIXTURE_ROOT / 'video_records.json'
    assert p.is_file(), "独立合成稳定期夹具必须随源码提供"
    recs = json.loads(p.read_text(encoding="utf-8"))["frames"]
    merged = br.merge_by_timestamp(recs)
    snaps = [{"ts": g["ts"], "text": g["text"], "type": g["type"]} for g in merged]

    HALL = set("⊙⊕Γ⊥√Σ±!！")

    def count(window):
        out = snaps if window == 0 else br.settle_snapshots(snaps, window=window)
        return sum(1 for s in out for c in s["text"] if c in HALL), out

    raw, _ = count(0)
    settled, out = count(3)
    assert settled < raw * 0.2, f"幻觉没降下来：{raw} → {settled}"
    # 真内容保留
    tail = "".join(s["text"] for s in out[-4:])
    assert "F_k" in tail or "F" in tail, f"末尾真内容丢了：{tail!r}"
