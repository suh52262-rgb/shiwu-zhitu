"""测量工具自身的护栏。

## 为什么测量工具也要测

这一轮反复出现的教训：**没有护栏的路径会静默失效，而且失效得看不出来**。

`check_isolation.py` 曾经把"公式隔离"与"图注去重"混算，
报出 87.8% 的零汉字占比，看上去像"判据换教材后失效了"——其实是脚本在说谎。
`build_retrieval_golden.py` 一旦被改成"从检索结果里采样"，
整个评测就变成循环论证，**而且数字会变得更好看**。

**测量工具的错误比被测系统的错误更危险**：后者会被发现，前者会被当成结论。
"""
import inspect
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _src(name: str) -> str:
    return (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- golden 构建

def test_golden_builder_is_non_circular():
    """**核心性质：gold 来自语料采样，不是来自检索结果。**

    如果反过来（先跑检索、再把返回的标成 gold），评测就永远 100%——
    而且**数字会更好看**，所以这种退化不会被自然发现。
    """
    src = _src("build_retrieval_golden")
    assert "block_store" in src, "没有从语料采样"
    assert "retrieve(" not in src, \
        "golden 构建脚本里出现了 retrieve——这会让 gold 来自检索结果（循环论证）"


def test_golden_builder_filters_junk_sources():
    """**从垃圾源段落出的题，评测出来的数字也没意义。**

    实测第一版采到了版权页与 OCR 乱码公式块。
    """
    src = _src("build_retrieval_golden")
    assert "_is_prose" in src, "没有散文筛选"
    for must in ("JUNK", "TRANSITION", "EXERCISE"):
        assert must in src, f"缺少 {must} 过滤（版权页/过渡句/习题片段）"


def test_transition_and_exercise_gold_are_rejected():
    """归因分析里记过的两类坏 gold，必须真的被拒。"""
    sys_path_added = str(ROOT / "scripts")
    import sys as _sys

    if sys_path_added not in _sys.path:
        _sys.path.insert(0, sys_path_added)
    import build_retrieval_golden as b

    # 过渡句：本身回答不了问题
    assert not b._is_prose("前面我们只对时空相对性作了定性的讨论，下面推导一些定量化的公式。看另外一个理想实验。假定")
    # 版权页
    assert not b._is_prose("版)为基础的教学改革项目《新概念力学》1977年获国家级教学成果奖一等奖，1998年获国家")
    # 正常散文应当通过
    assert b._is_prose(
        "此式表明，愈从系统提取能量，即系统的总能量E愈低，动能E愈大。"
        "在椭圆轨道的情形下，E不是常量，如果我们用它在一个周期里的平均值E来代替它，"
        "则可证明，上述结论对于椭圆轨道也成立。")


def test_golden_dataset_shape():
    """数据集结构：每条必须带 gold 的 (doc_id, page, block_id, text)。"""
    p = EVALUATION / "datasets/retrieval_golden_cases.json"
    cs = json.loads(p.read_text(encoding="utf-8"))["cases"]
    assert len(cs) >= 300, f"golden set 只有 {len(cs)} 条"
    for c in cs[:50]:
        assert c["query"].strip(), f"{c['id']} 问题为空"
        g = c["gold"]
        for k in ("doc_id", "page", "block_id", "text"):
            assert k in g, f"{c['id']} 的 gold 缺 {k}"
    ids = [c["id"] for c in cs]
    assert len(ids) == len(set(ids)), "id 有重复"


# ---------------------------------------------------------------- 隔离核对

def test_isolation_check_splits_reasons():
    """**必须按 note 拆分隔离原因。**

    `indexable=False` 有两种来源（公式过滤器 / 图注去重）。混算会把图注
    （`'图1-3温标有赖于测温物质'`）当成"误伤的公式碎片"，
    得出"判据失效"的错误结论——**实测踩过**。
    """
    src = _src("check_isolation")
    assert "坏公式" in src, "没有按 note 筛选公式隔离"
    assert "deduped" in src or "图注" in src, "没有把图注去重单独分出来"


# ---------------------------------------------------------------- 评测报告

def test_only_full_run_updates_full_baseline():
    """**单套件运行不得覆盖全量基线。**

    实测：跑一次 `--suite retrieval_golden` 之后，基线里只剩 3 个指标——
    看的人会以为诊断/检索/提示的指标不存在。
    """
    src = _src("run_eval")
    assert 'is_full = (args.suite == "all")' in src, "没有区分全量与单套件"
    assert 'latest-{mode}.json" if is_full' in src, "单套件仍会写 latest-full"


def test_report_declares_metrics_not_produced(monkeypatch, tmp_path):
    """**未产出的指标要显式列出**，不能静默缺席。

    看报告的人分不清"没跑"与"没实现"，那是最容易造成误解的地方。
    """
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("measurement_report_runner", ROOT / "scripts/run_eval.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "load_cases", lambda _: {"problem_id": "p", "cases": []})
    monkeypatch.setattr(runner, "load_problem_graph", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "graph", "--offline"])
    with pytest.raises(SystemExit) as result:
        runner.main()
    assert result.value.code == 0
    report = json.loads((tmp_path / "latest-suite-graph.json").read_text(encoding="utf-8"))
    for name in ("node_recall", "alternative_path_acceptance", "structure_valid_rate"):
        assert report["metrics"][name]["value"] is None
        assert any(name in entry for entry in report["not_produced"])


# ---------------------------------------------------------------- 配对比较

def test_paired_compare_forces_expansion_on():
    """配对脚本**必须强制打开扩展**。

    扩展默认关闭，而 `expand_query` 尊重那个开关——不强制打开的话，
    实验组跑的还是基线，会得到 b=c=0 的假结果。
    **实测因此差点得出"扩展完全无效"的错误结论。**
    """
    src = _src("paired_retrieval_compare")
    assert "qe.ENABLED = True" in src, "没有强制打开扩展"


def test_paired_compare_alarms_on_zero_flips():
    """**零翻转要报警，不能报"无显著差异"。**

    "没有效果"和"没有执行"看起来一模一样。一个合理的先验
    （"300 条不可能一条都不变"）是唯一的警报器。
    """
    src = _src("paired_retrieval_compare")
    assert "零翻转" in src, "没有零翻转报警"
    assert "不是**'扩展无效'的证据" in src or "不是" in src.split("零翻转")[1][:400]
