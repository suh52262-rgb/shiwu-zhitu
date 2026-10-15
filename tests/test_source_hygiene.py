"""源码卫生：把"落盘与校验的顺序"变成**硬约束**。

## 这个文件为什么只剩一条
我最初想在这里加两条检查，**第一条被我自己跑出来的数据否掉了**：

```
想加的 ①：中文文本里不许用直引号当引号（检测 CJK + " + CJK）
实测结果：**全仓 902 处** —— backend/app/graph.py、config.py … 到处都是
          # …只降级为"本轮无教材依据"
          # …等于埋一个"到时候才发现"的故障
⇒ 而这些**完全无害**（注释里的直引号不参与语法）
⇒ 规则若上，就是在要求重写 902 行 —— **那不是约束，是噪音。**
```

**更根本的是**：这件事**自带即时反馈**，不需要检查：
```
注释里  # 把"这句话"当引号        → 无害
代码里  print("把"这句话"当引号")  → **语法错**，Python 当场拦 ⇒ 进不了仓库
⇒ 危险只发生在**生成代码的那一刻**，而那一刻本来就炸给我看。
```

## 所以真正值得约束的是另一件：**顺序**
```
我今天有一次写反了 —— `write_text` 在前、`ast.parse` 在后
⇒ **坏文件落了盘**，把 tests/ 里一个文件写坏，得 git checkout 恢复
⇒ 根因不是"不小心"，是**顺序写反了。靠"下次注意"修不好，得变成函数。**
```
⇒ 见 `scripts/safe_write.py`，以及下面这几条测试。
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_safe_write_refuses_broken_python(tmp_path):
    """**落盘和校验的顺序必须是「先验后写」。**

    `write_text` 在前、`ast.parse` 在后 ⇒ **坏文件落盘**（今天真发生过）。
    """
    from safe_write import SyntaxRejected, write_python

    target = tmp_path / "broken.py"
    with pytest.raises(SyntaxRejected) as ei:
        write_python(target, 'x = "中文"引号"错"')
    assert not target.exists(), "**语法错时不许落盘**"
    assert "未写入" in str(ei.value)
    assert "第 1 行" in str(ei.value), "错误信息要带行号，免得回读整个文件"
    assert "该行内容" in str(ei.value)


def test_safe_write_does_not_clobber_an_existing_file(tmp_path):
    """**失败时连旧文件都不许动** —— 否则"没写成"变成"写坏了"。"""
    from safe_write import SyntaxRejected, write_python

    target = tmp_path / "keep.py"
    target.write_text("OK = 1\n", encoding="utf-8")
    with pytest.raises(SyntaxRejected):
        write_python(target, "def broken(:\n")
    assert target.read_text(encoding="utf-8") == "OK = 1\n", "旧内容被破坏了"


def test_safe_write_writes_good_python_without_bom(tmp_path):
    """合法代码正常写；**且不带 BOM、用 LF**。

    BOM 会让 `ast.parse` 在别的文件上崩（`U+FEFF`）——
    PowerShell 的 `Set-Content -Encoding utf8` 会加，今天犯过三次。
    """
    from safe_write import write_python

    target = tmp_path / "ok.py"
    write_python(target, 'x = "中文「引号」"\n')
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "不许带 BOM"
    assert b"\r\n" not in raw, "用 LF"
    assert ast.parse(raw.decode("utf-8"))


def test_the_helper_parses_before_it_writes():
    """**静态钉住顺序** —— 光靠上面那条测不出"为什么对"。

    将来有人为了"性能"把 parse 挪到 write 后面，这条会拦住。
    """
    src = (ROOT / "scripts" / "safe_write.py").read_text(encoding="utf-8")
    ast.parse(src)
    i_parse = src.index("ast.parse(source)")
    i_write = src.index("return write_text(path, source)")
    assert i_parse < i_write, "**校验必须在落盘之前**"
