"""`scripts/safe_write.py` —— 生成代码时**先验语法再落盘**。

## 为什么要有这个文件
我今天在这个坑上栽了两次：
```
① 生成的 Python 脚本自己语法错（中文里嵌了直引号）
   ⇒ 脚本崩了，白丢一轮
② 更糟的一次：**写文件在 `ast.parse` 之前**
   ⇒ 坏文件落了盘，把 tests/ 里一个文件写坏，得 git checkout 恢复
```

**⇒ 根因不是"我不小心"，是"落盘和校验的顺序写反了"。**
**顺序错了，靠"下次注意"是修不好的** —— 所以把它变成一个函数。

## 用法
```python
from safe_write import write_python, write_text
write_python("backend/app/x.py", source)   # 语法错就抛，**不落盘**
write_text("docs/x.md", md)                # 非 Python，只做原子写
```
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

__all__ = ["SyntaxRejected", "write_python", "write_text"]


class SyntaxRejected(ValueError):
    """生成的代码语法不合法。**文件没有被写入。**"""


def write_python(path: str | os.PathLike, source: str) -> Path:
    """写 Python 源文件 —— **先 `ast.parse`，通过才落盘**。

    语法错时抛 `SyntaxRejected`，**并且保证目标文件一字未动**。
    错误信息里带**出错行**，方便直接定位（而不是回读整个文件）。

    ⚠️ **不要用 `path.write_text(...)` 直接写生成的代码** ——
    那正是我两次栽跟头的写法。
    """
    try:
        ast.parse(source)
    except SyntaxError as exc:
        line = (source.splitlines()[exc.lineno - 1].strip()
                if exc.lineno and exc.lineno <= len(source.splitlines()) else "")
        raise SyntaxRejected(
            f"生成的代码语法不合法，**未写入** {path}\n"
            f"  第 {exc.lineno} 行：{exc.msg}\n"
            f"  该行内容：{line[:100]}") from exc
    return write_text(path, source)


def write_text(path: str | os.PathLike, text: str) -> Path:
    """原子写文本（**UTF-8，无 BOM，LF**）。

    · **无 BOM**：PowerShell 的 `Set-Content -Encoding utf8` / `Out-File` 会加，
      而 BOM 会让 `ast.parse` 在别的文件上崩（今天犯过三次）
    · **LF**：与仓库既有文件一致
    · **原子**：先写临时文件再 `os.replace` —— 中途失败不会留半个文件
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


if __name__ == "__main__":  # 自检
    try:
        write_python(Path(tempfile.gettempdir()) / "_never.py", 'x = "中文"引号"错"')
    except SyntaxRejected as exc:
        print("  ✓ 语法错被挡住，未落盘")
        print("   ", str(exc).splitlines()[1].strip())
    else:
        print("  ✗ 没挡住 —— 这个工具不可信")
        sys.exit(1)
