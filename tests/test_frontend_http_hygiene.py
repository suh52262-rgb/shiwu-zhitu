"""前端 HTTP 层的一条卫生规则：**GET 必须走 `_get`**。

## 为什么值得单独钉一条
`httpx.get()` 每次调用都会新建 Client，而新建 Client 会调
`httpx.create_ssl_context()` —— 本机实测 **305ms**（标准库同功能只要 15ms）。
一次脚本重跑要打 3 个端点（题库 / 健康 / 画像），
而 Streamlit **每次交互都重跑整个脚本** ⇒ 每次按键白付近一秒。

实测（AppTest 驱动真前端脚本、打真后端）：
```
修前  一次重跑 1117 ms
修后  一次重跑  110–140 ms
副作用：全量离线回归 137 s → 72 s（AppTest 用例不再每次都付这 300ms）
```

⚠️ **不用计时断言**：那种测试在负载变化时会假红。
这里只钉**结构**——"GET 有没有走那道包装"是可判定的，
"这次跑了几毫秒"不是。
"""
import ast
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "streamlit_app.py"


def _httpx_get_lines(tree: ast.AST) -> list[int]:
    """所有 `httpx.get(...)` 调用的行号。"""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "httpx"):
            out.append(node.lineno)
    return out


def _wrapper_span(tree: ast.AST) -> tuple[int, int]:
    """`_get` 这个函数的行号范围。"""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_get":
            return node.lineno, max(getattr(n, "lineno", node.lineno)
                                    for n in ast.walk(node))
    raise AssertionError("找不到 `_get` —— 它被改名或删了，这条测试要跟着改")


def test_every_httpx_get_call_sits_inside_the_wrapper():
    """除 `_get` 自己那一次转发，**不许有第二处 `httpx.get`**。

    新加一个端点时顺手写 `httpx.get(...)`，就会**静默**把那 300ms 加回来 ——
    页面只是"感觉有点卡"，没有任何报错、没有任何测试会红。
    """
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    start, end = _wrapper_span(tree)
    outside = [n for n in _httpx_get_lines(tree) if not (start <= n <= end)]
    assert not outside, (
        f"这些行绕过了 `_get`，会把每次交互的 300ms 又加回来：{outside}")


def test_the_wrapper_actually_injects_a_prebuilt_ssl_context():
    """包装里必须真的塞了 `verify=` —— 否则它只是个改名，一点用没有。"""
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    start, end = _wrapper_span(tree)
    body = "\n".join(src.splitlines()[start - 1:end])
    assert "verify" in body and "_VERIFY" in body, f"包装里没注入 verify：\n{body}"
    assert "ssl.create_default_context()" in src, "没有预先建好的 SSL 上下文"


def test_the_context_is_built_once_per_run_not_per_request():
    """上下文要在**模块级建一次**，不能写在函数里 —— 写在里面就等于每次请求重付 15ms。"""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_VERIFY" for t in node.targets):
            return
    raise AssertionError("`_VERIFY` 不是模块级赋值（是不是挪进函数里了？）")
