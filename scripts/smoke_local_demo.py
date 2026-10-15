"""联调前自检：对着**正在运行**的本地服务跑一遍只读/零成本的检查。

用法（先把服务起起来：`启动学生端.cmd`）：
    .venv/Scripts/python.exe scripts/smoke_local_demo.py

## 它验什么

```
A 后端冒烟      健康 / 题库 8 道 / 单题 / 画像 / 教师端 / 演示回放
B 任务接口      **零模型成本**：故意传"不是视频"的文件 ⇒ 解码就失败，不调模型
                幂等回放 / 内容或参数不同 409 / 非法编号 422 / 按编号找回 404 不泄露存在性
C 清理与只读    两层上传清理是否如实、连续 GET 是否**逐字节只读**
                （自动查询每 3 秒打一次，查询若带副作用就糟了）
D 前端          AppTest 驱动**真实** streamlit_app.py，打**真实后端** ——
                回答"前端对着真后端能不能渲染"，而首页 HTTP 200 只证明进程活着
```

## 它**不**验什么（必须真浏览器，见 `docs/联调清单-真实浏览器-2026-10-09.md`）
```
· 自动查询的**节拍**（每 3 秒）与**终态停止** —— AppTest 不推进时间
· 布局外观、需要点鼠标的交互
· 真实模型质量（本脚本刻意不触发任何模型调用）
```

**会留下痕迹**：B 段建一个任务（会落一份任务 JSON）。这是刻意的 ——
"清理是否如实"必须有一个真任务才验得了。任务很快进终态，不需要人工收尾。
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
FRONTEND = "http://127.0.0.1:8501"
STUDENT, PROBLEM = "stu_demo", "phy_001"

_results: list[tuple[str, bool]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok))
    print(("  [OK ] " if ok else "  [FAIL] ") + name + ("  " + detail if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- A 后端冒烟

def backend_smoke() -> str | None:
    section("A 后端冒烟")
    try:
        health = httpx.get(f"{BASE}/health", timeout=10)
        record("GET /health 200", health.status_code == 200, health.text[:80])
    except Exception as exc:  # noqa: BLE001
        record("GET /health 可达", False, f"{type(exc).__name__}: {exc}")
        return None                      # 后端不通，后面都别跑了

    problems = httpx.get(f"{BASE}/problems", timeout=20).json().get("items") or []
    record("题库 8 道", len(problems) == 8, f"实际 {len(problems)}")
    if not problems:
        return None
    pid = problems[0]["problem_id"]
    record("单题详情 200", httpx.get(f"{BASE}/problems/{pid}", timeout=20).status_code == 200)
    record("学生画像 200",
           httpx.get(f"{BASE}/students/{STUDENT}/profile", timeout=20).status_code == 200)
    record("教师端统计 200",
           httpx.get(f"{BASE}/teacher/knowledge-points/stats", timeout=20).status_code == 200)
    record("演示回放列表 200",
           httpx.get(f"{BASE}/demo/recordings", timeout=20).status_code == 200)
    return pid


# ---------------------------------------------------------------- B 任务接口

def post_job(rid: str, content: bytes, **over):
    form = {"problem_id": PROBLEM, "student_id": STUDENT, "every_sec": "20",
            "client_request_id": rid, **over}
    return httpx.post(f"{BASE}/video/jobs", data=form,
                      files={"file": ("probe.mp4", content, "video/mp4")}, timeout=60)


def job_api() -> str | None:
    section("B 任务接口（**零模型成本**：无效文件在解码处就失败）")
    rid = str(uuid.uuid4())
    bogus = b"this is definitely not a video"
    r = post_job(rid, bogus)
    if r.status_code != 200:
        record("建任务 200", False, f"{r.status_code} {r.text[:200]}")
        return None
    job = r.json()
    job_id = job["job_id"]
    record("建任务 200 且 queued", job["status"] in {"queued", "running"}, f"status={job['status']}")
    record("响应带回上传编号", job.get("client_request_id") == rid)

    again = post_job(rid, bogus)
    record("同编号 + 同内容 ⇒ 幂等回放原任务",
           again.status_code == 200 and again.json()["job_id"] == job_id,
           f"status={again.status_code}")
    record("同编号 + 不同内容 ⇒ 409", post_job(rid, b"different").status_code == 409)
    record("同编号 + 不同参数 ⇒ 409",
           post_job(rid, bogus, problem_id="demo_01").status_code == 409)
    record("非法 UUID ⇒ 422", post_job("not-a-uuid", b"x").status_code == 422)

    q = {"student_id": STUDENT, "problem_id": PROBLEM}
    found = httpx.get(f"{BASE}/video/jobs/by-request/{rid}", params=q, timeout=20)
    record("本人按编号查得到", found.status_code == 200 and found.json()["job_id"] == job_id,
           f"status={found.status_code}")
    record("归属不符 ⇒ 404（不是 403，不泄露存在性）",
           httpx.get(f"{BASE}/video/jobs/by-request/{rid}",
                     params={"student_id": "someone_else", "problem_id": PROBLEM},
                     timeout=20).status_code == 404)
    record("不存在的编号 ⇒ 404",
           httpx.get(f"{BASE}/video/jobs/by-request/{uuid.uuid4()}", params=q,
                     timeout=20).status_code == 404)

    for _ in range(60):
        time.sleep(1)
        got = httpx.get(f"{BASE}/video/jobs/{job_id}", params=q, timeout=20).json()
        if got["status"] in {"succeeded", "failed", "interrupted"}:
            break
    err = (got.get("error") or {}).get("message") or ""
    record("无效视频 ⇒ failed（不是 succeeded）", got["status"] == "failed", f"error={err[:90]}")
    record("失败原因不含模型字样（说明没走到模型）",
           "识别" not in err and "model" not in err.lower())
    return job_id


# ---------------------------------------------------------------- C 清理与只读

def cleanup_and_readonly(job_id: str) -> None:
    section("C 上传清理与「查询只读」")
    q = {"student_id": STUDENT, "problem_id": PROBLEM}
    job = httpx.get(f"{BASE}/video/jobs/{job_id}", params=q, timeout=20).json()
    record("如实报告清理状态（四个字段都在）",
           all(k in job for k in ("upload_removed", "spool_removed",
                                  "analysis_upload_removed", "analysis_upload_cleanup_verified")),
           f"upload_removed={job.get('upload_removed')}")
    spool = ROOT / "data" / "video_jobs" / job_id / "source.video"
    record("队列暂存 source.video 已删", not spool.exists())

    path = ROOT / "data" / "video_jobs" / f"{job_id}.json"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    for _ in range(3):
        httpx.get(f"{BASE}/video/jobs/{job_id}", params=q, timeout=20)
        time.sleep(1.1)
    after = (path.read_bytes(), path.stat().st_mtime_ns)
    record("连续 3 次查询后任务文件逐字节未变", before == after)


# ---------------------------------------------------------------- D 前端

def frontend_live() -> None:
    section("D 前端（AppTest 驱动真前端脚本，打**真后端**）")
    try:
        alive = httpx.get(FRONTEND, timeout=10)
        record("Streamlit 首页可达（**只证明进程活着**）", alive.status_code == 200)
    except Exception as exc:  # noqa: BLE001
        record("Streamlit 首页可达", False, f"{type(exc).__name__}: {exc}")
        return

    from streamlit.testing.v1 import AppTest
    # ⚠️ 默认超时 3 秒 —— 平时打桩够用，**真网络不够**。
    at = AppTest.from_file(str(ROOT / "frontend" / "streamlit_app.py")).run(timeout=60)
    record("脚本对真后端渲染无异常", not at.exception, str(at.exception or ""))

    # 健康状态在**侧边栏**，而且不一定是 caption ——
    # 只查主区、或只查一两种元素，都会漏掉它（我这两版各漏过一次）。
    side = "\n".join(str(getattr(i, "value", ""))
                     for kind in ("markdown", "caption", "info", "warning",
                                  "error", "success", "text")
                     for i in (getattr(at.sidebar, kind, []) or []))
    record("侧边栏说后端已连接", "后端已连接" in side, "" if "后端已连接" in side else side[:100])

    options = []
    for box in at.selectbox:
        options = list(getattr(box, "options", []) or [])
    record("题目下拉 8 个选项", len(options) == 8, f"实际 {len(options)}")

    main = "\n".join([str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption])
    record("没有任务时不出现自动查询/找回的痕迹",
           "自动查询" not in main
           and not any(b.key in {"recover_video_job", "refresh_video_job"} for b in at.button))


def main() -> int:
    print("联调前自检（只读 / 零模型成本）")
    print(f"后端 {BASE}    前端 {FRONTEND}")
    if not backend_smoke():
        print("\n后端不可达 —— 先起服务（双击 启动学生端.cmd），别急着往下看。")
        return 2
    job_id = job_api()
    if job_id:
        cleanup_and_readonly(job_id)
    frontend_live()

    bad = [n for n, ok in _results if not ok]
    print(f"\n===== {len(_results) - len(bad)}/{len(_results)} 通过 =====")
    if bad:
        print("未通过：\n  · " + "\n  · ".join(bad))
    print("\n⚠️ 本脚本**不验**：自动查询节拍、终态停止、布局外观、真实模型质量。"
          "\n   那些要真浏览器 —— 见 docs/联调清单-真实浏览器-2026-10-09.md")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
