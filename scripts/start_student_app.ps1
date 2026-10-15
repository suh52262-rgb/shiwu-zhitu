# 视物知途 · 学生端一键启动
#
# 双击 `启动学生端.cmd`（它调本脚本）即可：
#   1. 起后端（uvicorn，8000）
#   2. 等 /health 就绪
#   3. 起学生端（streamlit，8501）—— **不加 --server.headless**，本机要它自动开浏览器
#   4. 打开浏览器
#
# ## 为什么用 PowerShell 而不是纯 cmd
# 第一版是纯 `.cmd`，实测有两个坑：
#   · `set /a _n` 在 `if (...)` 块里**不会实时生效**（`%_n%` 在解析时就展开了），
#     于是 `if %_n% lss 40` 展开成 `if lss 40` —— `40` 被当成命令
#   · `timeout` 在输入被重定向时报 `Input redirection is not supported`
# 这两个都是 cmd 的老问题。PowerShell 里循环、HTTP 检查、字符串都正常。

[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8501,
    [switch]$NoBrowser          # 开了它就不自动开浏览器（服务器/调试用）
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'

function Write-Step($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host $msg -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host $msg -ForegroundColor Red }

Write-Host ("=" * 60)
Write-Host "  视物知途 · 学生端" -ForegroundColor White
Write-Host ("=" * 60)

if (-not (Test-Path $py)) {
    Write-Err "[X] 找不到虚拟环境：$py"
    Read-Host "按回车退出"
    exit 1
}

function Test-Port([int]$p) {
    [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}

function Test-Healthy([int]$p) {
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$p/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -ne 200) { return $false }
        $ready = Invoke-RestMethod "http://127.0.0.1:$p/runtime/readiness" -TimeoutSec 2
        return $ready.ready -eq $true
    } catch { return $false }
}

# ---------------- 后端 ----------------
if (Test-Port $BackendPort) {
    Write-Warn2 "[=] 后端已在 $BackendPort 运行，跳过启动"
} else {
    Write-Step "[1/3] 启动后端  http://127.0.0.1:$BackendPort"
    # 用 Start-Process 开独立窗口，这样关窗口 = 停服务，符合直觉
    $previousWarmup=[Environment]::GetEnvironmentVariable('EMBEDDING_WARMUP_ENABLED','Process')
    if(!$env:EMBEDDING_WARMUP_ENABLED){$env:EMBEDDING_WARMUP_ENABLED='true'}
    Start-Process -FilePath $py `
        -ArgumentList '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
                      '--port', "$BackendPort" `
        -WorkingDirectory (Join-Path $root 'backend') `
        -WindowStyle Hidden
    [Environment]::SetEnvironmentVariable('EMBEDDING_WARMUP_ENABLED',$previousWarmup,'Process')
}

Write-Host "      等待后端就绪" -NoNewline
$ok = $false
foreach ($i in 1..120) {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline
    if (Test-Healthy $BackendPort) { $ok = $true; break }
}
Write-Host ""
if ($ok) {
    Write-Ok "      [OK] 后端已就绪"
} else {
    # **两种情况必须分开说**：
    #  · 端口没开   → 多半是启动失败（依赖、配置、端口被占）
    #  · 端口开着但不健康 → **进程活着但挂起了**
    #    （项目里有这类先例：P0-1 就是 DB 不可达时无限阻塞）
    # 原来只有一句"没起来"，对第二种情况是错的 —— 它确实起来了。
    if (Test-Port $BackendPort) {
        Write-Err "      [X] 后端**在监听但没响应** —— 进程可能已挂起"
        Write-Err "          这类挂起项目里有先例（P0-1：DB 不可达时无限阻塞）"
        Write-Err "          处理：先关掉那个后端窗口，再重跑本脚本"
    } else {
        Write-Err "      [X] 后端 120 秒内未就绪（包含本地模型预热） —— 看那个最小化的窗口里的报错"
    }
    Read-Host "按回车退出"
    exit 1
}

# ---------------- 学生端 ----------------
if (Test-Port $FrontendPort) {
    Write-Warn2 "[=] 学生端已在 $FrontendPort 运行，直接开浏览器"
} else {
    Write-Step "[2/3] 启动学生端  http://127.0.0.1:$FrontendPort"

    # ⚠️⚠️ **必须先把 Streamlit 的邮箱提问堵掉，否则前端永远起不来。**
    #
    # Streamlit 首次运行会**等于终端输入**：
    #     "Welcome to Streamlit! ... please enter your email address below."
    #     "Email: "
    # 而 `Start-Process` 的 stdin 不是终端 → **它就一直等下去**。
    # 实测就是这样挂住的，日志里除了那行 `Email:` 什么都没有。
    #
    # `--server.headless true` 也能跳过提问，**但它同时会阻止自动打开浏览器**
    # —— 两个效果是绑在一起的，本机给人用不能加。
    # `--browser.gatherUsageStats false` **实测跳不过提问**。
    #
    # **只有建一个空邮箱的 `credentials.toml` 是可靠的。**
    $credDir = Join-Path $env:USERPROFILE '.streamlit'
    $credFile = Join-Path $credDir 'credentials.toml'
    if (-not (Test-Path $credFile)) {
        New-Item -ItemType Directory -Force -Path $credDir | Out-Null
        [System.IO.File]::WriteAllText($credFile, "[general]`nemail = `"`"`n",
            (New-Object System.Text.UTF8Encoding $false))
        Write-Host "      已建 $credFile（跳过 Streamlit 的邮箱提问）" -ForegroundColor DarkGray
    }

    #  **不加 --server.headless** —— 加了它 Streamlit 就不会自动开浏览器，
    #  人在这台机器上等半天什么都不会弹（实测踩过）。
    #
    # ⚠️ **必须把输出留下来。**
    # 上一版用 `-WindowStyle Minimized` 静默启动，前端起不来时**什么都看不到** ——
    # 只能看到"40 秒内没起来"，原因一个字都没有。
    # **启动一个后台进程却不留输出，等于把自己弄瞎。**
    $logDir = Join-Path $root 'data\logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $feOut = Join-Path $logDir 'streamlit.out.log'
    $feErr = Join-Path $logDir 'streamlit.err.log'
    Start-Process -FilePath $py `
        -ArgumentList '-m', 'streamlit', 'run', 'frontend\streamlit_app.py',
                      '--server.port', "$FrontendPort",
                      '--server.address', '127.0.0.1',
                      # ⚠️ **必须加这个，否则前端永远起不来。**
                      # Streamlit 首次运行会**等于邮箱输入**：
                      #   "Welcome to Streamlit! ... Email:"
                      # 而 Start-Process 的 stdin 不是终端 → **永远等下去**。
                      # 实测就是这么挂住的，日志里只有那行 Email: 提示。
                      #
                      # 注意 `--server.headless true` 也能跳过提问，**但它同时
                      # 会阻止自动打开浏览器** —— 两个效果是绑在一起的，
                      # 本机给人用不能加。`--browser.gatherUsageStats false`
                      # 只关掉提问与统计上报，不影响开浏览器。
                      '--browser.gatherUsageStats', 'false' `
        -WorkingDirectory $root `
        -RedirectStandardOutput $feOut `
        -RedirectStandardError $feErr `
        -WindowStyle Hidden
    # ⚠️ **必须等前端也真的起来。**
    # 第一版只检查了后端健康就去开浏览器，实测出现过
    # "脚本报已启动、8501 却没在监听" —— 浏览器打开的是个死页面。
    # **报成功却指向死页面，比直接报错更糟。**
    Write-Host "      等待学生端就绪" -NoNewline
    $feOk = $false
    foreach ($i in 1..120) {
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline
        if (Test-Port $FrontendPort) { $feOk = $true; break }
    }
    Write-Host ""
    if ($feOk) {
        Write-Ok "      [OK] 学生端已就绪"
        Start-Sleep -Seconds 2      # 端口开了再缓一下，避免首屏白页
    } else {
        Write-Err "      [X] 学生端 40 秒内没起来 —— 看那个最小化的窗口里的报错"
        Write-Err "          后端是好的（http://127.0.0.1:$BackendPort），只是前端没起"
        Read-Host "按回车退出"
        exit 1
    }
}

# ---------------- 打开 ----------------
if ($NoBrowser) {
    Write-Warn2 "[3/3] 按 -NoBrowser 要求，不自动打开浏览器"
} else {
    Write-Step "[3/3] 打开浏览器"
    Start-Process "http://127.0.0.1:$FrontendPort"
}

Write-Host ""
Write-Host ("=" * 60)
Write-Ok   "  已启动"
Write-Host "    后端   http://127.0.0.1:$BackendPort"
Write-Host "    学生端 http://127.0.0.1:$FrontendPort"
Write-Host ""
Write-Host "  停止：关掉那两个最小化的窗口（或在此按 Ctrl+C 后手动关）" -ForegroundColor DarkGray
Write-Host ("=" * 60)
Start-Sleep -Seconds 3
