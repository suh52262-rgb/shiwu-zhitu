[CmdletBinding()]
param([int]$Port = 8503)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw '请先安装项目依赖。' }
try { Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 3 | Out-Null }
catch { throw '请先启动后端（启动学生端.cmd 也会启动后端）。' }
Push-Location $projectRoot
try { & $pythonPath -m streamlit run (Join-Path $projectRoot 'frontend\problem_import_app.py') --server.address 127.0.0.1 --server.port $Port }
finally { Pop-Location }
