[CmdletBinding()]
param([int]$Port=8504,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$reviewRoot=Split-Path -Parent $PSScriptRoot
$reviewPy=Join-Path $reviewRoot '.venv\Scripts\python.exe'
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw 'Review port already in use'}
Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 3 | Out-Null
$reviewLogs=Join-Path $reviewRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $reviewLogs | Out-Null
$reviewProcess=Start-Process -FilePath $reviewPy -ArgumentList @('-m','streamlit','run','frontend/error_review_app.py','--server.address','127.0.0.1','--server.port',"$Port",'--server.headless','true') -WorkingDirectory $reviewRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $reviewLogs 'error-review.out.log') -RedirectStandardError (Join-Path $reviewLogs 'error-review.err.log') -PassThru
$reviewOwnedFile=Join-Path $reviewLogs 'owned-processes.json'
$reviewOwned=@()
if(Test-Path -LiteralPath $reviewOwnedFile){$reviewOwned=@(Get-Content -LiteralPath $reviewOwnedFile -Raw -Encoding UTF8 | ConvertFrom-Json)}
$reviewOwned+=@{pid=$reviewProcess.Id;created=$reviewProcess.StartTime.ToUniversalTime().ToString('o');role='streamlit-error-review'}
ConvertTo-Json -InputObject @($reviewOwned) -Depth 4 | Set-Content -LiteralPath $reviewOwnedFile -Encoding UTF8
if(!$NoBrowser){Start-Process "http://127.0.0.1:$Port"}
Write-Host "REVIEW_STARTED http://127.0.0.1:$Port"
