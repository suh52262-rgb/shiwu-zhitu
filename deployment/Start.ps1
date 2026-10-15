[CmdletBinding()]
param([string]$PythonPath='', [switch]$NoBrowser, [ValidateSet('custom','demo')][string]$Profile='demo', [int]$BackendPort=8000, [int]$FrontendPort=8501, [switch]$StudentOnly)
$ErrorActionPreference='Stop'
$deployRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $deployRoot
$deployPy=Join-Path $deployRoot '.venv\Scripts\python.exe'
if($PythonPath){$deployPy=$PythonPath}
if($BackendPort -eq $FrontendPort -or $BackendPort -lt 1024 -or $BackendPort -gt 65535 -or $FrontendPort -lt 1024 -or $FrontendPort -gt 65535){throw 'Use distinct application ports between 1024 and 65535.'}
$releasePages=@(@{file='streamlit_app.py';port=$FrontendPort;key='STUDENT'})
if(!$StudentOnly){$releasePages=@(@{file='portal_app.py';port=($FrontendPort-1);key='OVERVIEW'},@{file='streamlit_app.py';port=$FrontendPort;key='STUDENT'},@{file='teacher_app.py';port=($FrontendPort+1);key='TEACHER'},@{file='problem_import_app.py';port=($FrontendPort+2);key='PROBLEMS'},@{file='material_import_app.py';port=($FrontendPort+3);key='MATERIALS'})}
$releasePorts=@($BackendPort)+@($releasePages|ForEach-Object {$_.port})
if(@($releasePorts|Select-Object -Unique).Count -ne $releasePorts.Count -or @($releasePorts|Where-Object {$_ -lt 1024 -or $_ -gt 65535}).Count){throw 'Backend and workspace ports must be distinct and in range.'}
foreach($deployPort in $releasePorts){
    if(Get-NetTCPConnection -LocalPort $deployPort -State Listen -ErrorAction SilentlyContinue){throw "Port $deployPort is already in use. Stop your own old instance first."}
}
$deployLogs=Join-Path $deployRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $deployLogs | Out-Null
$deployEnvironment=@{}
$deployEnvironmentNames=@('BACKEND_URL','TEACHER_BACKEND_URL','STREAMLIT_SERVER_HEADLESS','SHIWU_RUNTIME_PROFILE','LLM_PROVIDER','LLM_MODEL','PROMPT_DIAGNOSIS_VERSION','PROMPT_TUTOR_VERSION','PROMPT_REVIEW_VERSION','EMBEDDING_WARMUP_ENABLED')
foreach($demoRole in @('STEP_MATCHER','ANALYZE_STEP','DIAGNOSIS','TUTOR','REVIEW','GRAPH_GENERATOR','GRAPH_VALIDATOR')){$deployEnvironmentNames+=($demoRole+'_MODEL')}
foreach($releasePage in $releasePages){$deployEnvironmentNames+=('WORKSPACE_'+$releasePage.key+'_PORT')}
foreach($deployName in $deployEnvironmentNames){$deployEnvironment[$deployName]=[Environment]::GetEnvironmentVariable($deployName,'Process')}
try {
$env:STREAMLIT_SERVER_HEADLESS='true'
if(!$env:EMBEDDING_WARMUP_ENABLED){$env:EMBEDDING_WARMUP_ENABLED='true'}
foreach($releasePage in $releasePages){[Environment]::SetEnvironmentVariable(('WORKSPACE_'+$releasePage.key+'_PORT'),[string]$releasePage.port,'Process')}
if($Profile -eq 'demo'){
    $env:SHIWU_RUNTIME_PROFILE='demo'
    $env:LLM_PROVIDER='deepseek'
    $env:LLM_MODEL='deepseek-flash'
    foreach($demoRole in @('STEP_MATCHER','ANALYZE_STEP','DIAGNOSIS','TUTOR','REVIEW','GRAPH_GENERATOR','GRAPH_VALIDATOR')){
        [Environment]::SetEnvironmentVariable(($demoRole+'_MODEL'),'deepseek-flash','Process')
    }
    $env:PROMPT_DIAGNOSIS_VERSION='v5'
    $env:PROMPT_TUTOR_VERSION='v5'
    $env:PROMPT_REVIEW_VERSION='v4'
    Write-Host 'Demo profile: deepseek-flash / diagnosis v5 / tutor v5 / review v4. Historical v22 gate covers its frozen graphless text snapshot with an empty corpus; not this release.'
}
$env:BACKEND_URL="http://127.0.0.1:$BackendPort"
$env:TEACHER_BACKEND_URL=$env:BACKEND_URL
$deployArgs=@('-m','uvicorn','app.main:app','--app-dir','backend','--host','127.0.0.1','--port',"$BackendPort")
$deployBackend=Start-Process -FilePath $deployPy -ArgumentList $deployArgs -WorkingDirectory $deployRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $deployLogs 'backend.out.log') -RedirectStandardError (Join-Path $deployLogs 'backend.err.log') -PassThru
$deployOwned=@(@{pid=$deployBackend.Id;created=$deployBackend.StartTime.ToUniversalTime().ToString('o');role='uvicorn'})
ConvertTo-Json -InputObject @($deployOwned) -Depth 4|Set-Content -LiteralPath (Join-Path $deployLogs 'owned-processes.json') -Encoding utf8
$deployReady=$false
foreach($i in 1..120){
    try{ $r=Invoke-RestMethod "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2; if($r.status -eq 'ok'){$deployReady=$true;break} }catch{}
    Start-Sleep -Seconds 1
}
if(!$deployReady){throw 'Backend failed; inspect data/logs/backend.err.log'}
$warmup=Invoke-RestMethod "http://127.0.0.1:$BackendPort/runtime/readiness" -TimeoutSec 3
if(!$warmup.ready){throw 'Local embedding warmup failed; inspect backend logs before analysis.'}
foreach($releasePage in $releasePages){
$deployFrontend=Start-Process -FilePath $deployPy -ArgumentList @('-m','streamlit','run',"frontend/$($releasePage.file)",'--server.address','127.0.0.1','--server.port',"$($releasePage.port)",'--server.headless','true','--browser.gatherUsageStats','false') -WorkingDirectory $deployRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $deployLogs "frontend-$($releasePage.port).out.log") -RedirectStandardError (Join-Path $deployLogs "frontend-$($releasePage.port).err.log") -PassThru
$deployOwned+=@{pid=$deployFrontend.Id;created=$deployFrontend.StartTime.ToUniversalTime().ToString('o');role='streamlit'}
ConvertTo-Json -InputObject @($deployOwned) -Depth 4|Set-Content -LiteralPath (Join-Path $deployLogs 'owned-processes.json') -Encoding utf8
$deployReady=$false
foreach($i in 1..120){
    try{Invoke-RestMethod "http://127.0.0.1:$($releasePage.port)/_stcore/health" -TimeoutSec 2|Out-Null;$deployReady=$true;break}catch{}
    Start-Sleep -Seconds 1
}
if(!$deployReady){throw 'Frontend failed; inspect data/logs/frontend.err.log'}
}
$releaseEntryPort=$releasePages[0].port
if(!$NoBrowser){Start-Process "http://127.0.0.1:$releaseEntryPort"}
Write-Host "START_OK http://127.0.0.1:$releaseEntryPort"
} finally {
    foreach($deployName in $deployEnvironmentNames){[Environment]::SetEnvironmentVariable($deployName,$deployEnvironment[$deployName],'Process')}
}
