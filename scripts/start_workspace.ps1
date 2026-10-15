[CmdletBinding()]
param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
$workspaceRoot=Split-Path -Parent $PSScriptRoot
$workspacePython=Join-Path $workspaceRoot '.venv\Scripts\python.exe'
if(!(Test-Path -LiteralPath $workspacePython)){throw 'Project Python environment not found'}
try { $health=Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 3 }
catch { throw 'Start the project backend first (startup student launcher also starts it).'}
if($health.app -ne 'shiwu-zhitu'){throw 'Port 8000 is not this project backend'}
$workspaceLogs=Join-Path $workspaceRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $workspaceLogs | Out-Null
$workspaceRegistry=Join-Path $workspaceLogs 'workspace-processes.json'
$workspaceProcesses=@()
if(Test-Path -LiteralPath $workspaceRegistry){$workspaceProcesses=Get-Content -LiteralPath $workspaceRegistry -Raw | ConvertFrom-Json; $workspaceProcesses=@($workspaceProcesses)}
foreach($proxyName in @('http_proxy','https_proxy','all_proxy','no_proxy')){
    $proxyValue=[Environment]::GetEnvironmentVariable($proxyName,'Process')
    [Environment]::SetEnvironmentVariable($proxyName,$null,'Process')
    [Environment]::SetEnvironmentVariable($proxyName.ToUpperInvariant(),$null,'Process')
    if($proxyValue){[Environment]::SetEnvironmentVariable($proxyName,$proxyValue,'Process')}
}
foreach($entry in @(@{file='portal_app.py';port=8500},@{file='streamlit_app.py';port=8501},@{file='teacher_app.py';port=8502},@{file='problem_import_app.py';port=8503},@{file='material_import_app.py';port=8504})){
    $listener=Get-NetTCPConnection -LocalPort $entry.port -State Listen -ErrorAction SilentlyContinue
    if($listener){
        $existing=Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
        if($existing.CommandLine -notmatch [regex]::Escape($entry.file)){throw "Port $($entry.port) belongs to another app"}
    }else{
        $process=Start-Process -FilePath $workspacePython -ArgumentList @('-m','streamlit','run',"frontend/$($entry.file)",'--server.address','127.0.0.1','--server.port',"$($entry.port)",'--server.headless','true','--browser.gatherUsageStats','false') -WorkingDirectory $workspaceRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $workspaceLogs "workspace-$($entry.port).out.log") -RedirectStandardError (Join-Path $workspaceLogs "workspace-$($entry.port).err.log") -PassThru
        $workspaceProcesses+=@{pid=$process.Id;created=$process.StartTime.ToUniversalTime().ToString('o');file=$entry.file}
        ConvertTo-Json -InputObject @($workspaceProcesses) -Depth 4 | Set-Content -LiteralPath $workspaceRegistry -Encoding UTF8
    }
    $ready=$false
    for($step=0;$step -lt 30;$step++){
        try { if((Invoke-WebRequest "http://127.0.0.1:$($entry.port)/_stcore/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200){$ready=$true;break} }catch{}
        Start-Sleep -Seconds 1
    }
    if(!$ready){throw "Workspace $($entry.port) did not become ready; see data/logs"}
    Write-Host "WORKSPACE_READY http://127.0.0.1:$($entry.port) $($entry.file)"
}
if(!$NoBrowser){Start-Process 'http://127.0.0.1:8500'}
