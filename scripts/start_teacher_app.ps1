[CmdletBinding()]
param([int]$Port=8502,[int]$BackendPort=8000,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$teacherRoot=Split-Path -Parent $PSScriptRoot
$teacherPython=Join-Path $teacherRoot '.venv\Scripts\python.exe'
if(!(Test-Path -LiteralPath $teacherPython)){throw 'Project Python environment not found'}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw 'Teacher port is already in use'}
$teacherLogs=Join-Path $teacherRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $teacherLogs | Out-Null
$teacherBackend=if($env:TEACHER_BACKEND_URL){$env:TEACHER_BACKEND_URL.TrimEnd('/')}else{"http://127.0.0.1:$BackendPort"}
# Normalize process-local duplicate proxy keys before Windows Start-Process.
foreach($proxyName in @('http_proxy','https_proxy','all_proxy','no_proxy')){
    $proxyValue=[System.Environment]::GetEnvironmentVariable($proxyName,'Process')
    [System.Environment]::SetEnvironmentVariable($proxyName,$null,'Process')
    [System.Environment]::SetEnvironmentVariable($proxyName.ToUpperInvariant(),$null,'Process')
    if($proxyValue){[System.Environment]::SetEnvironmentVariable($proxyName,$proxyValue,'Process')}
}
function Save-TeacherProcess($process,$role){
    $file=Join-Path $teacherLogs 'owned-processes.json'
    $records=@()
    if(Test-Path -LiteralPath $file){$records=Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json; $records=@($records)}
    $records+=@{pid=$process.Id;created=$process.StartTime.ToUniversalTime().ToString('o');role=$role}
    ConvertTo-Json -InputObject @($records) -Depth 4 | Set-Content -LiteralPath $file -Encoding UTF8
}
function Teacher-BackendReady {
    try { $result=Invoke-RestMethod "$teacherBackend/health" -TimeoutSec 3; return $result.app -eq 'shiwu-zhitu' }
    catch { return $false }
}
if(!(Teacher-BackendReady)){
    if($env:TEACHER_BACKEND_URL){throw 'Configured backend is not ready; check its address'}
    if(Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue){throw 'Backend port belongs to an unavailable or different service'}
    $backend=Start-Process -FilePath $teacherPython -ArgumentList @('-m','uvicorn','app.main:app','--app-dir','backend','--host','127.0.0.1','--port',"$BackendPort") -WorkingDirectory $teacherRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $teacherLogs 'teacher-backend.out.log') -RedirectStandardError (Join-Path $teacherLogs 'teacher-backend.err.log') -PassThru
    Save-TeacherProcess $backend 'uvicorn-teacher-backend'
    $ready=$false
    for($index=0;$index -lt 30;$index++){
        if(Teacher-BackendReady){$ready=$true;break}
        Start-Sleep -Seconds 1
    }
    if(!$ready){throw 'Backend did not become ready; see data/logs/teacher-backend.err.log'}
}
$schema=Invoke-RestMethod "$teacherBackend/openapi.json" -TimeoutSec 5
if(!$schema.paths.PSObject.Properties['/teacher/error-evidence']){throw 'Backend is an older version; restart it with current project code'}
$previousTeacherBackend=$env:TEACHER_BACKEND_URL
try {
    $env:TEACHER_BACKEND_URL=$teacherBackend
    $process=Start-Process -FilePath $teacherPython -ArgumentList @('-m','streamlit','run','frontend/teacher_app.py','--server.address','127.0.0.1','--server.port',"$Port",'--server.headless','true','--browser.gatherUsageStats','false') -WorkingDirectory $teacherRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $teacherLogs "teacher-$Port.out.log") -RedirectStandardError (Join-Path $teacherLogs "teacher-$Port.err.log") -PassThru
} finally { $env:TEACHER_BACKEND_URL=$previousTeacherBackend }
Save-TeacherProcess $process 'streamlit-teacher'
$teacherReady=$false
for($index=0;$index -lt 20;$index++){
    try { $response=Invoke-WebRequest "http://127.0.0.1:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 2; if($response.StatusCode -eq 200){$teacherReady=$true;break} }catch{}
    Start-Sleep -Seconds 1
}
if(!$teacherReady){throw 'Teacher UI did not become ready; see data/logs'}
if(!$NoBrowser){Start-Process "http://127.0.0.1:$Port"}
Write-Host "TEACHER_STARTED http://127.0.0.1:$Port"
