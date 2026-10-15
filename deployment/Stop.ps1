$ErrorActionPreference='Stop'
$deployRoot=Split-Path -Parent $PSScriptRoot
$deployPidFile=Join-Path $deployRoot 'data\logs\owned-processes.json'
if(!(Test-Path -LiteralPath $deployPidFile)){Write-Host 'No owned processes recorded.';exit 0}
$deployEntries=Get-Content -LiteralPath $deployPidFile -Raw|ConvertFrom-Json
foreach($deployEntry in $deployEntries){
    $deployProcess=Get-Process -Id $deployEntry.pid -ErrorAction SilentlyContinue
    if(!$deployProcess){continue}
    $deployCreated=$deployProcess.StartTime.ToUniversalTime().Ticks
    $deployExpected=([DateTime]$deployEntry.created).ToUniversalTime().Ticks
    if($deployCreated -ne $deployExpected){throw "PID reused: $($deployEntry.pid); refusing to stop it."}
    # Windows venv launches a child interpreter; stop only the verified owner's process tree.
    & taskkill /PID $deployEntry.pid /T /F | Out-Null
    if($LASTEXITCODE -ne 0){throw "Cannot stop owned process $($deployEntry.pid)"}
}
Write-Host 'Application stopped; database volumes preserved. Use docker compose stop for data services.'
