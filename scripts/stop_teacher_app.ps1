[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$teacherRoot=Split-Path -Parent $PSScriptRoot
$teacherOwned=Join-Path $teacherRoot 'data\logs\owned-processes.json'
if(!(Test-Path -LiteralPath $teacherOwned)){return}
$remaining=@()
$teacherRecords=Get-Content -LiteralPath $teacherOwned -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($record in $teacherRecords){
    if($record.role -ne 'streamlit-teacher'){$remaining+=@($record);continue}
    $process=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if(!$process){continue}
    $command=Get-CimInstance Win32_Process -Filter "ProcessId=$($record.pid)" -ErrorAction SilentlyContinue
    $expectedTime=[datetimeoffset]$record.created
    if($process.StartTime.ToUniversalTime() -eq $expectedTime.UtcDateTime -and $command.CommandLine -match 'frontend[/\\]teacher_app\.py'){
        # Windows venv Python launches a second Python process. Stop only its
        # identified teacher children, never another Streamlit or backend.
        $children=@(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($record.pid)" | Where-Object { $_.CommandLine -match '-m streamlit run frontend[/\\]teacher_app\.py' })
        foreach($child in $children){Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue}
        Stop-Process -Id $record.pid -ErrorAction SilentlyContinue
        Wait-Process -Id $record.pid -Timeout 5 -ErrorAction SilentlyContinue
    }else{$remaining+=@($record)}
}
ConvertTo-Json -InputObject @($remaining) -Depth 4 | Set-Content -LiteralPath $teacherOwned -Encoding UTF8
Write-Host 'TEACHER_STOPPED; shared backend remains available'
