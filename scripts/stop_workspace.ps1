[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$workspaceRoot=Split-Path -Parent $PSScriptRoot
$registry=Join-Path $workspaceRoot 'data\logs\workspace-processes.json'
if(!(Test-Path -LiteralPath $registry)){return}
$records=Get-Content -LiteralPath $registry -Raw | ConvertFrom-Json
$remaining=@()
foreach($record in $records){
    $process=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if(!$process){continue}
    $info=Get-CimInstance Win32_Process -Filter "ProcessId=$($record.pid)"
    if($process.StartTime.ToUniversalTime() -eq ([datetimeoffset]$record.created).UtcDateTime -and $info.CommandLine -match [regex]::Escape("frontend/$($record.file)")){
        $children=Get-CimInstance Win32_Process -Filter "ParentProcessId=$($record.pid)"
        foreach($child in $children){if($child.CommandLine -match [regex]::Escape("frontend/$($record.file)")){Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue}}
        Stop-Process -Id $record.pid -ErrorAction SilentlyContinue
    }else{$remaining+=@($record)}
}
ConvertTo-Json -InputObject @($remaining) -Depth 4 | Set-Content -LiteralPath $registry -Encoding UTF8
Write-Host 'WORKSPACES_STOPPED; shared backend and independently started pages remain available'
