$ErrorActionPreference='Stop'
$deployRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $deployRoot
& '.\.venv\Scripts\python.exe' scripts/deployment_bundle.py restore --bundle $deployRoot
if($LASTEXITCODE -ne 0){throw 'Restore failed. It never deletes existing data; use a new empty instance.'}
