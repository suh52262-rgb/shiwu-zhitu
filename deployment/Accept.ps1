$ErrorActionPreference='Stop'
$deployRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $deployRoot
& '.\.venv\Scripts\python.exe' deployment/accept.py --bundle $deployRoot
if($LASTEXITCODE -ne 0){throw 'Deployment acceptance failed; inspect deployment/acceptance-result.json.'}
