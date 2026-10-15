[CmdletBinding()]
param([switch]$Services,[switch]$Online,[string]$PythonPath='')
$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent $PSScriptRoot
$releasePython=Join-Path $releaseRoot '.venv\Scripts\python.exe'
if($PythonPath){$releasePython=$PythonPath}
if(!(Test-Path -LiteralPath $releasePython)){throw 'Create the Python 3.11 environment using Setup.ps1 first.'}
$releaseArguments=@((Join-Path $PSScriptRoot 'preflight.py'),'--root',$releaseRoot)
if($Services){$releaseArguments+='--services'}
if($Online){$releaseArguments+='--online'}
& $releasePython @releaseArguments
if($LASTEXITCODE -ne 0){throw 'Preflight failed. Follow the action fields above; no model requests were sent.'}
