[CmdletBinding()]
param([int]$PostgresPort=5432,[int]$QdrantPort=6333,[switch]$SkipInstall,[string]$PythonPath='',[string]$ComposeProject='')
$ErrorActionPreference='Stop'
$deployRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $deployRoot
function Check-Exit($label){ if($LASTEXITCODE -ne 0){throw "$label failed ($LASTEXITCODE)"} }
if($PythonPath){
    if(!(Test-Path -LiteralPath $PythonPath -PathType Leaf)){throw 'Provided Python executable does not exist.'}
    $deployPy=$PythonPath
}else{
    if(!(Test-Path -LiteralPath '.venv\Scripts\python.exe')){
        & py -3.11 -m venv .venv
        Check-Exit 'Create Python 3.11 environment'
    }
    $deployPy=Join-Path $deployRoot '.venv\Scripts\python.exe'
}
& $deployPy scripts/deployment_bundle.py verify --bundle $deployRoot
Check-Exit 'Verify bundle before installing wheels'
if(!$SkipInstall){
    if(Test-Path -LiteralPath 'wheels'){
        & $deployPy -m pip install --no-compile --no-cache-dir --no-index --find-links wheels --progress-bar off -r requirements-deploy.lock.txt
    }else{
        & $deployPy -m pip install --no-compile --no-cache-dir --progress-bar off -r requirements-deploy.lock.txt
    }
    Check-Exit 'Install locked dependencies'
}
if(!(Test-Path -LiteralPath '.env')){
    $deployEnv=Get-Content -LiteralPath '.env.example' -Raw -Encoding UTF8
    $deployModel=(Join-Path $deployRoot 'models\bge-small-zh-v1.5').Replace('\','/')
    $deployEnv=$deployEnv.Replace('EMBEDDING_MODEL=models/bge-small-zh-v1.5',"EMBEDDING_MODEL=$deployModel")
    $deployEnv=$deployEnv.Replace('POSTGRES_PORT=5432',"POSTGRES_PORT=$PostgresPort")
    $deployEnv=$deployEnv.Replace('DEPLOY_POSTGRES_PORT=5432',"DEPLOY_POSTGRES_PORT=$PostgresPort")
    $deployEnv=$deployEnv.Replace('http://127.0.0.1:6333',"http://127.0.0.1:$QdrantPort")
    $deployEnv=$deployEnv.Replace('DEPLOY_QDRANT_PORT=6333',"DEPLOY_QDRANT_PORT=$QdrantPort")
    [IO.File]::WriteAllText((Join-Path $deployRoot '.env'),$deployEnv,(New-Object Text.UTF8Encoding $false))
    Write-Host 'Created .env; fill DEEPSEEK_API_KEY locally for paid model features.'
}
# Ports alone do not isolate Docker volumes. Persist a version-specific project.
$deployEnvPath=Join-Path $deployRoot '.env'
$deployEnvText=Get-Content -LiteralPath $deployEnvPath -Raw -Encoding UTF8
$deployProjectMatch=[regex]::Match($deployEnvText,'(?m)^COMPOSE_PROJECT_NAME=([^\r\n]+)')
if(!$ComposeProject){
    if($deployProjectMatch.Success){$ComposeProject=$deployProjectMatch.Groups[1].Value.Trim()}
    else{
        $deployVersion=(Get-Content -LiteralPath 'VERSION.json' -Raw -Encoding UTF8|ConvertFrom-Json).version
        $ComposeProject='shiwu-'+($deployVersion.ToLowerInvariant() -replace '[^a-z0-9_-]','-')
    }
}
if($ComposeProject -notmatch '^[a-z0-9][a-z0-9_-]+$'){throw 'Invalid Compose project name.'}
if($deployProjectMatch.Success){
    $deployEnvText=[regex]::Replace($deployEnvText,'(?m)^COMPOSE_PROJECT_NAME=[^\r\n]+',"COMPOSE_PROJECT_NAME=$ComposeProject")
}else{$deployEnvText+="`nCOMPOSE_PROJECT_NAME=$ComposeProject`n"}
[IO.File]::WriteAllText($deployEnvPath,$deployEnvText,(New-Object Text.UTF8Encoding $false))
& docker compose -p $ComposeProject up -d --wait
Check-Exit 'Start isolated deployment services'
& $deployPy scripts/init_db.py
Check-Exit 'Initialize schema'
Write-Host 'SETUP_OK. Next: Restore.ps1, then Accept.ps1, then Start.ps1.'
