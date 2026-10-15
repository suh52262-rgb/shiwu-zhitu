[CmdletBinding()]
param([string]$PythonPath='')
$ErrorActionPreference='Stop'
$deployRoot=Split-Path -Parent $PSScriptRoot
$deployPy=Join-Path $deployRoot '.venv\Scripts\python.exe'
if($PythonPath){$deployPy=$PythonPath}
$deployNames=@('TEST_OFFLINE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','POSTGRES_HOST','POSTGRES_PORT','QWEN3VL_MODEL','DEEPSEEK_API_KEY','PROMPT_TUTOR_VERSION','PROMPT_DIAGNOSIS_VERSION','PROMPT_REVIEW_VERSION','PROMPT_STEP_MATCHER_VERSION')
$deployPrevious=@{}
foreach($n in $deployNames){$deployPrevious[$n]=[Environment]::GetEnvironmentVariable($n,'Process')}
try{
    $env:TEST_OFFLINE='1';$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1'
    $env:POSTGRES_HOST='localhost';$env:POSTGRES_PORT='5432'
    $env:QWEN3VL_MODEL=' ';$env:DEEPSEEK_API_KEY='offline-test-placeholder'
    # Unit-test registry defaults; demo prompts remain unchanged after the script.
    $env:PROMPT_TUTOR_VERSION='v4';$env:PROMPT_DIAGNOSIS_VERSION='v4'
    $env:PROMPT_REVIEW_VERSION='v2';$env:PROMPT_STEP_MATCHER_VERSION='v2'
    Push-Location -LiteralPath $deployRoot
    try{ & $deployPy -m pytest tests -m 'not real_db and not local_asset' -q; $deployTestExit=$LASTEXITCODE }finally{Pop-Location}
}finally{
    foreach($n in $deployNames){[Environment]::SetEnvironmentVariable($n,$deployPrevious[$n],'Process')}
}
if($deployTestExit -ne 0){throw "Offline regression failed ($deployTestExit)"}
