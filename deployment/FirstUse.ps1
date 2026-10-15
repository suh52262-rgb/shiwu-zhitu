[CmdletBinding()]
param(
    [ValidateSet('menu','guide','check','setup','restore','accept','start','logs','stop','overview')][string]$Action='menu',
    [switch]$NoPause,[switch]$NoOpen,[switch]$Json,
    [int]$BackendPort=8000,[int]$FrontendPort=8501
)
$ErrorActionPreference='Stop'
$firstRoot=Split-Path -Parent $PSScriptRoot
$firstPython=Join-Path $firstRoot '.venv\Scripts\python.exe'
$firstBundle=Test-Path -LiteralPath (Join-Path $firstRoot 'manifest.json')
Set-Location -LiteralPath $firstRoot

function Show-Guide {
    Write-Host '视物知途：首次使用'
    Write-Host '完整本地包：安装与初始化 → 环境检查 → 首次恢复教材 → 免费验收 → 启动五页。'
    Write-Host '源码候选：先安装依赖，再按 GitHub源码快速开始.md 准备数据库、合法题库、教材及BGE。'
    Write-Host '只看导览不需要数据库或密钥；在线识别、提示、复盘和建图需要自己的密钥并产生费用。'
    Write-Host '恢复教材仅用于新空实例，不会恢复学生历史。日常使用无需再次恢复。'
    Write-Host '详细说明：deployment\首次使用.md；网络模型故障时不要反复上传。'
}

function Show-Checks {
    $checks=@(
        @{item='Python环境';ready=(Test-Path -LiteralPath $firstPython);action='未建立时先选择安装；需要Python 3.11。'},
        @{item='Python启动器';ready=[bool](Get-Command py -ErrorAction SilentlyContinue);action='未找到py时自行安装Python 3.11，并勾选启动器。'},
        @{item='Docker命令';ready=[bool](Get-Command docker -ErrorAction SilentlyContinue);action='完整教学需Docker数据服务或自行配置PG/Qdrant；此项不代表Docker引擎已就绪。'},
        @{item='本地配置';ready=(Test-Path -LiteralPath (Join-Path $firstRoot '.env'));action='安装操作会从模板生成；只在本地填写密钥，不上传该文件。'},
        @{item='题库';ready=(Test-Path -LiteralPath (Join-Path $firstRoot 'data\problems'));action='源码候选不带题库；从导入页录入有权使用的题目。'}
    )
    if($Json){@{mode=$(if($firstBundle){'bundle'}else{'source'});checks=$checks;model_requests=0}|ConvertTo-Json -Depth 5}
    else{
        foreach($check in $checks){Write-Host (('{0}: {1}' -f $check.item,$(if($check.ready){'已找到'}else{'待准备'})));if(!$check.ready){Write-Host ('  '+$check.action)}}
        Write-Host '以上只检查文件和命令，不证明数据库、余额或模型生成可用。'
        if(Test-Path -LiteralPath $firstPython){
            & $firstPython (Join-Path $PSScriptRoot 'preflight.py') --root $firstRoot
            if($LASTEXITCODE -ne 0){Write-Host '详细检查有未就绪项；按action提示处理，不会自动修复或发起模型调用。'}
        }
    }
}

function Run-Action([string]$choice) {
    switch($choice){
        'guide' {Show-Guide}
        'check' {Show-Checks}
        'setup' {
            if($firstBundle){& (Join-Path $PSScriptRoot 'Setup.ps1')}
            else{
                if(!(Test-Path -LiteralPath $firstPython)){& py -3.11 -m venv .venv;if($LASTEXITCODE -ne 0){throw '建立Python 3.11环境失败。'}}
                $firstLock=Join-Path $firstRoot 'requirements-deploy.lock.txt'
                if(!(Test-Path -LiteralPath $firstLock)){$firstLock=Join-Path $firstRoot 'requirements.txt'}
                & $firstPython -m pip install -r $firstLock
                if($LASTEXITCODE -ne 0){throw '安装失败；保留错误，不要直接升级全部依赖。'}
                if(!(Test-Path -LiteralPath '.env')){Copy-Item -LiteralPath (Join-Path $PSScriptRoot '.env.deploy.example') -Destination '.env'}
                Write-Host '依赖已安装。源码不附带数据和模型，请按GitHub源码快速开始准备后端；此操作未启动数据库。'
            }
        }
        'restore' {
            if(!$firstBundle){throw '这是源码目录，不含教材恢复数据；请使用授权完整包或自行导入。'}
            $confirm=Read-Host '仅首次部署的新空实例可用。输入 RESTORE 确认恢复（不会覆盖非空库）'
            if($confirm -ne 'RESTORE'){Write-Host '已取消。';return}
            & (Join-Path $PSScriptRoot 'Restore.ps1')
        }
        'accept' {
            if(!$firstBundle){throw '源码目录不含完整数据清单；免费数据验收仅用于完整部署包。'}
            & (Join-Path $PSScriptRoot 'Accept.ps1')
        }
        'start' {
            if($firstBundle){& (Join-Path $PSScriptRoot 'Start.ps1') -BackendPort $BackendPort -FrontendPort $FrontendPort -NoBrowser:$NoOpen}
            else{
                Write-Host '源码模式需先按说明启动8000后端，再打开五页；本操作不会连接其他人的后端。'
                & (Join-Path $firstRoot 'scripts\start_workspace.ps1') -NoBrowser:$NoOpen
            }
        }
        'overview' {
            if(!(Test-Path -LiteralPath $firstPython)){throw '先安装Python依赖环境。'}
            Write-Host '只读导览将以前台方式运行；按Ctrl+C停止，不启动教学或模型。'
            & $firstPython -m streamlit run frontend/portal_app.py --server.address 127.0.0.1 --server.port ($FrontendPort-1) --browser.gatherUsageStats false
        }
        'logs' {
            $firstLogs=Join-Path $firstRoot 'data\logs'
            if(Test-Path -LiteralPath $firstLogs){Write-Host ('本机日志：'+$firstLogs);if(!$NoOpen){Invoke-Item -LiteralPath $firstLogs}}
            else{Write-Host '尚无日志；启动应用后日志保存在data\logs。'}
            Write-Host '日志可能包含作答和环境信息，公开分享前请脱敏；此处不自动上传或打包。'
        }
        'stop' {
            if($firstBundle){& (Join-Path $PSScriptRoot 'Stop.ps1')}
            else{& (Join-Path $firstRoot 'scripts\stop_workspace.ps1')}
            Write-Host '只停止所属页面/应用，不停止其他项目，也不删除数据库卷。'
            Write-Host '源码工作区的后端可能由独立启动器持有，须在原后端终端停止。'
        }
    }
}

try{
    if($Action -eq 'menu'){
        $mapping=@{'1'='guide';'2'='check';'3'='setup';'4'='restore';'5'='accept';'6'='start';'7'='logs';'8'='stop';'9'='overview'}
        while($true){
            Write-Host "`n视物知途 · 本地使用助手"
            Write-Host '1 使用说明  2 环境检查  3 安装/初始化  4 首次恢复教材  5 免费验收'
            Write-Host '6 启动工作区  7 查看日志目录  8 停止所属应用  9 只看导览  0 退出'
            $selection=Read-Host '请输入编号'
            if($selection -eq '0'){break}
            if($mapping.ContainsKey($selection)){
                try{Run-Action $mapping[$selection]}catch{Write-Host ('未完成：'+$_.Exception.Message) -ForegroundColor Red}
            }else{Write-Host '请输入0–9。'}
        }
    }else{Run-Action $Action}
}catch{
    Write-Host ('未完成：'+$_.Exception.Message) -ForegroundColor Red
    if(!$NoPause){Read-Host '按回车保留错误信息后退出'|Out-Null}
    exit 1
}
if(!$NoPause -and $Action -ne 'menu'){Read-Host '按回车退出'|Out-Null}
