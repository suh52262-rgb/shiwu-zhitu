import json
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.skipif(POWERSHELL is None,reason='PowerShell or pwsh runtime required for launcher integration')
def test_checks_do_not_install_or_print_config(tmp_path):
    folder=tmp_path/'deployment';folder.mkdir()
    shutil.copy2(ROOT/'deployment/FirstUse.ps1',folder/'FirstUse.ps1')
    (tmp_path/'.env').write_text('DEEPSEEK_API_KEY=private-check-fixture\n')
    (folder/'Setup.ps1').write_text("throw 'Unexpected installer call'")
    proc=subprocess.run([POWERSHELL,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(folder/'FirstUse.ps1'),'-Action','check','-Json','-NoPause'],capture_output=True,timeout=20)
    assert proc.returncode==0
    body=json.loads(proc.stdout.decode('utf-8',errors='replace'))
    assert body['mode']=='source' and body['model_requests']==0
    assert 'private-check-fixture' not in proc.stdout.decode(errors='replace')
    assert not (tmp_path/'.venv').exists()


@pytest.mark.skipif(POWERSHELL is None,reason='PowerShell or pwsh runtime required for launcher integration')
def test_log_action_never_dumps_log_contents(tmp_path):
    folder=tmp_path/'deployment';folder.mkdir()
    shutil.copy2(ROOT/'deployment/FirstUse.ps1',folder/'FirstUse.ps1')
    logs=tmp_path/'data/logs';logs.mkdir(parents=True)
    (logs/'backend.err.log').write_text('private-log-body-fixture')
    proc=subprocess.run([POWERSHELL,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(folder/'FirstUse.ps1'),'-Action','logs','-NoOpen','-NoPause'],capture_output=True,timeout=20)
    assert proc.returncode==0
    assert b'private-log-body-fixture' not in proc.stdout
