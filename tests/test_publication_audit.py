import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('publication_audit',Path(__file__).resolve().parents[1]/'scripts/publication_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_known_secret_report_does_not_echo_value():
    value=b'private-local-credential-for-test'
    result=audit.scan_text(b'key='+value,known=[value])
    assert result[0]['rule']=='known_local_credential'
    assert value.decode() not in str(result)


def test_pattern_and_url_findings_are_location_only():
    value=b'postgresql://operator:private-secret@db.local/example'
    result=audit.scan_text(b'first\n'+value)
    assert result[0]['line']==2
    assert result[0]['rule']=='credential_url_literal'
    assert 'private-secret' not in str(result)


def test_binary_and_oversize_are_not_claimed_clean(monkeypatch):
    assert audit.scan_bytes(b'\0binary')['content_status']=='binary_not_scanned'
    monkeypatch.setattr(audit,'LIMIT',3)
    assert audit.scan_bytes(b'larger')['content_status']=='oversize_not_scanned'


def test_templates_and_private_runtime_are_distinguished():
    assert audit.private_path('.env.example') is None
    assert audit.private_path('deployment/.env.deploy.example') is None
    assert audit.private_path('.env')=='environment_file'
    assert audit.private_path('data/video_runs/run/frame.png')=='runtime_or_private_material'
