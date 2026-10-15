import importlib.util
from pathlib import Path

import pytest

from scripts.check_test_report import check_report


def write_report(tmp_path, body):
    path = tmp_path / "report.xml"
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>", encoding="utf-8")
    return path


def test_ci_gate_allows_known_xfail_but_requires_passed_count(tmp_path):
    path = write_report(tmp_path, '<testcase name="ok"/><testcase name="known"><skipped type="pytest.xfail"/></testcase>')
    assert check_report(path, 1) == {"passed": 1, "xfailed": 1, "skipped": 0}
    with pytest.raises(ValueError, match="minimum=2"):
        check_report(path, 2)


@pytest.mark.parametrize("body", [
    '<testcase><skipped message="database unavailable"/></testcase>',
    '<testcase><failure message="broken"/></testcase>',
    '<testcase><error message="fixture failed"/></testcase>',
    '<testcase><skipped type="pytest.xfail"/></testcase><testcase><skipped type="pytest.xfail"/></testcase>',
])
def test_ci_gate_rejects_silent_loss_of_coverage(tmp_path, body):
    with pytest.raises(ValueError, match="CI test gate failed"):
        check_report(write_report(tmp_path, body), 0)


def test_service_smoke_requires_disposable_environment(monkeypatch):
    from scripts.ci_service_smoke import main
    monkeypatch.delenv("CI_ISOLATED_SERVICES", raising=False)
    with pytest.raises(RuntimeError, match="disposable CI services"):
        main()
