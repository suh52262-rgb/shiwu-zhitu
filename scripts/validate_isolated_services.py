"""Run integration checks in newly created Docker containers on random local ports.

Uses the current Python environment; CI tests dependency installation separately.
No existing container, volume, database or Qdrant collection is reused.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main():
    created = []
    evidence = ROOT / "artifacts"
    evidence.mkdir(exist_ok=True)
    log = (evidence / "services-local.log").open("w", encoding="utf-8")

    def run(args, *, env=None, timeout=180):
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout)
        log.write(result.stdout + result.stderr)
        log.flush()
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {args[0:3]}; see {log.name}")
        return result.stdout.strip()

    try:
        suffix = uuid4().hex[:12]
        pg = run(["docker", "run", "--detach", "--name", "shiwu-test-pg-" + suffix,
                  "--publish", "127.0.0.1::5432", "--env", "POSTGRES_USER=shiwu_ci",
                  "--env", "POSTGRES_PASSWORD=disposable_ci_password", "--env", "POSTGRES_DB=shiwu_ci",
                  "postgres:16-alpine"])
        if not re.fullmatch(r"[0-9a-f]{64}", pg):
            raise RuntimeError("Unexpected newly created PostgreSQL container ID")
        created.append(pg)
        qd = run(["docker", "run", "--detach", "--name", "shiwu-test-qd-" + suffix,
                  "--publish", "127.0.0.1::6333", "qdrant/qdrant:v1.12.4"])
        if not re.fullmatch(r"[0-9a-f]{64}", qd):
            raise RuntimeError("Unexpected newly created Qdrant container ID")
        created.append(qd)
        pg_port = run(["docker", "port", pg, "5432/tcp"]).split(":")[-1]
        qd_port = run(["docker", "port", qd, "6333/tcp"]).split(":")[-1]
        for _ in range(30):
            ready = subprocess.run(["docker", "exec", pg, "pg_isready", "-U", "shiwu_ci", "-d", "shiwu_ci"],
                                   capture_output=True, timeout=5).returncode == 0
            try:
                with urlopen(f"http://127.0.0.1:{qd_port}/readyz", timeout=2) as response:
                    ready = ready and response.status == 200
            except OSError:
                ready = False
            if ready:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Temporary services did not become ready")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "backend"), "PYTHONUTF8": "1",
               "CI_ISOLATED_SERVICES": "1", "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": pg_port,
               "POSTGRES_USER": "shiwu_ci", "POSTGRES_PASSWORD": "disposable_ci_password",
               "POSTGRES_DB": "shiwu_ci", "QDRANT_URL": f"http://127.0.0.1:{qd_port}",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        env.pop("TEST_OFFLINE", None)
        for args in (["scripts/init_db.py"], ["scripts/ci_service_smoke.py"],
                     ["-m", "pytest", "tests", "-m", "real_db", "-q", "--tb=short",
                      "--junitxml=artifacts/services.xml"],
                     ["scripts/check_test_report.py", "artifacts/services.xml", "--minimum", "26",
                      "--maximum-xfails", "0"],
                     ["-c", "from fastapi.testclient import TestClient; from app.main import app; "
                      "c=TestClient(app); assert c.get('/health').status_code==200; "
                      "assert c.get('/problems').status_code==200"]):
            run([sys.executable, *args], env=env)
        print("ISOLATED_SERVICES_OK: see artifacts/services.xml and services-local.log")
    finally:
        for container_id in reversed(created):
            run(["docker", "rm", "--force", "--volumes", container_id], timeout=30)
        log.close()


if __name__ == "__main__":
    main()
