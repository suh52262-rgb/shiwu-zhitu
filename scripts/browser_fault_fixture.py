"""Isolated, synthetic HTTP fixture for real Streamlit browser recovery acceptance.

Never imports business services, touches databases, or invokes models. Control it
by editing --out/control.json; every request is timestamped in requests.jsonl.
"""
import argparse
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18761)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    control_path = args.out / "control.json"
    if not control_path.exists():
        control_path.write_text(json.dumps({"mode": "running"}), encoding="utf-8")
    lock = threading.Lock()
    job_id = "a" * 32
    problem = json.loads((Path(__file__).resolve().parents[1] / "data/problems/demo_01/problem.json").read_text(encoding="utf-8"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.dispatch()

        def dispatch(self):
            control = json.loads(control_path.read_text(encoding="utf-8-sig"))
            mode = control.get("mode", "running")
            path = urlparse(self.path).path
            status, payload = 200, {}
            if path == "/health":
                payload = {"status": "ok"}
            elif path == "/problems":
                payload = [problem]
            elif "/profile" in path or "/graph" in path:
                status, payload = 404, {"detail": "Synthetic fixture: not available"}
            elif path.startswith("/video/jobs"):
                if self.command == "POST":
                    if mode == "delayed_post":
                        # The task is already queryable; only creation acknowledgement sleeps.
                        time.sleep(125)
                        payload = {"job_id": job_id, "student_id": "stu_demo", "problem_id": "demo_01", "status": "running", "phase": "recognizing"}
                    else:
                        status, payload = 503, {"detail": "Synthetic lost creation acknowledgement; task retained"}
                elif mode == "unknown":
                    status, payload = 503, {"detail": "Synthetic lookup unavailable"}
                elif mode == "absent":
                    status, payload = 404, {"detail": "Synthetic request not found"}
                else:
                    terminal = mode in {"failed", "interrupted"}
                    payload = {"job_id": job_id, "student_id": "stu_demo", "problem_id": "demo_01",
                               "status": mode if terminal else "running", "phase": "completed" if terminal else "recognizing",
                               "progress": {"completed": control.get("completed", 1), "total": 9},
                               "created_at": "2026-10-09T00:00:00+00:00", "updated_at": datetime.now(timezone.utc).isoformat()}
                    if terminal:
                        payload["error"] = {"message": "Synthetic acceptance task: " + mode}
            else:
                status, payload = 404, {"detail": "Synthetic fixture: unsupported endpoint"}
            entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
                     "method": self.command, "path": self.path, "status": status, "mode": mode,
                     "synthetic_model_substitute": True}
            with lock:
                with (args.out / "requests.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Expected when the browser frontend has already timed out.

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

