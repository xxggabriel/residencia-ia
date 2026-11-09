"""Internal HTTP distributor for eight independently resource-limited worker containers."""

import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class WorkerPool:
    def __init__(self, urls, opener=None):
        self.urls = tuple(url.rstrip("/") for url in urls)
        if len(self.urls) != 8 or len(set(self.urls)) != 8:
            raise ValueError("the standard pool requires eight distinct workers")
        self.opener = opener or urllib.request.urlopen
        self.guard = threading.Lock()

    def call(self, url, path, body=None):
        request = urllib.request.Request(url + path, data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener(request, timeout=10 if body is not None else 4) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read(65536))
            except (ValueError, UnicodeDecodeError):
                return error.code, {"error": "invalid_worker_response"}
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return 503, {"error": "worker_unavailable"}

    def snapshots(self, path):
        with ThreadPoolExecutor(max_workers=len(self.urls)) as executor:
            return dict(zip(self.urls, executor.map(lambda url: self.call(url, path), self.urls)))

    def get(self, path):
        if path in {"/ready", "/datasets-ready"}:
            snapshots = self.snapshots(path)
            ready = all(code == 200 and status.get("ready") for code, status in snapshots.values())
            metadata = {}
            if path == "/datasets-ready" and ready:
                admissions = [status for _code, status in snapshots.values()]
                first = admissions[0]
                ready = (isinstance(first.get("code_sha256"), str) and bool(first["code_sha256"])
                         and type(first.get("datasets")) is int and first["datasets"] > 0
                         and all(status.get("code_sha256") == first["code_sha256"]
                                 and status.get("datasets") == first["datasets"] for status in admissions))
                if ready:
                    metadata = {key: first[key] for key in ("code_sha256", "datasets")}
            return (200 if ready else 503), {"ready": ready, **metadata, "workers": {
                url: {"status_code": code, **status} for url, (code, status) in snapshots.items()}}
        if path == "/health":
            snapshots = self.snapshots(path)
            healthy = {url: status for url, (code, status) in snapshots.items()
                       if code == 200 and status.get("concurrency") == 1}
            ready = len(healthy) == len(self.urls)
            return (200 if ready else 503), {
                "status": "ok" if ready else "degraded", "concurrency": len(healthy),
                "configured_workers": len(self.urls),
                "active": sum(status["active"] for status in healthy.values()),
                "queue": sum(status["queue"] for status in healthy.values()),
                "workers": healthy,
            }
        # Job status lives on the volume shared by all execution containers.
        prefix, _, identity = path.rpartition("/")
        if prefix not in {"/runs", "/epoch-runs", "/zia-runs"} or len(identity) != 24 or any(
                c not in "0123456789abcdef" for c in identity):
            return 404, {"error": "not_found"}
        for url in self.urls:
            code, status = self.call(url, path)
            if code != 503:
                return code, status
        return 503, {"error": "no_available_worker"}

    def submit(self, path, body):
        if path not in {"/runs", "/epoch-runs", "/zia-runs"}:
            return 404, {"error": "not_found"}
        # Serialize placement, not execution. Admission reserves the backend slot before replying.
        with self.guard:
            snapshots = self.snapshots("/health")
            candidates = [(url, status) for url, (code, status) in snapshots.items()
                          if code == 200 and status.get("concurrency") == 1]
            candidates.sort(key=lambda item: item[1]["queue"])
            for url, _status in candidates:
                code, result = self.call(url, path, body)
                if code != 429:
                    # Never resend a timed-out POST: the backend may already have accepted it.
                    return code, result
            error = "pool_busy" if candidates else "no_available_worker"
            return (429 if candidates else 503), {"error": error}


class Handler(BaseHTTPRequestHandler):
    pool: WorkerPool

    def respond(self, code, status):
        payload = json.dumps(status).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.respond(*self.pool.get(self.path))

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("body size must be 1..65536")
            body = self.rfile.read(length)
            json.loads(body)
        except (ValueError, UnicodeDecodeError) as error:
            self.respond(400, {"error": str(error)})
            return
        self.respond(*self.pool.submit(self.path, body))


def serve():
    Handler.pool = WorkerPool(os.environ["BENCH_WORKER_URLS"].split(","))
    ThreadingHTTPServer(("0.0.0.0", 8088), Handler).serve_forever()


if __name__ == "__main__":
    serve()
