"""Run inside the worker; verify 261 local, MLflow and S3 reports, then emit JSON."""

import hashlib
import json
from pathlib import Path

from mlflow.tracking import MlflowClient

from continual_bench.adapters.services import S3Objects

manifest = json.loads(Path("/app/configs/campaign-controlled-v3.json").read_text())
expected = {r["run_key"] for r in manifest["configurations"]}
client = MlflowClient()
experiment = client.get_experiment_by_name("continual-bench-v1")
tracked = client.search_runs([experiment.experiment_id], max_results=1000)
by_key = {}
for run in tracked:
    key = run.data.tags.get("run_key")
    if key in expected:
        assert key not in by_key
        by_key[key] = run
assert len(by_key) == len(expected) == 261
objects = S3Objects()
records = []
for key in sorted(expected):
    directory = Path("/state/runs") / key
    job = json.loads((directory / "job.json").read_text())
    raw = (directory / "report.json").read_bytes()
    r = json.loads(raw)
    receipt = json.loads((directory / "receipt.json").read_text())
    remote = by_key[key]
    assert job["status"] == "succeeded" and r["complete"] and remote.info.status == "FINISHED"
    assert receipt["mlflow_run_id"] == remote.info.run_id
    for metric, value in r["metrics"].items():
        assert remote.data.metrics["summary_" + metric] == value
    data = (directory / "events.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == r["dataset"]["sha256"]
    stored = objects.client.get_object(Bucket=objects.bucket, Key=key + "/report.json")["Body"].read()
    assert hashlib.sha256(stored).digest() == hashlib.sha256(raw).digest()
    records.append(
        {
            "report": r,
            "publication_verified": True,
            "report_sha256_verified": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mlflow_run_id": remote.info.run_id,
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
        }
    )
print(json.dumps(records))
