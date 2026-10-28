"""Read-only end-to-end audit. Run in the worker after benchmark_versioned finishes."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import yaml
from mlflow.tracking import MlflowClient

from continual_bench.adapters.services import S3Objects
from continual_bench.adapters.versioning import digest_json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/app/configs/campaign-versioned-v4.json")
    parser.add_argument(
        "--limit", type=int, help="Audit only an initial subset; never marks the campaign complete"
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["configurations"]
    assert len(rows) == manifest.get("expected_runs", 289)
    if args.limit:
        rows = rows[: args.limit]
    expected = {row["run_key"]: row for row in rows}
    client = MlflowClient()
    experiments = {
        name: client.get_experiment_by_name(name) for name in sorted({r["experiment"] for r in rows})
    }
    assert all(experiments.values())
    runs = []
    token = None
    while True:
        page = client.search_runs(
            [e.experiment_id for e in experiments.values()],
            filter_string=f"tags.campaign = '{manifest['campaign']}'",
            max_results=1000,
            page_token=token,
        )
        runs.extend(page)
        token = page.token
        if not token:
            break
    by_key = {}
    for run in runs:
        key = run.data.tags.get("run_key")
        if key in expected:
            assert key not in by_key, f"duplicate run: {key}"
            by_key[key] = run
    assert set(by_key) == set(expected)
    registry = {}
    run_ids = {r.info.run_id for r in by_key.values()}
    token = None
    while True:
        page = client.search_model_versions(max_results=1000, page_token=token)
        for version in page:
            if version.run_id in run_ids:
                assert version.run_id not in registry, "duplicate registry version for run"
                registry[version.run_id] = version
        token = page.token
        if not token:
            break
    assert len(registry) == len(expected)
    objects = S3Objects()

    def read_uri(uri):
        parsed = urlparse(uri)
        assert parsed.scheme == "s3"
        return objects.client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()

    # Shared content is checked once. Run-specific links and inputs are checked below.
    shared = {}
    for key in expected:
        directory = Path("/state/runs") / key
        receipt = json.loads((directory / "receipt.json").read_text())
        for field in ("dataset_uri", "dataset_catalog_uri", "training_source_uri"):
            uri = receipt[field]
            if uri not in shared:
                shared[uri] = read_uri(uri)
        assert sha(shared[receipt["dataset_uri"]]) == receipt["dataset_version"]
        catalog = json.loads(shared[receipt["dataset_catalog_uri"]])
        assert catalog["dataset_version"] == receipt["dataset_version"]
        raw_uri = catalog.get("raw_source_uri")
        if raw_uri and raw_uri not in shared:
            shared[raw_uri] = read_uri(raw_uri)
        if raw_uri:
            assert sha(shared[raw_uri]) == catalog["source"]["raw_sha256"]

    archives = {key: value for key, value in shared.items() if key.endswith("/source.zip")}
    expected_sources = {row.get("source_sha256", manifest["source_sha256"]) for row in rows}
    assert len(archives) == len(expected_sources)
    archived_sources_by_sha = {}
    for uri, payload in archives.items():
        archive_sha = uri.rstrip("/").split("/")[-2]
        assert archive_sha in expected_sources
        archived_sources = {}
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            digest = hashlib.sha256()
            for name in sorted(archive.namelist()):
                digest.update(name.encode())
                contents = archive.read(name)
                digest.update(contents)
                archived_sources[name] = contents
        assert digest.hexdigest() == archive_sha
        archived_sources_by_sha[archive_sha] = archived_sources

    def verify(row):
        key = row["run_key"]
        directory = Path("/state/runs") / key
        job = json.loads((directory / "job.json").read_text())
        raw = (directory / "report.json").read_bytes()
        report = json.loads(raw)
        receipt = json.loads((directory / "receipt.json").read_text())
        tracked = by_key[key]
        expected_source = row.get("source_sha256", manifest["source_sha256"])
        archived_sources = archived_sources_by_sha[expected_source]
        assert job["status"] == "succeeded" and report["complete"]
        assert report["config"] == row["config"]
        assert report["environment"]["code_sha256"] == expected_source
        assert tracked.info.status == "FINISHED"
        assert receipt["mlflow_run_id"] == tracked.info.run_id
        assert receipt["mlflow_experiment_name"] == row["experiment"]
        assert tracked.info.experiment_id == experiments[row["experiment"]].experiment_id
        for metric, value in report["metrics"].items():
            assert tracked.data.metrics["summary_" + metric] == value
        metadata = report["model_versions"]
        dataset = report["dataset"]
        assert tracked.data.params["code_sha256"] == expected_source
        assert tracked.data.params["dataset"] == row["config"]["dataset"]["name"]
        assert tracked.data.params["dataset_version"] == dataset["sha256"]
        if manifest["campaign"] == "nonlinear-v5":
            assert tracked.data.params["model"] == row["config"]["model"]["name"]
            assert json.loads(tracked.data.params["model_params"]) == row["config"]["model"]["params"]
        assert tracked.data.params["model_state_sha256"] == metadata["state_sha256"]
        assert int(tracked.data.params["model_versions_count"]) == metadata["count"]
        inputs = tracked.inputs.dataset_inputs
        assert len(inputs) == 1
        assert inputs[0].dataset.name == dataset["name"]
        assert inputs[0].dataset.digest == dataset["sha256"][:32]
        assert receipt["dataset_uri"] in inputs[0].dataset.source
        assert sha((directory / "events.jsonl").read_bytes()) == dataset["sha256"]
        assert read_uri(f"s3://{objects.bucket}/{key}/report.json") == raw
        index_raw = read_uri(receipt["model_index_uri"])
        journal_uri = urlparse(receipt["model_states_uri"])
        journal_object = objects.client.get_object(
            Bucket=journal_uri.netloc, Key=journal_uri.path.lstrip("/")
        )
        journal = journal_object["Body"]
        journal_digest = hashlib.sha256()
        assert sha(index_raw) == metadata["index_sha256"]
        index = json.loads(index_raw)
        assert index["dataset_version"] == dataset["sha256"]
        assert index["code_sha256"] == expected_source
        assert len(index["versions"]) == metadata["count"] == 1 + len(report.get("updates", []))
        if row["config"]["policy"]["name"] == "online":
            assert len(report["updates"]) == dataset["samples"] - row["config"]["warmup"]
        used = sum(update["train_samples"] for update in report["updates"])
        assert tracked.data.metrics["training_used"] == used
        budget = row["config"]["training_budget"]
        assert budget is None or used <= budget
        parent = None
        offset = 0
        ids = []
        weights = []
        for ordinal, entry in enumerate(index["versions"]):
            assert entry["ordinal"] == ordinal and entry["parent_version"] == parent
            assert entry["dataset_version"] == dataset["sha256"]
            assert entry["code_sha256"] == expected_source
            assert entry["offset"] == offset
            state = journal.read(entry["size"])
            journal_digest.update(state)
            assert len(state) == entry["size"] and sha(state) == entry["state_sha256"]
            assert entry["version_id"] == digest_json({k: v for k, v in entry.items() if k != "version_id"})
            training = entry["training"]
            start, end = training["seq_range"]
            if ordinal and row["config"]["policy"]["name"] == "online":
                assert training["action"] == "learn_one"
                assert [start, end] == [
                    row["config"]["warmup"] + ordinal - 1,
                    row["config"]["warmup"] + ordinal,
                ]
            assert 0 <= start < end <= dataset["samples"]
            assert entry["training_view_version"] == digest_json(
                {"dataset_version": dataset["sha256"], "seq_range": [start, end]}
            )
            if "released_at" in training:
                assert training["released_at"] >= training["label_available_at"]
                assert (
                    training["label_available_at"]
                    == row["config"]["dataset"]["epoch"] + end - 1 + row["config"]["dataset"]["label_delay"]
                )
            parent = entry["version_id"]
            offset += entry["size"]
            ids.append(parent)
            weights.append(entry["state_sha256"])
        assert offset == journal_object["ContentLength"] and not journal.read(1)
        journal.close()
        assert journal_digest.hexdigest() == metadata["journal_sha256"]
        assert parent == metadata["final_version_id"]
        assert weights[-1] == metadata["state_sha256"]
        assert receipt["model_versions_count"] == metadata["count"]
        assert receipt["model_version_id"] == parent
        registered = registry[tracked.info.run_id]
        assert registered.status == "READY"
        assert registered.name == receipt["registered_model"]
        assert str(registered.version) == receipt["registered_model_version"]
        assert registered.tags["dataset_version"] == dataset["sha256"]
        assert registered.tags["model_sha256"] == metadata["state_sha256"]
        assert registered.tags["model_version_id"] == parent
        prefix = urlparse(registered.source).path.strip("/")
        model_info = yaml.safe_load(
            objects.client.get_object(Bucket="mlflow", Key=prefix + "/MLmodel")["Body"].read()
        )
        flavor = model_info["flavors"]["python_function"]
        for filename in ("conda.yaml", "requirements.txt", "python_env.yaml"):
            objects.client.head_object(Bucket="mlflow", Key=f"{prefix}/{filename}")
        requirements = (
            objects.client.get_object(Bucket="mlflow", Key=f"{prefix}/requirements.txt")["Body"]
            .read()
            .decode()
            .splitlines()
        )
        expected_requirements = {
            f"{name}=={report['environment']['packages'][name]}"
            for name in ("river", "numpy", "scipy", "mlflow-skinny", "pandas")
        }
        assert set(requirements) == expected_requirements
        required_code = (
            "__init__.py",
            "adapters/__init__.py",
            "adapters/river.py",
            "adapters/model_loader.py",
            "domain/__init__.py",
            "domain/contracts.py",
        )
        if "adapters/nonlinear.py" in archived_sources:
            required_code += ("adapters/nonlinear.py",)
        for filename in required_code:
            bundled = objects.client.get_object(
                Bucket="mlflow", Key=f"{prefix}/{flavor['code']}/continual_bench/{filename}"
            )["Body"].read()
            assert bundled == archived_sources[filename]
        data_path = flavor["data"]
        model_blob = objects.client.get_object(Bucket="mlflow", Key=f"{prefix}/{data_path}/model.pkl")[
            "Body"
        ].read()
        assert sha(model_blob) == metadata["state_sha256"]
        stored_meta = json.loads(
            objects.client.get_object(Bucket="mlflow", Key=f"{prefix}/{data_path}/model-metadata.json")[
                "Body"
            ].read()
        )
        assert stored_meta == metadata
        return {
            "report": report,
            "publication_verified": True,
            "sha256": sha(raw),
            "receipt": receipt,
            "artifact_sizes": {
                "model_states_bytes": journal_object["ContentLength"],
                "model_index_bytes": len(index_raw),
                "final_model_bytes": len(model_blob),
            },
            "job_started_at": job.get("started_at"),
            "job_finished_at": job.get("finished_at"),
            "snapshot_ids": ids,
            "state_hashes": weights,
        }

    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, record in enumerate(pool.map(verify, rows), 1):
            records.append(record)
            if number % 25 == 0:
                print(f"Verified {number}/{len(rows)} runs", file=sys.stderr, flush=True)
    # Exercise the Registry's authenticated download path and the packaged inference code.
    reloads = []
    contexts = sorted(
        {
            (
                r["config"]["dataset"]["name"],
                r["config"]["model"]["name"],
                r.get("source_sha256", manifest["source_sha256"]),
            )
            for r in rows
        }
    )
    for dataset, model_name, source_sha in contexts:
        record = next(
            r
            for r in records
            if r["report"]["config"]["dataset"]["name"] == dataset
            and r["report"]["config"]["model"]["name"] == model_name
            and r["report"]["environment"]["code_sha256"] == source_sha
        )
        probe = record["report"]["model_versions"]["inference_probe"]
        request = {
            "dataset": dataset,
            "model": model_name,
            "uri": record["receipt"]["model_uri"],
            "probe": probe,
        }
        checked = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("registry_inference_probe.py"))],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        reloads.append(json.loads(checked.stdout))
    snapshot_ids = {s for r in records for s in r.pop("snapshot_ids")}
    state_hashes = {s for r in records for s in r.pop("state_hashes")}
    result = {
        "campaign": manifest["campaign"],
        "complete_campaign": not args.limit,
        "runs_verified": len(records),
        "experiments": {k: v.experiment_id for k, v in experiments.items()},
        "datasets": dict(Counter(r["report"]["dataset"]["name"] for r in records)),
        "models": dict(Counter(r["report"]["config"]["model"]["name"] for r in records)),
        "dataset_versions": len({r["report"]["dataset"]["sha256"] for r in records}),
        "registered_model_families": len({r["receipt"]["registered_model"] for r in records}),
        "registered_final_versions": len(records),
        "snapshot_references": sum(r["report"]["model_versions"]["count"] for r in records),
        "unique_snapshot_ids": len(snapshot_ids),
        "unique_state_hashes": len(state_hashes),
        "registry_reloads": reloads,
        "source_sha256": manifest["source_sha256"],
        "source_revisions": sorted(expected_sources),
        "source_counts": dict(Counter(r["report"]["environment"]["code_sha256"] for r in records)),
        "records": records,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
