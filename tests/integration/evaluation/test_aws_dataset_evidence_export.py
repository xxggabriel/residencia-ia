import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from continual_bench.config import IterativeTrainingConfig
from continual_bench.evaluation.stream_metrics import ClassificationMetrics
from continual_bench.evaluation.tracking import metric_catalog
from scripts.aws import export_dataset_evidence as export


class FakeS3:
    """Deterministic versioned bytes; only the exporter's read interface exists."""

    def __init__(self):
        self.objects, self.reads, self.local_prediction_paths = {}, [], []

    def add(self, uri, payload):
        if isinstance(payload, dict):
            payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.objects[uri] = payload
        return hashlib.sha256(payload).hexdigest()

    def head(self, uri):
        payload = self.objects[uri]
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "ContentLength": len(payload),
            "ETag": '"' + digest[:32] + '"',
            "VersionId": "version-" + digest[:16],
            "Metadata": {"sha256": digest},
            "LastModified": "2026-01-01T00:00:00Z",
        }

    def download(self, uri, destination, head):
        self.reads.append(uri)
        if "/predictions/" in uri:
            for previous in self.local_prediction_paths:
                if previous.parent.parent != Path(destination).parent.parent:
                    assert not previous.exists(), "two cases retained predictions simultaneously"
            self.local_prediction_paths.append(Path(destination))
        Path(destination).write_bytes(self.objects[uri])
        return {"VersionId": head["VersionId"], "ETag": head["ETag"]}

    def list(self, prefix):
        return [
            {
                "uri": uri,
                "bytes": len(payload),
                "etag": self.head(uri)["ETag"],
                "verification": "listed_not_downloaded",
            }
            for uri, payload in sorted(self.objects.items())
            if uri.startswith(prefix)
        ]


def fixture_suite(tmp_path, *, full_selection=False):
    reader = FakeS3()
    bucket, campaign, release = "fixture-bucket", "fixture-full", "a" * 40
    source_sha, labels = "b" * 64, np.arange(20, dtype=np.uint8) % 2
    state_dir = tmp_path / "coordinator"
    state_dir.mkdir()
    units, verified, original_reports, campaign_units = [], {}, {}, {}
    datasets = {dataset: 24 for dataset in export.DATASET_ROWS} if full_selection else {"fixture": 24}
    policies = export.POLICIES if full_selection else ("frozen", "cara")
    for dataset, policy in ((dataset, policy) for dataset in datasets for policy in policies):
        stage = "cached" if policy in export.CACHED_POLICIES else "base"
        if full_selection:
            campaign = f"fixture-{export.digest(dataset)[:8]}-{stage}"
        for seed in (42, 43, 44):
            run_id = hashlib.sha256(f"{dataset}/{policy}/{seed}".encode()).hexdigest()[:20]
            prefix = f"s3://{bucket}/campaigns/{campaign}/runs/{run_id}/attempts/job-1"
            p = np.full(20, {42: 0.8, 43: 0.85, 44: 0.9}[seed])
            probabilities = np.column_stack((1 - p, p))
            table = pa.table(
                {
                    "seq": np.arange(4, 24, dtype=np.uint32),
                    "event_time_ms": np.arange(4, 24, dtype=np.int64),
                    "label": labels,
                    "prediction": np.ones(20, dtype=np.uint8),
                    "holdout": np.arange(4, 24) >= 20,
                    "model_end_seq": np.full(20, 4, np.uint32),
                    "p_0": probabilities[:, 0],
                    "p_1": probabilities[:, 1],
                }
            )
            part_path = tmp_path / "part.parquet"
            pq.write_table(table, part_path)
            part = part_path.read_bytes()
            part_uri = prefix + "/predictions/part-0000.parquet"
            part_sha = reader.add(part_uri, part)
            index = {
                "dataset_sha256": source_sha,
                "classes": ["0", "1"],
                "rows": 20,
                "fading_factor": 0.99,
                "fading_accuracy": [],
                "origin": "native",
                "parts": [{"file": "part-0000.parquet", "bytes": len(part), "rows": 20, "sha256": part_sha}],
            }
            counts, windows = {}, []
            for scope, start, end in (("prequential", 0, 16), ("holdout", 16, 20)):
                metric = ClassificationMetrics(2, 0.5)
                metric.add(labels[start:end], probabilities[start:end])
                counts[scope] = metric.metrics()[0]
            for start in range(0, 20, 4):
                metric = ClassificationMetrics(2, 0.5)
                metric.add(labels[start : start + 4], probabilities[start : start + 4])
                windows.append(
                    {
                        **metric.metrics()[0],
                        "end_seq": start + 8,
                        "scope": "prequential" if start < 16 else "holdout",
                        "start_time_ms": start + 4,
                        "end_time_ms": start + 7,
                    }
                )
            training = IterativeTrainingConfig(seed=seed).model_dump()
            history = [{"seed": seed, "checkpoint": {"file": None}}]
            contract = {
                "protocol": export.PROTOCOL,
                "release": {"git_sha": release},
                "stream_source": {"prepared_sha256": source_sha},
            }
            unit = {
                "run_id": run_id,
                "campaign_id": campaign,
                "params": {
                    "dataset": dataset,
                    "policy": policy,
                    "seed": seed,
                    "iterative_training": training,
                },
                "contract": contract,
                "contract_sha256": export.digest(contract),
            }
            report = {
                "complete": True,
                "artifact_mode": "metrics_only",
                "protocol": export.PROTOCOL,
                "dataset": dataset,
                "scope": "full",
                "policy": policy,
                "model_params": {"seed": seed},
                "contract": {
                    "artifact_mode": "metrics_only",
                    "input": {"sha256": source_sha, "rows": 24, "source": {"classes": ["0", "1"]}},
                    "iterative_training": training,
                    "holdout_start": 20,
                    "score_bins": 4096,
                    "decision_threshold": 0.5,
                    "block_rows": 4,
                    "decision_rule": "fixture",
                    "environment": {},
                },
                "metrics": {
                    **counts["prequential"],
                    "dataset_rows": 24,
                    "initial_train_rows": 4,
                    "training_used_post_warmup": 0,
                    "model_storage_bytes": 0,
                },
                "holdout": counts["holdout"],
                "windows": windows,
                "model_versions": [{"file": None}],
                "training_history": history,
                "costs": {"settings": {}},
                "alarms": [],
                "cost_sensitivity": [[0, 0], [1, 0]],
                "extension_metric_vector": [
                    None,
                    False,
                    0,
                    1.25,
                    {"complete_nested_vector": list(range(1000))},
                ],
            }
            if policy == "cara":
                report["retraining"] = {
                    "rules": {},
                    "alphas": [0, 1],
                    "cumulative_strategy_cost": {},
                    "setup": {
                        "offline_batches": 3,
                        "alpha_max": 1,
                        "focus_alpha": 1,
                        "timing": "fixture",
                        "online_batches": 3,
                    },
                }
            report_uri = prefix + "/report.json"
            report_sha = reader.add(report_uri, report)
            original_reports[run_id] = reader.objects[report_uri]
            reader.add(prefix + "/predictions-manifest.json", index)
            reader.add(prefix + "/training-history.json", {"version": 1, "fits": history})
            reader.add(
                prefix + "/artifacts/data/index.json",
                {"version": "run-artifacts-v1", "figures": [], "data": []},
            )
            reader.add(prefix + "/metric-definitions.json", metric_catalog()[1])
            reader.add(
                prefix + "/live-metrics.jsonl",
                (
                    json.dumps(
                        {
                            "step": 0,
                            "timestamp": 1,
                            "prefix": "attempts/job-1/progress",
                            "values": {"rows_processed": 0},
                        }
                    )
                    + "\n"
                ).encode(),
            )
            final = {
                "artifact_mode": "metrics_only",
                "model": None,
                "outputs_uri": prefix,
                "contract_sha256": unit["contract_sha256"],
                "report_uri": report_uri,
                "report_sha256": report_sha,
                "release": contract["release"],
                "tracking": {"complete": True, "required": True},
                "predictions_manifest_uri": prefix + "/predictions-manifest.json",
                "training_history_uri": prefix + "/training-history.json",
                "artifacts_index_uri": prefix + "/artifacts/data/index.json",
            }
            reader.add(f"s3://{bucket}/campaigns/{campaign}/runs/{run_id}/final.json", final)
            verified[run_id] = {
                "final_sha256": export.digest(final),
                "report_sha256": report_sha,
                "dataset": dataset,
                "policy": policy,
                "seed": seed,
            }
            units.append(unit)
            campaign_units.setdefault(campaign, []).append(unit)
    for campaign, selected in campaign_units.items():
        reader.add(f"s3://{bucket}/campaigns/{campaign}/plan.json", {"units": selected})
    stages = {"fixture": {"base": {"request": {"campaign_id": campaign, "release": release}}}}
    if full_selection:
        stages = {}
        for dataset in datasets:
            stages[dataset] = {}
            for stage, selected in export.STAGE_POLICIES.items():
                matrix = {
                    "kind": "stream",
                    "policies": list(selected),
                    "seeds": list(export.SEEDS),
                    "artifact_mode": "metrics_only",
                    "iterative_training": {"max_epochs": 100},
                }
                if stage == "cached":
                    matrix["metric_cache_sources"] = {
                        str(seed): {"uri": f"s3://{bucket}/cache/{dataset}/{seed}", "sha256": source_sha}
                        for seed in export.SEEDS
                    }
                stages[dataset][stage] = {
                    "request": {
                        "campaign_id": f"fixture-{export.digest(dataset)[:8]}-{stage}",
                        "release": release,
                        "dataset": dataset,
                        "scope": "full",
                        "matrix": matrix,
                    }
                }
    state = {
        "schema": "aws-ordered-dataset-suite-v1",
        "bucket": bucket,
        "region": "us-east-1",
        "release": release,
        "contract": {
            "requests": {dataset: records["base"]["request"] for dataset, records in stages.items()}
        },
        "verified_finals": verified,
        "datasets": stages,
    }
    export.save(state_dir / "suite-state.json", state)
    export.save(
        state_dir / "latest-status.json",
        {
            "complete": True,
            "verified_units": len(verified),
            "expected_units": len(verified),
            "expected_campaigns": len(campaign_units),
            "batch": {"active": 0},
            "campaigns": {
                campaign: {"complete": True, "expected_units": len(selected), "verified_units": len(selected)}
                for campaign, selected in campaign_units.items()
            },
        },
    )
    coordinates = export.expected_matrix(datasets, policies=policies)
    return reader, state_dir, original_reports, coordinates


def test_export_reconstructs_real_prediction_metrics_one_case_at_a_time_and_preserves_every_report_byte(
    tmp_path,
):
    reader, state_dir, reports, coordinates = fixture_suite(tmp_path)
    before = dict(reader.objects)
    output, scratch = tmp_path / "evidence", tmp_path / "scratch"
    manifest = export.export_evidence(
        state_dir, output, scratch, reader, expected_rows={"fixture": 24}, coordinates=coordinates
    )
    assert manifest["complete"] and manifest["coverage"]["cases"] == 6
    assert manifest["checks"]["all_prediction_parts_verified"]
    assert manifest["native_mlflow_api_readback"]["verified"] is False
    assert not list(scratch.iterdir())
    assert not list(output.rglob("*.parquet"))
    assert reader.objects == before
    for run_id, payload in reports.items():
        descriptor = export.read(output / "reports" / run_id / "source.json")
        recovered = b"".join(
            gzip.decompress((output / p["path"]).read_bytes()) for p in descriptor["blob"]["parts"]
        )
        assert recovered == payload
        decoded = json.loads(recovered)
        assert decoded["extension_metric_vector"][:3] == [None, False, 0]
        assert len(decoded["extension_metric_vector"][-1]["complete_nested_vector"]) == 1000
        qa = export.read(output / "cases" / run_id / "qa.json")
        assert qa["common_comparison"]["seq_range"] == [12, 20]
        assert qa["common_comparison"]["metrics"]["n"] == 8
        provenance = export.read(output / "cases" / run_id / "provenance.json")
        prediction = provenance["source_objects"]["predictions/part-0000.parquet"]
        assert prediction["version_id"].startswith("version-")
        assert prediction["verification"] == "downloaded_size_and_sha256_verified"
    assert all("/campaigns/" in uri for uri in reader.reads)
    assert (output / "comparison-mean-sd.csv").is_file()
    assert (output / "SHA256SUMS").is_file()
    predictions_read = [uri for uri in reader.reads if "/predictions/" in uri]
    repeated = export.export_evidence(
        state_dir, output, scratch, reader, expected_rows={"fixture": 24}, coordinates=coordinates
    )
    assert repeated["complete"]
    assert [uri for uri in reader.reads if "/predictions/" in uri] == predictions_read
    assert repeated["scratch"]["observed_peak_case_bytes"] == manifest["scratch"]["observed_peak_case_bytes"]
    run_id = next(iter(reports))
    qa_path = output / "cases" / run_id / "qa.json"
    changed = export.read(qa_path)
    changed["prediction_rows"] = 0
    export.save(qa_path, changed)
    with pytest.raises(ValueError, match="case audit files changed"):
        export.export_evidence(
            state_dir, output, scratch, reader, expected_rows={"fixture": 24}, coordinates=coordinates
        )


def test_selected_four_dataset_export_covers_132_cases_eight_campaigns_and_four_common_cohorts(tmp_path):
    reader, state_dir, reports, coordinates = fixture_suite(tmp_path, full_selection=True)
    before = dict(reader.objects)
    state_bytes = (state_dir / "suite-state.json").read_bytes()
    output, scratch = tmp_path / "evidence", tmp_path / "scratch"
    manifest = export.export_evidence(
        state_dir,
        output,
        scratch,
        reader,
        expected_rows={dataset: 24 for dataset in export.DATASET_ROWS},
        coordinates=coordinates,
        scratch_bytes=1024**2,
    )
    assert manifest["coverage"] == {
        "cases": 132,
        "campaigns": 8,
        "cohorts": 4,
        "datasets": sorted(export.DATASET_ROWS),
        "policies": sorted(export.POLICIES),
        "seeds": [42, 43, 44],
    }
    cohorts = export.read(output / "cohorts.json")["cohorts"]
    assert {row["dataset"] for row in cohorts} == set(export.DATASET_ROWS)
    assert all(row["seq_range"] == [12, 20] for row in cohorts)
    with (output / "comparison-mean-sd.csv").open() as stream:
        aggregates = list(csv.DictReader(stream))
    assert {row["dataset"] for row in aggregates} == set(export.DATASET_ROWS)
    assert {row["policy"] for row in aggregates} == set(export.POLICIES)
    assert all(int(row["n_defined"]) in (0, 3) for row in aggregates)
    assert len(reports) == len(list((output / "cases").glob("*/qa.json"))) == 132
    assert manifest["scratch"]["observed_peak_case_bytes"] <= 1024**2
    assert not list(scratch.iterdir()) and not list(output.rglob("*.parquet"))
    assert reader.objects == before and (state_dir / "suite-state.json").read_bytes() == state_bytes


@pytest.mark.parametrize("mutation", ["zia", "seed", "policy", "cached", "campaign_count", "active"])
def test_selected_export_refuses_incomplete_or_expanded_scope_before_any_s3_read(tmp_path, mutation):
    reader, state_dir, _, coordinates = fixture_suite(tmp_path, full_selection=True)
    state = export.read(state_dir / "suite-state.json")
    status = export.read(state_dir / "latest-status.json")
    if mutation == "zia":
        state["datasets"]["fraud"] = {}
    elif mutation == "seed":
        state["datasets"]["electricity"]["base"]["request"]["matrix"]["seeds"] = [42, 43]
        state["contract"]["requests"]["electricity"] = state["datasets"]["electricity"]["base"]["request"]
    elif mutation == "policy":
        coordinates.remove(("electricity", "oracle", 44))
    elif mutation == "cached":
        state["datasets"]["electricity"]["cached"]["request"]["matrix"]["metric_cache_sources"].pop("44")
    elif mutation == "campaign_count":
        status["campaigns"].pop(next(iter(status["campaigns"])))
    else:
        status["batch"]["active"] = 1
    export.save(state_dir / "suite-state.json", state)
    export.save(state_dir / "latest-status.json", status)
    state_bytes = (state_dir / "suite-state.json").read_bytes()
    with pytest.raises(ValueError):
        export.export_evidence(
            state_dir,
            tmp_path / "evidence",
            tmp_path / "scratch",
            reader,
            expected_rows={dataset: 24 for dataset in export.DATASET_ROWS},
            coordinates=coordinates,
        )
    assert reader.reads == [] and (state_dir / "suite-state.json").read_bytes() == state_bytes
    assert not (tmp_path / "scratch").exists()


def test_lossless_gzip_shards_preserve_nulls_vectors_and_exact_byte_ranges(tmp_path):
    source = tmp_path / "source.json"
    payload = export.canonical({"vector": [None, False, True, *range(12000)]})
    source.write_bytes(payload)
    output = tmp_path / "export"
    blob = export.gzip_blob(source, output, "report.full.json", max_file_bytes=4096)
    assert len(blob["parts"]) > 2
    assert all(p["bytes"] <= 4096 for p in blob["parts"])
    work = tmp_path / "scratch"
    work.mkdir()
    export.restore_blob(blob, output, work / "report.json", export.ScratchBudget(work, 1024**2))
    assert (work / "report.json").read_bytes() == payload
    part = output / blob["parts"][0]["path"]
    part.write_bytes(b"modified")
    with pytest.raises(ValueError, match="shard changed"):
        export.restore_blob(blob, output, work / "again.json", export.ScratchBudget(work, 1024**2))


def test_scratch_guard_refuses_an_object_before_downloading_and_rejects_version_changes(tmp_path):
    reader = FakeS3()
    uri = "s3://fixture/campaigns/test/object.json"
    reader.add(uri, b"12345")
    budget = export.ScratchBudget(tmp_path, 4)
    with pytest.raises(ValueError, match="scratch budget"):
        export.download_verified(reader, uri, tmp_path / "object", budget)
    assert reader.reads == []
    budget = export.ScratchBudget(tmp_path, 100)
    original = reader.download

    def altered(uri, destination, head):
        result = original(uri, destination, head)
        return {**result, "VersionId": "different-version"}

    reader.download = altered
    with pytest.raises(ValueError, match="version differs"):
        export.download_verified(reader, uri, tmp_path / "object", budget)


def test_coordinator_final_mutation_and_unfinished_campaign_block_export(tmp_path):
    reader, state_dir, _, coordinates = fixture_suite(tmp_path)
    status = export.read(state_dir / "latest-status.json")
    status["complete"] = False
    export.save(state_dir / "latest-status.json", status)
    with pytest.raises(ValueError, match="must finish"):
        export.export_evidence(
            state_dir,
            tmp_path / "evidence",
            tmp_path / "scratch",
            reader,
            expected_rows={"fixture": 24},
            coordinates=coordinates,
        )
    assert reader.reads == []
    status["complete"] = True
    export.save(state_dir / "latest-status.json", status)
    uri = next(uri for uri in reader.objects if uri.endswith("/final.json"))
    final = json.loads(reader.objects[uri])
    final["tracking"]["complete"] = False
    reader.add(uri, final)
    with pytest.raises(ValueError, match="final manifest changed"):
        export.export_evidence(
            state_dir,
            tmp_path / "evidence",
            tmp_path / "scratch",
            reader,
            expected_rows={"fixture": 24},
            coordinates=coordinates,
        )


def test_live_progress_ledger_rejects_raw_rows_and_retains_only_the_native_metric_schema(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "timestamp": 1,
                "prefix": "attempts/job-1/progress",
                "values": {"rows_processed": 20.0},
            }
        )
        + "\n"
    )
    assert export.validate_live_ledger(path) == 1
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "timestamp": 1,
                "prefix": "attempts/job-1/progress",
                "values": {"raw_rows": [[1, 2, 3]]},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="scalar metric"):
        export.validate_live_ledger(path)


def test_cli_adapter_is_read_only_paginates_and_pins_a_version(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        operation = command[2]
        params = json.loads(command[command.index("--cli-input-json") + 1])
        if operation == "list-objects-v2":
            if not params.get("ContinuationToken"):
                value = {
                    "Contents": [{"Key": "campaigns/a/one", "Size": 1}],
                    "IsTruncated": True,
                    "NextContinuationToken": "next",
                }
            else:
                value = {"Contents": [{"Key": "campaigns/a/two", "Size": 1}], "IsTruncated": False}
        else:
            assert params["VersionId"] == "version-real"
            Path(command[-1]).write_bytes(b"data")
            value = {"VersionId": "version-real", "ETag": '"etag"'}
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

    reader = export.ReadOnlyS3("us-east-1", runner=runner)
    assert len(reader.list("s3://bucket/campaigns/a/")) == 2
    reader.download("s3://bucket/campaigns/a/one", tmp_path / "object", {"VersionId": "version-real"})
    with pytest.raises(ValueError, match="mutation"):
        reader._call("delete-object", {"Bucket": "bucket", "Key": "any"})
    assert [c[2] for c in calls] == ["list-objects-v2", "list-objects-v2", "get-object"]
