"""Admission, immutable identities and completion checks without AWS calls."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "run_dataset_campaigns", Path(__file__).parents[1] / "scripts/aws/run_dataset_campaigns.py"
)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)
SHA = "a" * 40
HASH = "b" * 64
BUCKET = "test-bucket"
MACHINE = "arn:aws:states:us-east-1:123456789012:stateMachine:benchmark"


def receipts():
    smoke = {
        "schema": "ordered-dataset-suite-qa-v1",
        "complete": True,
        "scope": "smoke",
        "code": {"code_sha256": HASH},
        "coverage": {
            "expected_cases": 231,
            "passed_cases": 231,
            "datasets": list(suite.DATASET_ROWS),
            "policies": list(suite.POLICIES),
            "seeds": [42, 43, 44],
        },
        "checks": {
            key: True
            for key in (
                "prediction_integrity",
                "metrics_reconstructed",
                "no_model_files",
                "tracking_receipts_complete",
                "native_tracking_catalog_verified",
            )
        },
    }
    deployment = {
        "complete": True,
        "git_sha": SHA,
        "smoke_code_sha256": HASH,
        "workflow": {"databaseId": 123, "headSha": SHA, "status": "completed", "conclusion": "success"},
    }
    return smoke, deployment


def base_request(dataset="electricity"):
    return {
        "campaign_id": "suite-" + dataset[:20].replace("_", "-"),
        "dataset": dataset,
        "scope": "full",
        "release": SHA,
        "matrix": {
            "kind": "stream",
            "policies": list(suite.BASE_POLICIES),
            "seeds": [42, 43, 44],
            "artifact_mode": "metrics_only",
            "iterative_training": {"max_epochs": 100},
        },
        "max_parallel": 32,
        "queue_timeout_seconds": 86400,
    }


@pytest.mark.parametrize("field", ["passed_cases", "seeds", "policies", "datasets"])
def test_smoke_gate_rejects_partial_matrix(field):
    smoke, deployment = receipts()
    smoke["coverage"][field] = 230 if field == "passed_cases" else []
    with pytest.raises(ValueError, match="231 smoke"):
        suite.validate_smoke(smoke, deployment, SHA)


def test_smoke_gate_requires_native_metrics_and_code_binding():
    smoke, deployment = receipts()
    suite.validate_smoke(smoke, deployment, SHA)
    smoke["checks"]["native_tracking_catalog_verified"] = False
    with pytest.raises(ValueError, match="native metric"):
        suite.validate_smoke(smoke, deployment, SHA)
    smoke["checks"]["native_tracking_catalog_verified"] = True
    deployment["smoke_code_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="bind"):
        suite.validate_smoke(smoke, deployment, SHA)


def test_smoke_gate_rejects_missing_code_hash_even_if_both_receipts_omit_it():
    smoke, deployment = receipts()
    smoke["code"].pop("code_sha256")
    deployment.pop("smoke_code_sha256")
    with pytest.raises(ValueError, match="bind"):
        suite.validate_smoke(smoke, deployment, SHA)


@pytest.mark.parametrize("mutation", ["online", "seed", "epochs", "scope", "escape"])
def test_request_reader_admits_only_full_three_seed_non_online_suite(tmp_path, mutation):
    entries = []
    for dataset in suite.DATASET_ROWS:
        request = base_request(dataset)
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps({"request": request}))
        entries.append({"dataset": dataset, "campaign_id": request["campaign_id"], "file": path.name})
    (tmp_path / "index.json").write_text(json.dumps({"release": SHA, "stage": "base", "requests": entries}))
    assert len(suite.read_requests(tmp_path, SHA)) == 7
    path = tmp_path / "electricity.json"
    request = json.loads(path.read_text())
    if mutation == "online":
        request["request"]["matrix"]["policies"][0] = "online"
    elif mutation == "seed":
        request["request"]["matrix"]["seeds"] = [42]
    elif mutation == "epochs":
        request["request"]["matrix"]["iterative_training"]["max_epochs"] = 3
    elif mutation == "scope":
        request["request"]["scope"] = "smoke"
    else:
        entries[0]["file"] = "../escaped.json"
        (tmp_path / "index.json").write_text(
            json.dumps({"release": SHA, "stage": "base", "requests": entries})
        )
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        suite.read_requests(tmp_path, SHA)


def test_cached_request_requires_three_seed_artifacts_and_keeps_base_unchanged():
    base = base_request()
    before = copy.deepcopy(base)
    caches = {str(seed): {"uri": f"s3://{BUCKET}/cache/{seed}", "sha256": HASH} for seed in suite.SEEDS}
    result = suite.cached_request(base, caches)
    assert base == before and result["campaign_id"] != base["campaign_id"]
    assert len(result["campaign_id"]) <= 40 and result["matrix"]["policies"] == list(suite.CACHED_POLICIES)
    assert result["matrix"]["metric_cache_sources"] == caches
    caches.pop("44")
    with pytest.raises(ValueError, match="three"):
        suite.cached_request(base, caches)


class ExecutionClient:
    def __init__(self, request, *, existing=True, uncertain=False, different=False):
        self.request, self.existing, self.uncertain, self.different = request, existing, uncertain, different
        self.calls = []

    def call(self, service, operation, params):
        self.calls.append((service, operation, params))
        if operation == "describe-execution":
            if not self.existing:
                raise RuntimeError("ExecutionDoesNotExist")
            request = self.request if not self.different else {"different": True}
            return {"input": json.dumps({"request": request}), "status": "RUNNING"}
        assert operation == "start-execution"
        self.existing = True
        if self.uncertain:
            raise RuntimeError("transport closed after request")
        return {"executionArn": suite.execution_arn(MACHINE, self.request["campaign_id"])}


def test_existing_execution_is_reused_only_for_exact_input():
    request = base_request()
    client = ExecutionClient(request)
    assert suite.start_request(client, request, MACHINE)["reused"]
    assert [c[1] for c in client.calls] == ["describe-execution"]
    client.different = True
    with pytest.raises(ValueError, match="different request"):
        suite.start_request(client, request, MACHINE)


def test_uncertain_submission_recovers_original_identity_without_second_start():
    request = base_request()
    client = ExecutionClient(request, existing=False, uncertain=True)
    result = suite.start_request(client, request, MACHINE)
    assert result["reused"]
    assert [c[1] for c in client.calls].count("start-execution") == 1
    assert result["executionArn"].endswith(":" + request["campaign_id"])


def valid_run():
    run_id = "0" * 20
    unit = {
        "run_id": run_id,
        "contract_sha256": HASH,
        "params": {"policy": "frozen", "seed": 42},
        "contract": {"stream_source": {"prepared_sha256": HASH}},
    }
    uri = f"s3://{BUCKET}/campaigns/suite/runs/{run_id}/attempts/local-0"
    final = {
        "contract_sha256": HASH,
        "artifact_mode": "metrics_only",
        "model": None,
        "tracking": {"required": True, "complete": True},
        "labels": {"seed": "42"},
        "attempt": {"checkpoint": None},
        "outputs_uri": uri,
        "report_uri": uri + "/report.json",
        "report_sha256": HASH,
    }
    report = {
        "dataset": "electricity",
        "scope": "full",
        "policy": "frozen",
        "contract": {
            "input": {"rows": 45_312, "sha256": HASH},
            "iterative_training": {"seed": 42},
            "artifact_mode": "metrics_only",
        },
        "metrics": {
            "dataset_rows": 45_312,
            "initial_train_rows": 1000,
            "n": 39_781,
            "model_storage_bytes": 0,
        },
        "holdout": {"n": 4531},
        "model_versions": [{"file": None}],
        "training_history": [],
    }
    return unit, final, report


class ReportClient:
    def __init__(self, report, *, forbidden=False):
        self.report, self.forbidden = report, forbidden
        self.calls = []

    def verified_json(self, bucket, key, sha256):
        return self.report

    def call(self, service, operation, params):
        self.calls.append((service, operation, params))
        assert operation == "list-objects-v2"
        if "ContinuationToken" not in params:
            return {"Contents": [{"Key": params["Prefix"] + "report.json"}], "NextContinuationToken": "next"}
        key = "models/state.json" if self.forbidden else "predictions/part-0001.parquet"
        return {"Contents": [{"Key": params["Prefix"] + key}]}


@pytest.mark.parametrize("mutation", ["rows", "source", "seed", "weights", "checkpoint", "tracking"])
def test_final_evidence_checks_coverage_source_seed_models_and_native_receipt(mutation):
    unit, final, report = valid_run()
    client = ReportClient(report)
    assert suite.verify_final(client, BUCKET, unit, final, "electricity")["artifact_objects"] == 2
    if mutation == "rows":
        report["metrics"]["n"] -= 1
    elif mutation == "source":
        report["contract"]["input"]["sha256"] = "c" * 64
    elif mutation == "seed":
        report["contract"]["iterative_training"]["seed"] = 43
    elif mutation == "weights":
        client.forbidden = True
    elif mutation == "checkpoint":
        final["attempt"]["checkpoint"] = {"uri": "s3://test/checkpoint"}
    else:
        final["tracking"]["complete"] = False
    with pytest.raises(ValueError):
        suite.verify_final(client, BUCKET, unit, final, "electricity")


def test_pagination_does_not_silently_drop_later_artifacts():
    _, _, report = valid_run()
    client = ReportClient(report, forbidden=True)
    rows = suite.all_items(
        client,
        "s3api",
        "list-objects-v2",
        {"Prefix": "p/"},
        "Contents",
        input_token="ContinuationToken",
        output_token="NextContinuationToken",
    )
    assert len(rows) == 2 and rows[-1]["Key"] == "p/models/state.json"


def test_execution_success_without_complete_summary_is_not_accepted():
    request = base_request()
    client = ExecutionClient(request)
    client.call = lambda *args: {"input": json.dumps({"request": request}), "status": "SUCCEEDED"}
    client.optional_json = lambda *args: None
    # A workflow may briefly have no plan while RUNNING; SUCCEEDED without one is invalid.
    with pytest.raises(ValueError, match="without.*plan"):
        suite.inspect_campaign(client, BUCKET, {"request": request, "executionArn": "x"}, "electricity", {})


class RuntimeClient:
    def __init__(self):
        self.definition = "arn:aws:batch:us-east-1:123456789012:job-definition/benchmark:1"
        self.image = "repo@sha256:" + HASH
        self.resources = [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "4096"}]
        self.compute_type, self.max_vcpus, self.quota = "FARGATE_SPOT", 64, 64
        self.queue_actions = [
            {"action": "CANCEL", "state": "RUNNABLE", "maxTimeSeconds": 86400, "reason": reason}
            for reason in sorted(suite.QUEUE_WAIT_REASONS)
        ]
        self.external = False

    def json_object(self, bucket, key):
        return {
            "git_sha": SHA,
            "benchmark_compute": "batch-fargate-spot",
            "job_definitions": {"benchmark": self.definition},
            "images": {"runtime": {"repository": "repo", "digest": "sha256:" + HASH}},
        }

    def call(self, service, operation, params):
        if operation == "describe-stacks":
            return {
                "Stacks": [
                    {
                        "StackStatus": "UPDATE_COMPLETE",
                        "StackId": "stack",
                        "Outputs": [
                            {"OutputKey": k, "OutputValue": v}
                            for k, v in {
                                "GitSha": SHA,
                                "BenchmarkJobDefinitionArn": self.definition,
                                "JobQueueArn": "queue",
                                "BenchmarkStateMachineArn": MACHINE,
                            }.items()
                        ],
                    }
                ]
            }
        if operation == "describe-job-definitions":
            assert params == {"jobDefinitions": [self.definition]}
            return {
                "jobDefinitions": [
                    {
                        "status": "ACTIVE",
                        "platformCapabilities": ["FARGATE"],
                        "containerProperties": {"image": self.image, "resourceRequirements": self.resources},
                    }
                ]
            }
        if operation == "describe-job-queues":
            assert params == {"jobQueues": ["queue"]}
            return {
                "jobQueues": [
                    {
                        "state": "ENABLED",
                        "status": "VALID",
                        "jobStateTimeLimitActions": self.queue_actions,
                        "computeEnvironmentOrder": [{"computeEnvironment": "compute"}],
                    }
                ]
            }
        if operation == "describe-compute-environments":
            assert params == {"computeEnvironments": ["compute"]}
            return {
                "computeEnvironments": [
                    {
                        "state": "ENABLED",
                        "status": "VALID",
                        "ecsClusterArn": "own",
                        "computeResources": {"type": self.compute_type, "maxvCpus": self.max_vcpus},
                    }
                ]
            }
        if operation == "list-service-quotas":
            return {"Quotas": [{"QuotaCode": "L-36FBB829", "Value": self.quota}]}
        if operation == "list-clusters":
            return {"clusterArns": ["external"] if self.external else []}
        if operation == "list-tasks":
            return {"taskArns": ["task"]}
        if operation == "describe-tasks":
            return {
                "tasks": [
                    {
                        "launchType": "FARGATE",
                        "lastStatus": "RUNNING",
                        "taskArn": "task",
                        "cpu": "2048",
                        "capacityProviderName": "FARGATE_SPOT",
                    }
                ]
            }
        raise AssertionError((service, operation, params))


@pytest.mark.parametrize(
    "mutation", ["cpu", "memory", "image", "compute_type", "max_vcpus", "quota", "external"]
)
def test_preflight_checks_physical_resources_quota_image_and_regional_usage(mutation):
    client = RuntimeClient()
    assert suite.preflight(client, BUCKET, "project", SHA)["resources"] == {"VCPU": "2", "MEMORY": "4096"}
    if mutation == "cpu":
        client.resources[0]["value"] = "4"
    elif mutation == "memory":
        client.resources[1]["value"] = "16384"
    elif mutation == "image":
        client.image = "repo@sha256:" + "c" * 64
    elif mutation == "compute_type":
        client.compute_type = "FARGATE"
    elif mutation == "max_vcpus":
        client.max_vcpus = 128
    elif mutation == "quota":
        client.quota = 32
    else:
        client.external = True
    with pytest.raises(ValueError):
        suite.preflight(client, BUCKET, "project", SHA)


def test_status_does_not_start_cached_stage_without_all_three_cara_caches(monkeypatch):
    base = base_request()
    state = {
        "bucket": BUCKET,
        "queue": "queue",
        "state_machine": MACHINE,
        "datasets": {"electricity": {"base": {"request": base, "executionArn": "arn"}}},
    }
    caches = {"42": {"uri": "uri", "sha256": HASH}, "43": {"uri": "uri2", "sha256": HASH}}
    monkeypatch.setattr(
        suite,
        "inspect_campaign",
        lambda *args: {
            "matrix_caches": caches,
            "complete": False,
            "expected_units": 21,
            "verified_units": 2,
        },
    )
    monkeypatch.setattr(suite, "batch_snapshot", lambda *args: {"active": 0})
    calls = []
    monkeypatch.setattr(suite, "start_request", lambda *args: calls.append(args) or {"executionArn": "new"})
    client = RuntimeClient()
    assert not suite.status(client, state, advance=True)["complete"]
    assert calls == [] and "cached" not in state["datasets"]["electricity"]
    caches["44"] = {"uri": "uri3", "sha256": HASH}
    suite.status(client, state, advance=True)
    assert len(calls) == 1 and "cached" in state["datasets"]["electricity"]


@pytest.mark.parametrize("reason", sorted(suite.QUEUE_WAIT_REASONS))
def test_preflight_refuses_each_queue_cancellation_before_full_wait(reason):
    client = RuntimeClient()
    next(action for action in client.queue_actions if action["reason"] == reason)["maxTimeSeconds"] = 3600
    with pytest.raises(ValueError, match="at least 86400"):
        suite.preflight(client, BUCKET, "project", SHA)


def test_preflight_records_real_queue_actions_at_full_wait():
    client = RuntimeClient()
    result = suite.preflight(client, BUCKET, "project", SHA)
    assert result["queue_admission"] == {
        "minimum_queue_wait_seconds": 86400,
        "job_state_time_limit_actions": client.queue_actions,
    }


@pytest.mark.parametrize(
    "mutation", ["absent", "missing_reason", "wrong_state", "duplicate", "additional_short"]
)
def test_queue_time_limit_contract_fails_closed_for_missing_or_ambiguous_actions(mutation):
    client = RuntimeClient()
    if mutation == "absent":
        client.queue_actions.clear()
    elif mutation == "missing_reason":
        client.queue_actions.pop()
    elif mutation == "wrong_state":
        client.queue_actions[0]["state"] = "RUNNING"
    elif mutation == "duplicate":
        client.queue_actions.append(dict(client.queue_actions[0]))
    else:
        client.queue_actions.append(
            {
                "action": "CANCEL",
                "state": "RUNNABLE",
                "reason": "OTHER:CAPACITY",
                "maxTimeSeconds": 3600,
            }
        )
    with pytest.raises(ValueError):
        suite.preflight(client, BUCKET, "project", SHA)


def test_advance_rechecks_live_queue_before_starting_any_campaign(monkeypatch):
    client = RuntimeClient()
    client.queue_actions[0]["maxTimeSeconds"] = 3600
    state = {"queue": "queue", "datasets": {}}
    starts = []
    monkeypatch.setattr(suite, "start_request", lambda *args: starts.append(args))
    with pytest.raises(ValueError, match="at least 86400"):
        suite.status(client, state, advance=True)
    assert starts == []
