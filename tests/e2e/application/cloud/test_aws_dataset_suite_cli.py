"""Admission, immutable identities and completion checks without AWS calls."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "run_dataset_campaigns", Path(__file__).parents[4] / "scripts/aws/run_dataset_campaigns.py"
)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)
SHA = "a" * 40
HASH = "b" * 64
BUCKET = "test-bucket"
MACHINE = "arn:aws:states:us-east-1:123456789012:stateMachine:benchmark"


def receipts(datasets=None):
    datasets = list(suite.DATASET_ROWS) if datasets is None else list(datasets)
    cases = [
        {
            "case_id": f"{dataset}/{policy}/{seed}",
            "dataset": dataset,
            "policy": policy,
            "seed": seed,
            "complete": True,
            "native_tracking": {"complete": True},
        }
        for dataset in datasets
        for policy in suite.POLICIES
        for seed in suite.SEEDS
    ]
    smoke = {
        "schema": "ordered-dataset-suite-qa-v1",
        "complete": True,
        "scope": "smoke",
        "code": {"code_sha256": HASH},
        "coverage": {
            "expected_cases": len(cases),
            "passed_cases": len(cases),
            "datasets": datasets,
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
                "native_publication_markers_verified",
                "fit_seeds_verified",
            )
        },
        "cases": cases,
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


def base_state():
    state = {
        "bucket": BUCKET,
        "release": SHA,
        "queue": "queue",
        "state_machine": MACHINE,
        "datasets": {
            dataset: {"base": {"request": base_request(dataset), "executionArn": "arn-" + dataset}}
            for dataset in suite.DATASET_ROWS
        },
    }
    state["contract"] = {
        "requests": {dataset: stages["base"]["request"] for dataset, stages in state["datasets"].items()}
    }
    return state


def write_request_index(directory, datasets=None):
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for dataset in suite.DATASET_ROWS if datasets is None else datasets:
        request = base_request(dataset)
        path = directory / f"{dataset}.json"
        suite.save_json(path, {"request": request})
        entries.append({"dataset": dataset, "campaign_id": request["campaign_id"], "file": path.name})
    suite.save_json(directory / "index.json", {"release": SHA, "stage": "base", "requests": entries})


@pytest.mark.parametrize("field", ["passed_cases", "seeds", "policies", "datasets"])
def test_smoke_gate_rejects_partial_matrix(field):
    smoke, deployment = receipts()
    smoke["coverage"][field] = 230 if field == "passed_cases" else []
    with pytest.raises(ValueError, match="declared smoke"):
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


def test_complete_seven_dataset_smoke_is_a_valid_superset_of_the_four_dataset_aws_suite():
    smoke, deployment = receipts(sorted(suite.QA_DATASETS))
    assert len(smoke["cases"]) == 231
    suite.validate_smoke(smoke, deployment, SHA)
    assert len(suite.DATASET_ROWS) * len(suite.POLICIES) * len(suite.SEEDS) == 132


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incomplete", "native", "selection"])
def test_superset_smoke_requires_every_declared_coordinate_and_native_receipt(mutation):
    smoke, deployment = receipts(sorted(suite.QA_DATASETS))
    if mutation == "missing":
        smoke["cases"].pop()
    elif mutation == "duplicate":
        smoke["cases"][-1] = copy.deepcopy(smoke["cases"][0])
    elif mutation == "incomplete":
        smoke["cases"][0]["complete"] = False
    elif mutation == "native":
        smoke["cases"][0]["native_tracking"]["complete"] = False
    else:
        smoke, deployment = receipts([dataset for dataset in suite.QA_DATASETS if dataset != "airlines"])
    with pytest.raises(ValueError, match="declared smoke"):
        suite.validate_smoke(smoke, deployment, SHA)


@pytest.mark.parametrize("dataset", ["fraud", "ecommerce", "social"])
def test_zia_requests_cannot_be_admitted_or_submitted(tmp_path, dataset):
    write_request_index(tmp_path, [*suite.DATASET_ROWS, dataset])
    with pytest.raises(ValueError, match="invalid full base"):
        suite.read_requests(tmp_path, SHA)
    client = ExecutionClient(base_request(dataset))
    with pytest.raises(ValueError, match="invalid full base"):
        suite.start_request(client, base_request(dataset), MACHINE)
    assert client.calls == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "implicit_dataset", "zia_kind", "policy"])
def test_request_index_requires_exact_explicit_stream_selection(tmp_path, mutation):
    write_request_index(tmp_path)
    index = suite.read_json(tmp_path / "index.json")
    request_path = tmp_path / "electricity.json"
    document = suite.read_json(request_path)
    if mutation == "missing":
        index["requests"].pop()
    elif mutation == "duplicate":
        index["requests"].append(index["requests"][0])
    elif mutation == "implicit_dataset":
        document["request"].pop("dataset")
        document["request"]["snapshot"] = {"dataset": "electricity", "scope": "full"}
    elif mutation == "zia_kind":
        document["request"]["matrix"]["kind"] = "zia"
    else:
        document["request"]["matrix"]["policies"].pop()
    suite.save_json(tmp_path / "index.json", index)
    suite.save_json(request_path, document)
    with pytest.raises(ValueError):
        suite.read_requests(tmp_path, SHA)


@pytest.mark.parametrize("mutation", ["online", "seed", "epochs", "scope", "escape"])
def test_request_reader_admits_only_full_three_seed_non_online_suite(tmp_path, mutation):
    entries = []
    for dataset in suite.DATASET_ROWS:
        request = base_request(dataset)
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps({"request": request}))
        entries.append({"dataset": dataset, "campaign_id": request["campaign_id"], "file": path.name})
    (tmp_path / "index.json").write_text(json.dumps({"release": SHA, "stage": "base", "requests": entries}))
    assert len(suite.read_requests(tmp_path, SHA)) == len(suite.DATASET_ROWS) == 4
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
        "params": {
            "dataset": "electricity",
            "policy": "frozen",
            "seed": 42,
            "iterative_training": {"seed": 42, "max_epochs": 100},
        },
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
            "iterative_training": {"seed": 42, "max_epochs": 100},
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
    state = base_state()
    caches = {"42": {"uri": "uri", "sha256": HASH}, "43": {"uri": "uri2", "sha256": HASH}}
    monkeypatch.setattr(
        suite,
        "inspect_campaign",
        lambda _client, _bucket, _record, dataset, _verified: {
            "matrix_caches": caches if dataset == "electricity" else {},
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
    state = base_state()
    starts = []
    monkeypatch.setattr(suite, "start_request", lambda *args: starts.append(args))
    with pytest.raises(ValueError, match="at least 86400"):
        suite.status(client, state, advance=True)
    assert starts == []


def test_historical_seven_dataset_state_is_rejected_without_changes_or_aws_calls():
    state = base_state()
    for dataset in sorted(suite.QA_DATASETS - set(suite.DATASET_ROWS)):
        state["datasets"][dataset] = {"base": {"request": base_request(dataset)}}
    before = copy.deepcopy(state)
    client = ExecutionClient(base_request())
    with pytest.raises(ValueError, match="separate state directory"):
        suite.status(client, state, advance=True)
    assert state == before and client.calls == []


@pytest.mark.parametrize("missing", ["contract", "requests"])
def test_selected_state_requires_request_contract_before_any_changes_or_aws_calls(missing):
    state = base_state()
    if missing == "contract":
        state.pop("contract")
    else:
        state["contract"].pop("requests")
    before = copy.deepcopy(state)
    client = ExecutionClient(base_request())
    with pytest.raises(ValueError, match="request contract"):
        suite.status(client, state, advance=True)
    assert state == before and client.calls == []


@pytest.mark.parametrize("active", [0, 1])
def test_complete_selected_suite_requires_eight_campaigns_132_units_and_zero_active_jobs(monkeypatch, active):
    state = base_state()
    for stages in state["datasets"].values():
        request = suite.cached_request(
            stages["base"]["request"],
            {str(seed): {"uri": f"s3://{BUCKET}/cache/{seed}", "sha256": HASH} for seed in suite.SEEDS},
        )
        stages["cached"] = {"request": request, "executionArn": "cached"}

    def inspect(_client, _bucket, record, dataset, verified):
        policies = record["request"]["matrix"]["policies"]
        for policy in policies:
            for seed in suite.SEEDS:
                run_id = f"{dataset}/{policy}/{seed}"
                verified[run_id] = {"dataset": dataset, "policy": policy, "seed": seed}
        return {
            "complete": True,
            "matrix_caches": {},
            "expected_units": len(policies) * len(suite.SEEDS),
            "verified_units": len(policies) * len(suite.SEEDS),
        }

    monkeypatch.setattr(suite, "inspect_campaign", inspect)
    monkeypatch.setattr(suite, "batch_snapshot", lambda *args: {"active": active})
    result = suite.status(None, state)
    assert result["complete"] is (active == 0)
    assert result["verified_units"] == result["expected_units"] == 132
    assert len(result["campaigns"]) == result["expected_campaigns"] == 8
    assert sum(r["expected_units"] for r in result["campaigns"].values()) == 84 + 48
    for dataset in suite.DATASET_ROWS:
        assert {
            row["seed"]
            for row in state["verified_finals"].values()
            if row["dataset"] == dataset and row["policy"] == "cara"
        } == set(suite.SEEDS)


def test_start_base_reports_only_the_selected_84_base_and_48_deferred_units(tmp_path, monkeypatch, capsys):
    requests_dir = tmp_path / "requests"
    write_request_index(requests_dir)
    smoke, deployment = receipts(sorted(suite.QA_DATASETS))
    suite.save_json(tmp_path / "smoke.json", smoke)
    suite.save_json(tmp_path / "deployment.json", deployment)
    monkeypatch.setattr(suite, "AwsCli", lambda _region: None)
    monkeypatch.setattr(suite, "verify_workflow", lambda value: value["workflow"])
    monkeypatch.setattr(suite, "preflight", lambda *args: {"queue": "queue", "state_machine": MACHINE})
    starts = []
    monkeypatch.setattr(
        suite,
        "start_request",
        lambda _client, request, _machine: starts.append(request) or {"executionArn": "arn"},
    )
    suite.main(
        [
            "start-base",
            "--state-dir",
            str(tmp_path / "state"),
            "--requests-dir",
            str(requests_dir),
            "--smoke-receipt",
            str(tmp_path / "smoke.json"),
            "--deployment-receipt",
            str(tmp_path / "deployment.json"),
            "--release",
            SHA,
            "--bucket",
            BUCKET,
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "submitted_base_campaigns": 4,
        "base_units": 84,
        "cached_units_deferred": 48,
        "complete": False,
    }
    assert {request["dataset"] for request in starts} == set(suite.DATASET_ROWS)
    state = suite.read_json(tmp_path / "state" / "suite-state.json")
    assert set(state["datasets"]) == set(state["contract"]["requests"]) == set(suite.DATASET_ROWS)


class PartialSubmissionClient:
    """Service protocol fixture with real coordinator calls and no benchmark output."""

    def __init__(self, *, fail_submission=None):
        self.fail_submission = fail_submission
        self.executions, self.submissions, self.calls = {}, [], []

    def record_execution(self, request):
        arn = suite.execution_arn(MACHINE, request["campaign_id"])
        self.executions[arn] = {
            "executionArn": arn,
            "input": suite.canonical({"request": request}),
            "status": "RUNNING",
        }
        return arn

    def call(self, service, operation, params):
        self.calls.append((service, operation, copy.deepcopy(params)))
        if operation == "describe-execution":
            if params["executionArn"] not in self.executions:
                raise RuntimeError("ExecutionDoesNotExist")
            return self.executions[params["executionArn"]]
        if operation == "start-execution":
            request = json.loads(params["input"])["request"]
            self.submissions.append(request)
            if len(self.submissions) == self.fail_submission:
                raise RuntimeError("submission unavailable before acceptance")
            return {"executionArn": self.record_execution(request)}
        if operation == "list-jobs":
            name = params["filters"][0]["values"][0]
            return {
                "jobSummaryList": [
                    {"jobId": arn, "jobName": name.removesuffix("*") + "0" * 20}
                    for arn, execution in self.executions.items()
                    if name == "cb-" + json.loads(execution["input"])["request"]["campaign_id"] + "-*"
                ]
            }
        if operation == "describe-jobs":
            return {
                "jobs": [
                    {
                        "jobId": arn,
                        "jobName": "cb-"
                        + json.loads(self.executions[arn]["input"])["request"]["campaign_id"]
                        + "-"
                        + "0" * 20,
                        "status": "RUNNING",
                        "container": {
                            "resourceRequirements": [
                                {"type": "VCPU", "value": "2"},
                                {"type": "MEMORY", "value": "4096"},
                            ],
                        },
                    }
                    for arn in params["jobs"]
                ]
            }
        if operation == "describe-job-queues":
            return RuntimeClient().call(service, operation, params)
        raise AssertionError((service, operation, params))

    def optional_json(self, bucket, key):
        self.calls.append(("s3", "optional-json", {"bucket": bucket, "key": key}))
        assert key.endswith("/plan.json")
        return None  # The running fixture jobs have not published their plans yet.


def test_cli_monitors_and_resumes_after_the_second_base_submission_fails(tmp_path, monkeypatch, capsys):
    requests_dir, state_dir = tmp_path / "requests", tmp_path / "state"
    write_request_index(requests_dir)
    smoke, deployment = receipts()
    suite.save_json(tmp_path / "smoke.json", smoke)
    suite.save_json(tmp_path / "deployment.json", deployment)
    client = PartialSubmissionClient(fail_submission=2)
    monkeypatch.setattr(suite, "AwsCli", lambda _region: client)
    monkeypatch.setattr(suite, "verify_workflow", lambda value: value["workflow"])
    monkeypatch.setattr(suite, "preflight", lambda *args: {"queue": "queue", "state_machine": MACHINE})
    arguments = [
        "start-base",
        "--state-dir",
        str(state_dir),
        "--requests-dir",
        str(requests_dir),
        "--smoke-receipt",
        str(tmp_path / "smoke.json"),
        "--deployment-receipt",
        str(tmp_path / "deployment.json"),
        "--release",
        SHA,
        "--bucket",
        BUCKET,
    ]
    with pytest.raises(RuntimeError, match="ExecutionDoesNotExist"):
        suite.main(arguments)
    saved = suite.read_json(state_dir / "suite-state.json")
    assert len(saved["datasets"]) == len(client.executions) == 1
    assert len(saved["contract"]["requests"]) == 4
    first = next(iter(saved["datasets"]))
    original_contract = copy.deepcopy(saved["contract"])
    calls_before_status, submissions_before_status = len(client.calls), len(client.submissions)

    suite.main(["status", "--state-dir", str(state_dir)])
    display = json.loads(capsys.readouterr().out)
    latest = suite.read_json(state_dir / "latest-status.json")
    assert display["complete"] is latest["complete"] is False
    assert display["expected_units"] == 132 and display["expected_campaigns"] == 8
    assert latest["expected_units"] == 132 and latest["expected_campaigns"] == 8
    assert len(latest["campaigns"]) == latest["batch"]["active"] == 1
    assert latest["batch"]["reserved_vcpus_running_or_starting"] == 2
    assert latest["batch"]["jobs"][0]["resources"] == {"VCPU": "2", "MEMORY": "4096"}
    assert len(client.submissions) == submissions_before_status
    assert [operation for _, operation, _ in client.calls[calls_before_status:]] == [
        "describe-execution",
        "optional-json",
        "list-jobs",
        "describe-jobs",
    ]

    suite.main(arguments)
    resumed = suite.read_json(state_dir / "suite-state.json")
    assert resumed["contract"] == original_contract
    assert set(resumed["datasets"]) == set(suite.DATASET_ROWS)
    assert len(client.executions) == 4 and len(client.submissions) == 5
    assert sum(request["dataset"] == first for request in client.submissions) == 1
    assert {request["dataset"] for request in client.submissions[submissions_before_status:]} == set(
        suite.DATASET_ROWS
    ) - {first}
    suite.main(arguments)
    assert len(client.executions) == 4 and len(client.submissions) == 5


@pytest.mark.parametrize("registered", [0, 1, 3])
def test_partial_status_uses_the_whole_contract_and_only_queries_registered_campaigns(registered):
    state = base_state()
    state["datasets"] = dict(list(state["datasets"].items())[:registered])
    client = PartialSubmissionClient()
    for records in state["datasets"].values():
        records["base"]["executionArn"] = client.record_execution(records["base"]["request"])
    result = suite.status(client, state)
    assert result["complete"] is False
    assert result["expected_units"] == 132 and result["expected_campaigns"] == 8
    assert len(result["campaigns"]) == result["batch"]["active"] == registered
    assert result["batch"]["reserved_vcpus_running_or_starting"] == registered * 2
    assert (
        len([operation for _, operation, _ in client.calls if operation == "describe-execution"])
        == registered
    )
    assert client.submissions == []


@pytest.mark.parametrize("registered", [0, 1, 3])
def test_partial_advance_can_start_only_cached_stages_of_registered_bases(monkeypatch, registered):
    state = base_state()
    state["datasets"] = dict(list(state["datasets"].items())[:registered])
    client = PartialSubmissionClient()
    for records in state["datasets"].values():
        records["base"]["executionArn"] = client.record_execution(records["base"]["request"])
    monkeypatch.setattr(
        suite,
        "inspect_campaign",
        lambda *args: {
            "matrix_caches": {
                str(seed): {"uri": f"s3://{BUCKET}/cache/{seed}", "sha256": HASH} for seed in suite.SEEDS
            },
            "complete": False,
            "expected_units": 21,
            "verified_units": 3,
        },
    )
    result = suite.status(client, state, advance=True)
    assert (
        result["complete"] is False and result["expected_units"] == 132 and result["expected_campaigns"] == 8
    )
    assert len(client.submissions) == registered
    assert {request["dataset"] for request in client.submissions} == set(state["datasets"])
    assert all(request["matrix"]["policies"] == list(suite.CACHED_POLICIES) for request in client.submissions)
    assert all(set(records) == {"base", "cached"} for records in state["datasets"].values())


@pytest.mark.parametrize(
    "mutation",
    ["release", "scope", "seed", "policy", "epochs", "artifact_mode", "kind", "dataset", "duplicate_id"],
)
def test_invalid_unregistered_planned_request_fails_before_aws_or_state_changes(mutation):
    state = base_state()
    state["datasets"] = dict(list(state["datasets"].items())[:1])
    missing = next(dataset for dataset in suite.DATASET_ROWS if dataset not in state["datasets"])
    request = state["contract"]["requests"][missing]
    if mutation == "release":
        request["release"] = "c" * 40
    elif mutation == "scope":
        request["scope"] = "smoke"
    elif mutation == "seed":
        request["matrix"]["seeds"] = [42, 43]
    elif mutation == "policy":
        request["matrix"]["policies"].pop()
    elif mutation == "epochs":
        request["matrix"]["iterative_training"]["max_epochs"] = 3
    elif mutation == "artifact_mode":
        request["matrix"]["artifact_mode"] = "full"
    elif mutation == "kind":
        request["matrix"]["kind"] = "zia"
    elif mutation == "dataset":
        request["dataset"] = "fraud"
    else:
        request["campaign_id"] = next(iter(state["datasets"].values()))["base"]["request"]["campaign_id"]
    before = copy.deepcopy(state)
    client = PartialSubmissionClient()
    with pytest.raises(ValueError):
        suite.status(client, state, advance=True)
    assert state == before and client.calls == [] and client.submissions == []


@pytest.mark.parametrize(
    "mutation", ["empty", "stage", "altered_base", "duplicate_cached", "missing_planned"]
)
def test_partial_submission_state_keeps_registered_and_planned_identity_guards(mutation):
    state = base_state()
    state["contract"] = copy.deepcopy(state["contract"])
    state["datasets"] = dict(list(state["datasets"].items())[:1])
    dataset, stages = next(iter(state["datasets"].items()))
    unregistered = next(name for name in suite.DATASET_ROWS if name not in state["datasets"])
    if mutation == "empty":
        state["datasets"][dataset] = {}
    elif mutation == "stage":
        stages["unexpected"] = copy.deepcopy(stages["base"])
    elif mutation == "altered_base":
        stages["base"]["request"]["max_parallel"] = 16
    elif mutation == "duplicate_cached":
        request = suite.cached_request(
            stages["base"]["request"],
            {str(seed): {"uri": f"s3://{BUCKET}/cache/{seed}", "sha256": HASH} for seed in suite.SEEDS},
        )
        request["campaign_id"] = state["contract"]["requests"][unregistered]["campaign_id"]
        stages["cached"] = {"request": request, "executionArn": "cached"}
    else:
        state["contract"]["requests"].pop(unregistered)
    before = copy.deepcopy(state)
    client = PartialSubmissionClient()
    with pytest.raises(ValueError):
        suite.status(client, state, advance=True)
    assert state == before and client.calls == [] and client.submissions == []
