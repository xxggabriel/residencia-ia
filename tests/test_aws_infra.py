"""CloudFormation, IAM, ASL and CI/CD contracts of the AWS platform (offline)."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CFN = ROOT / "infra" / "cloudformation"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "aws" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATES = script("validate_templates")
ASL = script("validate_asl")


def template(relative):
    return TEMPLATES.load(CFN / relative)


def statements(role):
    for policy in role["Properties"].get("Policies", []):
        if isinstance(policy, dict) and "PolicyDocument" in policy:
            yield from policy["PolicyDocument"]["Statement"]


@pytest.mark.parametrize("path", TEMPLATES.TEMPLATES, ids=lambda p: p.name)
def test_templates_respect_security_invariants(path):
    assert TEMPLATES.check(path) == []


def test_invariant_checker_catches_public_ips_and_unretained_state(tmp_path):
    execution = (CFN / "modules" / "execution.yaml").read_text()
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        execution.replace("AssignPublicIp: DISABLED", "AssignPublicIp: ENABLED", 1).replace(
            "\nResources:\n", "\nResources:\n  Gateway:\n    Type: AWS::EC2::InternetGateway\n", 1
        )
    )
    errors = TEMPLATES.check(broken)
    assert any("AssignPublicIp" in e for e in errors) and any("InternetGateway" in e for e in errors)
    persistent = (
        (CFN / "persistent.yaml").read_text().replace("DeletionPolicy: Retain", "DeletionPolicy: Delete", 1)
    )
    (tmp_path / "p.yaml").write_text(persistent)
    assert any("retained" in e for e in TEMPLATES.check(tmp_path / "p.yaml"))


def test_job_definitions_match_the_reference_campaign():
    resources = template("modules/execution.yaml")["Resources"]
    root = template("root.yaml")["Parameters"]
    params = template("modules/execution.yaml")["Parameters"]
    assert (params["BenchmarkVcpus"]["Default"], params["BenchmarkMemoryMiB"]["Default"]) == ("1", 2048)
    assert (root["BenchmarkVcpus"]["Default"], root["BenchmarkMemoryMiB"]["Default"]) == ("1", 2048)
    benchmark = resources["BenchmarkJobDefinition"]["Properties"]["ContainerProperties"]
    assert benchmark["Command"][:4] == ["continual-bench", "aws", "run-unit", "--unit-uri"]
    assert benchmark["User"] == "1000" and benchmark["NetworkConfiguration"]["AssignPublicIp"] == "DISABLED"
    prepare = resources["PrepareJobDefinition"]["Properties"]["ContainerProperties"]
    assert prepare["ResourceRequirements"] != benchmark["ResourceRequirements"]
    queue = resources["JobQueue"]["Properties"]
    assert {a["State"] for a in queue["JobStateTimeLimitActions"]} == {"RUNNABLE"}
    assert resources["ComputeEnvironment"]["Properties"]["ComputeResources"]["Type"] == {
        "Ref": "FargateCapacity"
    }
    assert "AWS::ECS::Service" not in {r["Type"] for r in resources.values()}


def test_worker_roles_reach_s3_only_through_the_gateway_endpoint():
    resources = template("modules/execution.yaml")["Resources"]
    for name in ("BenchmarkJobRole", "PrepareJobRole"):
        for statement in statements(resources[name]):
            actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
            if any(a.startswith("s3:") for a in actions):
                assert statement["Condition"]["StringEquals"]["aws:SourceVpce"] == {
                    "Ref": "S3GatewayEndpointId"
                }
            assert "iam:PassRole" not in actions


def test_roles_that_check_absent_keys_can_list_the_bucket():
    """HeadObject on an absent key returns 404 only with s3:ListBucket; otherwise 403 breaks idempotency."""
    resources = template("modules/execution.yaml")["Resources"]
    for name in ("BenchmarkJobRole", "PrepareJobRole", "AcquisitionRole", "ControlRole"):
        listing = [s for s in statements(resources[name]) if s["Action"] == "s3:ListBucket"]
        assert listing and "s3:prefix" not in json.dumps(listing), name


def test_only_tagged_jobs_can_be_terminated():
    resources = template("modules/execution.yaml")["Resources"]
    for role in ("ControlRole", "StatesRole"):
        for statement in statements(resources[role]):
            actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
            if "batch:TerminateJob" in actions:
                assert statement["Condition"] == {
                    "StringEquals": {"aws:ResourceTag/cb-stack": {"Ref": "Project"}}
                }
    submit = next(s for s in statements(resources["StatesRole"]) if s.get("Sid") == "SubmitToThisQueueOnly")
    assert {"Ref": "JobQueue"} in submit["Resource"] and len(submit["Resource"]) == 3


def test_roles_are_separate_and_passrole_is_scoped():
    execution = template("modules/execution.yaml")["Resources"]
    cicd = template("cicd.yaml")["Resources"]
    principals = {}
    for resources in (execution, cicd):
        for name, resource in resources.items():
            if resource["Type"] == "AWS::IAM::Role":
                principals[name] = resource["Properties"]["AssumeRolePolicyDocument"]["Statement"][0][
                    "Principal"
                ]
    assert {
        "PipelineRole",
        "TestRole",
        "BuildRole",
        "DeployRole",
        "ReleaseRole",
        "CloudFormationExecutionRole",
        "StatesRole",
        "TaskExecutionRole",
        "BenchmarkJobRole",
        "PrepareJobRole",
        "AcquisitionRole",
        "ControlRole",
    } <= set(principals)
    deploy_pass = [s for s in statements(cicd["DeployRole"]) if s["Action"] == "iam:PassRole"]
    assert deploy_pass[0]["Resource"] == {"Fn::GetAtt": "CloudFormationExecutionRole.Arn"}
    cfn_pass = next(
        s for s in statements(cicd["CloudFormationExecutionRole"]) if s["Action"] == "iam:PassRole"
    )
    assert "iam:PassedToService" in cfn_pass["Condition"]["StringEquals"]


def test_network_is_private_with_required_endpoints():
    resources = template("modules/network.yaml")["Resources"]
    services = {
        json.dumps(r["Properties"]["ServiceName"])
        for r in resources.values()
        if r["Type"] == "AWS::EC2::VPCEndpoint"
    }
    for service in (
        "s3",
        "ecr.api",
        "ecr.dkr",
        "logs",
        "sagemaker.api",
        "kinesis-streams",
        "kms",
        "secretsmanager",
        "sagemaker.featurestore-runtime",
    ):
        assert any(service in s for s in services), service
    s3 = resources["S3GatewayEndpoint"]["Properties"]["PolicyDocument"]["Statement"]
    assert any("starport-layer-bucket" in json.dumps(statement["Resource"]) for statement in s3)
    assert resources["Vpc"]["Properties"]["EnableDnsHostnames"] is True
    assert all(
        r["Properties"].get("MapPublicIpOnLaunch") is False
        for r in resources.values()
        if r["Type"] == "AWS::EC2::Subnet"
    )
    routes = [r for r in resources.values() if r["Type"] == "AWS::EC2::Route"]
    assert routes == []  # only the local route and the gateway endpoint's prefix-list route exist


def test_bucket_policy_admits_writers_outside_the_vpc():
    policy = template("persistent.yaml")["Resources"]["DataBucketPolicy"]["Properties"]["PolicyDocument"]
    assert "aws:SourceVpce" not in json.dumps(policy)
    billing = template("persistent.yaml")["Resources"]["BillingBucketPolicy"]["Properties"]["PolicyDocument"]
    assert "bcm-data-exports.amazonaws.com" in json.dumps(billing)


def test_pipeline_triggers_on_push_to_main_only():
    pipeline = template("cicd.yaml")["Resources"]["Pipeline"]["Properties"]
    assert pipeline["PipelineType"] == "V2"
    (trigger,) = pipeline["Triggers"]
    assert trigger["GitConfiguration"]["Push"] == [{"Branches": {"Includes": [{"Ref": "BranchName"}]}}]
    assert "PullRequest" not in trigger["GitConfiguration"]
    assert template("cicd.yaml")["Parameters"]["BranchName"]["Default"] == "main"
    assert not [r for r in template("cicd.yaml")["Resources"].values() if r["Type"] == "AWS::Events::Rule"]
    stages = [s["Name"] for s in pipeline["Stages"]]
    assert stages == ["Source", "Verify", "Build", "Deploy"]
    text = json.dumps(pipeline)
    assert "startExecution" not in text and "dataset" not in text.lower()


def test_pipeline_artifact_prefix_matches_codepipeline_key_layout():
    """CodePipeline writes artifacts under the first 20 characters of the pipeline name."""
    cicd = template("cicd.yaml")
    persistent = template("persistent.yaml")
    pipeline_name = cicd["Parameters"]["Project"]["Default"] + "-release"
    prefix = cicd["Parameters"]["PipelineArtifactPrefix"]["Default"]
    assert prefix == pipeline_name[:20]
    assert persistent["Parameters"]["PipelineArtifactPrefix"]["Default"] == prefix
    rules = persistent["Resources"]["ArtifactsBucket"]["Properties"]["LifecycleConfiguration"]["Rules"]
    (expiry,) = [r for r in rules if r["Id"] == "pipeline-artifacts"]
    assert expiry["Prefix"] == {"Fn::Sub": "${PipelineArtifactPrefix}/"}
    text = json.dumps(cicd["Resources"])
    assert "/pipeline/*" not in text
    for name in ("PipelineRole", "TestRole", "BuildRole", "DeployRole", "ReleaseRole"):
        assert "${Arn}/${PipelineArtifactPrefix}/*" in json.dumps(cicd["Resources"][name]), name


def test_buildspecs_never_acquire_data_or_start_campaigns():
    for path in (ROOT / "infra" / "buildspecs").glob("*.yml"):
        text = path.read_text()
        for forbidden in ("start-execution", "start_campaign", "start_dataset", "aws acquire", "submit-job"):
            assert forbidden not in text, (path.name, forbidden)


def test_optional_modules_are_explicit_and_stopped_by_default():
    root = template("root.yaml")
    for flag in (
        "EnableStreaming",
        "EnableFeatureStore",
        "EnableOnlineStore",
        "EnableMl",
        "EnableEndpoint",
        "EnableBi",
    ):
        assert root["Parameters"][flag]["Default"] == "false"
    for name in ("Streaming", "Features", "Ml", "Bi", "Billing"):
        assert root["Resources"][name]["Condition"]
    flink = template("modules/streaming.yaml")["Resources"]["WindowsApplication"]["Properties"]
    assert "RunConfiguration" not in flink
    features = template("modules/features.yaml")["Resources"]["SocialFeatureGroup"]["Properties"]
    assert features["OnlineStoreConfig"]["Fn::If"][0] == "OnlineStore"


def test_billing_export_includes_resources_and_split_costs():
    export = template("modules/billing.yaml")["Resources"]["CostAndUsageExport"]["Properties"]["Export"]
    table = export["DataQuery"]["TableConfigurations"]["COST_AND_USAGE_REPORT"]
    assert table["INCLUDE_RESOURCES"] == "TRUE" and table["INCLUDE_SPLIT_COST_ALLOCATION_DATA"] == "TRUE"
    assert "split_line_item_split_cost" in export["DataQuery"]["QueryStatement"]


def test_state_machines_validate_offline():
    for path in (ROOT / "infra" / "statemachines").glob("*.asl.json"):
        assert ASL.validate_file(path) == []


def test_asl_validator_rejects_unsafe_definitions(tmp_path):
    definition = json.loads((ROOT / "infra" / "statemachines" / "benchmark.asl.json").read_text())
    broken = copy.deepcopy(definition)
    states = broken["States"]["InlineRuns"]["ItemProcessor"]["States"]
    states["UnitFailed"] = {"Type": "Fail", "Error": "X"}
    del states["Submit"]["Parameters"]["Tags"]
    states["Reconcile"]["Retry"] = []
    broken["States"]["Orphan"] = {"Type": "Succeed"}
    path = tmp_path / "benchmark.asl.json"
    path.write_text(json.dumps(broken))
    errors = "\n".join(ASL.validate_file(path))
    assert (
        "Fail inside a Map" in errors
        and "tags" in errors
        and "transient" in errors
        and "unreachable" in errors
    )
    path.write_text(json.dumps(definition).replace("${JobQueue}", "${UnknownQueue}"))
    assert any("undefined substitutions" in e for e in ASL.validate_file(path))


def test_runtime_parameters_pin_build_outputs():
    compose = script("runtime_parameters").compose
    build = {
        "git_sha": "a" * 40,
        "control": {"key": "lambda/a/control.zip", "code_sha256": "abc="},
        "images": {
            "runtime": {"repository": "r", "digest": "sha256:" + "1" * 64},
            "ml": {"repository": "m", "digest": "sha256:" + "2" * 64},
        },
    }
    values = {p["ParameterKey"]: p["ParameterValue"] for p in compose(build, {"EnableStreaming": "true"})}
    assert values["RuntimeImage"] == "r@sha256:" + "1" * 64 and values["EnableStreaming"] == "true"
    with pytest.raises(SystemExit):
        compose(build, {"RuntimeImage": "r:latest"})


def test_release_manifest_composition_refuses_mismatched_deploys():
    release = script("write_release")
    build = {
        "images": {
            "runtime": {"repository": "1.dkr.ecr.us-east-1.amazonaws.com/r", "digest": "sha256:" + "1" * 64},
            "ml": {"repository": "1.dkr.ecr.us-east-1.amazonaws.com/m", "digest": "sha256:" + "2" * 64},
        }
    }
    outputs = {
        "GitSha": "a" * 40,
        "RuntimeImage": "1.dkr.ecr.us-east-1.amazonaws.com/r@sha256:" + "1" * 64,
        "MlImage": "1.dkr.ecr.us-east-1.amazonaws.com/m@sha256:" + "2" * 64,
        "BenchmarkJobDefinitionArn": "arn:aws:batch:us-east-1:123456789012:job-definition/cb-benchmark:4",
        "PrepareJobDefinitionArn": "arn:aws:batch:us-east-1:123456789012:job-definition/cb-prepare:4",
        "ControlFunctionVersionArn": "arn:aws:lambda:us-east-1:123456789012:function:cb-control:9",
    }
    manifest = release.compose(outputs, build, "a" * 40, "stack", "exec-1")
    from continual_bench.application.cloud.contracts import ReleaseManifest

    assert ReleaseManifest(**manifest).job_definitions["benchmark"].endswith(":4")
    with pytest.raises(SystemExit):
        release.compose({**outputs, "GitSha": "b" * 40}, build, "a" * 40, "stack", None)
    with pytest.raises(SystemExit):
        release.compose(
            {**outputs, "BenchmarkJobDefinitionArn": outputs["BenchmarkJobDefinitionArn"][:-2]},
            build,
            "a" * 40,
            "stack",
            None,
        )


def test_control_lambda_dependencies_come_from_the_aws_lock():
    subset = script("build_control_lambda").locked_subset(ROOT / "requirements-aws.lock")
    names = {line.split("==")[0] for line in subset.splitlines() if "==" in line}
    assert names == {
        "pydantic",
        "pydantic-core",
        "annotated-types",
        "typing-extensions",
        "typing-inspection",
        "pyyaml",
    }
    assert subset.count("--hash=sha256:") >= 6


def test_aws_lock_excludes_service_stack():
    lock = (ROOT / "requirements-aws.lock").read_text()
    for package in ("mlflow", "zenml", "confluent-kafka", "apache-airflow", "s3fs"):
        assert f"\n{package}==" not in lock
    for package in ("boto3", "duckdb", "pyarrow", "xgboost-cpu", "pandas", "scipy"):
        assert f"\n{package}==" in lock
