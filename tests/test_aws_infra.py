"""CloudFormation, IAM, ASL and CI/CD contracts of the AWS platform (offline)."""

import copy
import importlib.util
import json
import subprocess
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


def test_tagged_submission_can_authorize_the_job_queue_and_definitions():
    role = template("modules/execution.yaml")["Resources"]["StatesRole"]
    submit = next(s for s in statements(role) if s.get("Sid") == "SubmitToThisQueueOnly")
    tag = next(s for s in statements(role) if s["Action"] == "batch:TagResource")
    assert tag["Resource"] == [
        *submit["Resource"],
        {"Fn::Sub": "arn:${AWS::Partition}:batch:${AWS::Region}:${AWS::AccountId}:job/*"},
    ]
    assert tag["Condition"] == {"StringEquals": {"aws:RequestTag/cb-stack": {"Ref": "Project"}}}


def test_worker_can_list_and_register_versions_in_project_model_groups():
    """ListModelPackages/CreateModelPackage are evaluated on model-package/<group>/*."""
    role = template("modules/execution.yaml")["Resources"]["BenchmarkJobRole"]
    registry = next(
        p["Fn::If"][1] for p in role["Properties"]["Policies"] if isinstance(p, dict) and "Fn::If" in p
    )
    sid = "ListAndRegisterVersionsInProjectGroups"
    versions = next(s for s in registry["PolicyDocument"]["Statement"] if s.get("Sid") == sid)
    assert {"sagemaker:ListModelPackages", "sagemaker:CreateModelPackage"} <= set(versions["Action"])
    arns = json.dumps(versions["Resource"])
    assert "model-package/${Project}-*" in arns and "model-package-group/${Project}-*" in arns


def test_control_can_read_back_write_once_analytics_rows():
    role = template("modules/execution.yaml")["Resources"]["ControlRole"]
    read = next(s for s in statements(role) if s.get("Sid") == "ReadReleasesAndSnapshots")
    write = next(s for s in statements(role) if s.get("Sid") == "WriteCampaignStateManifestsAndAnalytics")
    assert {"Fn::Sub": "${DataBucketArn}/analytics/*"} in read["Resource"]
    assert {"Fn::Sub": "${DataBucketArn}/analytics/*"} in write["Resource"]


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
        "GitHubActionsRole",
        "CloudFormationExecutionRole",
        "StatesRole",
        "TaskExecutionRole",
        "BenchmarkJobRole",
        "PrepareJobRole",
        "AcquisitionRole",
        "ControlRole",
    } <= set(principals)
    deploy_pass = [s for s in statements(cicd["GitHubActionsRole"]) if s["Action"] == "iam:PassRole"]
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


def test_runtime_can_reuse_retained_log_groups_after_stack_deletion():
    root = template("root.yaml")
    execution = template("modules/execution.yaml")
    resources = execution["Resources"]
    reuse = "ReuseRuntimeLogGroups"
    create = "CreateRuntimeLogGroups"
    assert root["Parameters"][reuse]["Default"] == execution["Parameters"][reuse]["Default"] == "false"
    assert root["Resources"]["Execution"]["Properties"]["Parameters"][reuse] == {"Ref": reuse}
    assert execution["Conditions"][create] == {"Fn::Equals": [{"Ref": reuse}, "false"]}

    names = {}
    for group in ("JobLogGroup", "AcquisitionLogGroup"):
        resource = resources[group]
        assert resource["Condition"] == create
        assert resource["DeletionPolicy"] == "Retain"
        names[group] = {"Fn::If": [create, {"Ref": group}, resource["Properties"]["LogGroupName"]]}

    for definition in ("BenchmarkJobDefinition", "PrepareJobDefinition"):
        logging = resources[definition]["Properties"]["ContainerProperties"]["LogConfiguration"]
        assert logging["Options"]["awslogs-group"] == names["JobLogGroup"]
    assert execution["Outputs"]["JobLogGroupName"]["Value"] == names["JobLogGroup"]
    acquisition_logs = resources["AcquisitionProject"]["Properties"]["LogsConfig"]["CloudWatchLogs"]
    assert acquisition_logs["GroupName"] == names["AcquisitionLogGroup"]


def test_bucket_policy_admits_writers_outside_the_vpc():
    policy = template("persistent.yaml")["Resources"]["DataBucketPolicy"]["Properties"]["PolicyDocument"]
    assert "aws:SourceVpce" not in json.dumps(policy)
    billing = template("persistent.yaml")["Resources"]["BillingBucketPolicy"]["Properties"]["PolicyDocument"]
    assert "bcm-data-exports.amazonaws.com" in json.dumps(billing)


def test_github_actions_role_trusts_only_the_main_deploy_workflow():
    cicd = template("cicd.yaml")
    role = cicd["Resources"]["GitHubActionsRole"]["Properties"]
    (trust,) = role["AssumeRolePolicyDocument"]["Statement"]
    assert trust["Action"] == "sts:AssumeRoleWithWebIdentity"
    condition = json.dumps(trust["Condition"])
    assert "environment:${GitHubEnvironment}" in condition
    assert "deploy.yml@refs/heads/${BranchName}" in condition
    assert "StringLike" not in trust["Condition"]
    assert cicd["Parameters"]["BranchName"]["Default"] == "main"
    resources = cicd["Resources"].values()
    assert not [r for r in resources if r["Type"].startswith(("AWS::CodePipeline", "AWS::CodeBuild"))]
    text = json.dumps(role)
    assert "startExecution" not in text and "batch:SubmitJob" not in text


def _pass_role_statements(role):
    for policy in role["Properties"]["Policies"]:
        for statement in policy["PolicyDocument"]["Statement"]:
            if statement.get("Action") == "iam:PassRole":
                yield statement


def test_cfn_execution_role_can_pass_every_runtime_role():
    """Nested stacks need the execution role to pass itself; module roles go to their own services."""
    role = template("cicd.yaml")["Resources"]["CloudFormationExecutionRole"]
    assert role["Properties"]["RoleName"] == {"Fn::Sub": "${Project}-cfn-execution"}
    statements = list(_pass_role_statements(role))
    (self_pass,) = [s for s in statements if s.get("Sid") == "PassItselfToNestedStacks"]
    assert self_pass["Resource"]["Fn::Sub"].endswith(":role/${Project}/${Project}-cfn-execution")
    assert self_pass["Condition"] == {"StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}}
    assert any(r["Type"] == "AWS::CloudFormation::Stack" for r in template("root.yaml")["Resources"].values())

    (services,) = [s for s in statements if s.get("Sid") == "PassProjectRolesToTheirServices"]
    allowed = set(services["Condition"]["StringEquals"]["iam:PassedToService"])
    for module in (CFN / "modules").glob("*.yaml"):
        for name, resource in template(f"modules/{module.name}")["Resources"].items():
            if resource["Type"] != "AWS::IAM::Role":
                continue
            for trust in resource["Properties"]["AssumeRolePolicyDocument"]["Statement"]:
                principal = trust["Principal"]["Service"]
                for service in [principal] if isinstance(principal, str) else principal:
                    assert service in allowed, (module.name, name, service)


def test_cfn_execution_role_can_lookup_and_clean_up_runtime_roles_without_iam_path():
    role = template("cicd.yaml")["Resources"]["CloudFormationExecutionRole"]
    (fallback,) = [
        statement
        for statement in statements(role)
        if statement.get("Sid") == "LookupAndCleanUpRuntimeRolesByName"
    ]
    assert fallback["Effect"] == "Allow"
    assert set(fallback["Action"]) == {
        "iam:GetRole",
        "iam:GetRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:DeleteRole",
    }

    prefix = "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/"
    expected = {
        prefix + resource["Properties"]["RoleName"]["Fn::Sub"]
        for module in (CFN / "modules").glob("*.yaml")
        for resource in template(f"modules/{module.name}")["Resources"].values()
        if resource["Type"] == "AWS::IAM::Role"
    }
    assert {resource["Fn::Sub"] for resource in fallback["Resource"]} == expected


def test_workflows_deploy_from_main_and_never_acquire_data_or_start_campaigns():
    workflows = ROOT / ".github" / "workflows"
    deploy = (workflows / "deploy.yml").read_text()
    assert "branches: [main]" in deploy and "id-token: write" in deploy
    assert "build_images.sh" in deploy and "deploy_change_set.sh" in deploy
    for path in workflows.glob("*.yml"):
        text = path.read_text()
        assert "AWS_SECRET_ACCESS_KEY" not in text, path.name
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


def test_billing_can_subscribe_to_an_existing_service_anomaly_monitor():
    root = template("root.yaml")
    billing = template("modules/billing.yaml")
    param = "ExistingServiceAnomalyMonitorArn"
    create = "CreateServiceAnomalyMonitor"
    assert root["Parameters"][param]["Default"] == billing["Parameters"][param]["Default"] == ""
    assert root["Resources"]["Billing"]["Properties"]["Parameters"][param] == {"Ref": param}
    assert billing["Conditions"][create] == {"Fn::Equals": [{"Ref": param}, ""]}
    resources = billing["Resources"]
    assert resources["ServiceAnomalyMonitor"]["Condition"] == create
    assert resources["AnomalySubscription"]["Properties"]["MonitorArnList"] == [
        {"Fn::If": [create, {"Ref": "ServiceAnomalyMonitor"}, {"Ref": param}]}
    ]


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
    # Step Functions rejects a state name reused in another Map processor (DUPLICATE_STATE_NAME).
    reused = copy.deepcopy(definition)
    distributed = reused["States"]["DistributedRuns"]["ItemProcessor"]
    distributed["States"]["Route"] = distributed["States"].pop("DistributedRoute")
    path.write_text(json.dumps(reused))
    assert any("duplicate state name 'Route'" in e for e in ASL.validate_file(path))


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


def test_release_uploads_use_the_data_buckets_kms_key(monkeypatch, tmp_path):
    release = script("write_release")
    root = template("root.yaml")
    execution = template("modules/execution.yaml")
    assert root["Outputs"]["DataKeyArn"]["Value"] == {"Fn::GetAtt": "Execution.Outputs.DataKeyArn"}
    assert execution["Outputs"]["DataKeyArn"]["Value"] == {"Ref": "DataKeyArn"}
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/example"
    sha = "a" * 40
    build_output = tmp_path / "build-output.json"
    build_output.write_text("{}")
    uploads = []

    def upload(*args, check=True):
        assert args[:2] == ("s3api", "put-object")
        uploads.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(release, "aws", upload)
    monkeypatch.setattr(release, "stack_outputs", lambda _: {"DataKeyArn": key_arn})
    monkeypatch.setattr(release, "compose", lambda *_: {"images": {}})
    monkeypatch.setattr(release.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        release.sys,
        "argv",
        [
            "write_release.py",
            "--stack",
            "stack",
            "--bucket",
            "data",
            "--git-sha",
            sha,
            "--build-output",
            str(build_output),
        ],
    )
    release.main()

    assert [args[args.index("--key") + 1] for args in uploads] == [
        f"releases/{sha}/release.json",
        "releases/latest.json",
    ]
    for args in uploads:
        assert args[args.index("--server-side-encryption") + 1] == "aws:kms"
        assert args[args.index("--ssekms-key-id") + 1] == key_arn
    assert uploads[0][uploads[0].index("--if-none-match") + 1] == "*"
    assert "--if-none-match" not in uploads[1]


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
