"""CloudFormation, IAM, ASL and CI/CD contracts of the AWS platform (offline)."""

import copy
import importlib.util
import json
import os
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
    assert (params["BenchmarkVcpus"]["Default"], params["BenchmarkMemoryMiB"]["Default"]) == ("2", 2048)
    assert (root["BenchmarkVcpus"]["Default"], root["BenchmarkMemoryMiB"]["Default"]) == ("2", 2048)
    benchmark = resources["BenchmarkJobDefinition"]["Properties"]["ContainerProperties"]
    assert benchmark["Command"][:4] == ["continual-bench", "aws", "run-unit", "--unit-uri"]
    assert benchmark["User"] == "1000" and benchmark["NetworkConfiguration"]["AssignPublicIp"] == "DISABLED"
    benchmark_env = {item["Name"]: item["Value"] for item in benchmark["Environment"]}
    assert benchmark_env["OMP_NUM_THREADS"] == "2"
    assert benchmark_env["OPENBLAS_NUM_THREADS"] == "2"
    assert benchmark_env["MKL_NUM_THREADS"] == "2"
    assert benchmark_env["BENCH_BENCHMARK_VCPUS"] == {"Ref": "BenchmarkVcpus"}
    prepare = resources["PrepareJobDefinition"]["Properties"]["ContainerProperties"]
    assert prepare["ResourceRequirements"] != benchmark["ResourceRequirements"]
    queue = resources["JobQueue"]["Properties"]
    assert {a["State"] for a in queue["JobStateTimeLimitActions"]} == {"RUNNABLE"}
    assert resources["ComputeEnvironment"]["Properties"]["ComputeResources"]["Type"] == {
        "Ref": "FargateCapacity"
    }
    assert "AWS::ECS::Service" not in {r["Type"] for r in resources.values()}


def test_release_revisions_survive_runtime_updates():
    resources = template("modules/execution.yaml")["Resources"]
    for name in ("BenchmarkJobDefinition", "PrepareJobDefinition"):
        assert resources[name]["Properties"]["ResourceRetentionPolicy"]["SkipDeregisterOnUpdate"] is True
    assert resources["ControlVersion"]["UpdateReplacePolicy"] == "Retain"


def test_distributed_map_writes_to_the_encrypted_artifacts_bucket():
    resources = template("modules/execution.yaml")["Resources"]
    substitutions = resources["BenchmarkStateMachine"]["Properties"]["DefinitionSubstitutions"]
    assert substitutions["MapResultsBucket"] == {"Ref": "ArtifactsBucketName"}
    definition = json.loads((ROOT / "infra" / "statemachines" / "benchmark.asl.json").read_text())
    writer = definition["States"]["DistributedRuns"]["ResultWriter"]["Parameters"]
    assert writer["Bucket"] == "${MapResultsBucket}" and "map-results/" in writer["Prefix.$"]
    role = list(statements(resources["StatesRole"]))
    write = next(s for s in role if s.get("Sid") == "WriteDistributedMapResults")
    assert write["Resource"]["Fn::Sub"].endswith("${ArtifactsBucketName}/map-results/*")


def test_dataset_launcher_builds_the_execution_name(tmp_path, monkeypatch):
    aws = tmp_path / "aws"
    aws.write_text('#!/usr/bin/env bash\ncase "$1" in\ncloudformation) echo arn:dataset;;\n'
                   'stepfunctions) printf "%s\\n" "$@";;\nesac\n')
    aws.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', os.defpath)}")
    output = subprocess.check_output(
        ["bash", "scripts/aws/start_dataset.sh", "infra/examples/dataset-social-smoke.json", "retry1"],
        cwd=ROOT, text=True,
    )
    assert "social-smoke-download--retry1" in output.splitlines()


def test_worker_roles_reach_s3_only_through_the_gateway_endpoint():
    resources = template("modules/execution.yaml")["Resources"]
    for name in ("BenchmarkJobRole", "PrepareJobRole"):
        for statement in statements(resources[name]):
            actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
            if statement.get("Sid") == "ReadModelArchiveForRegistration":
                assert name == "BenchmarkJobRole" and actions == ["s3:GetObject"]
                assert statement["Condition"] == {
                    "StringEquals": {"aws:CalledViaFirst": "sagemaker.amazonaws.com"}
                }
                assert statement["Resource"]["Fn::Sub"].endswith("/attempts/*/model/model.tar.gz")
            elif any(a.startswith("s3:") for a in actions):
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


def test_model_registration_reads_only_the_project_inference_repository():
    execution = template("modules/execution.yaml")
    role = execution["Resources"]["BenchmarkJobRole"]
    registry = next(
        policy["Fn::If"] for policy in role["Properties"]["Policies"] if "Fn::If" in policy
    )
    assert registry[0] == "ModelRegistry"
    image = next(
        statement for statement in registry[1]["PolicyDocument"]["Statement"]
        if statement.get("Sid") == "ReadInferenceImageForRegistration"
    )
    assert set(image["Action"]) == {"ecr:BatchGetImage", "ecr:DescribeImages"}
    assert image["Resource"] == {"Ref": "MlRepositoryArn"}
    root = template("root.yaml")["Resources"]["Execution"]
    assert root["Properties"]["Parameters"]["MlRepositoryArn"] == {
        "Fn::ImportValue": {"Fn::Sub": "${Project}-MlRepositoryArn"}
    }


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


def test_pipeline_triggers_on_push_to_main_only():
    pipeline = template("cicd.yaml")["Resources"]["Pipeline"]["Properties"]
    assert pipeline["PipelineType"] == "V2"
    assert pipeline["ExecutionMode"] == "SUPERSEDED"
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


def test_sagemaker_experiments_iam_and_vpce_permissions():
    required_actions = {
        "sagemaker:CreateExperiment",
        "sagemaker:DescribeExperiment",
        "sagemaker:UpdateExperiment",
        "sagemaker:CreateTrial",
        "sagemaker:DescribeTrial",
        "sagemaker:UpdateTrial",
        "sagemaker:CreateTrialComponent",
        "sagemaker:DescribeTrialComponent",
        "sagemaker:UpdateTrialComponent",
        "sagemaker:AssociateTrialComponent",
        "sagemaker:BatchPutMetrics",
        "sagemaker:AddTags",
    }
    expected_resources = {
        "arn:${AWS::Partition}:sagemaker:${AWS::Region}:${AWS::AccountId}:experiment/${Project}-*",
        "arn:${AWS::Partition}:sagemaker:${AWS::Region}:${AWS::AccountId}:experiment-trial/${Project}-*",
        "arn:${AWS::Partition}:sagemaker:${AWS::Region}:${AWS::AccountId}:experiment-trial-component/${Project}-*",
    }
    execution_role = template("modules/execution.yaml")["Resources"]["BenchmarkJobRole"]
    ml_role = template("modules/ml.yaml")["Resources"]["SageMakerRole"]
    for role in (execution_role, ml_role):
        exp_stmts = [
            s
            for s in statements(role)
            if isinstance(s.get("Action"), list) and "sagemaker:BatchPutMetrics" in s["Action"]
        ]
        assert exp_stmts, f"Missing SageMaker Experiments statement in {role['Properties']['RoleName']}"
        stmt = exp_stmts[0]
        assert required_actions <= set(stmt["Action"])
        subs = {r["Fn::Sub"] for r in stmt["Resource"]}
        assert subs == expected_resources

    vpce_stmt = template("modules/network.yaml")["Resources"]["SageMakerApiEndpoint"]["Properties"][
        "PolicyDocument"
    ]["Statement"][0]
    assert required_actions <= set(vpce_stmt["Action"])


def test_sagemaker_mlflow_app_is_serverless_and_wired_into_vpc_and_batch():
    execution = template("modules/execution.yaml")
    network = template("modules/network.yaml")
    root = template("root.yaml")

    app = execution["Resources"]["MlflowApp"]
    assert app["Type"] == "AWS::SageMaker::MlflowApp"
    assert app["Condition"] == "ModelRegistry"
    assert app["Properties"]["ArtifactStoreUri"] == {"Fn::Sub": "s3://${DataBucketName}/mlflow"}
    assert app["Properties"]["ModelRegistrationMode"] == "AutoModelRegistrationDisabled"

    role = execution["Resources"]["MlflowAppRole"]
    assert role["Condition"] == "ModelRegistry"
    assert role["Properties"]["RoleName"] == {"Fn::Sub": "${Project}-mlflow"}

    env = {
        item["Name"]: item["Value"]
        for item in execution["Resources"]["BenchmarkJobDefinition"]["Properties"]["ContainerProperties"][
            "Environment"
        ]
    }
    assert env["MLFLOW_TRACKING_URI"] == {
        "Fn::If": ["ModelRegistry", {"Fn::GetAtt": "MlflowApp.Arn"}, ""]
    }
    assert root["Outputs"]["MlflowAppArn"]["Value"] == {"Fn::GetAtt": "Execution.Outputs.MlflowAppArn"}

    mlflow_vpce = network["Resources"]["SageMakerMlflowEndpoint"]
    assert mlflow_vpce["Condition"] == "ModelRegistry"
    assert mlflow_vpce["Properties"]["ServiceName"] == {"Fn::Sub": "aws.sagemaker.${AWS::Region}.mlflow"}
    assert mlflow_vpce["Properties"]["PrivateDnsEnabled"] is True


def test_timeouts_are_standardized_to_14400_seconds():
    from continual_bench.application.cloud.contracts import CampaignRequest

    execution_params = template("modules/execution.yaml")["Parameters"]
    root_params = template("root.yaml")["Parameters"]
    assert execution_params["BenchmarkTimeoutSeconds"]["Default"] == 14400
    assert execution_params["PrepareTimeoutSeconds"]["Default"] == 14400
    assert root_params["BenchmarkTimeoutSeconds"]["Default"] == 14400
    assert root_params["PrepareTimeoutSeconds"]["Default"] == 14400
    req = CampaignRequest(
        campaign_id="c1",
        matrix={"kind": "experiment_matrix", "path": "configs/matrices/binary-comparison.yaml"},
    )
    assert req.run_timeout_seconds == 14400

    for profile_name in ("training", "tuning", "processing"):
        profile = json.loads((ROOT / "infra" / "sagemaker" / "profiles" / f"{profile_name}.json").read_text())
        assert profile["max_runtime_seconds"] == 14400

    dataset_asl = json.loads((ROOT / "infra" / "statemachines" / "dataset.asl.json").read_text())
    assert dataset_asl["States"]["Prepare"]["TimeoutSeconds"] == 14400


def test_deploy_change_set_skips_when_empty_changeset(tmp_path):
    mock_aws = tmp_path / "aws"
    log_file = tmp_path / "aws_calls.log"
    mock_aws.write_text(f"""#!/usr/bin/env python3
import sys

args = sys.argv[1:]
with open({repr(str(log_file))}, "a") as f:
    f.write(" ".join(args) + "\\n")

if "describe-stacks" in args:
    print("UPDATE_COMPLETE")
    sys.exit(0)
if "create-change-set" in args:
    sys.exit(0)
if "wait" in args and "change-set-create-complete" in args:
    sys.exit(255)
if "describe-change-set" in args and "StatusReason" in args:
    print("The submitted information didn't contain changes. "
          "Submit different information to create a change set.")
    sys.exit(0)
if "delete-change-set" in args:
    sys.exit(0)
sys.exit(0)
""")
    mock_aws.chmod(0o755)

    template_file = tmp_path / "root.yaml"
    template_file.write_text("AWSTemplateFormatVersion: '2010-09-09'\\nResources: {}\\n")
    params_file = tmp_path / "params.json"
    params_file.write_text("[]")

    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "aws" / "deploy_change_set.sh"),
            "my-stack",
            str(template_file),
            str(params_file),
            "arn:aws:iam::123456789012:role/exec",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "no changes for" in proc.stdout
    calls = log_file.read_text()
    assert "delete-change-set" in calls
    assert "execute-change-set" not in calls


def test_deploy_change_set_executes_nested_stack_changes(tmp_path):
    mock_aws = tmp_path / "aws"
    log_file = tmp_path / "aws_calls.log"
    mock_aws.write_text(f"""#!/usr/bin/env python3
import sys

args = sys.argv[1:]
with open({repr(str(log_file))}, "a") as f:
    f.write(" ".join(args) + "\\n")

if "describe-stacks" in args:
    print("UPDATE_COMPLETE")
    sys.exit(0)
if "create-change-set" in args:
    sys.exit(0)
if "wait" in args and "change-set-create-complete" in args:
    sys.exit(0)
if "describe-change-set" in args and "length(Changes)" in " ".join(args):
    print("1")
    sys.exit(0)
if "describe-change-set" in args and "Changes[]." in " ".join(args):
    print("----------------------\\n| DescribeChangeSet |\\n----------------------")
    sys.exit(0)
if "execute-change-set" in args:
    sys.exit(0)
if "wait" in args and "stack-update-complete" in args:
    sys.exit(0)
sys.exit(0)
""")
    mock_aws.chmod(0o755)

    template_file = tmp_path / "root.yaml"
    template_file.write_text("AWSTemplateFormatVersion: '2010-09-09'\\nResources: {}\\n")
    params_file = tmp_path / "params.json"
    params_file.write_text("[]")

    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "aws" / "deploy_change_set.sh"),
            "my-stack",
            str(template_file),
            str(params_file),
            "arn:aws:iam::123456789012:role/exec",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "deployed" in proc.stdout
    calls = log_file.read_text()
    assert "execute-change-set" in calls
    assert "wait stack-update-complete" in calls


def test_deploy_change_set_skips_when_num_changes_zero(tmp_path):
    mock_aws = tmp_path / "aws"
    log_file = tmp_path / "aws_calls.log"
    mock_aws.write_text(f"""#!/usr/bin/env python3
import sys

args = sys.argv[1:]
with open({repr(str(log_file))}, "a") as f:
    f.write(" ".join(args) + "\\n")

if "describe-stacks" in args:
    print("UPDATE_COMPLETE")
    sys.exit(0)
if "create-change-set" in args:
    sys.exit(0)
if "wait" in args and "change-set-create-complete" in args:
    sys.exit(0)
if "describe-change-set" in args and "length(Changes)" in " ".join(args):
    print("0")
    sys.exit(0)
if "delete-change-set" in args:
    sys.exit(0)
sys.exit(0)
""")
    mock_aws.chmod(0o755)

    template_file = tmp_path / "root.yaml"
    template_file.write_text("AWSTemplateFormatVersion: '2010-09-09'\\nResources: {}\\n")
    params_file = tmp_path / "params.json"
    params_file.write_text("[]")

    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "aws" / "deploy_change_set.sh"),
            "my-stack",
            str(template_file),
            str(params_file),
            "arn:aws:iam::123456789012:role/exec",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "skipping execution" in proc.stdout
    calls = log_file.read_text()
    assert "delete-change-set" in calls
    assert "execute-change-set" not in calls


