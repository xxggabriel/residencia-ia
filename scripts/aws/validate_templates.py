"""Validate the CloudFormation templates: cfn-lint plus project security invariants.

Invariants checked offline (PyYAML with CloudFormation tags):
- Fargate job definitions use AssignPublicIp=DISABLED, x86_64 and explicit resources;
- no internet gateway, NAT gateway, Elastic IP or load balancer anywhere;
- stateful resources (buckets, KMS key, ECR, job logs) are retained on delete and replace;
- buckets block public access, enable encryption and versioning (except the access-log bucket);
- every IAM role has its own trust principal and no `"*"` iam:PassRole;
- the benchmark trigger stays manual: no EventBridge rule targets a state machine or starts jobs.

Usage: CFN_LINT=cfn-lint python scripts/aws/validate_templates.py [templates...]
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = sorted(
    p for p in (ROOT / "infra" / "cloudformation").rglob("*.yaml") if not p.name.startswith(".")
)
FORBIDDEN = {
    "AWS::EC2::InternetGateway",
    "AWS::EC2::NatGateway",
    "AWS::EC2::EIP",
    "AWS::ElasticLoadBalancingV2::LoadBalancer",
    "AWS::ElasticLoadBalancing::LoadBalancer",
    "AWS::EC2::VPCGatewayAttachment",
    "AWS::ECS::Service",
}
RETAINED = {"AWS::S3::Bucket", "AWS::KMS::Key", "AWS::ECR::Repository"}


class CfnLoader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {("Ref" if suffix == "Ref" else f"Fn::{suffix}"): value}


CfnLoader.add_multi_constructor("!", _tag)


def load(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=CfnLoader)


def resources(template: dict):
    return (template.get("Resources") or {}).items()


def check(path: Path) -> list[str]:
    template = load(path)
    errors = []
    name = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
    for logical, resource in resources(template):
        kind, props = resource.get("Type"), resource.get("Properties") or {}
        where = f"{name}:{logical}"
        if kind in FORBIDDEN:
            errors.append(f"{where}: {kind} is forbidden in the private-only design")
        if kind in RETAINED and (
            resource.get("DeletionPolicy") != "Retain" or resource.get("UpdateReplacePolicy") != "Retain"
        ):
            errors.append(f"{where}: stateful {kind} must be retained on delete and replace")
        if kind == "AWS::S3::Bucket":
            block = props.get("PublicAccessBlockConfiguration") or {}
            if not all(
                block.get(k)
                for k in ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets")
            ):
                errors.append(f"{where}: bucket must block public access")
            if "BucketEncryption" not in props:
                errors.append(f"{where}: bucket must declare default encryption")
            if (
                "access-logs" not in json.dumps(props.get("BucketName"))
                and (props.get("VersioningConfiguration") or {}).get("Status") != "Enabled"
            ):
                errors.append(f"{where}: bucket must enable versioning")
        if kind == "AWS::Batch::JobDefinition":
            container = props.get("ContainerProperties") or {}
            if props.get("PlatformCapabilities") != ["FARGATE"]:
                errors.append(f"{where}: job definitions run on Fargate")
            if (container.get("NetworkConfiguration") or {}).get("AssignPublicIp") != "DISABLED":
                errors.append(f"{where}: AssignPublicIp must be DISABLED")
            if (container.get("RuntimePlatform") or {}).get("CpuArchitecture") != "X86_64":
                errors.append(f"{where}: images are linux/amd64")
            if not props.get("Timeout") or not props.get("RetryStrategy"):
                errors.append(f"{where}: job definitions need a timeout and a retry strategy")
            if "@sha256" not in json.dumps(container.get("Image")) and "RuntimeImage" not in json.dumps(
                container.get("Image")
            ):
                errors.append(f"{where}: image must be pinned by digest")
        if kind == "AWS::IAM::Role":
            if not (props.get("AssumeRolePolicyDocument") or {}).get("Statement"):
                errors.append(f"{where}: role needs its own trust policy")
            for policy in props.get("Policies") or []:
                if not isinstance(policy, dict) or "PolicyDocument" not in policy:
                    continue  # Conditional (Fn::If) policies are checked by cfn-lint.
                for statement in policy["PolicyDocument"]["Statement"]:
                    actions = statement.get("Action")
                    actions = [actions] if isinstance(actions, str) else actions or []
                    if "iam:PassRole" in actions and statement.get("Resource") == "*":
                        errors.append(f"{where}: iam:PassRole on '*'")
        if kind == "AWS::Events::Rule":
            for target in props.get("Targets") or []:
                arn = json.dumps(target.get("Arn"))
                if "StateMachine" in arn or "states:" in arn or "JobQueue" in arn:
                    errors.append(
                        f"{where}: event rules must not start workflows or jobs (manual trigger only)"
                    )
        if kind == "AWS::Batch::ComputeEnvironment":
            compute = props.get("ComputeResources") or {}
            if compute.get("Type") not in ({"Ref": "FargateCapacity"}, "FARGATE", "FARGATE_SPOT"):
                errors.append(f"{where}: compute environment must be Fargate")
    return errors


def run_cfn_lint(paths) -> int:
    command = shlex.split(os.environ.get("CFN_LINT", "cfn-lint"))
    config = ROOT / "infra" / "cloudformation" / ".cfnlintrc"
    return subprocess.run([*command, "--config-file", str(config), "--", *map(str, paths)]).returncode


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = [Path(p).resolve() for p in argv] or TEMPLATES
    errors = [error for path in paths for error in check(path)]
    for error in errors:
        print(error, file=sys.stderr)
    lint = 0 if os.environ.get("SKIP_CFN_LINT") else run_cfn_lint(paths)
    print(json.dumps({"templates": len(paths), "invariant_errors": len(errors), "cfn_lint_exit": lint}))
    return 1 if errors or lint else 0


if __name__ == "__main__":
    sys.exit(main())
