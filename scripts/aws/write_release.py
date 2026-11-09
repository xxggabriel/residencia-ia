"""Write the release manifest after a successful deploy (stdlib + AWS CLI only).

releases/<sha>/release.json is created once with `If-None-Match: *`. A re-run of the same
commit is accepted when the existing manifest has the same images, job definition revisions
and control function; any difference fails instead of rewriting evidence. Then
releases/latest.json points to the commit. Campaigns resolve the release once at planning, so
moving `latest` never changes a campaign that already started.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile

STABLE_FIELDS = ("version", "git_sha", "stack", "region", "images", "job_definitions", "control_function")


def aws(*args, check=True):
    result = subprocess.run(["aws", *args, "--output", "json"], capture_output=True, text=True)
    if check and result.returncode:
        raise SystemExit(result.stderr)
    return result


def stack_outputs(stack: str) -> dict:
    stacks = json.loads(aws("cloudformation", "describe-stacks", "--stack-name", stack).stdout)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def compose(outputs: dict, build: dict, git_sha: str, stack: str, execution_id: str | None) -> dict:
    if outputs.get("GitSha") != git_sha:
        raise SystemExit(f"runtime stack is at {outputs.get('GitSha')}, not {git_sha}")
    images = {}
    for role, key in (("runtime", "RuntimeImage"), ("ml", "MlImage")):
        repository, digest = outputs[key].split("@")
        built = build["images"][role]
        if (repository, digest) != (built["repository"], built["digest"]):
            raise SystemExit(f"{role} image deployed {outputs[key]} differs from the build output")
        images[role] = {"repository": repository, "digest": digest, "tag": f"git-{git_sha}"}
    manifest = {
        "version": "release-v1",
        "git_sha": git_sha,
        "stack": stack,
        "region": "us-east-1",
        "images": images,
        "job_definitions": {
            "benchmark": outputs["BenchmarkJobDefinitionArn"],
            "prepare": outputs["PrepareJobDefinitionArn"],
        },
        "control_function": outputs["ControlFunctionVersionArn"],
        "pipeline_execution_id": execution_id or None,
    }
    validate(manifest)
    return manifest


def validate(manifest: dict):
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["git_sha"]):
        raise SystemExit("invalid git sha")
    for name, arn in manifest["job_definitions"].items():
        if not re.fullmatch(r"arn:aws:batch:[a-z0-9-]+:\d{12}:job-definition/[\w-]+:\d+", arn):
            raise SystemExit(f"{name} job definition must be revision-qualified: {arn}")
    if not re.fullmatch(
        r"arn:aws:lambda:[a-z0-9-]+:\d{12}:function:[\w-]+:\d+", manifest["control_function"]
    ):
        raise SystemExit("control function must be version-qualified")
    for role, image in manifest["images"].items():
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image["digest"]):
            raise SystemExit(f"{role} digest invalid")


def ecr_digest(repository: str, tag: str) -> str:
    name = repository.split("/", 1)[1]
    detail = json.loads(
        aws("ecr", "describe-images", "--repository-name", name, "--image-ids", f"imageTag={tag}").stdout
    )["imageDetails"][0]
    return detail["imageDigest"]


def put_once(bucket: str, key: str, manifest: dict, kms_key_arn: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as body:
        json.dump(manifest, body, indent=2, sort_keys=True)
    created = aws(
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        body.name,
        "--content-type",
        "application/json",
        "--server-side-encryption",
        "aws:kms",
        "--ssekms-key-id",
        kms_key_arn,
        "--if-none-match",
        "*",
        check=False,
    )
    if created.returncode == 0:
        return "created"
    if "PreconditionFailed" not in created.stderr and "412" not in created.stderr:
        raise SystemExit(created.stderr)
    with tempfile.NamedTemporaryFile(suffix=".json") as existing_file:
        aws("s3api", "get-object", "--bucket", bucket, "--key", key, existing_file.name)
        existing = json.loads(open(existing_file.name).read())
    differences = [f for f in STABLE_FIELDS if existing.get(f) != manifest.get(f)]
    if differences:
        raise SystemExit(f"release {key} already exists with different {differences}; refusing to rewrite")
    return "unchanged"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--build-output", required=True)
    parser.add_argument("--pipeline-execution-id", default="")
    args = parser.parse_args()
    build = json.loads(open(args.build_output).read())
    outputs = stack_outputs(args.stack)
    manifest = compose(outputs, build, args.git_sha, args.stack, args.pipeline_execution_id)
    kms_key_arn = outputs["DataKeyArn"]
    for role, image in manifest["images"].items():
        if ecr_digest(image["repository"], image["tag"]) != image["digest"]:
            raise SystemExit(f"ECR tag {image['tag']} of {role} does not point to {image['digest']}")
    key = f"releases/{args.git_sha}/release.json"
    status = put_once(args.bucket, key, manifest, kms_key_arn)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as pointer:
        json.dump({"git_sha": args.git_sha, "release_key": key}, pointer)
    aws(
        "s3api",
        "put-object",
        "--bucket",
        args.bucket,
        "--key",
        "releases/latest.json",
        "--body",
        pointer.name,
        "--content-type",
        "application/json",
        "--server-side-encryption",
        "aws:kms",
        "--ssekms-key-id",
        kms_key_arn,
    )
    json.dump({"release": key, "status": status, **manifest}, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
