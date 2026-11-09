#!/usr/bin/env bash
# Roll the runtime stack back to a previous release by digest (images, control Lambda and
# templates of that commit), without rebuilding. Running campaigns keep their own release.
# Usage: scripts/aws/rollback.sh GIT_SHA
set -euo pipefail
GIT_SHA="${1:?git sha of a published release}"
[[ "$GIT_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "expected a full Git SHA" >&2; exit 1; }
cd "$(dirname "$0")/../.."
PROJECT="${PROJECT:-continual-bench}"
DATA="$(aws cloudformation list-exports --query "Exports[?Name=='$PROJECT-DataBucket'].Value" --output text)"
DATA_KEY_ARN="$(aws cloudformation list-exports --query "Exports[?Name=='$PROJECT-DataKeyArn'].Value" --output text)"
ARTIFACTS="$(aws cloudformation list-exports --query "Exports[?Name=='$PROJECT-ArtifactsBucket'].Value" --output text)"
ROLE="$(aws cloudformation describe-stacks --stack-name "$PROJECT-cicd" --query "Stacks[0].Outputs[?OutputKey=='CloudFormationExecutionRoleArn'].OutputValue" --output text)"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
POINTER_ETAG="$(aws s3api head-object --bucket "$DATA" --key releases/latest.json --query ETag --output text)"
aws s3 cp "s3://$DATA/releases/$GIT_SHA/release.json" "$WORK/release.json" --quiet
aws cloudformation describe-stacks --stack-name "$PROJECT-runtime" +  --query 'Stacks[0].Parameters' --output json > "$WORK/current-parameters.json"
python3 - "$WORK" "$ARTIFACTS" "$GIT_SHA" <<'PY'
import json, subprocess, sys, pathlib
work, bucket, sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
release = json.loads((work / "release.json").read_text())
if release["git_sha"] != sha:
    raise SystemExit("release manifest does not match the requested SHA")
from scripts.aws.runtime_parameters import PINNED
current = json.loads((work / "current-parameters.json").read_text())
if not current:
    raise SystemExit("cannot preserve the current runtime configuration")
(work / "overrides.json").write_text(json.dumps({
    p["ParameterKey"]: p["ParameterValue"] for p in current if p["ParameterKey"] not in PINNED
}))
listing = subprocess.run(["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", f"lambda/{sha}/",
                          "--query", "Contents[0].Key", "--output", "text"], capture_output=True, text=True, check=True)
key = listing.stdout.strip()
head = json.loads(subprocess.run(["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
                                 capture_output=True, text=True, check=True).stdout)
(work / "build-output.json").write_text(json.dumps({
    "git_sha": sha, "images": {r: {"repository": i["repository"], "digest": i["digest"]} for r, i in release["images"].items()},
    "control": {"key": key, "code_sha256": head["Metadata"]["code-sha256"]}}))
PY
verify_release() {
  python3 - "$WORK/release.json" <<'PY'
import json, subprocess, sys

def aws(*args):
    return json.loads(subprocess.check_output(["aws", *args, "--output", "json"], text=True))

with open(sys.argv[1]) as file:
    release = json.load(file)
if aws("lambda", "get-function-configuration", "--function-name", release["control_function"])["State"] != "Active":
    raise SystemExit("the release's control function is not active")
for name, arn in release["job_definitions"].items():
    jobs = aws("batch", "describe-job-definitions", "--job-definitions", arn)
    if not any(j["jobDefinitionArn"] == arn and j["status"] == "ACTIVE" for j in jobs["jobDefinitions"]):
        raise SystemExit(f"the release's {name} job definition is not active")
for image in release["images"].values():
    aws("ecr", "describe-images", "--repository-name", image["repository"].split("/", 1)[1],
        "--image-ids", f"imageDigest={image['digest']}")
PY
}
verify_release
python3 scripts/aws/runtime_parameters.py "$WORK/build-output.json" "$WORK/overrides.json" > "$WORK/runtime-parameters.json"
# Templates and ASL of that commit (identical inputs re-package to the same S3 objects).
git archive "$GIT_SHA" infra | tar -x -C "$WORK"
aws cloudformation package --template-file "$WORK/infra/cloudformation/root.yaml" --s3-bucket "$ARTIFACTS" \
  --s3-prefix "templates/$GIT_SHA" --output-template-file "$WORK/packaged-root.yaml" >/dev/null
ARTIFACTS_BUCKET= RUNTIME_CONFIG_KEY= bash scripts/aws/deploy_change_set.sh "$PROJECT-runtime" \
  "$WORK/packaged-root.yaml" "$WORK/runtime-parameters.json" "$ROLE" "$GIT_SHA"
verify_release
[[ "$(RUNTIME_STACK="$PROJECT-runtime" bash scripts/aws/outputs.sh GitSha)" == "$GIT_SHA" ]]
aws s3api put-object --bucket "$DATA" --key releases/latest.json --content-type application/json \
  --if-match "$POINTER_ETAG" \
  --server-side-encryption aws:kms --ssekms-key-id "$DATA_KEY_ARN" \
  --body <(printf '{"git_sha": "%s", "release_key": "releases/%s/release.json", "rollback": true}' "$GIT_SHA" "$GIT_SHA") >/dev/null
echo "runtime rolled back to $GIT_SHA; releases/latest.json points to it"
