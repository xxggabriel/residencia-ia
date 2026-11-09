#!/usr/bin/env bash
# Build, validate and publish the runtime and ml images for one commit, idempotently.
#   - tag: git-<40-hex SHA>, immutable in ECR; an existing tag is reused, never rebuilt;
#   - platform: linux/amd64, dependencies from requirements-aws.lock with hashes;
#   - validation before push: runtime selftest (imports, x86_64, no MLflow/ZenML/Kafka) and the
#     ml image's SageMaker contract (/ping, /invocations) on a tiny fixture model;
#   - output: build-output.json with repository, tag and resolved digest per image.
# Requires RUNTIME_REPOSITORY and ML_REPOSITORY (…amazonaws.com/<name>) and Docker.
set -euo pipefail
GIT_SHA="${1:?git sha}"
[[ "$GIT_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid sha: $GIT_SHA" >&2; exit 2; }
TAG="git-$GIT_SHA"
REGISTRY="${RUNTIME_REPOSITORY%%/*}"
OUT="${BUILD_OUTPUT:-build-output.json}"
aws ecr get-login-password --region "${AWS_REGION:-us-east-1}" | docker login --username AWS --password-stdin "$REGISTRY"

digest_of() {  # repository-uri tag -> digest or empty
  local name="${1#*/}"
  aws ecr describe-images --repository-name "$name" --image-ids imageTag="$2" \
    --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true
}

publish() {  # target repository-uri
  local target="$1" repo="$2" digest
  digest="$(digest_of "$repo" "$TAG")"
  if [[ -n "$digest" && "$digest" != "None" ]]; then
    echo "reuse $repo:$TAG ($digest)" >&2
  else
    docker buildx build --platform linux/amd64 -f docker/aws/Dockerfile --target "$target" \
      --build-arg GIT_SHA="$GIT_SHA" --provenance=false -t "$repo:$TAG" --load . >&2
    if [[ "$target" == runtime ]]; then
      docker run --rm --platform linux/amd64 "$repo:$TAG" continual-bench aws selftest --role runtime --require-x86-64 >&2
    else
      bash scripts/aws/check_inference_image.sh "$repo:$TAG" >&2
    fi
    docker push "$repo:$TAG" >&2
    digest="$(digest_of "$repo" "$TAG")"
  fi
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "could not resolve digest for $repo:$TAG" >&2; exit 1; }
  printf '%s' "$digest"
}

RUNTIME_DIGEST="$(publish runtime "$RUNTIME_REPOSITORY")"
ML_DIGEST="$(publish ml "$ML_REPOSITORY")"
python3 - "$OUT" <<PY
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
state = json.loads(path.read_text()) if path.is_file() else {}
state.update({"git_sha": "$GIT_SHA", "images": {
    "runtime": {"repository": "$RUNTIME_REPOSITORY", "tag": "$TAG", "digest": "$RUNTIME_DIGEST"},
    "ml": {"repository": "$ML_REPOSITORY", "tag": "$TAG", "digest": "$ML_DIGEST"}}})
path.write_text(json.dumps(state, indent=2, sort_keys=True))
PY
echo "runtime $RUNTIME_REPOSITORY@$RUNTIME_DIGEST"
echo "ml      $ML_REPOSITORY@$ML_DIGEST"
