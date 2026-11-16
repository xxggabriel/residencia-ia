#!/usr/bin/env bash
# Usage: deploy_native_pipeline.sh <runtime-image@sha256:...> <ml-image@sha256:...> <git-sha>
set -euo pipefail

RUNTIME_IMAGE="${1:?runtime image digest URI required}"
ML_IMAGE="${2:?ML image digest URI required}"
GIT_SHA="${3:?Git revision required}"
PROJECT="${PROJECT:-continual-bench}"
REGION="${AWS_REGION:-us-east-1}"
AWS=(aws --region "$REGION")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! "$GIT_SHA" =~ ^[0-9a-f]{40}$ || ! "$RUNTIME_IMAGE" =~ @sha256:[0-9a-f]{64}$ \
      || ! "$ML_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "Specify two image digests and the 40-character source revision" >&2
  exit 2
fi

stack_export() {
  "${AWS[@]}" cloudformation list-exports --query "Exports[?Name=='$PROJECT-$1'].Value | [0]" --output text
}

DATA_BUCKET="$(stack_export DataBucket)"
ARTIFACTS_BUCKET="$(stack_export ArtifactsBucket)"
KEY_ARN="$(stack_export DataKeyArn)"
APP_ARN="$("${AWS[@]}" cloudformation describe-stacks --stack-name "$PROJECT-runtime" \
  --query "Stacks[0].Outputs[?OutputKey=='MlflowAppArn'].OutputValue | [0]" --output text)"
ACCOUNT="$("${AWS[@]}" sts get-caller-identity --query Account --output text)"
PARTITION="${KEY_ARN#arn:}"
PARTITION="${PARTITION%%:*}"
if [[ "$APP_ARN" != arn:*:mlflow-app/* || "$DATA_BUCKET" == "None" || "$ARTIFACTS_BUCKET" == "None" ]]; then
  echo "The runtime stack and its MLflow app and persistent buckets must already exist" >&2
  exit 1
fi

ROLE="arn:$PARTITION:iam::$ACCOUNT:role/$PROJECT/$PROJECT-studio-native-job"
DEFINITION="$(mktemp)"
trap 'unlink "$DEFINITION"' EXIT
python3 "$SCRIPT_DIR/native_pipeline.py" --bucket "$DATA_BUCKET" --role "$ROLE" \
  --runtime-image "$RUNTIME_IMAGE" --ml-image "$ML_IMAGE" --git-sha "$GIT_SHA" --key-arn "$KEY_ARN" \
  --mlflow-arn "$APP_ARN" > "$DEFINITION"
SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$DEFINITION")"
KEY="templates/studio-native/$SHA.json"

"${AWS[@]}" s3 cp "$DEFINITION" "s3://$ARTIFACTS_BUCKET/$KEY" --content-type application/json --quiet
"${AWS[@]}" cloudformation deploy --template-file "$ROOT/infra/cloudformation/studio-native.yaml" \
  --stack-name "$PROJECT-studio-native" --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset \
  --parameter-overrides Project="$PROJECT" DataBucket="$DATA_BUCKET" ArtifactsBucket="$ARTIFACTS_BUCKET" \
  KeyArn="$KEY_ARN" DefinitionKey="$KEY" MlflowAppArn="$APP_ARN"
