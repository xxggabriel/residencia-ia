#!/usr/bin/env bash
# Usage: deploy_studio_catalog.sh <domain-id> <project-id>
set -euo pipefail

DOMAIN_ID="${1:?Unified Studio domain ID required}"
PROJECT_ID="${2:?Unified Studio project ID required}"
PROJECT="${PROJECT:-continual-bench}"
REGION="${AWS_REGION:-us-east-1}"
AWS=(aws --region "$REGION")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

stack_export() {
  "${AWS[@]}" cloudformation list-exports --query "Exports[?Name=='$PROJECT-$1'].Value | [0]" --output text
}

DATA_BUCKET="$(stack_export DataBucket)"
ARTIFACTS_BUCKET="$(stack_export ArtifactsBucket)"
KEY_ARN="$(stack_export DataKeyArn)"
APP_ARN="$("${AWS[@]}" cloudformation describe-stacks --stack-name "$PROJECT-runtime" \
  --query "Stacks[0].Outputs[?OutputKey=='MlflowAppArn'].OutputValue | [0]" --output text)"
PORTAL="$("${AWS[@]}" datazone get-domain --identifier "$DOMAIN_ID" --query portalUrl --output text)"
if [[ "$APP_ARN" != arn:*:mlflow-app/* || "$DATA_BUCKET" == "None" || "$ARTIFACTS_BUCKET" == "None" ]]; then
  echo "Deploy the benchmark persistent and runtime stacks before the catalog" >&2
  exit 1
fi
APP_ID="${APP_ARN##*/}"
MLFLOW_UI="https://$(printf '%s' "$APP_ID" | tr '[:upper:]' '[:lower:]').mlflow.sagemaker.$REGION.app.aws"

BUNDLE="$(mktemp -d)"
trap 'unlink "$BUNDLE/catalog.zip"; rmdir "$BUNDLE"' EXIT
(cd "$ROOT/infra/sagemaker/catalog" && zip -q "$BUNDLE/catalog.zip" catalog.py)
SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$BUNDLE/catalog.zip")"
CODE_KEY="templates/studio-catalog/$SHA.zip"
"${AWS[@]}" s3 cp "$BUNDLE/catalog.zip" "s3://$ARTIFACTS_BUCKET/$CODE_KEY" \
  --sse aws:kms --sse-kms-key-id "$KEY_ARN" --only-show-errors

STACK="$PROJECT-studio-catalog"
"${AWS[@]}" cloudformation deploy --template-file "$ROOT/infra/cloudformation/studio-catalog.yaml" \
  --stack-name "$STACK" --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides Project="$PROJECT" DataBucket="$DATA_BUCKET" ArtifactsBucket="$ARTIFACTS_BUCKET" \
  KeyArn="$KEY_ARN" CodeKey="$CODE_KEY" CatalogDomainId="$DOMAIN_ID" CatalogProjectId="$PROJECT_ID" \
  CatalogPortalUrl="$PORTAL" MlflowAppArn="$APP_ARN" MlflowUiUrl="$MLFLOW_UI"

ROLE="$("${AWS[@]}" cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='PublisherRoleArn'].OutputValue | [0]" --output text)"
if ! MEMBER="$("${AWS[@]}" datazone get-group-profile --domain-identifier "$DOMAIN_ID" \
  --group-identifier "$ROLE" --query id --output text 2>&1)"; then
  if [[ "$MEMBER" != *ResourceNotFoundException* ]]; then
    printf '%s\n' "$MEMBER" >&2
    exit 1
  fi
  MEMBER="$("${AWS[@]}" datazone create-group-profile --domain-identifier "$DOMAIN_ID" \
    --role-principal-arn "$ROLE" --query id --output text)"
fi
MEMBERSHIP="$("${AWS[@]}" datazone list-project-memberships --domain-identifier "$DOMAIN_ID" \
  --project-identifier "$PROJECT_ID" \
  --query "members[?memberDetails.group.groupId=='$MEMBER'].designation | [0]" --output text)"
if [[ "$MEMBERSHIP" == "None" ]]; then
  "${AWS[@]}" datazone create-project-membership --domain-identifier "$DOMAIN_ID" \
    --project-identifier "$PROJECT_ID" --member "groupIdentifier=$MEMBER" --designation PROJECT_CONTRIBUTOR
fi
