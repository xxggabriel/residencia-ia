#!/usr/bin/env bash
# Start a SageMaker Studio campaign from the same JSON used by start_campaign.sh.
# Usage: RELEASE_SHA=<published SHA> scripts/aws/start_studio_campaign.sh <campaign.json> [dataset.json]
set -euo pipefail

CAMPAIGN="${1:?campaign JSON required}"
DATASET="${2:-}"
PROJECT="${PROJECT:-continual-bench}"
REGION="${AWS_REGION:-us-east-1}"
SCRIPTS="$(dirname "$0")"
REQUEST_DIR="$(mktemp -d)"
trap 'rm -f "$REQUEST_DIR/campaign.json" "$REQUEST_DIR/dataset.json"; rmdir "$REQUEST_DIR"' EXIT
PREPARE=(python3 "$SCRIPTS/campaign_request.py" "$CAMPAIGN" "$REQUEST_DIR/campaign.json")
if [[ -n "$DATASET" ]]; then
  PREPARE+=( "$DATASET" "$REQUEST_DIR/dataset.json" )
fi
CAMPAIGN_ID="$("${PREPARE[@]}")"

AWS=(aws --region "$REGION")
BUCKET="$("${AWS[@]}" cloudformation list-exports \
  --query "Exports[?Name=='$PROJECT-DataBucket'].Value | [0]" --output text)"
KEY_ARN="$("${AWS[@]}" cloudformation list-exports \
  --query "Exports[?Name=='$PROJECT-DataKeyArn'].Value | [0]" --output text)"
if [[ "$BUCKET" == "None" || "$KEY_ARN" == "None" ]]; then
  echo "Persistent stack exports for $PROJECT were not found in $REGION" >&2
  exit 1
fi

REQUEST_ID="$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')"

upload() {
  "${AWS[@]}" s3api put-object --bucket "$BUCKET" --key "$2" --body "$1" \
    --content-type application/json --server-side-encryption aws:kms --ssekms-key-id "$KEY_ARN" \
    --if-none-match '*' --query VersionId --output text
}

CAMPAIGN_KEY="studio/requests/$REQUEST_ID/campaign.json"
CAMPAIGN_VERSION="$(upload "$REQUEST_DIR/campaign.json" "$CAMPAIGN_KEY")"
PARAMS=( "Name=CampaignRequestUri,Value=s3://$BUCKET/$CAMPAIGN_KEY" \
         "Name=CampaignRequestVersion,Value=$CAMPAIGN_VERSION" )
if [[ -n "$DATASET" ]]; then
  DATASET_KEY="studio/requests/$REQUEST_ID/dataset.json"
  DATASET_VERSION="$(upload "$REQUEST_DIR/dataset.json" "$DATASET_KEY")"
  PARAMS+=( "Name=DatasetRequestUri,Value=s3://$BUCKET/$DATASET_KEY" \
            "Name=DatasetRequestVersion,Value=$DATASET_VERSION" )
fi

"${AWS[@]}" sagemaker start-pipeline-execution --pipeline-name "$PROJECT-campaign" \
  --pipeline-execution-display-name "$CAMPAIGN_ID" --client-request-token "$REQUEST_ID" \
  --pipeline-parameters "${PARAMS[@]}"
