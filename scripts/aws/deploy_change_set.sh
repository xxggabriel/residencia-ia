#!/usr/bin/env bash
# Create, describe and execute a change set for the runtime stack. Idempotent: an empty change
# set (same commit re-run, nothing changed) is deleted and the deploy succeeds without changes.
# Usage: deploy_change_set.sh STACK PACKAGED_TEMPLATE PARAMETERS_JSON CFN_ROLE_ARN GIT_SHA
set -euo pipefail
STACK="$1"; TEMPLATE="$2"; PARAMETERS="$3"; ROLE="$4"; GIT_SHA="$5"
if [[ -n "${ARTIFACTS_BUCKET:-}" && -n "${RUNTIME_CONFIG_KEY:-}" ]] && \
   aws s3api head-object --bucket "$ARTIFACTS_BUCKET" --key "$RUNTIME_CONFIG_KEY" >/dev/null 2>&1; then
  aws s3 cp "s3://$ARTIFACTS_BUCKET/$RUNTIME_CONFIG_KEY" /tmp/runtime-overrides.json --quiet
  python3 scripts/aws/runtime_parameters.py "${BUILD_OUTPUT:-$(dirname "$PARAMETERS")/build-output.json}" \
    /tmp/runtime-overrides.json > /tmp/runtime-parameters.json
  PARAMETERS=/tmp/runtime-parameters.json
fi
TYPE=UPDATE
STATUS="$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE)"
if [[ "$STATUS" == ROLLBACK_COMPLETE ]]; then
  echo "Stack $STACK is in ROLLBACK_COMPLETE state; deleting before recreating..."
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
  STATUS=NONE
fi
[[ "$STATUS" == NONE || "$STATUS" == REVIEW_IN_PROGRESS || "$STATUS" == DELETE_COMPLETE ]] && TYPE=CREATE
NAME="deploy-${GIT_SHA:0:12}-$(date -u +%Y%m%d%H%M%S)"
aws cloudformation create-change-set --stack-name "$STACK" --change-set-name "$NAME" \
  --change-set-type "$TYPE" --template-body "file://$TEMPLATE" --parameters "file://$PARAMETERS" \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --role-arn "$ROLE" \
  --include-nested-stacks --tags Key=cb-stack,Value=continual-bench >/dev/null
if ! aws cloudformation wait change-set-create-complete --stack-name "$STACK" --change-set-name "$NAME"; then
  REASON="$(aws cloudformation describe-change-set --stack-name "$STACK" --change-set-name "$NAME" --query StatusReason --output text)"
  if [[ "$REASON" == *"didn't contain changes"* || "$REASON" == *"No updates are to be performed"* ]]; then
    aws cloudformation delete-change-set --stack-name "$STACK" --change-set-name "$NAME"
    echo "no changes for $GIT_SHA; runtime stack already matches"
    exit 0
  fi
  echo "change set failed: $REASON" >&2
  exit 1
fi
# Inspect if the change set contains 0 changes
NUM_CHANGES="$(aws cloudformation describe-change-set --stack-name "$STACK" --change-set-name "$NAME" \
  --query 'length(Changes)' --output text 2>/dev/null || echo 0)"
if [[ "$NUM_CHANGES" == "0" || "$NUM_CHANGES" == "None" ]]; then
  aws cloudformation delete-change-set --stack-name "$STACK" --change-set-name "$NAME"
  echo "change set has 0 resource changes; skipping execution for $GIT_SHA (runtime stack already matches)"
  exit 0
fi
aws cloudformation describe-change-set --stack-name "$STACK" --change-set-name "$NAME" \
  --query 'Changes[].ResourceChange.{action:Action,resource:LogicalResourceId,type:ResourceType,replacement:Replacement}' --output table
aws cloudformation execute-change-set --stack-name "$STACK" --change-set-name "$NAME"
if [[ "$TYPE" == CREATE ]]; then
  aws cloudformation wait stack-create-complete --stack-name "$STACK"
else
  aws cloudformation wait stack-update-complete --stack-name "$STACK"
fi
echo "deployed $GIT_SHA to $STACK"
