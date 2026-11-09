#!/usr/bin/env bash
# Remove billed runtime resources (VPC endpoints, Batch, Lambda, state machines, optional
# modules, Flink, endpoints) while keeping data, models, images and billing history.
#   1. stop the Flink application and any running campaign executions (with reconciliation);
#   2. delete the runtime stack (stateful resources inside it are retained by policy);
#   3. optionally delete the CI/CD stack (--with-cicd). The persistent stack is never deleted here.
set -euo pipefail
PROJECT="${PROJECT:-continual-bench}"
cd "$(dirname "$0")/../.."
if aws cloudformation describe-stacks --stack-name "$PROJECT-runtime" >/dev/null 2>&1; then
  if APP="$(bash scripts/aws/outputs.sh FlinkApplicationName 2>/dev/null)" && [[ -n "$APP" && "$APP" != None ]]; then
    aws kinesisanalyticsv2 stop-application --application-name "$APP" --force >/dev/null 2>&1 || true
  fi
  SM="$(bash scripts/aws/outputs.sh BenchmarkStateMachineArn)"
  for execution in $(aws stepfunctions list-executions --state-machine-arn "$SM" --status-filter RUNNING --query 'executions[].name' --output text); do
    bash scripts/aws/cancel_campaign.sh "${execution%%--*}" "$execution" || true
  done
  aws cloudformation delete-stack --stack-name "$PROJECT-runtime"
  aws cloudformation wait stack-delete-complete --stack-name "$PROJECT-runtime"
fi
if [[ "${1:-}" == --with-cicd ]]; then
  aws cloudformation delete-stack --stack-name "$PROJECT-cicd"
  aws cloudformation wait stack-delete-complete --stack-name "$PROJECT-cicd"
fi
echo "runtime removed; retained: $PROJECT-persistent (buckets, KMS, ECR), job/acquisition log groups, Model Registry"
