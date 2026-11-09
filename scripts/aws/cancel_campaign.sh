#!/usr/bin/env bash
# Stop a campaign execution and reconcile: terminate every non-terminal job tagged/named for the
# campaign and wait until Batch reports terminal states and ECS reports STOPPED tasks.
# Usage: scripts/aws/cancel_campaign.sh CAMPAIGN_ID [EXECUTION_NAME]
set -euo pipefail
CAMPAIGN="${1:?campaign id}"; EXECUTION="${2:-$1}"
SM="$(bash "$(dirname "$0")/outputs.sh" BenchmarkStateMachineArn)"
EXECUTION_ARN="${SM/:stateMachine:/:execution:}:$EXECUTION"
export BENCH_JOB_QUEUE="$(bash "$(dirname "$0")/outputs.sh" JobQueueArn)"
export BENCH_BUCKET="$(aws cloudformation list-exports --query "Exports[?Name=='continual-bench-DataBucket'].Value" --output text)"
continual-bench aws cancel --campaign "$CAMPAIGN" --execution-arn "$EXECUTION_ARN" --checks 20 --interval 30
