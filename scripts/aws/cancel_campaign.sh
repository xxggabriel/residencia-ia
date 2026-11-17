#!/usr/bin/env bash
# Cancel the logical campaign, stop its workflow, then verify instance and volume cleanup.
# Usage: scripts/aws/cancel_campaign.sh CAMPAIGN_ID [EXECUTION_NAME]
set -euo pipefail
CAMPAIGN="${1:?campaign id}"; EXECUTION="${2:-$1}"
SM="$(bash "$(dirname "$0")/outputs.sh" BenchmarkStateMachineArn)"
EXECUTION_ARN="${SM/:stateMachine:/:execution:}:$EXECUTION"
WORKFLOW="${SM##*:}"
export BENCH_STACK="${BENCH_STACK:-${WORKFLOW%-benchmark}}"
export BENCH_JOB_QUEUE="$(bash "$(dirname "$0")/outputs.sh" JobQueueArn)"
export BENCH_KMS_KEY_ID="$(bash "$(dirname "$0")/outputs.sh" DataKeyArn)"
export BENCH_BUCKET="$(aws cloudformation list-exports --query "Exports[?Name=='${BENCH_STACK}-DataBucket'].Value" --output text)"
continual-bench aws cancel --campaign "$CAMPAIGN" --execution-arn "$EXECUTION_ARN" --checks 20 --interval 30
