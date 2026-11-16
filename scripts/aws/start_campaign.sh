#!/usr/bin/env bash
# Manually start a benchmark campaign; the execution name is the campaign_id (plus an optional
# retry suffix: `<campaign_id>--<suffix>`). Completed runs of the campaign are reused on retry.
# Usage: RELEASE_SHA=<published SHA> scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json [suffix]
set -euo pipefail
INPUT="${1:?input json}"; SUFFIX="${2:-}"
SCRIPTS="$(dirname "$0")"
REQUEST_FILE="$(mktemp)"
trap 'rm -f "$REQUEST_FILE"' EXIT
CAMPAIGN="$(python3 "$SCRIPTS/campaign_request.py" "$INPUT" "$REQUEST_FILE")"
ARN="$(bash "$SCRIPTS/outputs.sh" BenchmarkStateMachineArn)"
aws stepfunctions start-execution --state-machine-arn "$ARN" --name "$CAMPAIGN${SUFFIX:+--$SUFFIX}" \
  --input "file://$REQUEST_FILE"
