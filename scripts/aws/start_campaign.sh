#!/usr/bin/env bash
# Manually start a benchmark campaign; the execution name is the campaign_id (plus an optional
# retry suffix: `<campaign_id>--<suffix>`). Completed runs of the campaign are reused on retry.
# Usage: scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json [suffix]
set -euo pipefail
INPUT="${1:?input json}"; SUFFIX="${2:-}"
ARN="$(bash "$(dirname "$0")/outputs.sh" BenchmarkStateMachineArn)"
CAMPAIGN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request"]["campaign_id"])' "$INPUT")"
aws stepfunctions start-execution --state-machine-arn "$ARN" --name "$CAMPAIGN${SUFFIX:+--$SUFFIX}" --input "file://$INPUT"
