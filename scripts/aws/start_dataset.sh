#!/usr/bin/env bash
# Manually start the dataset pipeline. The execution name is derived from the snapshot so a
# concurrent duplicate is refused by Step Functions; a later re-run adds a suffix and resumes.
# Usage: scripts/aws/start_dataset.sh infra/examples/dataset-social-smoke.json [suffix]
set -euo pipefail
INPUT="${1:?input json}"; SUFFIX="${2:-}"
ARN="$(bash "$(dirname "$0")/outputs.sh" DatasetStateMachineArn)"
NAME="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["request"]; print("-".join((r["dataset"], r["scope"], r.get("mode", "download"))))' "$INPUT")${SUFFIX:+--$SUFFIX}"
aws stepfunctions start-execution --state-machine-arn "$ARN" --name "$NAME" --input "file://$INPUT"
