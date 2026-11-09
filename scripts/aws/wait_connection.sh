#!/usr/bin/env bash
# Wait until a CodeConnections connection is AVAILABLE (after the GitHub App authorization).
set -euo pipefail
ARN="${1:?connection arn}"
for _ in $(seq 1 60); do
  STATUS="$(aws codeconnections get-connection --connection-arn "$ARN" --query Connection.ConnectionStatus --output text)"
  echo "$STATUS"; [[ "$STATUS" == AVAILABLE ]] && exit 0
  sleep 10
done
echo "connection still $STATUS" >&2; exit 1
