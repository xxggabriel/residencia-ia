#!/usr/bin/env bash
# Explicit activation of the managed Flink application (bills KPUs only while RUNNING).
# Usage: scripts/aws/flink.sh start|stop|status [--restore-latest]
set -euo pipefail
APP="$(bash "$(dirname "$0")/outputs.sh" FlinkApplicationName)"
case "${1:?start|stop|status}" in
  start)
    RESTORE=SKIP_RESTORE_FROM_SNAPSHOT; [[ "${2:-}" == --restore-latest ]] && RESTORE=RESTORE_FROM_LATEST_SNAPSHOT
    aws kinesisanalyticsv2 start-application --application-name "$APP" \
      --run-configuration "{\"ApplicationRestoreConfiguration\": {\"ApplicationRestoreType\": \"$RESTORE\"}}" ;;
  stop)
    # A graceful stop takes a snapshot so the next start can resume from its checkpoint state.
    aws kinesisanalyticsv2 stop-application --application-name "$APP" --no-force ;;
  status)
    aws kinesisanalyticsv2 describe-application --application-name "$APP" --query ApplicationDetail.ApplicationStatus --output text ;;
esac
