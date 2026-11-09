#!/usr/bin/env bash
# Publish optional-module code keyed by commit (idempotent): Glue ETL script and the PyFlink
# application zip with its Kinesis connector JAR (verified against Maven Central's SHA-1 file).
set -euo pipefail
GIT_SHA="${1:?git sha}"
BUCKET="${ARTIFACTS_BUCKET:?ARTIFACTS_BUCKET}"
KINESIS_CONNECTOR_VERSION="${KINESIS_CONNECTOR_VERSION:-5.0.0-1.20}"
put_once() {  # local key
  if aws s3api head-object --bucket "$BUCKET" --key "$2" >/dev/null 2>&1; then echo "reuse s3://$BUCKET/$2"; return; fi
  aws s3api put-object --bucket "$BUCKET" --key "$2" --body "$1" --if-none-match '*' >/dev/null
  echo "uploaded s3://$BUCKET/$2"
}
put_once infra/glue/materialize_features.py "glue/$GIT_SHA/materialize_features.py"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/app/lib"
cp infra/flink/event_time_windows.py "$WORK/app/"
BASE="https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kinesis/$KINESIS_CONNECTOR_VERSION"
JAR="flink-sql-connector-kinesis-$KINESIS_CONNECTOR_VERSION.jar"
curl -fsSL --retry 4 -o "$WORK/app/lib/flink-sql-connector-kinesis.jar" "$BASE/$JAR"
EXPECTED="$(curl -fsSL --retry 4 "$BASE/$JAR.sha1" | cut -c1-40)"
ACTUAL="$(sha1sum "$WORK/app/lib/flink-sql-connector-kinesis.jar" | cut -c1-40)"
[[ "$EXPECTED" == "$ACTUAL" ]] || { echo "connector checksum mismatch" >&2; exit 1; }
(cd "$WORK/app" && TZ=UTC touch -t 202601010000 event_time_windows.py lib lib/*.jar && zip -qX -r ../event_time_windows.zip .)
put_once "$WORK/event_time_windows.zip" "flink/$GIT_SHA/event_time_windows.zip"
