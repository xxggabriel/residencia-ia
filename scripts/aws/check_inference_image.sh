#!/usr/bin/env bash
# SageMaker inference contract of the ml image: train a fixture model with the image itself,
# serve it on :8080, require /ping 200 and /invocations probabilities that sum to one.
set -euo pipefail
IMAGE="${1:?image}"
PLATFORM="${PLATFORM:-linux/amd64}"
WORK="$(mktemp -d)"
trap 'docker rm -f cb-inference-check >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
mkdir -p "$WORK/input/data/train" "$WORK/input/config" "$WORK/model" "$WORK/output"
docker run --rm --platform "$PLATFORM" -v "$WORK:/work" --entrypoint python "$IMAGE" -c '
from pathlib import Path
from continual_bench.application.cloud.fixtures import fixture_parquet
fixture_parquet(Path("/work/input/data/train/smoke.parquet"))'
echo '{"validation_fraction": "0.1", "max_iterations": "10"}' > "$WORK/input/config/hyperparameters.json"
docker run --rm --platform "$PLATFORM" -v "$WORK:/opt/ml" "$IMAGE" train
docker run -d --name cb-inference-check --platform "$PLATFORM" -p 18080:8080 -v "$WORK/model:/opt/ml/model:ro" "$IMAGE" serve >/dev/null
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:18080/ping >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:18080/ping
RESPONSE="$(curl -fsS -H 'Content-Type: application/json' -d '{"instances": [[0.0], [3.0]]}' http://127.0.0.1:18080/invocations)"
python3 -c 'import json,sys; p=json.loads(sys.argv[1])["probabilities"]; assert len(p)==2 and all(abs(sum(r)-1)<1e-6 for r in p), p; print("inference contract ok", p)' "$RESPONSE"
