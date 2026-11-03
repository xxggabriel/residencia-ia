#!/usr/bin/env bash
# Execute on the homelab only. Paths are explicit; never read or print credentials.
set -euo pipefail
: "${WEEK09_ROOT:?Set the isolated absolute working directory}"
: "${WEEK09_SERVICE_ENV:?Set the existing Compose environment file path}"
case "$(hostname)" in gabriel) ;; *) echo 'Run this script via SSH on the authorized homelab.' >&2; exit 2;; esac
if [[ "${1:-}" == unit ]]; then
  timeout 240 docker run --rm --network none --cpus 2 --memory 2g --pids-limit 128 \
    --read-only --tmpfs /tmp:rw,size=256m -v "$WEEK09_ROOT/code:/work:ro" -w /work \
    -e PYTHONPATH=/work/src -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python continual-bench-worker:0.1.0 -m pytest -q -p no:cacheprovider
elif [[ "${1:-}" == services ]]; then
  timeout 300 docker compose --env-file "$WEEK09_SERVICE_ENV" -f "$WEEK09_ROOT/compose.yaml" \
    run --rm --no-deps --entrypoint python validation -m pytest -q -p no:cacheprovider \
    -m services tests/test_week09_services.py
else
  echo 'Usage: run_week09_homelab.sh unit|services' >&2; exit 2
fi
