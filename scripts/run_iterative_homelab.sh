#!/usr/bin/env bash
# Run on the homelab. Trains the 17 catalog datasets (100 epochs/update, patience 10) in 8 containers
# of 2 vCPU / 2 GiB each; workers claim datasets from a shared queue (atomic mkdir), heaviest first.
set -euo pipefail
ROOT="${ITER_ROOT:?absolute working directory}"
IMAGE="${ITER_IMAGE:-continual-bench-datasets:semana09-v2}"
WORKERS="${ITER_WORKERS:-8}"
LEGACY_VOLUME="${ITER_LEGACY_VOLUME:-continual-bench_worker-data}"
DATASETS=(core50 clear100 clear10 split_cifar100 split_cifar10 airlines permuted_mnist split_mnist
          insects electricity gas sea agrawal stagger hyperplane rbf_drift friedman_drift)
# MLflow credentials come from the existing service env file; values are never printed or passed on a
# command line (docker copies them from this shell's environment).
ENV_FILE="${ITER_SERVICE_ENV:-$HOME/.config/continual-bench/benchmark.env}"
set -a; . "$ENV_FILE"; set +a
export MLFLOW_TRACKING_USERNAME="${WEB_USER:?missing in service env}"
export MLFLOW_TRACKING_PASSWORD="${MLFLOW_PASSWORD:?missing in service env}"
MLFLOW_URI="${ITER_MLFLOW_URI:-http://192.168.100.27:15000}"
CAMPAIGN="${ITER_CAMPAIGN:-$(date +%Y%m%d-%H%M)}"
OUT="state/out-$CAMPAIGN"
mkdir -p "$ROOT/$OUT" "$ROOT/state/cache"
args=(); for d in "${DATASETS[@]}"; do args+=(--dataset "$d"); done
for i in $(seq 1 "$WORKERS"); do
  docker rm -f "iter-worker-$i" >/dev/null 2>&1 || true
  docker run -d --name "iter-worker-$i" --cpus 2 --memory 2g --memory-swap 2g --pids-limit 256 \
    --security-opt no-new-privileges:true --read-only --tmpfs /tmp:rw,size=256m \
    -v "$ROOT/code:/work:ro" -v "$ROOT/state:/state" -v "$LEGACY_VOLUME:/legacy:ro" -w /work \
    -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONHASHSEED=0 \
    -e OMP_NUM_THREADS=2 -e BENCH_DATASET_DOWNLOAD=true -e MLFLOW_TRACKING_URI="$MLFLOW_URI" \
    -e MLFLOW_TRACKING_USERNAME -e MLFLOW_TRACKING_PASSWORD -e BENCH_GIT_REVISION="${ITER_GIT:-uncommitted}" \
    --entrypoint python "$IMAGE" -m continual_bench.cli iterative-worker \
    --cache /state/cache --real-directory /legacy/datasets --out "/$OUT" --campaign "$CAMPAIGN" --worker "w$i" "${args[@]}"
done
docker ps --filter name=iter-worker --format '{{.Names}}\t{{.Status}}'
