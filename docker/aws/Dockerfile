# syntax=docker/dockerfile:1.7
# AWS managed profile images, Linux x86_64, dependencies from requirements-aws.lock (hashes).
#   runtime: Batch benchmark/preparation jobs, acquisition CodeBuild, SageMaker Processing.
#            Entry point: `continual-bench aws ...` (job definitions set the command).
#   ml:      SageMaker Training/Tuning (`train`) and Inference (`serve` on :8080).
# Both share the dependency layer; they differ only in entry point, so they are separate tags.
# No MLflow, ZenML, Kafka, MinIO or Airflow is installed.
# Use the official mirror to avoid Docker Hub rate limits in CodeBuild.
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
# Releases are always linux/amd64. IMAGE_PLATFORM=linux/arm64 exists only for native smoke
# checks on Apple Silicon, where amd64 emulation (QEMU) cannot run pyarrow.
ARG IMAGE_PLATFORM=linux/amd64
FROM --platform=${IMAGE_PLATFORM} ${PYTHON_IMAGE} AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    BENCH_PROJECT_ROOT=/app
WORKDIR /app
COPY requirements-aws.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-aws.lock
COPY pyproject.toml ./
COPY configs/contracts/metric-definitions.json configs/contracts/image-datasets.lock.json \
     configs/contracts/real-streams.lock.json configs/contracts/temporal-vision.lock.json ./configs/contracts/
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin bench
COPY configs ./configs
ARG GIT_SHA=uncommitted
ENV BENCH_GIT_REVISION=${GIT_SHA}
LABEL org.opencontainers.image.source="continual-bench" org.opencontainers.image.revision="${GIT_SHA}"

FROM base AS runtime
LABEL org.opencontainers.image.title="continual-bench-runtime"
CMD ["continual-bench", "aws", "selftest", "--role", "runtime", "--require-x86-64"]

FROM base AS ml
LABEL org.opencontainers.image.title="continual-bench-ml"
EXPOSE 8080
ENTRYPOINT ["python", "-m", "continual_bench.orchestration.sagemaker"]
CMD ["serve"]
