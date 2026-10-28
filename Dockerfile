FROM python:3.12.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ZENML_ANALYTICS_OPT_IN=false ZENML_CONFIG_PATH=/state/zen-config \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml ./
COPY src ./src
COPY configs/metric-definitions.json ./configs/metric-definitions.json
RUN pip install --no-cache-dir --no-deps .
COPY configs ./configs
COPY tests ./tests
COPY examples ./examples
COPY scripts ./scripts
COPY dags ./dags
ENV PYTHONPATH=/app
CMD ["continual-bench", "serve"]
