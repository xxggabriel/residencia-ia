FROM ghcr.io/mlflow/mlflow:v3.17.2
RUN pip install --no-cache-dir 'mlflow[auth]==3.17.2' 'boto3==1.43.118'
COPY docker/mlflow-auth.ini /etc/mlflow/auth.ini
ENV MLFLOW_AUTH_CONFIG_PATH=/etc/mlflow/auth.ini
