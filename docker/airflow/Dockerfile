FROM apache/airflow:3.3.2-python3.12
# Airflow and its providers stay in the upstream constrained environment.
COPY dags /opt/airflow/dags
COPY src/continual_bench/application/project_suite.py /opt/airflow/shared/continual_bench/application/project_suite.py
ENV PYTHONPATH=/opt/airflow/shared
COPY configs/base/tracking.yaml /opt/airflow/configs/base/tracking.yaml
