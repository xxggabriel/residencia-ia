FROM apache/airflow:3.3.2-python3.12
# Airflow and its providers stay in the upstream constrained environment.
COPY dags /opt/airflow/dags
COPY configs/tracking-metrics-profile.json /opt/airflow/configs/tracking-metrics-profile.json
