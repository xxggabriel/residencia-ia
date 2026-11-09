"""AWS profile use cases: snapshots, planning, one-run workers, lifecycle, records and costs.

Ports (object store, Batch, Model Registry, metrics) are injected; boto3 stays in
`infrastructure.aws`. Nothing here imports MLflow, ZenML, Kafka, MinIO or Airflow.
"""
