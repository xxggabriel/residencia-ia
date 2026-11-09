"""SageMaker Feature Store record I/O (smoke-scale writes and online reads).

Bulk offline loads go through the Glue ETL job, which writes the versioned materialized
snapshot; this adapter covers record-level ingestion and the online read used by the smoke.
"""

from .session import client as make_client


def _record(features: dict) -> list[dict]:
    return [
        {"FeatureName": name, "ValueAsString": repr(value) if isinstance(value, float) else str(value)}
        for name, value in features.items()
    ]


class FeatureStoreRecords:
    def __init__(self, group: str, *, runtime=None):
        self.group = group
        self.runtime = runtime or make_client("sagemaker-featurestore-runtime")

    def put(self, rows, *, target_stores=None) -> int:
        count = 0
        for row in rows:
            arguments = {"FeatureGroupName": self.group, "Record": _record(row)}
            if target_stores:
                arguments["TargetStores"] = list(target_stores)
            self.runtime.put_record(**arguments)
            count += 1
        return count

    def get(self, record_id: str, features=None) -> dict | None:
        arguments = {"FeatureGroupName": self.group, "RecordIdentifierValueAsString": record_id}
        if features:
            arguments["FeatureNames"] = list(features)
        response = self.runtime.get_record(**arguments)
        if "Record" not in response:
            return None
        return {item["FeatureName"]: item["ValueAsString"] for item in response["Record"]}
