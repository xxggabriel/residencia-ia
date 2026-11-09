"""Feature Store smoke: write prepared rows, read them back (online when enabled), check parity
and point-in-time correctness against the prepared Parquet.

Usage: python scripts/aws/feature_store_smoke.py --group continual-bench-zia-social-edges \
         --parquet state/zia-v1/social/smoke.parquet [--rows 50] [--online]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def rows(parquet, limit):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet).slice(0, limit).to_pylist()
    return [
        {
            "event_id": f"edge-{i}",
            "event_time": float(r["event_time"]) / 1000,
            **{k: v for k, v in r.items() if k != "event_time"},
        }
        for i, r in enumerate(table)
    ]


def main():
    from continual_bench.application.cloud.features import assert_no_leakage, parity, point_in_time
    from continual_bench.infrastructure.aws.feature_store import FeatureStoreRecords

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    prepared = rows(args.parquet, args.rows)
    store = FeatureStoreRecords(args.group)
    written = store.put(
        prepared, target_stores=["OnlineStore", "OfflineStore"] if args.online else ["OfflineStore"]
    )
    result = {
        "written": written,
        "offline": "records reach the offline store within minutes; query via Athena",
    }
    if args.online:
        read = [store.get(r["event_id"]) for r in prepared]
        materialized = [
            {**{k: float(v) for k, v in r.items() if k not in ("event_id",)}, "event_id": r["event_id"]}
            for r in read
            if r
        ]
        features = [k for k in prepared[0] if k not in ("event_id", "event_time", "label")]
        result["parity"] = parity(prepared, materialized, features, tolerance=1e-6)
        joined = point_in_time(
            [{"entity_id": r["event_id"], "event_time": r["event_time"]} for r in prepared],
            [{"entity_id": r["event_id"], "event_time": float(r["event_time"])} for r in materialized],
        )
        result["leakage_checked_rows"] = assert_no_leakage(
            [
                {"event_time": j["event_time"], "feature_time": (j["features"] or {}).get("event_time")}
                for j in joined
            ]
        )
    print(json.dumps(result, indent=2))
    if args.online and not result["parity"]["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
