"""Pinned Zia Data Labs inputs and bounded chronological Parquet preparation."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from continual_bench.infrastructure.storage import atomic_write

DATASETS = {
    "fraud": {
        "repo": "FreeFraudDetection50M",
        "revision": "8d2baf070c1e40a353bd6f947a785119291f9554",
        "file": "FreeSyntheticDataFraudDetection50M.parquet",
        "sha256": "d5bec836f43aedf3380ccd726e1d65edf6f9090cfde0466ed6fce1ebd5a79b21",
        "rows": 50_000_000,
        "classes": ["legitimate", "fraud"],
        "model": "xgboost",
    },
    "social": {
        "repo": "FreeSocialNetwork100M",
        "revision": "8518bf04c105df75ab72c13bbec1e12b270eb92f",
        "file": "social_network_edges_100M.parquet",
        "sha256": "35460cf50ae36472e748c58c43aba0da7e85e6765f976daf1f1e92c45a4b69fa",
        "nodes": "social_network_nodes_3M.parquet",
        "nodes_sha256": "f1250d093dcf3ed5b4837e2f85fb4a8d5a5d3818e5d82e995c37eefb744235a4",
        "rows": 100_000_000,
        "classes": ["follow", "friend", "block", "mute"],
        "model": "multinomial_logistic",
    },
    "ecommerce": {
        "repo": "FreeEcommerceOrders50M",
        "revision": "d4f02e6ba4fd10c1c03a544d98f289908d427750",
        "file": "ecommerce_50M.parquet",
        "sha256": "163589081eaeca83cccacc0405f59d4be149615ddd622069362cb0797689b326",
        "rows": 50_000_000,
        "classes": ["other_status", "returned"],
        "model": "xgboost",
    },
}
SCOPES = ("smoke", "full")
SMOKE_ROWS = 10_000
FORMAT = "zia-temporal-v1"

# Nominal inputs to the tree models have fixed hashed integer encodings.
FRAUD = """
    transaction_id AS event_id, epoch_ms("timestamp") AS event_time,
    CAST(fraud_label AS UTINYINT) AS label,
    amount, account_age_days, prev_transaction_amt,
    is_international::INT AS international,
    (shipping_zip <> billing_zip)::INT AS zip_mismatch,
    (hash(currency) % 64)::INT AS currency_code,
    (hash(merchant_category) % 256)::INT AS merchant_category_code,
    (hash(card_type) % 64)::INT AS card_type_code,
    (hash(location_country) % 256)::INT AS country_code,
    hour("timestamp") AS hour_of_day, dayofweek("timestamp") AS weekday
"""
ECOMMERCE = """
    order_id AS event_id, epoch_ms(order_date::DATE) AS event_time,
    (order_status = 'returned')::UTINYINT AS label,
    quantity, unit_price, order_total, shipping_cost, discount_applied::INT AS discount,
    (hash(product_category) % 256)::INT AS category_code,
    (hash(payment_method) % 64)::INT AS payment_code,
    dayofweek(order_date::DATE) AS weekday
"""
SOCIAL = """
    edge_id AS event_id, epoch_ms(created_date) AS event_time,
    (list_position(['follow', 'friend', 'block', 'mute'], edge_type) - 1)::UTINYINT AS label,
    substring(source_user_id, 2)::INT AS src, substring(target_user_id, 2)::INT AS dst
"""


def sha256_file(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def input_paths(root, dataset, scope):
    folder = Path(root) / dataset
    return folder / f"{scope}.parquet", folder / f"{scope}.json"


def load_manifest(root, dataset, scope):
    from pyarrow.parquet import ParquetFile

    path, manifest_path = input_paths(root, dataset, scope)
    manifest = json.loads(manifest_path.read_text())
    spec = DATASETS[dataset]
    expected_rows = SMOKE_ROWS if scope == "smoke" else spec["rows"]
    with ParquetFile(path) as source:
        names = source.schema_arrow.names
        if (manifest["format"] != FORMAT or manifest["dataset"] != dataset or manifest["scope"] != scope
                or manifest["source"] != spec or manifest["rows"] != expected_rows
                or source.metadata.num_rows != expected_rows or manifest["bytes"] != path.stat().st_size
                or names != ["event_time", "label", *manifest["features"]]):
            raise ValueError(f"prepared input contract mismatch: {path}")
    return path, manifest


def social_features(source, nodes_path, destination):
    """Join static attributes in memory and expose degree only through the previous date."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    node_ids = np.full(3_000_001, -1, dtype=np.int8)
    joined = np.zeros(len(node_ids), dtype=np.int32)
    with pq.ParquetFile(nodes_path) as nodes:
        if nodes.metadata.num_rows != 3_000_000:
            raise ValueError("unexpected social node count")
        for batch in nodes.iter_batches(columns=["user_id", "region_id", "join_date"], use_threads=False):
            ids = pc.cast(pc.utf8_slice_codeunits(batch[0], 1), pa.int32()).to_numpy()
            node_ids[ids] = batch[1].to_numpy()
            joined[ids] = pc.cast(batch[2], pa.int32()).to_numpy()
    if np.count_nonzero(node_ids >= 0) != 3_000_000:
        raise ValueError("duplicate or missing social node attributes")
    incoming = np.zeros(len(node_ids), dtype=np.int32)
    outgoing = np.zeros_like(incoming)
    pending_src, pending_dst = [], []
    current_day = None
    features = ["same_region", "source_age", "target_age",
                "source_in", "source_out", "target_in", "target_out"]
    schema = pa.schema([("event_time", pa.int64()), ("label", pa.uint8()),
                        *((name, pa.float32()) for name in features)])
    with pq.ParquetFile(source) as edges, pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in edges.iter_batches(batch_size=65_536, use_threads=False):
            times, labels, src, dst = (column.to_numpy() for column in batch.columns)
            days = times // 86_400_000
            boundaries = np.r_[0, np.flatnonzero(np.diff(days)) + 1, len(days)]
            values = np.empty((len(days), len(features)), dtype=np.float32)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                day = days[start]
                if day != current_day:
                    for previous in pending_src:
                        np.add.at(outgoing, previous, 1)
                    for previous in pending_dst:
                        np.add.at(incoming, previous, 1)
                    pending_src, pending_dst = [], []
                    current_day = day
                a, b = src[start:end], dst[start:end]
                if (np.any(a < 0) or np.any(b < 0) or np.any(node_ids[a] < 0) or np.any(node_ids[b] < 0)
                        or np.any(joined[a] > day) or np.any(joined[b] > day)):
                    raise ValueError("edge endpoint absent or created after the edge")
                values[start:end, 0] = node_ids[a] == node_ids[b]
                values[start:end, 1] = (day - joined[a]) / 3650
                values[start:end, 2] = (day - joined[b]) / 3650
                for column, count in enumerate((incoming[a], outgoing[a], incoming[b], outgoing[b]), 3):
                    values[start:end, column] = np.log1p(count) / 10
                pending_src.append(a.copy())
                pending_dst.append(b.copy())
            writer.write_batch(pa.RecordBatch.from_arrays(
                [pa.array(times), pa.array(labels), *(pa.array(values[:, i]) for i in range(len(features)))],
                schema=schema,
            ))


def prepare(root, dataset, scope):
    import duckdb
    from pyarrow.parquet import ParquetFile

    root = Path(root).resolve()
    spec = DATASETS[dataset]
    path, manifest_path = input_paths(root, dataset, scope)
    if manifest_path.exists():
        return load_manifest(root, dataset, scope)[1]
    raw = root / "raw" / spec["file"]
    expected_rows = SMOKE_ROWS if scope == "smoke" else spec["rows"]
    nodes = root / "raw" / spec["nodes"] if "nodes" in spec else None
    if sha256_file(raw) != spec["sha256"] or (nodes and sha256_file(nodes) != spec["nodes_sha256"]):
        raise ValueError("pinned source checksum mismatch")
    with ParquetFile(raw) as source:
        if source.metadata.num_rows != spec["rows"]:
            raise ValueError("unexpected raw dataset row count")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".prepare-") as temp:
        temp = Path(temp)
        output = temp / "prepared.parquet"
        sorted_path = temp / "edges.parquet" if nodes else output
        projection = {"fraud": FRAUD, "social": SOCIAL, "ecommerce": ECOMMERCE}[dataset]
        limit = f"LIMIT {SMOKE_ROWS}" if scope == "smoke" else ""
        with duckdb.connect(config={"threads": 1, "memory_limit": "768MB",
                                     "temp_directory": str(temp / "spill")}) as con:
            con.from_parquet(str(raw)).create_view("source_rows")
            con.execute(f"""
                COPY (SELECT * EXCLUDE (event_id) FROM
                      (SELECT {projection} FROM (SELECT * FROM source_rows {limit}))
                      ORDER BY event_time, event_id)
                TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)
            """, [str(sorted_path)])
        if nodes:
            social_features(sorted_path, nodes, output)
        with ParquetFile(output) as parquet:
            if parquet.metadata.num_rows != expected_rows:
                raise ValueError("prepared dataset row count differs from the input")
            features = parquet.schema_arrow.names[2:]
        manifest = {
            "format": FORMAT, "dataset": dataset, "scope": scope, "source": spec,
            "license": "CC-BY-NC-4.0", "rows": expected_rows, "features": features,
            "sha256": sha256_file(output), "bytes": output.stat().st_size,
        }
        output.replace(path)
        atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
    return manifest
