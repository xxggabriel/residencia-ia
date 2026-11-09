"""Synthetic fixtures for the CI smoke: a published smoke snapshot and a release manifest.

The fixture follows the real layout and verification path (`verify_snapshot`, `materialize`,
`load_manifest`), but its raw section is marked as a fixture: no public file is downloaded.
"""

import hashlib
import json
from pathlib import Path

from continual_bench.datasets.zia import DATASETS, FORMAT, SMOKE_ROWS
from continual_bench.infrastructure.object_store import put_json, put_json_once

from . import layout
from .contracts import ReleaseManifest, SnapshotRef
from .snapshots import SNAPSHOT_VERSION

FIXTURE_SHA = "0" * 39 + "1"


def fixture_parquet(path: Path, dataset="social") -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from continual_bench.datasets.zia import sha256_file

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    classes = len(DATASETS[dataset]["classes"])
    timestamps = np.arange(SMOKE_ROWS, dtype=np.int64) // 13
    labels = (np.arange(SMOKE_ROWS) % classes).astype(np.uint8)
    noise = rng.normal(0, 0.05, SMOKE_ROWS).astype(np.float32)
    pq.write_table(
        pa.table({"event_time": timestamps, "label": labels, "feature": labels.astype(np.float32) + noise}),
        path,
    )
    manifest = {
        "dataset": dataset,
        "scope": "smoke",
        "source": DATASETS[dataset],
        "format": FORMAT,
        "license": "CC-BY-NC-4.0",
        "features": ["feature"],
        "rows": SMOKE_ROWS,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def publish_fixture_snapshot(store, workdir: Path, dataset="social") -> SnapshotRef:
    snapshot = SnapshotRef(dataset=dataset, revision=DATASETS[dataset]["revision"], scope="smoke")
    parquet = Path(workdir) / "fixture" / dataset / "smoke.parquet"
    manifest = fixture_parquet(parquet, dataset)
    parquet_key, manifest_key = (
        layout.prepared_key(snapshot, "parquet"),
        layout.prepared_key(snapshot, "json"),
    )
    if store.head(parquet_key) is None:
        store.put_file(parquet_key, parquet, if_absent=True, sha256=manifest["sha256"])
        store.put_file(manifest_key, parquet.with_suffix(".json"), if_absent=True)
    stored_parquet, stored_manifest = store.head(parquet_key), store.head(manifest_key)
    record = {
        "version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot": snapshot.model_dump(),
        "license": "CC-BY-NC-4.0",
        "format": FORMAT,
        "rows": SMOKE_ROWS,
        "features": ["feature"],
        "schema": [
            {"name": "event_time", "type": "int64"},
            {"name": "label", "type": "uint8"},
            {"name": "feature", "type": "float"},
        ],
        "ordering": ["event_time", "event_id"],
        "raw": {"fixture": True, "files": []},
        "prepared": {
            "parquet": {
                "key": parquet_key,
                "sha256": stored_parquet.sha256,
                "bytes": stored_parquet.size,
                "version_id": stored_parquet.version_id,
            },
            "manifest": {
                "key": manifest_key,
                "sha256": stored_manifest.sha256,
                "bytes": stored_manifest.size,
            },
            "sha256": stored_parquet.sha256,
            "bytes": stored_parquet.size,
        },
        "transform": {"version": FORMAT, "code_sha256": None, "image_digest": None},
    }
    put_json_once(store, layout.snapshot_manifest_key(snapshot), record)
    data = store.get_bytes(layout.snapshot_manifest_key(snapshot))
    put_json_once(
        store,
        layout.snapshot_marker_key(snapshot),
        {
            "snapshot_id": snapshot.snapshot_id,
            "manifest_key": layout.snapshot_manifest_key(snapshot),
            "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "manifest_version_id": None,
            "prepared_sha256": stored_parquet.sha256,
        },
    )
    return snapshot


def publish_fixture_release(store, git_sha=FIXTURE_SHA, account="000000000000") -> ReleaseManifest:
    image = {
        "repository": f"{account}.dkr.ecr.us-east-1.amazonaws.com/continual-bench-runtime",
        "digest": "sha256:" + hashlib.sha256(git_sha.encode()).hexdigest(),
        "tag": f"git-{git_sha}",
    }
    release = ReleaseManifest(
        git_sha=git_sha,
        stack="continual-bench-local",
        images={
            "runtime": image,
            "ml": {**image, "repository": image["repository"].replace("runtime", "ml")},
        },
        job_definitions={
            "benchmark": f"arn:aws:batch:us-east-1:{account}:job-definition/cb-benchmark:1",
            "prepare": f"arn:aws:batch:us-east-1:{account}:job-definition/cb-prepare:1",
        },
        control_function=f"arn:aws:lambda:us-east-1:{account}:function:cb-control:1",
    )
    put_json_once(store, layout.release_key(git_sha), release.model_dump(mode="json"))
    put_json(store, layout.LATEST_RELEASE, {"git_sha": git_sha})
    return release
