"""Explicitly acquire, validate, and optionally publish pinned OpenML and INSECTS sources."""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from continual_bench.datasets.arff import file_sha256
from continual_bench.datasets.openml import REAL_SOURCES, validate_source
from continual_bench.datasets.real_streams import SPECS as STREAM_SPECS
from continual_bench.datasets.real_streams import _download
from continual_bench.datasets.real_streams import _validate as validate_stream
from continual_bench.infrastructure.storage import atomic_write

INSECTS_SOURCES = (
    "insects",
    "insects_abrupt_imbalanced",
    "insects_incremental_abrupt_reoccurring_imbalanced",
)
ALL_DATASETS = tuple(sorted([*REAL_SOURCES, *INSECTS_SOURCES]))


def _prepare_openml(name, directory, source_directory=None):
    destination = directory / f"{name}.arff"
    if destination.exists():
        receipt = validate_source(destination, name)
    else:
        fd, filename = tempfile.mkstemp(dir=directory, prefix=f".pending-{name}-")
        os.close(fd)
        temporary = Path(filename)
        try:
            if source_directory:
                shutil.copyfile(Path(source_directory) / destination.name, temporary)
            else:
                with urllib.request.urlopen(REAL_SOURCES[name]["url"], timeout=90) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
            receipt = validate_source(temporary, name)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    receipt.update(
        {
            "source": REAL_SOURCES[name]["url"],
            "openml_id": REAL_SOURCES[name]["openml_id"],
            "openml_version": REAL_SOURCES[name]["version"],
            "license": REAL_SOURCES[name]["license"],
            "file": destination.name,
        }
    )
    atomic_write(directory / f"{name}.admission.json", json.dumps(receipt, indent=2).encode())
    return receipt


def _validate_insects_file(name, path):
    spec = STREAM_SPECS[name]
    actual_bytes = Path(path).stat().st_size
    if spec.get("expected_bytes") is not None and actual_bytes != spec["expected_bytes"]:
        raise ValueError(f"Pinned {name} byte length mismatch: {actual_bytes} != {spec['expected_bytes']}")
    raw_sha256 = file_sha256(path)
    if spec.get("expected_sha256") is not None and raw_sha256 != spec["expected_sha256"]:
        raise ValueError(f"Pinned {name} SHA-256 mismatch")
    admission = validate_stream(name, path)
    return raw_sha256, actual_bytes, admission


def _prepare_insects(name, directory, source_directory=None):
    spec = STREAM_SPECS[name]
    destination = directory / spec["filename"]
    if destination.exists():
        raw_sha256, actual_bytes, admission = _validate_insects_file(name, destination)
    else:
        fd, filename = tempfile.mkstemp(dir=directory, prefix=f".pending-{name}-")
        os.close(fd)
        temporary = Path(filename)
        try:
            if source_directory:
                shutil.copyfile(Path(source_directory) / spec["filename"], temporary)
            else:
                _download(spec["url"], temporary, max_bytes=spec["max_bytes"], timeout=300)
            raw_sha256, actual_bytes, admission = _validate_insects_file(name, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    receipt = {
        "name": name,
        "variant": spec["variant"],
        "source": spec["source"],
        "url": spec["url"],
        "license": spec["license"],
        "license_reference": spec["license_reference"],
        "file": destination.name,
        "raw_sha256": raw_sha256,
        "raw_bytes": actual_bytes,
        "raw_rows": admission["rows"],
        "labels": admission["labels"],
        "changepoints": list(spec["changepoints"]) if spec.get("changepoints") else None,
        "target_transformation": spec.get("target_transformation"),
        "binary_distribution_per_regime": admission.get("binary_distribution_per_regime"),
    }
    atomic_write(directory / f"{name}.admission.json", json.dumps(receipt, indent=2).encode())
    return receipt


def prepare(name, directory, source_directory=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if name in REAL_SOURCES:
        return _prepare_openml(name, directory, source_directory=source_directory)
    if name in INSECTS_SOURCES:
        return _prepare_insects(name, directory, source_directory=source_directory)
    raise ValueError(f"unknown real source {name}")


def publish_s3(receipt, directory, bucket, prefix="datasets/real-sources", *, client=None):
    if client is None:
        import boto3

        client = boto3.client("s3")
    directory = Path(directory)
    prefix = prefix.strip("/")
    data_file = directory / receipt["file"]
    admission_file = directory / f"{receipt['name']}.admission.json"
    rel = f"{receipt['name']}/{receipt['file']}"
    data_key = f"{prefix}/{rel}" if prefix else rel
    admission_key = (
        f"{prefix}/{receipt['name']}/{admission_file.name}"
        if prefix
        else f"{receipt['name']}/{admission_file.name}"
    )
    client.upload_file(str(data_file), bucket, data_key)
    client.upload_file(str(admission_file), bucket, admission_key)
    return {
        "data_uri": f"s3://{bucket}/{data_key}",
        "admission_uri": f"s3://{bucket}/{admission_key}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=os.environ.get("BENCH_DATASETS", "/state/datasets"))
    parser.add_argument("--source-directory", help="Copy pre-downloaded pinned files instead of HTTP")
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument(
        "--publish-s3-bucket",
        help="Optional S3 bucket to publish validated files and receipts",
    )
    parser.add_argument("--publish-s3-prefix", default="datasets/real-sources")
    args = parser.parse_args()
    for name in args.datasets:
        receipt = prepare(name, args.directory, args.source_directory)
        summary = {key: receipt[key] for key in ("name", "raw_sha256", "raw_rows", "license")}
        if args.publish_s3_bucket:
            summary["published"] = publish_s3(
                receipt, args.directory, args.publish_s3_bucket, args.publish_s3_prefix
            )
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
