"""Explicitly acquire and validate pinned OpenML sources before running benchmarks."""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from continual_bench.adapters.local import atomic_write
from continual_bench.adapters.real_sources import REAL_SOURCES, validate_source


def prepare(name, directory, source_directory=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
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
    receipt.update({"source": REAL_SOURCES[name]["url"], "openml_id": REAL_SOURCES[name]["openml_id"],
                    "openml_version": REAL_SOURCES[name]["version"],
                    "license": REAL_SOURCES[name]["license"]})
    atomic_write(directory / f"{name}.admission.json", json.dumps(receipt, indent=2).encode())
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=os.environ.get("BENCH_DATASETS", "/state/datasets"))
    parser.add_argument("--source-directory", help="Copy pre-downloaded pinned ARFF files instead of HTTP")
    parser.add_argument("--datasets", nargs="+", choices=sorted(REAL_SOURCES), default=sorted(REAL_SOURCES))
    args = parser.parse_args()
    for name in args.datasets:
        receipt = prepare(name, args.directory, args.source_directory)
        print(json.dumps({key: receipt[key] for key in ("name", "raw_sha256", "raw_rows", "license")}))


if __name__ == "__main__":
    main()
