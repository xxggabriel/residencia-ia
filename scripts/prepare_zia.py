"""Run only on the machine holding the pinned Hugging Face files."""

import argparse
import json
import subprocess
from pathlib import Path

from continual_bench.datasets.zia import DATASETS, SCOPES, prepare, sha256_file


def download(root, dataset):
    spec = DATASETS[dataset]
    files = [(spec["file"], spec["sha256"])]
    if "nodes" in spec:
        files.append((spec["nodes"], spec["nodes_sha256"]))
    folder = root / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in files:
        path = folder / name
        if path.is_file() and sha256_file(path) == expected:
            continue
        partial = path.with_suffix(".partial")
        url = f"https://huggingface.co/datasets/ziadatalabs/{spec['repo']}/resolve/{spec['revision']}/{name}"
        if not partial.is_file() or sha256_file(partial) != expected:
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "4", "--connect-timeout", "15",
                 "--continue-at", "-", "--output", str(partial), url], check=True,
            )
        if sha256_file(partial) != expected:
            partial.unlink()
            raise ValueError(f"downloaded checksum mismatch: {name}")
        partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("--download", action="store_true", help="Fetch full source files on the homelab")
    args = parser.parse_args()
    if args.download:
        download(args.root, args.dataset)
    print(json.dumps(prepare(args.root, args.dataset, args.scope), indent=2))


if __name__ == "__main__":
    main()
