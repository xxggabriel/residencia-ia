"""Build the control Lambda zip deterministically and publish it once per content digest.

Contents: the `continual_bench` package, the configuration files the planner reads, and the
pure-Python/manylinux wheels of pydantic and PyYAML pinned (with hashes) in requirements-aws.lock.
boto3 comes from the Lambda runtime. Fixed timestamps and ordering make the zip reproducible,
so re-running the same commit yields the same CodeSha256 and no new Lambda version.

Usage: python scripts/aws/build_control_lambda.py GIT_SHA [--bucket B] [--output build/control.zip]
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "typing-extensions",
    "typing-inspection",
    "pyyaml",
)
CONFIGS = ("configs/base", "configs/matrices", "configs/contracts")
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def locked_subset(lock: Path, names=PACKAGES) -> str:
    """Requirement blocks (name==version + hashes) for the given packages from a uv export."""
    blocks, current = [], []
    for line in lock.read_text().splitlines():
        if re.match(r"^[A-Za-z0-9]", line):
            if current:
                blocks.append(current)
            current = [line]
        elif line.startswith("    --hash") and current:
            current.append(line)
    if current:
        blocks.append(current)
    wanted = []
    for block in blocks:
        name = re.split(r"[=; ]", block[0], maxsplit=1)[0].lower()
        if name in names:
            wanted.append(
                "\n".join(
                    line.rstrip(" \\") + (" \\" if i < len(block) - 1 else "") for i, line in enumerate(block)
                )
            )
    missing = set(names) - {re.split(r"[=; ]", w, maxsplit=1)[0].lower() for w in wanted}
    if missing:
        raise SystemExit(f"packages missing from {lock}: {sorted(missing)}")
    return "\n".join(wanted) + "\n"


def install_dependencies(target: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as requirements:
        requirements.write(locked_subset(ROOT / "requirements-aws.lock"))
    if importlib.util.find_spec("pip") is not None:  # CodeBuild images ship pip
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--platform",
            "manylinux2014_x86_64",
            "--python-version",
            "3.12",
            "--implementation",
            "cp",
            "--only-binary=:all:",
        ]
    elif shutil.which("uv"):  # uv-managed virtualenvs have no pip
        command = [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python-platform",
            "x86_64-manylinux2014",
            "--python-version",
            "3.12",
            "--only-binary",
            ":all:",
        ]
    else:
        raise SystemExit("pip or uv is required to fetch the locked Lambda wheels")
    subprocess.run(
        [*command, "--no-deps", "--require-hashes", "--target", str(target), "-r", requirements.name],
        check=True,
    )


def files(staging: Path):
    for path in sorted(staging.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
            yield path


def build(output: Path) -> dict:
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        install_dependencies(staging)
        for path in (ROOT / "src" / "continual_bench").rglob("*.py"):
            target = staging / path.relative_to(ROOT / "src")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        for folder in CONFIGS:
            for path in (ROOT / folder).rglob("*"):
                if path.is_file():
                    target = staging / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files(staging):
                info = zipfile.ZipInfo(path.relative_to(staging).as_posix(), FIXED_TIME)
                info.external_attr = 0o644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    data = output.read_bytes()
    return {
        "path": str(output),
        "bytes": len(data),
        "sha256_hex": hashlib.sha256(data).hexdigest(),
        "code_sha256": base64.b64encode(hashlib.sha256(data).digest()).decode(),
    }


def publish(bucket: str, key: str, path: Path, code_sha256: str) -> str:
    head = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key], capture_output=True, text=True
    )
    if head.returncode == 0:
        existing = json.loads(head.stdout).get("Metadata", {}).get("code-sha256")
        if existing != code_sha256:
            raise SystemExit(f"s3://{bucket}/{key} holds a different zip ({existing}); refusing to overwrite")
        return "reused"
    subprocess.run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(path),
            "--metadata",
            f"code-sha256={code_sha256}",
            "--if-none-match",
            "*",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return "uploaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("git_sha")
    parser.add_argument("--bucket")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "control.zip")
    parser.add_argument("--build-output", type=Path, default=Path("build-output.json"))
    args = parser.parse_args()
    result = build(args.output)
    key = f"lambda/{args.git_sha}/control-{result['sha256_hex'][:16]}.zip"
    if args.bucket:
        result["upload"] = publish(args.bucket, key, args.output, result["code_sha256"])
    state = json.loads(args.build_output.read_text()) if args.build_output.is_file() else {}
    state["control"] = {"key": key, "code_sha256": result["code_sha256"], "bytes": result["bytes"]}
    args.build_output.write_text(json.dumps(state, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
