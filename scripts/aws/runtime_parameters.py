"""Compose CloudFormation parameters for the runtime stack from build outputs and overrides.

Usage: python scripts/aws/runtime_parameters.py build-output.json [overrides.json] > params.json
Overrides (optional JSON object) carry operator choices such as EnableStreaming or
MonthlyBudgetUsd; image digests, Git SHA and the control Lambda come only from the build.
"""

import json
import sys
from pathlib import Path

PINNED = {"GitSha", "RuntimeImage", "MlImage", "ControlCodeKey", "ControlCodeSha256"}


def compose(build: dict, overrides: dict | None = None) -> list[dict]:
    images = build["images"]
    values = {
        "GitSha": build["git_sha"],
        "RuntimeImage": f"{images['runtime']['repository']}@{images['runtime']['digest']}",
        "MlImage": f"{images['ml']['repository']}@{images['ml']['digest']}",
        "ControlCodeKey": build["control"]["key"],
        "ControlCodeSha256": build["control"]["code_sha256"],
    }
    for key, value in (overrides or {}).items():
        if key in PINNED:
            raise SystemExit(f"{key} is pinned by the build and cannot be overridden")
        values[key] = str(value)
    return [{"ParameterKey": k, "ParameterValue": v} for k, v in sorted(values.items())]


if __name__ == "__main__":
    build = json.loads(Path(sys.argv[1]).read_text())
    overrides = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    print(json.dumps(compose(build, overrides), indent=2))
