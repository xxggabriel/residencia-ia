"""CI smoke of the AWS profile with fixtures: plan → four overlapping unit processes → consolidate."""

import json
import subprocess
import sys


def test_local_smoke_runs_four_overlapping_units(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "continual_bench.cli",
            "aws",
            "local-smoke",
            "--root",
            str(tmp_path),
            "--timeout",
            "600",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    summary = json.loads(result.stdout)
    assert summary["complete"] and summary["succeeded"] == summary["total"] == 4
    assert summary["exit_codes"] == [0, 0, 0, 0] and summary["overlapping"]
