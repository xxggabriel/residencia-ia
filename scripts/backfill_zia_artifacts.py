"""Complete selected, finished Zia experiments from retained reports without fitting models."""

import argparse
import json
from pathlib import Path

from continual_bench.evaluation.environment import encoded
from continual_bench.evaluation.zia_publication import publish_artifacts
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.orchestration.zia_jobs import DATA_ROOT, JOB_ROOT


def main():
    from mlflow.tracking import MlflowClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_ids", nargs="+")
    parser.add_argument("--jobs", type=Path, default=JOB_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    client = MlflowClient()
    for experiment_id in args.experiment_ids:
        experiment = client.get_experiment(experiment_id)
        if not experiment.name.startswith("continual-bench/zia-"):
            raise ValueError(f"not a Zia experiment: {experiment_id}")
        for run in client.search_runs([experiment_id], filter_string="attributes.status = 'FINISHED'"):
            key = run.data.tags["zia_run_key"]
            if len(key) != 24 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("invalid Zia run key")
            directory = args.jobs / key
            with run_lock(args.jobs / ".job-claims", key):
                status = json.loads((directory / "job.json").read_text())
                receipt = json.loads((directory / "receipt.json").read_text())
                if status["status"] != "succeeded" or receipt["mlflow_run_id"] != run.info.run_id:
                    raise ValueError(f"local Zia receipt differs from MLflow: {key}")
                report = json.loads((directory / "report.json").read_text())
                result = publish_artifacts(
                    client, directory, run.info.run_id, report, args.data, derived=True,
                    elapsed_seconds=status["finished_at"] - status["started_at"],
                )
                receipt.update(result)
                atomic_write(directory / "receipt.json", encoded(receipt))
                status["receipt"] = receipt
                atomic_write(directory / "job.json", encoded(status))
                print(experiment_id, run.info.run_id, result["run_artifacts"]["artifact_files"], flush=True)


if __name__ == "__main__":
    main()
