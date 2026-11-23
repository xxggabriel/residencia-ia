"""Run a versioned corrective campaign, retaining metrics rather than model checkpoints.

The same Python entry point runs on a workstation, a batch worker, or another
managed executor. Credentials and resource reservations belong to the caller.
Completed units may be reused; an interrupted unit restarts its evaluation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from continual_bench.application.matrix import ExperimentMatrix, case_key, describe
from continual_bench.config import Config
from continual_bench.config.loader import load_model
from continual_bench.evaluation.environment import environment
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files
from continual_bench.infrastructure.storage import atomic_write, run_lock

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "corrective-benchmarks-v1"


def save(path, data):
    atomic_write(path, json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity():
    env = environment()
    revision = env["git_revision"]
    if revision == "uncommitted":
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    return {
        "git_revision": revision,
        "code_sha256": env["code_sha256"],
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replication_cli_sha256": hashlib.sha256(
            (PROJECT / "scripts/replicate_upf.py").read_bytes()
        ).hexdigest(),
    }, env


def make_plan(args):
    matrix_path = Path(args.matrix).resolve()
    matrix = load_model(ExperimentMatrix, matrix_path)
    expanded = matrix.plan()
    assert not expanded["skipped"], "a corrective campaign must not silently omit configurations"
    if tuple(matrix.seeds) != tuple(args.seeds):
        raise ValueError("matrix and replication seeds must match the same three explicit seeds")
    cases = []
    for config in expanded["cases"]:
        if config.artifact_mode != "metrics_only":
            raise ValueError("all campaign cases require artifact_mode=metrics_only")
        if config.policy.name == "online":
            raise ValueError("Online training is deferred to the next campaign")
        cases.append(
            {
                "case_key": case_key(config),
                "coordinates": describe(config),
                "config": config.model_dump(mode="json"),
            }
        )
    jobs = []
    if "upf" in args.families:
        for name in args.replication_datasets:
            jobs.append({"job_id": "upf-" + name, "family": "upf", "dataset": name})
    if "paired" in args.families:
        for timing in ("same-batch", "next-batch"):
            for name in ("electricity", "airlines"):
                jobs.append(
                    {"job_id": timing + "-" + name, "family": "paired", "timing": timing, "dataset": name}
                )
    code, env = code_identity()
    sources = []
    data_directory = getattr(args, "data_directory", Path("state/datasets"))
    for admission in sorted(data_directory.glob("*.admission.json")):
        item = read(admission)
        data_path = data_directory / item["file"]
        with data_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["raw_sha256"]:
            raise ValueError("input differs from its admitted content hash")
        sources.append({**item, "file": data_path.name})
    return {
        "schema": SCHEMA,
        "campaign_id": args.output.name,
        "code": code,
        "environment": env,
        "data_sources": sources,
        "exclusions": [
            {
                "family": "matrix",
                "policy": "online",
                "reason": "requested_for_next_round",
                "seeds": args.seeds,
                "configurations": len(matrix.datasets) * len(matrix.models),
                "runs": len(matrix.datasets) * len(matrix.models) * len(matrix.seeds),
            }
        ],
        "resource_contract": {
            "parallel_cases": args.parallel_cases,
            "thread_environment": env["resources"]["thread_environment"],
            "container_or_host_metrics": "shared_with_concurrent_work; process_metrics_are_per_unit",
        },
        "matrix": {
            "spec_path": str(matrix_path.relative_to(PROJECT)),
            "spec_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "matrix_id": expanded["matrix"],
            "skipped": expanded["skipped"],
            "cases": cases if "matrix" in args.families else [],
        },
        "replications": {
            "jobs": jobs,
            "trials": len(args.seeds),
            "seeds": args.seeds,
            "full_source": True,
            "monte_carlo_samples": args.samples,
        },
        "tracking_required": args.tracking,
        "retention": "metrics_only; no persisted training state or weights",
    }


def all_jobs(plan):
    matrix_jobs = [
        {"job_id": "matrix-" + c["case_key"], "family": "matrix", **c} for c in plan["matrix"]["cases"]
    ]
    reps = plan["replications"]["jobs"]
    return reps + matrix_jobs


def assert_tracking_receipts(value):
    if isinstance(value, dict):
        if "mlflow" in value and value["mlflow"] is not None:
            receipt = value["mlflow"]
            if not isinstance(receipt, dict) or receipt.get("complete") is not True:
                raise RuntimeError("incomplete MLflow publication")
        for child in value.values():
            assert_tracking_receipts(child)
    elif isinstance(value, list):
        for child in value:
            assert_tracking_receipts(child)


def execute_job(args):
    plan = read(args.output / "campaign-plan.json")
    code, env = code_identity()
    if code != plan["code"] or env["packages"] != plan["environment"]["packages"]:
        raise ValueError("code or dependency contract changed after campaign planning")
    job = next(j for j in all_jobs(plan) if j["job_id"] == args.execute_job)
    started = now()
    result_path = args.output / "job-results" / (job["job_id"] + ".json")
    if job["family"] == "matrix":
        from continual_bench.application.runner import run
        from continual_bench.infrastructure.benchmark_tracking import publish_local_benchmark

        config = Config.model_validate(job["config"])
        root = args.output / "matrix" / "cases" / job["case_key"] / "binary"
        report = run(config, str(root))
        if report.get("complete") is not True:
            raise RuntimeError("evaluation did not finish the planned stream")
        if report.get("observed") != config.dataset.samples:
            raise RuntimeError("evaluation did not consume every configured example")
        directory = root / report["run_key"]
        verify_metrics_only_files(directory)
        receipt = None
        if plan["tracking_required"]:
            receipt = publish_local_benchmark(
                directory, report, experiment_name="continual-bench/corrective/" + plan["campaign_id"]
            )
            if receipt.get("complete") is not True:
                raise RuntimeError("incomplete matrix publication")
        result = {
            "status": "passed",
            "report": str((directory / "report.json").relative_to(args.output)),
            "tracking": receipt,
        }
    else:
        output = args.output / "replications" / (job["job_id"] + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PROJECT / "scripts/replicate_upf.py"),
            job["dataset"],
            "--seeds",
            *map(str, plan["replications"]["seeds"]),
            "--full-source",
            "--samples",
            str(plan["replications"]["monte_carlo_samples"]),
            "--metrics-only",
            "--output",
            str(output),
        ]
        if job["dataset"] in {"electricity", "airlines"}:
            command.extend(["--" + job["dataset"], str(args.data_directory / (job["dataset"] + ".arff"))])
        if job["family"] == "paired":
            command.extend(["--compare-oracles", "--decision-timing", job["timing"]])
        child_env = dict(os.environ, BENCH_STATE=str(args.output / "replications" / "tracking"))
        if not plan["tracking_required"]:
            child_env.pop("MLFLOW_TRACKING_URI", None)
        subprocess.run(command, cwd=PROJECT, env=child_env, check=True)
        values = read(output)
        if plan["tracking_required"]:
            assert_tracking_receipts(values)
        verify_metrics_only_files(args.output / "replications")
        result = {"status": "passed", "output": str(output.relative_to(args.output))}
    result.update(job_id=job["job_id"], code=code, started_at=started, finished_at=now())
    target = args.output / result.get("report", result.get("output"))
    result["artifact_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    save(result_path, result)


def run_campaign(args, plan):
    required_sources = {
        c["coordinates"]["dataset"]
        for c in plan["matrix"]["cases"]
        if c["coordinates"]["dataset"].startswith("insects_")
    }
    required_sources.update(
        j["dataset"] for j in plan["replications"]["jobs"] if j["dataset"] in {"electricity", "airlines"}
    )
    sources = {s["name"]: s for s in plan["data_sources"]}
    if required_sources - sources.keys():
        raise ValueError("every real source needs a validated admission before execution")
    for case in plan["matrix"]["cases"]:
        name = case["coordinates"]["dataset"]
        if name in sources and case["config"]["dataset"]["samples"] != sources[name]["raw_rows"]:
            raise ValueError("the corrective matrix must consume the admitted source in full")
    if plan["tracking_required"]:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise ValueError("--tracking requires an explicitly configured MLflow service")
        from mlflow.tracking import MlflowClient

        MlflowClient().search_experiments(max_results=1)
    status_path = args.output / "campaign-status.json"
    status = (
        read(status_path) if status_path.exists() else {"complete": False, "matrix": {}, "replications": {}}
    )
    jobs = all_jobs(plan)
    lock = threading.Lock()

    def category(job):
        return "matrix" if job["family"] == "matrix" else "replications"

    def key(job):
        return job.get("case_key", job["job_id"])

    def mark(job, value):
        with lock:
            status[category(job)][key(job)] = value
            status["complete"] = False
            status["updated_at"] = now()
            save(status_path, status)

    def launch(job):
        result_path = args.output / "job-results" / (job["job_id"] + ".json")
        if result_path.exists():
            previous = read(result_path)
            artifact = args.output / previous.get("report", previous.get("output"))
            if (
                previous["code"] != plan["code"]
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != previous["artifact_sha256"]
            ):
                raise ValueError("retained unit differs from its campaign receipt")
            mark(job, previous)
            return
        mark(job, {"status": "running", "started_at": now()})
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--execute-job",
            job["job_id"],
            "--data-directory",
            str(args.data_directory),
        ]
        log = args.output / "logs" / (job["job_id"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            mark(job, {"status": "failed", "exit_code": completed.returncode, "finished_at": now()})
        else:
            mark(job, read(result_path))
        print(job["job_id"], status[category(job)][key(job)]["status"], flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_cases) as pool:
        for future in concurrent.futures.as_completed([pool.submit(launch, j) for j in jobs]):
            future.result()
    status["complete"] = all(status[category(j)][key(j)]["status"] == "passed" for j in jobs)
    status["finished_at"] = now()
    save(status_path, status)
    for family in ("upf", "same-batch", "next-batch"):
        combined = []
        selected = [
            j
            for j in plan["replications"]["jobs"]
            if (j["family"] == "upf" and family == "upf") or j.get("timing") == family
        ]
        for job in selected:
            if status["replications"][job["job_id"]]["status"] == "passed":
                combined.extend(read(args.output / "replications" / (job["job_id"] + ".json")))
        if selected and combined:
            save(args.output / "replications" / (family + ".json"), combined)
    return 0 if status["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", default="configs/matrices/insects-corrected.yaml")
    parser.add_argument("--data-directory", type=Path, default=Path("state/datasets"))
    parser.add_argument("--parallel-cases", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--families", nargs="+", choices=("matrix", "upf", "paired"), default=["matrix", "upf", "paired"]
    )
    parser.add_argument(
        "--replication-datasets",
        nargs="+",
        choices=("gauss", "circles", "electricity", "airlines"),
        default=["electricity", "airlines", "gauss", "circles"],
    )
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.parallel_cases < 1 or args.samples < 2 or len(set(args.seeds)) != 3:
        parser.error("positive parallelism, three distinct seeds and at least two samples are required")
    args.output = args.output.resolve()
    args.data_directory = args.data_directory.resolve()
    os.environ["BENCH_DATASETS"] = str(args.data_directory)
    if args.execute_job:
        execute_job(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args)
    path = args.output / "campaign-plan.json"
    if path.exists():
        previous = read(path)
        if any(
            previous[k] != plan[k]
            for k in (
                "code",
                "data_sources",
                "matrix",
                "replications",
                "resource_contract",
                "tracking_required",
            )
        ):
            raise ValueError("changed campaign contract requires a new output directory")
        plan = previous
    else:
        save(path, plan)
    print(
        json.dumps(
            {
                "matrix_cases": len(plan["matrix"]["cases"]),
                "replication_jobs": len(plan["replications"]["jobs"]),
                "tracking": plan["tracking_required"],
            }
        ),
        flush=True,
    )
    if args.plan_only:
        return 0
    with run_lock(args.output, "campaign"):
        return run_campaign(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
