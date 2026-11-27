"""Plan and validate the common ordered-stream benchmark without model retention.

Each case runs in a separate process. The caller reserves container CPU and RAM;
``--workers`` controls processes inside that allocation, not a cloud reservation.
Interrupted cases restart from the beginning. Completed metric evidence is reused
only after its plan identity and file hashes have been checked.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from continual_bench.config import IterativeTrainingConfig, TrackingConfig
from continual_bench.config.loader import load_model
from continual_bench.datasets.streams import load_stream_manifest, sha256_file
from continual_bench.evaluation.environment import environment
from continual_bench.infrastructure.storage import atomic_write, run_lock

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "ordered-dataset-suite-v1"
DATASETS = (
    "electricity",
    "airlines",
    "insects_abrupt_imbalanced",
    "insects_incremental_abrupt_reoccurring_imbalanced",
    "fraud",
    "ecommerce",
    "social",
)
POLICIES = (
    "frozen",
    "drift",
    "drift_adwin",
    "drift_ddm",
    "incremental",
    "periodic",
    "cara",
    "cara_cumulative",
    "cara_periodic",
    "upf",
    "oracle",
)
SEEDS = (42, 43, 44)
BASE_POLICIES = POLICIES[:7]
CACHED_POLICIES = POLICIES[7:]
COST_POLICIES = frozenset(("cara", *CACHED_POLICIES))


def save(path, value):
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        "launcher_sha256": sha256_file(__file__),
        "verifier_sha256": sha256_file(PROJECT / "scripts/verify_dataset_suite.py"),
    }, env


def make_plan(args):
    if tuple(args.seeds) != SEEDS:
        raise ValueError("the suite requires seeds 42, 43 and 44 in that order")
    if len(set(args.datasets)) != len(args.datasets) or not set(args.datasets) <= set(DATASETS):
        raise ValueError("unknown or repeated dataset")
    allowed = set(POLICIES) | {"online"}
    if len(set(args.policies)) != len(args.policies) or not set(args.policies) <= allowed:
        raise ValueError("unknown or repeated policy")
    if "online" in args.policies and not args.online_only:
        raise ValueError("Online requires --online-only and is excluded from the AWS suite")
    if args.online_only and args.policies != ["online"]:
        raise ValueError("--online-only requires --policies online")
    settings = load_model(TrackingConfig, args.tracking_config).model_dump(mode="json")
    training = IterativeTrainingConfig(max_epochs=args.max_epochs, patience=args.max_epochs)
    sources = {}
    for dataset in args.datasets:
        path, manifest = load_stream_manifest(args.root, dataset, args.scope)
        if sha256_file(path) != manifest["sha256"]:
            raise ValueError(f"prepared source checksum differs for {dataset}")
        sources[dataset] = {"manifest": manifest, "manifest_sha256": digest(manifest)}
    code, env = code_identity()
    cases = []
    for dataset in args.datasets:
        for policy in args.policies:
            for seed in args.seeds:
                config = {
                    "dataset": dataset,
                    "scope": args.scope,
                    "policy": policy,
                    "seed": seed,
                    "artifact_mode": "metrics_only",
                    "input_sha256": sources[dataset]["manifest"]["sha256"],
                    "iterative_training": {**training.model_dump(mode="json"), "seed": seed},
                    "tracking_settings": {**deepcopy(settings), "retraining": {"seed": seed}},
                }
                cases.append({"case_id": digest(config)[:24], "config": config})
    return {
        "schema": SCHEMA,
        "code": code,
        "environment": env,
        "scope": args.scope,
        "sources": sources,
        "cases": cases,
        "tracking_required": args.tracking,
        "resource_contract": {"workers": args.workers, "reservation": "supplied_by_caller"},
        "excluded_policies": [] if args.online_only else ["online"],
        "seeds": list(SEEDS),
        "protocol": {
            "name": "ordered-stream-batch-prequential-metrics-v2",
            "label_availability": "labels become available after the complete prediction block",
            "holdout": "final temporal 10 percent, adjusted to keep timestamp ties together",
            "primary_comparison": "common prequential suffix after every policy calibration phase",
            "ranking": "uniform_histogram_4096_bins",
            "seed_interpretation": "computational repetitions of the same real ordered source",
            "scope_warning": "smoke uses bounded source prefixes and its declared epoch budget",
        },
    }


def execute_case(args):
    from continual_bench.evaluation.stream_benchmark import run
    from continual_bench.evaluation.stream_artifacts import tracking_report, write_zia_artifacts
    from continual_bench.evaluation.stream_predictions import prediction_manifest

    if __package__:
        from .verify_dataset_suite import verify_case
    else:
        from verify_dataset_suite import verify_case

    plan = read(args.output / "suite-plan.json")
    code, env = code_identity()
    if code != plan["code"] or env["packages"] != plan["environment"]["packages"]:
        raise ValueError("code or package versions changed after suite planning")
    case = next(c for c in plan["cases"] if c["case_id"] == args.execute_case)
    config, key = case["config"], case["case_id"]
    directory = args.output / "cases" / key
    result_path = args.output / "results" / (key + ".json")
    if result_path.exists():
        result = read(result_path)
        if result["case"] != case or result["code"] != code:
            raise ValueError("retained case receipt differs from the suite plan")
        for path, expected in result["files"].items():
            if sha256_file(directory / path) != expected:
                raise ValueError("retained case evidence changed")
        verify_case(directory, case)
        return
    # An incomplete attempt has no reusable training state in this mode.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    started = now()
    kwargs = {"iterative_config": config["iterative_training"], "artifact_mode": "metrics_only"}
    if config["policy"] in COST_POLICIES:
        cache = args.output / "matrix-metrics" / config["dataset"] / f"seed-{config['seed']}.json.gz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        kwargs["matrix_metrics_cache"] = cache
    publisher = None
    try:
        if plan["tracking_required"]:
            if not os.environ.get("MLFLOW_TRACKING_URI"):
                raise ValueError("tracking requires an explicitly configured MLflow endpoint")
            from mlflow.tracking import MlflowClient

            from continual_bench.evaluation.stream_publication import publish
            from continual_bench.infrastructure.stream_tracking import begin

            client = MlflowClient()
            run_id, experiment = begin(
                client, directory, config["dataset"], config["scope"], config["policy"], args.output.name
            )
            # Retain the real run ID before the first subsequent tracking write,
            # so a metadata or publication failure also terminates this parent.
            publisher = (client, run_id, experiment, publish)
            for tag, value in {"seed": str(config["seed"]), "artifact_mode": "metrics_only"}.items():
                client.set_tag(run_id, tag, value)
        report = run(
            args.root,
            config["dataset"],
            config["scope"],
            config["policy"],
            directory,
            config["tracking_settings"],
            **kwargs,
        )
        predictions = prediction_manifest(directory, report)
        adapted = tracking_report(report, predictions)
        write_zia_artifacts(directory, report, adapted)
        receipt = None
        if publisher is not None:
            client, run_id, experiment, publish = publisher
            receipt = publish(client, directory, run_id, experiment, report, args.root)
            if receipt.get("complete") is not True:
                raise ValueError("MLflow publication is incomplete")
        qa = verify_case(directory, case)
        save(directory / "suite-case-qa.json", qa)
        files = {
            p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        }
        save(
            result_path,
            {
                "status": "passed",
                "case": case,
                "code": code,
                "started_at": started,
                "finished_at": now(),
                "files": files,
                "tracking": receipt,
            },
        )
    except BaseException:
        if publisher is not None:
            publisher[0].set_terminated(publisher[1], "FAILED")
        raise


def run_suite(args, plan):
    state = {"complete": False, "cases": {}, "planned": len(plan["cases"])}
    lock = threading.Lock()

    def launch(case):
        key = case["case_id"]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(args.root),
            "--output",
            str(args.output),
            "--execute-case",
            key,
        ]
        log = args.output / "logs" / (key + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        result = {
            "status": "passed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "finished_at": now(),
        }
        with lock:
            state["cases"][key] = result
            save(args.output / "suite-status.json", state)
        coordinates = {k: case["config"][k] for k in ("dataset", "policy", "seed", "scope")}
        print(json.dumps({"case_id": key, **coordinates, **result}), flush=True)

    def launch_group(cases):
        # A single CARA case populates the immutable metrics cache before the other
        # cost policies reuse it. No model or optimizer state is shared or retained.
        for case in sorted(cases, key=lambda c: c["config"]["policy"] != "cara"):
            launch(case)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        groups = {}
        for case in plan["cases"]:
            group = case["config"]["dataset"], case["config"]["seed"]
            groups.setdefault(group, []).append(case)
        futures = [pool.submit(launch_group, cases) for cases in groups.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    state["complete"] = all(c["status"] == "passed" for c in state["cases"].values())
    state["finished_at"] = now()
    save(args.output / "suite-status.json", state)
    return 0 if state["complete"] else 1


def write_requests(args, plan):
    """Write inputs only; never deploy infrastructure or start an AWS execution."""
    from continual_bench.application.cloud.contracts import CampaignRequest
    from continual_bench.datasets.pinned_stream import DATASETS as ZIA

    if len(args.release or "") != 40 or any(c not in "0123456789abcdef" for c in args.release):
        raise ValueError("--write-requests requires an immutable 40-character --release SHA")
    if "online" in {c["config"]["policy"] for c in plan["cases"]}:
        raise ValueError("Online must not be included in AWS requests")
    args.write_requests.mkdir(parents=True, exist_ok=True)
    written = []
    metric_sources = read(args.metric_cache_sources) if args.metric_cache_sources else {}
    for dataset, source in plan["sources"].items():
        configs = [c["config"] for c in plan["cases"] if c["config"]["dataset"] == dataset]
        training = {k: v for k, v in configs[0]["iterative_training"].items() if k != "seed"}
        policies = BASE_POLICIES if args.request_stage == "base" else CACHED_POLICIES
        selected = [p for p in policies if p in {c["policy"] for c in configs}]
        if not selected:
            continue
        suffix = hashlib.sha256(dataset.encode()).hexdigest()[:6]
        campaign_id = f"ds-{suffix}-{plan['scope']}-{args.request_stage}-{args.release[:8]}"
        request = {
            "campaign_id": campaign_id,
            "release": args.release,
            "dataset": dataset,
            "scope": plan["scope"],
            "tracking": Path(args.tracking_config).resolve().relative_to(PROJECT).as_posix(),
            "matrix": {
                "kind": "zia" if dataset in ZIA else "stream",
                "policies": selected,
                "seeds": list(SEEDS),
                "artifact_mode": "metrics_only",
                "iterative_training": training,
            },
            "max_parallel": 32,
            "max_attempts": 2,
            "queue_timeout_seconds": 86400,
            "run_timeout_seconds": 14 * 24 * 3600,
        }
        if args.request_stage == "cached":
            cache_sources = metric_sources.get(dataset)
            if not cache_sources or set(cache_sources) != {str(s) for s in SEEDS}:
                raise ValueError("cached requests require all three CARA metrics-cache receipts")
            request["matrix"]["metric_cache_sources"] = cache_sources
        if dataset in ZIA:
            request["snapshot"] = {
                "dataset": dataset,
                "scope": plan["scope"],
                "revision": ZIA[dataset]["revision"],
                "preparation_version": "zia-temporal-v1",
            }
        else:
            request["source_digests"] = {dataset: source["manifest"]["sha256"]}
        CampaignRequest.model_validate(request)
        target = args.write_requests / (dataset + ".json")
        save(target, {"request": request})
        written.append({"dataset": dataset, "campaign_id": campaign_id, "file": target.name})
    save(
        args.write_requests / "index.json",
        {
            "requests": written,
            "release": args.release,
            "stage": args.request_stage,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--policies", nargs="+", choices=(*POLICIES, "online"), default=list(POLICIES))
    parser.add_argument("--seeds", nargs=3, type=int, default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument(
        "--tracking-config", type=Path, default=PROJECT / "configs/base/tracking-dataset-suite.yaml"
    )
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--online-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-case", help=argparse.SUPPRESS)
    parser.add_argument("--write-requests", type=Path)
    parser.add_argument("--request-stage", choices=("base", "cached"), default="base")
    parser.add_argument("--metric-cache-sources", type=Path)
    parser.add_argument("--release")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.max_epochs <= 100:
        parser.error("positive workers and 1..100 epochs are required")
    if args.scope == "full" and args.max_epochs != 100 and not args.execute_case:
        parser.error("full suite requires the declared 100-epoch training budget")
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.execute_case:
        execute_case(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args)
    target = args.output / "suite-plan.json"
    if target.exists():
        previous = read(target)
        if any(previous[k] != plan[k] for k in ("code", "sources", "cases", "tracking_required")):
            raise ValueError("changed suite contract requires a new output directory")
        plan = previous
    else:
        save(target, plan)
    print(
        json.dumps({"cases": len(plan["cases"]), "scope": plan["scope"], "tracking": args.tracking}),
        flush=True,
    )
    if args.write_requests:
        write_requests(args, plan)
    if args.plan_only:
        return 0
    with run_lock(args.output, "dataset-suite"):
        return run_suite(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
