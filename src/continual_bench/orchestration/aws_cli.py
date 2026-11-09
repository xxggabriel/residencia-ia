"""`continual-bench aws <action>`: entry points of CodeBuild, Batch jobs and operators."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _store(args):
    if getattr(args, "local_root", None):
        from continual_bench.infrastructure.object_store import LocalObjectStore

        return LocalObjectStore(args.local_root)
    from continual_bench.infrastructure.aws.s3 import S3ObjectStore

    bucket = args.bucket or os.environ.get("BENCH_BUCKET")
    if not bucket:
        raise SystemExit("--bucket or BENCH_BUCKET is required (or --local-root for offline runs)")
    return S3ObjectStore(bucket, kms_key_id=os.environ.get("BENCH_KMS_KEY_ID"))


def _emit(value):
    print(json.dumps(value, indent=2, default=str))


def acquire(args):
    from continual_bench.application.cloud.contracts import DatasetRequest
    from continual_bench.application.cloud.snapshots import acquire as run

    request = DatasetRequest(
        dataset=args.dataset,
        scope=args.scope,
        revision=args.revision,
        mode=args.mode,
        import_uri=args.import_uri,
    )
    store = _store(args)
    importer = getattr(store, "download_uri", None)
    with tempfile.TemporaryDirectory(dir=args.workdir) as workdir:
        result = run(
            store,
            request.snapshot(),
            workdir,
            mode=request.mode,
            import_uri=request.import_uri,
            importer=importer,
        )
    _emit(
        {k: result[k] for k in ("raw_manifest_uri", "created")}
        | {
            "files": [
                {k: f.get(k) for k in ("name", "sha256", "bytes", "retrieval", "version_id")}
                for f in result["files"]
            ]
        }
    )


def prepare(args):
    from continual_bench.application.cloud.contracts import DatasetRequest
    from continual_bench.application.cloud.snapshots import prepare_snapshot
    from continual_bench.evaluation.system_metrics import SystemMetricsSampler

    snapshot = DatasetRequest(
        dataset=args.dataset, scope=args.scope, revision=args.revision, mode="prepare"
    ).snapshot()
    with (
        tempfile.TemporaryDirectory(dir=args.workdir) as workdir,
        SystemMetricsSampler(workdir, 5) as sampler,
    ):
        result = prepare_snapshot(
            _store(args), snapshot, workdir, image_digest=os.environ.get("BENCH_IMAGE_DIGEST")
        )
    _emit(
        {
            "reused": result["reused"],
            "rows": result["prepared"]["rows"],
            "sha256": result["prepared"]["sha256"],
            "resources": sampler.report()["summary"],
        }
    )


def publish_snapshot(args):
    from continual_bench.application.cloud.contracts import DatasetRequest
    from continual_bench.application.cloud.snapshots import publish_snapshot as run

    snapshot = DatasetRequest(
        dataset=args.dataset, scope=args.scope, revision=args.revision, mode="prepare"
    ).snapshot()
    result = run(_store(args), snapshot)
    _emit({k: result[k] for k in ("manifest_uri", "marker_uri", "created", "snapshot_id")})


def publish_matrix_source(args):
    from continual_bench.application.cloud.matrix_sources import publish_source

    if not args.local_root and not os.environ.get("BENCH_KMS_KEY_ID"):
        raise SystemExit("BENCH_KMS_KEY_ID must contain the data bucket KMS key ARN")
    with tempfile.TemporaryDirectory() as workdir:
        _emit(publish_source(
            _store(args), args.dataset, cache=args.cache, workdir=workdir, examples=args.examples,
            seed=args.seed, real_directory=args.real_directory, download=args.download,
        ))


def run_unit(args):
    from continual_bench.application.cloud.worker import run_unit as execute
    from continual_bench.application.cloud.worker import sagemaker_registrar
    from continual_bench.infrastructure.aws.emf import emit

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    store = _store(args)
    registry = None
    tracker = None
    if os.environ.get("BENCH_MODEL_REGISTRY", "false") == "true" and not args.local_root:
        from continual_bench.infrastructure.aws.experiments import SageMakerExperimentsTracker
        from continual_bench.infrastructure.aws.model_registry import SageMakerModelRegistry

        registry = sagemaker_registrar(SageMakerModelRegistry(), os.environ["BENCH_INFERENCE_IMAGE"])
        model_names = {
            "social": "multinomial-logistic",
            "fraud": "fraud-booster",
            "ecommerce": "return-booster",
        }

        def tracker(unit):
            dataset = unit.params.get("dataset", "social")
            policy = unit.params.get("policy", "frozen")
            return SageMakerExperimentsTracker(
                project=os.environ.get("BENCH_STACK", "continual-bench"),
                dataset=dataset,
                policy=policy,
                model_name=model_names.get(dataset, "zia-model"),
                run_id=unit.run_id,
                job_name=unit.job_name,
                git_sha=os.environ.get("BENCH_GIT_REVISION", unit.contract["release"]["git_sha"]),
                stack=os.environ.get("BENCH_STACK", "continual-bench"),
            )

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="cb-run-"))
    workdir.mkdir(parents=True, exist_ok=True)
    code = execute(store, args.unit_uri, workdir, registry=registry, tracker=tracker, emit=emit)
    sys.exit(code)


def selftest(args):
    """Image contract: imports, CLI entry point, locked runtime and scientific stack."""
    import importlib
    import platform

    checks = {"python": platform.python_version(), "machine": platform.machine()}
    modules = {
        "runtime": [
            "continual_bench.evaluation.zia",
            "continual_bench.application.cloud.worker",
            "continual_bench.infrastructure.aws.s3",
            "continual_bench.infrastructure.aws.experiments",
            "continual_bench.models.training.loop",
            "duckdb",
            "pyarrow",
            "xgboost",
            "boto3",
        ],
        "ml": [
            "continual_bench.orchestration.sagemaker",
            "continual_bench.infrastructure.aws.experiments",
            "continual_bench.models.zia",
            "continual_bench.models.training.loop",
            "pandas",
            "xgboost",
            "scipy",
        ],
    }[args.role]
    for name in modules:
        importlib.import_module(name)
    forbidden = [m for m in ("mlflow", "zenml", "confluent_kafka", "airflow") if m in sys.modules]
    if forbidden:
        raise SystemExit(f"AWS profile imported forbidden services: {forbidden}")
    if args.require_x86_64 and platform.machine() != "x86_64":
        raise SystemExit(f"image architecture is {platform.machine()}, expected x86_64")
    _emit({**checks, "role": args.role, "imports": modules, "status": "ok"})


def local_smoke(args):
    """CI smoke with fixtures: plan → four parallel unit processes → consolidate, no AWS calls."""
    from continual_bench.application.cloud import layout, lifecycle, planning
    from continual_bench.application.cloud.contracts import CampaignRequest
    from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
    from continual_bench.application.cloud.snapshots import verify_snapshot
    from continual_bench.infrastructure.object_store import LocalObjectStore, get_json

    root = Path(args.root).resolve()
    store = LocalObjectStore(root / "store")
    snapshot = publish_fixture_snapshot(store, root, args.dataset)
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id=args.campaign, snapshot=snapshot.model_dump(), max_parallel=args.parallel
    )
    plan = planning.build_plan(
        request, release, snapshot_manifest=verify_snapshot(store, snapshot), project_root="."
    )
    published = planning.publish_plan(store, plan)
    codes, running, pending = {}, {}, list(enumerate(published["items"]))
    while pending or running:
        while pending and len(running) < published["concurrency"]:
            index, item = pending.pop(0)
            env = {
                **os.environ,
                "AWS_BATCH_JOB_ID": f"local-{index}",
                "AWS_BATCH_JOB_ATTEMPT": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "BENCH_IMAGE_DIGEST": release.images["runtime"].digest,
            }
            running[index] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "continual_bench.cli",
                    "aws",
                    "run-unit",
                    "--local-root",
                    str(root / "store"),
                    "--unit-uri",
                    store.uri(layout.unit_key(item["campaign_id"], item["run_id"])),
                    "--workdir",
                    str(root / "work" / item["run_id"]),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
            )
        for index, process in list(running.items()):
            if process.poll() is not None:
                codes[index] = running.pop(index).returncode
        time.sleep(0.2)
    items = published["items"]
    results = [
        {"run_id": item["run_id"], "status": "SUCCEEDED" if codes[i] == 0 else "FAILED"}
        for i, item in enumerate(items)
    ]
    summary = lifecycle.consolidate(store, args.campaign, results)
    windows = []
    for index, item in enumerate(items):
        status = get_json(
            store,
            layout.attempt_status_key(
                item["campaign_id"], item["run_id"], layout.attempt_key_for(f"local-{index}", 1)
            ),
        )
        windows.append((status["started_at"], status["finished_at"]))
    codes = [codes[i] for i in range(len(items))]
    overlap = max(start for start, _ in windows) < min(end for _, end in windows)
    _emit({**summary, "exit_codes": codes, "overlapping": overlap, "store": str(root / "store")})
    if not summary["complete"]:
        sys.exit(1)


def plan_command(args):
    from continual_bench.orchestration.aws_control import Context, plan_campaign

    request = json.loads(Path(args.request).read_text())
    _emit(plan_campaign(Context(), {"request": request}))


def refresh_prices(args):
    from continual_bench.infrastructure.aws.pricing import fetch

    snapshot = fetch(args.region)
    Path(args.output).write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    _emit({k: v.get("status") for k, v in snapshot["prices"].items()})


def query(args):
    from continual_bench.infrastructure.aws.athena import run_query

    sql = Path(args.sql).read_text()
    for name, value in (item.split("=", 1) for item in args.param):
        if not value.replace("-", "").replace("_", "").isalnum():
            raise SystemExit(f"parameter {name} must be alphanumeric with - or _")
        sql = sql.replace("${" + name + "}", value)
    _emit(run_query(sql, workgroup=args.workgroup, database=args.database))


def reconcile(args):
    from continual_bench.application.cloud.lifecycle import reconcile_campaign
    from continual_bench.infrastructure.aws.batch import BatchJobs

    store, jobs = _store(args), BatchJobs(args.queue or os.environ["BENCH_JOB_QUEUE"])
    for attempt in range(args.checks):
        report = reconcile_campaign(store, jobs, args.campaign, args.reason)
        if report["clean"]:
            break
        time.sleep(args.interval)
    _emit(report)
    sys.exit(0 if report["clean"] else 1)


def cancel(args):
    from continual_bench.infrastructure.aws.session import client

    client("stepfunctions").stop_execution(
        executionArn=args.execution_arn,
        error="CampaignCancelled",
        cause=f"cancelled by operator for campaign {args.campaign}",
    )
    args.reason = "operator cancellation"
    reconcile(args)


def configure(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest="aws_command", required=True)

    def add(name, handler, *options, store=True):
        command = sub.add_parser(name)
        if store:
            command.add_argument("--bucket")
            command.add_argument("--local-root")
        for option in options:
            option(command)
        command.set_defaults(aws_handler=handler)
        return command

    def dataset(command):
        command.add_argument("--dataset", required=True, choices=["fraud", "social", "ecommerce"])
        command.add_argument("--scope", required=True, choices=["smoke", "full"])
        command.add_argument("--revision", default="pinned")
        command.add_argument("--workdir", default=None)

    add(
        "acquire",
        acquire,
        dataset,
        lambda c: (
            c.add_argument("--mode", default="download", choices=["download", "import"]),
            c.add_argument("--import-uri"),
        ),
    )
    add("prepare", prepare, dataset)
    add("publish-snapshot", publish_snapshot, dataset)
    matrix_source = add("publish-matrix-source", publish_matrix_source)
    matrix_source.add_argument("--dataset", required=True)
    matrix_source.add_argument("--cache", default="state/datasets")
    matrix_source.add_argument("--real-directory")
    matrix_source.add_argument("--examples", type=int, default=100)
    matrix_source.add_argument("--seed", type=int, default=42)
    matrix_source.add_argument("--download", action="store_true")
    add(
        "run-unit",
        run_unit,
        lambda c: (c.add_argument("--unit-uri", required=True), c.add_argument("--workdir")),
    )
    add(
        "selftest",
        selftest,
        lambda c: (
            c.add_argument("--role", choices=["runtime", "ml"], default="runtime"),
            c.add_argument("--require-x86-64", action="store_true"),
        ),
        store=False,
    )
    add(
        "local-smoke",
        local_smoke,
        lambda c: (
            c.add_argument("--root", default="state/aws-local-smoke"),
            c.add_argument("--campaign", default="local-smoke"),
            c.add_argument("--dataset", choices=["fraud", "social", "ecommerce"], default="social"),
            c.add_argument("--parallel", type=int, default=4),
            c.add_argument("--timeout", type=int, default=900),
        ),
        store=False,
    )
    add("plan", plan_command, lambda c: c.add_argument("--request", required=True), store=False)
    add(
        "refresh-prices",
        refresh_prices,
        lambda c: (
            c.add_argument("--region", default="us-east-1"),
            c.add_argument("--output", default="configs/contracts/aws-prices.us-east-1.json"),
        ),
        store=False,
    )
    add(
        "query",
        query,
        lambda c: (
            c.add_argument("sql"),
            c.add_argument("--param", action="append", default=[]),
            c.add_argument("--workgroup", default="continual-bench"),
            c.add_argument("--database", default="continual_bench"),
        ),
        store=False,
    )

    def stop(command):
        command.add_argument("--campaign", required=True)
        command.add_argument("--queue")
        command.add_argument("--checks", type=int, default=10)
        command.add_argument("--interval", type=int, default=30)

    add("reconcile", reconcile, stop, lambda c: c.add_argument("--reason", default="operator reconciliation"))
    add("cancel", cancel, stop, lambda c: c.add_argument("--execution-arn", required=True))


def handle(args):
    args.aws_handler(args)
