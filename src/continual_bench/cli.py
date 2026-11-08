"""Command-line entry point. Command pattern: one registered handler per subcommand."""

import argparse
import json
from pathlib import Path
from typing import Callable

Configure = Callable[[argparse.ArgumentParser], None]
Handler = Callable[[argparse.Namespace], None]
COMMANDS: dict[str, tuple[Configure, Handler]] = {}


def command(name: str, configure: Configure = lambda parser: None):
    def register(handler):
        COMMANDS[name] = (configure, handler)
        return handler

    return register


def emit(value):
    print(json.dumps(value, indent=2))


def _run_options(parser):
    parser.add_argument("--state", default="state")
    parser.add_argument("--services", action="store_true")
    parser.add_argument("--zenml", action="store_true")


def _execute(config, args):
    if args.zenml:
        from .orchestration.zenml_pipeline import execute_pipeline

        return execute_pipeline(config, args.state, services=args.services)
    from .application.runner import run

    return run(config, args.state, services=args.services)


def _summary(result):
    return {k: result[k] for k in ("run_key", "protocol", "metrics", "complete")}


@command("run", lambda p: (p.add_argument("config"), _run_options(p)))
def run_command(args):
    from .config import load_config

    emit(_summary(_execute(load_config(args.config), args)))


def _smoke_options(parser):
    _run_options(parser)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--samples", default=600, type=int)


@command("smoke", _smoke_options)
def smoke_command(args):
    from .config import Config

    results = []
    for dataset in ("sea", "agrawal", "stagger"):
        for mode in ("streaming", "incremental", "offline"):
            policies = (["none"] if mode == "offline"
                        else (["none", "periodic", "drift"] if args.baselines else ["drift"]))
            for policy in policies:
                config = Config.model_validate({
                    "name": "smoke",
                    "mode": mode,
                    "dataset": {"name": dataset, "samples": args.samples, "label_delay": 3},
                    "warmup": 50,
                    "metric_window": 50,
                    "policy": {"name": policy, "every": 100, "cooldown": 50, "min_samples": 50,
                               "min_window": 25},
                })
                results.append(_summary(_execute(config, args)))
                print(json.dumps({"dataset": dataset, "mode": mode, "policy": policy, **results[-1]}),
                      flush=True)
    Path(args.state).mkdir(parents=True, exist_ok=True)
    (Path(args.state) / "smoke-summary.json").write_text(json.dumps(results, indent=2))


@command("serve")
def serve_command(args):
    from .orchestration.server import serve

    serve()


@command("serve-datasets")
def serve_datasets_command(args):
    from .orchestration.epoch_server import serve

    serve()


def _campaign_options(parser):
    parser.add_argument("config")
    parser.add_argument("--state", default="state/task_evaluation")
    parser.add_argument("--services", action="store_true")
    parser.add_argument("--zenml", action="store_true")


@command("campaign", _campaign_options)
def campaign_command(args):
    from .application.campaign import Campaign, execute_campaign
    from .config import load_model

    specification = load_model(Campaign, args.config)
    executor = None
    if args.zenml:
        from .orchestration.zenml_pipeline import execute_pipeline as executor
    emit(execute_campaign(specification, args.state, services=args.services, zenml=args.zenml,
                          executor=executor))


@command("typed", lambda p: (p.add_argument("config"),
                             p.add_argument("--state", default="state/task_evaluation-typed")))
def typed_command(args):
    from .application.components import default_registry, make_source
    from .config import TypedConfig, load_model
    from .evaluation.typed import run_typed

    specification = load_model(TypedConfig, args.config)
    source = make_source(default_registry(), specification.dataset)
    result = run_typed(source, args.state, **specification.model_dump(exclude={"dataset"}))
    emit({k: result[k] for k in ("run_key", "metrics", "complete")})


def _datasets_options(parser):
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--cache", default="state/datasets")
    parser.add_argument("--state", default="state/dataset-epochs")
    parser.add_argument("--real-directory")
    parser.add_argument("--download", action="store_true")


@command("datasets", _datasets_options)
def datasets_command(args):
    from .config import EpochConfig
    from .evaluation.epochs import run_epochs, smoke_all

    if args.dataset == "all":
        result = smoke_all(args.state, args.cache, download=args.download, real_directory=args.real_directory)
    else:
        result = run_epochs(EpochConfig(dataset=args.dataset), args.state, args.cache,
                            download=args.download, real_directory=args.real_directory)
    emit(result)


def _epoch_job_options(parser):
    parser.add_argument("config")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--real-directory", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--services", action="store_true")


@command("epoch-job", _epoch_job_options)
def epoch_job_command(args):
    from .config import EpochConfig, load_model

    config = load_model(EpochConfig, args.config)
    if args.services:
        from .orchestration.epoch_pipeline import execute_epoch_pipeline

        result = execute_epoch_pipeline(config, args.state, args.cache,
                                        real_directory=args.real_directory, download=args.download)
    else:
        from .evaluation.epochs import run_epochs

        result = run_epochs(config, args.state, args.cache, real_directory=args.real_directory,
                            download=args.download)
    emit(result)


@command("components", lambda p: p.add_argument("--plugin", action="append", default=[]))
def components_command(args):
    from .application.components import default_registry

    registry = default_registry(tuple(args.plugin))
    emit({
        "datasets": [{"name": s.name, "task": s.task or "declared_by_manifest", "family": s.family}
                     for s in registry.sources],
        "models": [{"name": m.name, "tasks": list(m.tasks)} for m in registry.models],
        "detectors": list(registry.catalogs["detector"].names()),
        "policies": list(registry.catalogs["policy"].names()),
    })


def _matrix_options(parser):
    parser.add_argument("config", help="YAML or JSON ExperimentMatrix")
    parser.add_argument("--state", default="state/matrices")
    parser.add_argument("--cache", default="state/datasets")
    parser.add_argument("--real-directory")
    parser.add_argument("--dry-run", action="store_true", help="only print the expanded plan")


@command("matrix", _matrix_options)
def matrix_command(args):
    from .application.matrix import ExperimentMatrix, case_key, describe, execute_matrix
    from .config import load_model

    matrix = load_model(ExperimentMatrix, args.config)
    if args.dry_run:
        plan = matrix.plan()
        emit({"matrix": plan["matrix"], "evaluator": plan["evaluator"], "cases": len(plan["cases"]),
              "planned": [{"case": case_key(c), **describe(c)} for c in plan["cases"]],
              "skipped": plan["skipped"]})
        return
    journal = execute_matrix(matrix, args.state, cache=args.cache, real_directory=args.real_directory)
    emit({"complete": journal["complete"], "cases": len(journal["cases"]),
          "skipped": len(journal["skipped"])})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="continual-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (configure, _) in COMMANDS.items():
        configure(sub.add_parser(name))
    args = parser.parse_args(argv)
    COMMANDS[args.command][1](args)


if __name__ == "__main__":
    main()
