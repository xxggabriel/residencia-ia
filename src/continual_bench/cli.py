import argparse
import json
from pathlib import Path

from .application.config import Config, load_config
from .application.runner import run


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("config")
    execute.add_argument("--state", default="state")
    execute.add_argument("--services", action="store_true")
    execute.add_argument("--zenml", action="store_true")
    matrix = sub.add_parser("smoke")
    matrix.add_argument("--state", default="state")
    matrix.add_argument("--services", action="store_true")
    matrix.add_argument("--zenml", action="store_true")
    matrix.add_argument("--baselines", action="store_true")
    matrix.add_argument("--samples", default=600, type=int)
    sub.add_parser("serve")
    campaign = sub.add_parser("campaign")
    campaign.add_argument("config")
    campaign.add_argument("--state", default="state/week09")
    campaign.add_argument("--services", action="store_true")
    campaign.add_argument("--zenml", action="store_true")
    typed = sub.add_parser("typed")
    typed.add_argument("config")
    typed.add_argument("--state", default="state/week09-typed")
    datasets = sub.add_parser("datasets")
    datasets.add_argument("--dataset", default="all")
    datasets.add_argument("--cache", default="state/datasets")
    datasets.add_argument("--state", default="state/dataset-epochs")
    datasets.add_argument("--real-directory")
    datasets.add_argument("--download", action="store_true")
    epoch_job = sub.add_parser("epoch-job")
    epoch_job.add_argument("config")
    epoch_job.add_argument("--cache", required=True)
    epoch_job.add_argument("--state", required=True)
    epoch_job.add_argument("--real-directory", required=True)
    epoch_job.add_argument("--download", action="store_true")
    epoch_job.add_argument("--services", action="store_true")
    sub.add_parser("serve-datasets")
    args = parser.parse_args()
    if args.command == "serve-datasets":
        from .application.epoch_server import serve

        serve()
        return
    if args.command == "epoch-job":
        from .application.epoch_benchmark import EpochConfig, run_epochs

        config = EpochConfig.model_validate_json(Path(args.config).read_text())
        if args.services:
            from .adapters.epoch_pipeline import execute_epoch_pipeline

            result = execute_epoch_pipeline(config, args.state, args.cache,
                                            real_directory=args.real_directory, download=args.download)
        else:
            result = run_epochs(config, args.state, args.cache, real_directory=args.real_directory,
                                download=args.download)
        print(json.dumps(result, indent=2))
        return
    if args.command == "datasets":
        from .application.epoch_benchmark import EpochConfig, run_epochs, smoke_all

        if args.dataset == "all":
            result = smoke_all(args.state, args.cache, download=args.download,
                               real_directory=args.real_directory)
        else:
            result = run_epochs(EpochConfig(dataset=args.dataset), args.state, args.cache,
                                download=args.download, real_directory=args.real_directory)
        print(json.dumps(result, indent=2))
        return
    if args.command == "typed":
        from .application.registry import default_registry, make_source
        from .application.typed_benchmark import run_typed
        from .application.typed_config import TypedConfig

        specification = TypedConfig.model_validate_json(Path(args.config).read_text())
        source = make_source(default_registry(), specification.dataset)
        result = run_typed(source, args.state, **specification.model_dump(exclude={"dataset"}))
        print(json.dumps({k: result[k] for k in ("run_key", "metrics", "complete")}, indent=2))
        return
    if args.command == "campaign":
        from .application.campaign import Campaign, execute_campaign

        specification = Campaign.model_validate_json(Path(args.config).read_text())
        result = execute_campaign(specification, args.state, services=args.services, zenml=args.zenml)
        print(json.dumps(result, indent=2))
        return
    if args.command == "serve":
        from .application.server import serve

        serve()
        return

    def execute(config):
        if args.zenml:
            from .adapters.zenml_pipeline import execute_pipeline

            return execute_pipeline(config, args.state, services=args.services)
        return run(config, args.state, services=args.services)

    if args.command == "run":
        result = execute(load_config(args.config))
        print(json.dumps({k: result[k] for k in ("run_key", "protocol", "metrics", "complete")}, indent=2))
    else:
        results = []
        for dataset in ("sea", "agrawal", "stagger"):
            for mode in ("streaming", "incremental", "offline"):
                policies = (
                    ["none"]
                    if mode == "offline"
                    else (["none", "periodic", "drift"] if args.baselines else ["drift"])
                )
                for policy in policies:
                    config = Config.model_validate(
                        {
                            "name": "smoke",
                            "mode": mode,
                            "dataset": {"name": dataset, "samples": args.samples, "label_delay": 3},
                            "warmup": 50,
                            "metric_window": 50,
                            "policy": {
                                "name": policy,
                                "every": 100,
                                "cooldown": 50,
                                "min_samples": 50,
                                "min_window": 25,
                            },
                        }
                    )
                    result = execute(config)
                    results.append({k: result[k] for k in ("run_key", "protocol", "metrics", "complete")})
                    print(
                        json.dumps({"dataset": dataset, "mode": mode, "policy": policy, **results[-1]}),
                        flush=True,
                    )
        Path(args.state).mkdir(parents=True, exist_ok=True)
        (Path(args.state) / "smoke-summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
