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
    args = parser.parse_args()
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
