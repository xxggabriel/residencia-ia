"""CLI helper for Makefile `bench` / `run-experiment` / `bench-paper` targets."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _airflow_request(
    base_url: str,
    path: str,
    *,
    username: str,
    password: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _get_airflow_jwt(base_url: str, username: str, password: str) -> str | None:
    """Authenticate against Airflow 3 FAB auth token endpoint if available."""
    try:
        res = _airflow_request(
            base_url,
            "/auth/token",
            username="",
            password="",
            payload={"username": username, "password": password},
        )
        return res.get("access_token")
    except Exception:
        return None


def _airflow_api(
    base_url: str,
    path: str,
    *,
    username: str,
    password: str,
    jwt_token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    elif username and password:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run benchmark campaign locally or via Airflow.")
    parser.add_argument("--env", default="docker")
    parser.add_argument("--config", default="")
    parser.add_argument("--protocol", default="cara")
    parser.add_argument("--dataset", default="electricity")
    parser.add_argument("--campaign", default="")
    parser.add_argument("--model", default="logistic")
    parser.add_argument("--policies", default="")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--save-checkpoints", default="false")
    parser.add_argument("--dag", default="benchmark_campaign")
    parser.add_argument("--wait", default="1")
    args = parser.parse_args(argv)

    dotenv = _load_dotenv()
    save_ckpt = str(args.save_checkpoints).strip().lower() in {"1", "true", "yes", "on"}
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.env == "local":
        os.environ["BENCH_ENV"] = "local"
        os.environ["BENCH_SAVE_MODEL_CHECKPOINTS"] = "true" if save_ckpt else "false"
        os.environ["BENCH_GENERATE_PLOTS"] = "false"

        from continual_bench.protocols.base import CampaignConfig
        from continual_bench.protocols.campaign import load_campaign_config, run_campaign

        if args.config:
            cfg = load_campaign_config(args.config)
            updates: dict[str, Any] = {}
            if args.campaign:
                updates["name"] = args.campaign
            if policies:
                updates["policies"] = policies
            if args.seeds and args.seeds != "42,43,44,45,46":
                updates["seeds"] = seeds
            if updates:
                cfg = cfg.model_copy(update=updates)
        else:
            cfg = CampaignConfig(
                name=args.campaign or f"{args.protocol}-{args.dataset}",
                protocol=args.protocol,  # type: ignore[arg-type]
                dataset={"name": args.dataset, "protocol": args.protocol},  # type: ignore[arg-type]
                model={"name": args.model},  # type: ignore[arg-type]
                seeds=seeds,
                policies=policies,
            )
        result = run_campaign(cfg, env_name="local", save_model_checkpoints=save_ckpt)
        print(
            json.dumps(
                {
                    "env": "local",
                    "protocol": result.protocol,
                    "dataset": result.dataset,
                    "artifacts": result.artifacts,
                    "mlflow_receipt": result.mlflow_receipt,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    bind_ip = os.environ.get("WEB_BIND_IP") or dotenv.get("WEB_BIND_IP") or "127.0.0.1"
    airflow_url = os.environ.get("BENCH_AIRFLOW_URL") or f"http://{bind_ip}:18081"
    minio_url = f"http://{bind_ip}:19001"
    mlflow_url = f"http://{bind_ip}:15000"
    user = os.environ.get("WEB_USER") or dotenv.get("WEB_USER") or "admin"
    password = os.environ.get("WEB_PASSWORD") or dotenv.get("WEB_PASSWORD") or ""

    conf: dict[str, Any] = {
        "env": args.env,
        "config_path": args.config,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "campaign": args.campaign,
        "model": args.model,
        "policies": policies,
        "seeds": seeds if (not args.config or args.seeds != "42,43,44,45,46") else [],
        "save_checkpoints": save_ckpt,
    }
    jwt_token = _get_airflow_jwt(airflow_url, user, password)
    # Unpause DAG first if paused
    try:
        req = urllib.request.Request(
            f"{airflow_url}/api/v2/dags/{args.dag}",
            data=json.dumps({"is_paused": False}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {jwt_token}"} if jwt_token else {}),
            },
            method="PATCH",
        )
        urllib.request.urlopen(req, timeout=15).close()
    except Exception:
        pass

    triggered = _airflow_api(
        airflow_url,
        f"/api/v2/dags/{args.dag}/dagRuns",
        username=user,
        password=password,
        jwt_token=jwt_token,
        payload={"logical_date": datetime.now(timezone.utc).isoformat(), "conf": conf},
    )
    dag_run_id = triggered.get("dag_run_id")
    print(f"Airflow DAG triggered: dag={args.dag} dag_run_id={dag_run_id}")
    print(f"Airflow UI: {airflow_url}/dags/{args.dag}/runs/{dag_run_id}")
    print(f"MinIO UI:   {minio_url}")
    print(f"MLflow UI:  {mlflow_url}")

    if str(args.wait).strip() not in {"1", "true", "yes"}:
        return 0

    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        state_resp = _airflow_api(
            airflow_url,
            f"/api/v2/dags/{args.dag}/dagRuns/{dag_run_id}",
            username=user,
            password=password,
            jwt_token=jwt_token,
        )
        state = str(state_resp.get("state", "")).lower()
        if state == "success":
            print(f"DAG run {dag_run_id} succeeded.")
            return 0
        if state == "failed":
            print(f"DAG run {dag_run_id} failed.", file=sys.stderr)
            return 1
        time.sleep(5)
    print(f"Timed out waiting for DAG run {dag_run_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
