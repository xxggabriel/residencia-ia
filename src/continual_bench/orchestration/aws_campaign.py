"""Generic, parallel, and Spot-resilient AWS Batch campaign orchestration."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from continual_bench.backends.aws import _parse_s3_uri
from continual_bench.models.training.callbacks import GracefulTerminationHandler
from continual_bench.protocols.base import CampaignConfig, SpotInterruptedError
from continual_bench.protocols.campaign import (
    create_campaign_tracker,
    finalize_campaign_result,
    resolve_campaign_settings,
)
from continual_bench.protocols.cara import CaraProtocol
from continual_bench.protocols.common import canonical_dataset_name
from continual_bench.protocols.upf import UpfProtocol, _AbsoluteStalenessMatrix
from continual_bench.retraining.matrix import PerformanceMatrix

EXIT_INTERRUPTED = 75

# Valid AWS Fargate vCPU -> Memory (MiB) combinations.
FARGATE_MEMORY_CHOICES: dict[str, tuple[int, ...]] = {
    "0.25": (512, 1024, 2048),
    "0.5": (1024, 2048, 3072, 4096),
    "1": tuple(range(2048, 8193, 1024)),
    "2": tuple(range(4096, 16385, 1024)),
    "4": tuple(range(8192, 30721, 1024)),
    "8": tuple(range(16384, 61441, 4096)),
    "16": tuple(range(32768, 122881, 8192)),
}

_DEFAULT_MEMORY_FOR_VCPU: dict[str, int] = {
    "0.25": 512,
    "0.5": 1024,
    "1": 2048,
    "2": 4096,
    "4": 8192,
    "8": 16384,
    "16": 32768,
}


def _normalize_vcpus(vcpus: str | int | float) -> str:
    if isinstance(vcpus, str):
        stripped = vcpus.strip()
        if stripped in FARGATE_MEMORY_CHOICES:
            return stripped
        num = float(stripped)
    else:
        num = float(vcpus)
    if num in (0.25, 0.5):
        return str(num)
    if num.is_integer():
        return str(int(num))
    return str(num)


def validate_fargate_resources(vcpus: str | int | float, memory_mib: int) -> tuple[str, int]:
    """Validate and normalize a Fargate (vCPU, Memory MiB) pair."""
    vcpu_str = _normalize_vcpus(vcpus)
    if vcpu_str not in FARGATE_MEMORY_CHOICES:
        valid = sorted(FARGATE_MEMORY_CHOICES.keys(), key=float)
        raise ValueError(f"Invalid Fargate vCPU {vcpus!r}; expected one of {valid}")
    mem = int(memory_mib)
    allowed = FARGATE_MEMORY_CHOICES[vcpu_str]
    if mem not in allowed:
        raise ValueError(
            f"Invalid Fargate memory {mem} MiB for {vcpu_str} vCPU; "
            f"allowed range is {allowed[0]}..{allowed[-1]} MiB"
        )
    return vcpu_str, mem


def resolve_campaign_resources(
    config: CampaignConfig,
    *,
    vcpus: str | int | float | None = None,
    memory_mib: int | None = None,
) -> tuple[str, int]:
    """Determine Fargate vCPU and Memory MiB for a campaign, honoring explicit overrides."""
    if vcpus is not None:
        vcpu_str = _normalize_vcpus(vcpus)
        if vcpu_str not in FARGATE_MEMORY_CHOICES:
            validate_fargate_resources(vcpu_str, memory_mib or 4096)
        mem = int(memory_mib) if memory_mib is not None else _DEFAULT_MEMORY_FOR_VCPU[vcpu_str]
        return validate_fargate_resources(vcpu_str, mem)

    if memory_mib is not None:
        mem = int(memory_mib)
        for candidate in ("2", "4", "8", "1", "16", "0.5", "0.25"):
            if mem in FARGATE_MEMORY_CHOICES[candidate]:
                return candidate, mem
        raise ValueError(f"No valid Fargate vCPU supports {mem} MiB memory")

    params = config.model.params or {}
    n_jobs = int(params.get("n_jobs") or params.get("nthread") or 1)
    canonical = canonical_dataset_name(config.dataset.name)
    if n_jobs >= 8:
        return "8", 16384
    if n_jobs >= 4 or canonical == "airlines":
        return "4", 8192
    return "2", 4096


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _read_text_uri_or_path(ref: str, *, s3_client: Any | None = None) -> str:
    if ref.startswith("s3://"):
        bucket, key = _parse_s3_uri(ref)
        client = s3_client
        if client is None:
            import boto3  # type: ignore[import-untyped]

            client = boto3.client("s3")
        resp = client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read().decode("utf-8")
    return Path(ref).read_text(encoding="utf-8")


def _parse_json_arg(raw: str | dict[str, Any] | None, *, name: str) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return parsed


def _parse_policies(raw: str | list[str] | None) -> list[str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return [str(p).strip() for p in raw if str(p).strip()]
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _parse_seeds(raw: str | list[int] | None, single_seed: int | None = None) -> list[int] | None:
    if single_seed is not None:
        return [int(single_seed)]
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return [int(s) for s in raw]
    return [int(s.strip()) for s in str(raw).split(",") if s.strip()]


def resolve_campaign_config(
    *,
    config_ref: str | None = None,
    payload_uri: str | None = None,
    protocol: str | None = None,
    dataset: str | None = None,
    campaign: str | None = None,
    model: str | None = None,
    model_params: str | dict[str, Any] | None = None,
    policies: str | list[str] | None = None,
    seeds: str | list[int] | None = None,
    seed: int | None = None,
    staleness_mode: str | None = None,
    overrides: str | dict[str, Any] | None = None,
    output_dir: str | None = None,
    tracking_uri: str | None = None,
    s3_client: Any | None = None,
) -> CampaignConfig:
    """Resolve a ``CampaignConfig`` from a payload URI, YAML/JSON config, and CLI overrides."""
    raw_dict: dict[str, Any] = {}
    if payload_uri and payload_uri != "s3://unset":
        loaded = json.loads(_read_text_uri_or_path(payload_uri, s3_client=s3_client))
        if isinstance(loaded, dict) and "config" in loaded and isinstance(loaded["config"], dict):
            raw_dict = dict(loaded["config"])
        elif isinstance(loaded, dict):
            raw_dict = dict(loaded)
    elif config_ref:
        text = _read_text_uri_or_path(config_ref, s3_client=s3_client)
        if config_ref.lower().endswith(".json"):
            raw_dict = json.loads(text)
        else:
            raw_dict = yaml.safe_load(text) or {}

    parsed_overrides = _parse_json_arg(overrides, name="--overrides")
    if parsed_overrides:
        raw_dict = _deep_merge(raw_dict, parsed_overrides)

    if protocol:
        raw_dict["protocol"] = protocol
    if dataset:
        ds_dict = dict(raw_dict.get("dataset") or {})
        ds_dict["name"] = dataset
        if protocol:
            ds_dict["protocol"] = protocol
        elif "protocol" in raw_dict:
            ds_dict["protocol"] = raw_dict["protocol"]
        raw_dict["dataset"] = ds_dict
    elif protocol and isinstance(raw_dict.get("dataset"), dict):
        raw_dict["dataset"]["protocol"] = protocol

    if campaign:
        raw_dict["name"] = campaign

    if model or model_params:
        m_dict = dict(raw_dict.get("model") or {})
        if model:
            m_dict["name"] = model
        parsed_mp = _parse_json_arg(model_params, name="--model-params")
        if parsed_mp:
            m_dict["params"] = {**dict(m_dict.get("params") or {}), **parsed_mp}
        raw_dict["model"] = m_dict

    parsed_policies = _parse_policies(policies)
    if parsed_policies is not None:
        raw_dict["policies"] = parsed_policies

    parsed_seeds = _parse_seeds(seeds, single_seed=seed)
    if parsed_seeds is not None:
        raw_dict["seeds"] = parsed_seeds
    elif "seeds" not in raw_dict:
        raw_dict["seeds"] = [0, 1, 2, 3, 4]

    if staleness_mode:
        raw_dict["staleness_mode"] = staleness_mode
    if output_dir:
        raw_dict["output_dir"] = output_dir
    if tracking_uri:
        raw_dict["tracking_uri"] = tracking_uri

    if "name" not in raw_dict:
        proto = raw_dict.get("protocol", "cara")
        ds_name = (raw_dict.get("dataset") or {}).get("name", "dataset")
        raw_dict["name"] = f"{proto}-{ds_name}"

    if "policies" not in raw_dict:
        raw_dict["policies"] = []

    return CampaignConfig(**raw_dict)


def config_fingerprint(config: CampaignConfig) -> str:
    """Short SHA-256 fingerprint of matrix-affecting campaign configuration fields."""
    ds = (
        config.dataset.model_dump(mode="json", exclude_unset=True)
        if hasattr(config.dataset, "model_dump")
        else dict(config.dataset)
    )
    mdl = (
        config.model.model_dump(mode="json", exclude_unset=True)
        if hasattr(config.model, "model_dump")
        else dict(config.model)
    )
    payload = {
        "protocol": config.protocol,
        "dataset": ds,
        "model": mdl,
        "staleness_mode": config.staleness_mode,
        "kernel_gamma": config.kernel_gamma,
        "policies": list(config.policies),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def campaign_seed_rel_path(campaign_name: str, seed: int, fingerprint: str = "") -> str:
    prefix = f"campaigns/{campaign_name}/{fingerprint}" if fingerprint else f"campaigns/{campaign_name}"
    return f"{prefix}/seeds/seed_{seed}_matrix.json.gz"


def campaign_checkpoint_rel_path(campaign_name: str, seed: int, fingerprint: str = "") -> str:
    prefix = f"campaigns/{campaign_name}/{fingerprint}" if fingerprint else f"campaigns/{campaign_name}"
    return f"{prefix}/checkpoints/seed_{seed}_matrix.json.gz"


def campaign_config_rel_path(campaign_name: str) -> str:
    return f"campaigns/{campaign_name}/campaign_config.json"


def _stage_dataset_if_needed(
    config: CampaignConfig,
    backend: Any,
    workdir: Path,
) -> CampaignConfig:
    """Download prepared dataset from S3 into ``workdir`` when not present locally."""
    from continual_bench.datasets.streams_unified import resolve_dataset_path

    src = config.dataset.source_path
    if src and src.startswith("s3://"):
        bucket, key = _parse_s3_uri(src)
        client = backend._get_s3_client()
        if client is None:
            import boto3  # type: ignore[import-untyped]

            client = boto3.client("s3")
        local_path = workdir / "datasets" / Path(key).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.is_file():
            resp = client.get_object(Bucket=bucket, Key=key)
            local_path.write_bytes(resp["Body"].read())
        return config.model_copy(
            update={"dataset": config.dataset.model_copy(update={"source_path": str(local_path)})}
        )

    try:
        resolve_dataset_path(config.dataset.name, source_path=src)
        return config
    except FileNotFoundError:
        pass

    s3_bucket = getattr(backend, "s3_bucket", None) or os.environ.get("BENCH_BUCKET")
    client = backend._get_s3_client() if hasattr(backend, "_get_s3_client") else None
    if not s3_bucket or client is None:
        return config

    canonical = canonical_dataset_name(config.dataset.name)
    prefixes = (
        f"datasets/stream-sources/{canonical}/full/",
        f"datasets/{canonical}/prepared/",
        f"datasets/matrix/{canonical}/",
    )
    for prefix in prefixes:
        if not hasattr(client, "list_objects_v2"):
            continue
        resp = client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        candidates = [
            obj["Key"]
            for obj in resp.get("Contents", [])
            if obj["Key"].endswith((".parquet", ".csv", ".arff"))
        ]
        if candidates:
            key = sorted(candidates)[-1]
            local_path = workdir / "datasets" / f"{canonical}{Path(key).suffix}"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.is_file():
                obj_resp = client.get_object(Bucket=s3_bucket, Key=key)
                local_path.write_bytes(obj_resp["Body"].read())
            return config.model_copy(
                update={"dataset": config.dataset.model_copy(update={"source_path": str(local_path)})}
            )
    return config


def serialize_seed_result(protocol: str, seed_result: tuple[Any, ...]) -> bytes:
    """Serialize a CARA or UPF seed matrix tuple to gzipped JSON (zero-checkpoint compliant)."""
    if protocol == "cara":
        seed, offline_cost, online_cost, accuracy_online, detector_retrains, fit_times, sample_counts = (
            seed_result
        )
        payload = {
            "protocol": "cara",
            "seed": int(seed),
            "offline_cost": np.asarray(offline_cost, dtype=np.float64).tolist(),
            "online_cost": np.asarray(online_cost, dtype=np.float64).tolist(),
            "accuracy_online": np.asarray(accuracy_online, dtype=np.float64).tolist(),
            "detector_retrains": {k: [int(x) for x in v] for k, v in detector_retrains.items()},
            "fit_times": [float(x) for x in fit_times],
            "sample_counts": [int(x) for x in sample_counts],
        }
    elif protocol == "upf":
        seed, matrix, detector_schedules = seed_result
        payload = {
            "protocol": "upf",
            "seed": int(seed),
            "is_absolute": isinstance(matrix, _AbsoluteStalenessMatrix),
            "gamma": getattr(matrix, "_gamma", None),
            "matrix": {
                "err": np.asarray(matrix.err, dtype=np.float64).tolist(),
                "offline": int(matrix.offline),
                "points": [np.asarray(p, dtype=np.float64).tolist() for p in matrix.points],
                "queries": [np.asarray(q, dtype=np.float64).tolist() for q in matrix.queries],
                "sub_loss": {
                    f"{i},{j}": np.asarray(v, dtype=np.float64).tolist()
                    for (i, j), v in matrix.sub_loss.items()
                },
                "feature_means": np.asarray(matrix.feature_means, dtype=np.float64).tolist(),
                "fit_seconds": np.asarray(matrix.fit_seconds, dtype=np.float64).tolist(),
                "rows": np.asarray(matrix.rows, dtype=np.int64).tolist(),
                "paper_timing": bool(matrix.paper_timing),
            },
            "detector_schedules": {k: [int(x) for x in v] for k, v in detector_schedules.items()},
        }
    else:
        raise ValueError(f"Unsupported protocol {protocol!r}")
    return gzip.compress(json.dumps(payload).encode("utf-8"))


def deserialize_seed_result(data: bytes) -> tuple[Any, ...]:
    """Deserialize a gzipped JSON seed matrix tuple back into native protocol objects."""
    payload = json.loads(gzip.decompress(data).decode("utf-8"))
    proto = payload["protocol"]
    if proto == "cara":
        return (
            int(payload["seed"]),
            np.asarray(payload["offline_cost"], dtype=np.float64),
            np.asarray(payload["online_cost"], dtype=np.float64),
            np.asarray(payload["accuracy_online"], dtype=np.float64),
            {str(k): [int(x) for x in v] for k, v in payload["detector_retrains"].items()},
            [float(x) for x in payload["fit_times"]],
            [int(x) for x in payload["sample_counts"]],
        )
    if proto == "upf":
        m = payload["matrix"]
        mat = PerformanceMatrix(
            err=np.asarray(m["err"], dtype=np.float64),
            offline=int(m["offline"]),
            points=[np.asarray(p, dtype=np.float64) for p in m["points"]],
            queries=[np.asarray(q, dtype=np.float64) for q in m["queries"]],
            sub_loss={
                tuple(int(x) for x in k.split(",")): np.asarray(v, dtype=np.float64)
                for k, v in m["sub_loss"].items()
            },
            feature_means=np.asarray(m["feature_means"], dtype=np.float64),
            fit_seconds=np.asarray(m["fit_seconds"], dtype=np.float64),
            rows=np.asarray(m["rows"], dtype=np.int64),
            paper_timing=bool(m.get("paper_timing", True)),
        )
        if payload.get("is_absolute"):
            from continual_bench.protocols.upf import _to_absolute_staleness_matrix

            mat = _to_absolute_staleness_matrix(mat, gamma=payload.get("gamma"))
        elif payload.get("gamma") is not None:
            mat._gamma = float(payload["gamma"])
        return (
            int(payload["seed"]),
            mat,
            {str(k): [int(x) for x in v] for k, v in payload["detector_schedules"].items()},
        )
    raise ValueError(f"Unsupported protocol {proto!r}")


def run_seed_aws(
    config: CampaignConfig,
    seed: int,
    *,
    env_name: str = "aws",
    output_dir: str | None = None,
    tracking_uri: str | None = None,
    workdir: str | Path | None = None,
    s3_client: Any | None = None,
    checkpoint_interval_seconds: float | None = None,
    stop_handler: GracefulTerminationHandler | None = None,
) -> dict[str, Any]:
    """Execute a single seed matrix build with S3 Spot checkpointing and resume."""
    from continual_bench.backends import resolve_backend

    work_path = Path(workdir) if workdir is not None else (Path(tempfile.gettempdir()) / "cb-run")
    work_path.mkdir(parents=True, exist_ok=True)

    settings = resolve_campaign_settings(
        config,
        env_name=env_name,
        platform="aws" if (env_name == "aws" or str(output_dir or "").startswith("s3://")) else None,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        save_model_checkpoints=False,
    )
    backend = resolve_backend(platform=settings.platform, settings=settings)
    if hasattr(backend, "staging_dir"):
        backend.staging_dir = work_path / "staging"
    if s3_client is not None and hasattr(backend, "_s3_client"):
        backend._s3_client = s3_client

    fp = config_fingerprint(config)
    seed_rel = campaign_seed_rel_path(config.name, seed, fingerprint=fp)
    if backend.exists(seed_rel):
        return {"campaign": config.name, "seed": seed, "status": "reused", "path": seed_rel}

    config = _stage_dataset_if_needed(config, backend, work_path)
    ckpt_rel = campaign_checkpoint_rel_path(config.name, seed, fingerprint=fp)
    interval = (
        float(checkpoint_interval_seconds)
        if checkpoint_interval_seconds is not None
        else float(os.environ.get("BENCH_CHECKPOINT_INTERVAL_SECONDS", "30"))
    )

    handler = stop_handler or GracefulTerminationHandler()
    with handler:
        try:
            if config.protocol == "cara":
                proto = CaraProtocol()
                context = proto._prepare_context(config)
                seed_result = proto.build_seed_matrix(
                    config,
                    seed,
                    context=context,
                    checkpoint_backend=backend,
                    checkpoint_rel_path=ckpt_rel,
                    stop_requested=lambda: handler.terminated,
                    checkpoint_interval_seconds=interval,
                )
            elif config.protocol == "upf":
                proto_upf = UpfProtocol()
                seed_result = proto_upf.build_seed_matrix(
                    config,
                    seed,
                    checkpoint_backend=backend,
                    checkpoint_rel_path=ckpt_rel,
                    stop_requested=lambda: handler.terminated,
                    checkpoint_interval_seconds=interval,
                )
            else:
                raise ValueError(f"Unsupported protocol {config.protocol!r}")
        except SpotInterruptedError:
            return {
                "campaign": config.name,
                "seed": seed,
                "status": "interrupted",
                "exit_code": EXIT_INTERRUPTED,
                "checkpoint_path": ckpt_rel,
            }

    written = backend.write_bytes(seed_rel, serialize_seed_result(config.protocol, seed_result))
    violations = backend.verify_no_checkpoints()
    if violations:
        raise RuntimeError(f"Checkpoint policy violation: {violations}")
    return {
        "campaign": config.name,
        "seed": seed,
        "status": "completed",
        "exit_code": 0,
        "output": written,
    }


def consolidate_aws(
    config: CampaignConfig,
    *,
    env_name: str = "aws",
    output_dir: str | None = None,
    tracking_uri: str | None = None,
    s3_client: Any | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Consolidate per-seed matrices from S3, score grids, compare to paper, and publish to MLflow."""
    settings = resolve_campaign_settings(
        config,
        env_name=env_name,
        platform="aws" if (env_name == "aws" or str(output_dir or "").startswith("s3://")) else None,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        save_model_checkpoints=False,
    )
    backend, tracker, resolved_run_id = create_campaign_tracker(
        config, settings, run_id=run_id or config.name
    )
    if s3_client is not None and hasattr(backend, "_s3_client"):
        backend._s3_client = s3_client

    fp = config_fingerprint(config)
    seeds = list(config.seeds) if config.seeds else [0, 1, 2, 3, 4]
    seed_matrices: list[tuple[Any, ...]] = []
    missing: list[int] = []
    for s in seeds:
        raw = backend.read_bytes(campaign_seed_rel_path(config.name, s, fingerprint=fp))
        if raw is None:
            missing.append(s)
        else:
            seed_matrices.append(deserialize_seed_result(raw))
    if missing:
        raise FileNotFoundError(f"Missing seed matrices in storage for seeds: {missing}")

    tracker.start_campaign()
    try:
        if config.protocol == "cara":
            proto = CaraProtocol()
            result = proto.score_and_finalize(
                config,
                seed_matrices,
                backend,
                tracker,
                gamma=config.kernel_gamma,
            )
        elif config.protocol == "upf":
            proto_upf = UpfProtocol()
            result = proto_upf.score_and_finalize(
                config,
                seed_matrices,
                backend,
                tracker,
            )
        else:
            raise ValueError(f"Unsupported protocol {config.protocol!r}")
        final = finalize_campaign_result(config, result, backend, tracker, settings, resolved_run_id)
    except BaseException:
        tracker.fail_campaign()
        raise

    return {
        "campaign": config.name,
        "run_id": resolved_run_id,
        "protocol": final.protocol,
        "dataset": final.dataset,
        "seeds": seeds,
        "artifacts": final.artifacts,
        "mlflow_receipt": final.mlflow_receipt,
    }


def _common_cli_flags(
    *,
    payload_uri: str,
    env_name: str,
    output_dir: str | None,
    tracking_uri: str | None,
) -> list[str]:
    flags = ["--payload-uri", payload_uri, "--env", env_name]
    if output_dir:
        flags.extend(["--output-dir", output_dir])
    if tracking_uri:
        flags.extend(["--tracking-uri", tracking_uri])
    return flags


def submit_aws_campaign(
    config: CampaignConfig,
    *,
    job_queue: str,
    job_definition: str,
    env_name: str = "aws",
    output_dir: str | None = None,
    tracking_uri: str | None = None,
    vcpus: str | int | float | None = None,
    memory_mib: int | None = None,
    dry_run: bool = False,
    batch_client: Any | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Plan and optionally submit 1 AWS Batch Fargate Spot job per seed + 1 consolidate job."""
    resolved_vcpus, resolved_mem = resolve_campaign_resources(
        config, vcpus=vcpus, memory_mib=memory_mib
    )
    settings = resolve_campaign_settings(
        config,
        env_name=env_name,
        platform="aws" if (env_name == "aws" or str(output_dir or "").startswith("s3://")) else None,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        save_model_checkpoints=False,
    )
    out_str = str(settings.output_dir)
    if out_str.startswith("s3://"):
        bucket, prefix = _parse_s3_uri(out_str)
        rel_cfg = campaign_config_rel_path(config.name)
        cfg_key = f"{prefix}/{rel_cfg}" if prefix else rel_cfg
        payload_uri = f"s3://{bucket}/{cfg_key}"
    else:
        bucket, cfg_key = "", ""
        payload_uri = str(Path(out_str) / campaign_config_rel_path(config.name))

    if not dry_run:
        body = json.dumps(
            config.model_dump(mode="json", exclude_unset=True), indent=2, sort_keys=True
        ).encode("utf-8")
        if out_str.startswith("s3://"):
            s3 = s3_client
            if s3 is None:
                import boto3  # type: ignore[import-untyped]

                s3 = boto3.client("s3")
            s3.put_object(Bucket=bucket, Key=cfg_key, Body=body)
        else:
            p = Path(payload_uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)

    base_flags = _common_cli_flags(
        payload_uri=payload_uri,
        env_name=env_name,
        output_dir=output_dir or (out_str if out_str.startswith("s3://") else None),
        tracking_uri=tracking_uri or settings.tracking_uri or None,
    )
    resource_reqs = [
        {"type": "VCPU", "value": resolved_vcpus},
        {"type": "MEMORY", "value": str(resolved_mem)},
    ]
    env_overrides = [
        {"name": "BENCH_ENV", "value": env_name},
        {"name": "BENCH_BENCHMARK_VCPUS", "value": resolved_vcpus},
        {"name": "BENCH_SAVE_MODEL_CHECKPOINTS", "value": "false"},
    ]

    seeds = list(config.seeds) if config.seeds else [0, 1, 2, 3, 4]
    seed_jobs: list[dict[str, Any]] = []
    for s in seeds:
        seed_jobs.append(
            {
                "jobName": f"{config.name}-seed-{s}",
                "jobQueue": job_queue,
                "jobDefinition": job_definition,
                "parameters": {"payload_uri": payload_uri},
                "containerOverrides": {
                    "command": [
                        "continual-bench",
                        "aws",
                        "run-seed",
                        "--seed",
                        str(s),
                        *base_flags,
                    ],
                    "resourceRequirements": resource_reqs,
                    "environment": env_overrides,
                },
                "tags": {
                    "cb-campaign": config.name,
                    "cb-protocol": config.protocol,
                    "cb-seed": str(s),
                },
                "propagateTags": True,
            }
        )

    consolidate_job: dict[str, Any] = {
        "jobName": f"{config.name}-consolidate",
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "parameters": {"payload_uri": payload_uri},
        "containerOverrides": {
            "command": [
                "continual-bench",
                "aws",
                "consolidate",
                *base_flags,
            ],
            "resourceRequirements": resource_reqs,
            "environment": env_overrides,
        },
        "tags": {
            "cb-campaign": config.name,
            "cb-protocol": config.protocol,
            "cb-stage": "consolidate",
        },
        "propagateTags": True,
    }

    if dry_run:
        return {
            "dry_run": True,
            "campaign": config.name,
            "protocol": config.protocol,
            "dataset": config.dataset.name,
            "seeds": seeds,
            "resources": {"vcpus": resolved_vcpus, "memory_mib": resolved_mem},
            "payload_uri": payload_uri,
            "seed_jobs": seed_jobs,
            "consolidate_job": consolidate_job,
        }

    batch = batch_client
    if batch is None:
        import boto3  # type: ignore[import-untyped]

        batch = boto3.client("batch")

    submitted_seeds: list[dict[str, str]] = []
    for job_req in seed_jobs:
        resp = batch.submit_job(**job_req)
        submitted_seeds.append({"jobName": job_req["jobName"], "jobId": resp["jobId"]})

    consolidate_req = {
        **consolidate_job,
        "dependsOn": [{"jobId": item["jobId"], "type": "SEQUENTIAL"} for item in submitted_seeds],
    }
    cons_resp = batch.submit_job(**consolidate_req)
    return {
        "dry_run": False,
        "campaign": config.name,
        "protocol": config.protocol,
        "dataset": config.dataset.name,
        "seeds": seeds,
        "resources": {"vcpus": resolved_vcpus, "memory_mib": resolved_mem},
        "payload_uri": payload_uri,
        "seed_jobs": submitted_seeds,
        "consolidate_job": {"jobName": consolidate_job["jobName"], "jobId": cons_resp["jobId"]},
    }
