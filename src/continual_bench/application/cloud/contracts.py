"""Validated inputs and records of the AWS profile (dataset pipeline, releases and campaigns)."""

import hashlib
import json
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from continual_bench.config import Strict
from continual_bench.core.policies import ZiaPolicy
from continual_bench.datasets.zia import DATASETS, FORMAT, SCOPES

CAMPAIGN_ID = r"^[a-z0-9][a-z0-9-]{0,39}$"
GIT_SHA = r"^[0-9a-f]{40}$"
DIGEST = r"^sha256:[0-9a-f]{64}$"
ZIA_POLICIES = ("frozen", "drift", "incremental", "periodic")
EVALUATION_PROTOCOL = "zia-batch-prequential-holdout-v1"
STREAM_METRICS_PROTOCOL = "ordered-stream-batch-prequential-metrics-v2"
OFFLINE_MATRIX_FAMILIES = ("river_abrupt", "river_moving", "river_regression")
# AWS Batch on FARGATE_SPOT; metrics-only attempts restart evaluation without model checkpoints.
SPOT_BACKEND = "batch-fargate-spot"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class SnapshotRef(Strict):
    dataset: Literal["fraud", "social", "ecommerce"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    preparation_version: str = Field(default=FORMAT, pattern=r"^[a-z0-9.-]{1,64}$")
    scope: Literal["smoke", "full"]

    @model_validator(mode="after")
    def admitted(self):
        if self.revision != DATASETS[self.dataset]["revision"]:
            raise ValueError("revision is not admitted by the dataset catalog")
        if self.preparation_version != FORMAT:
            raise ValueError(f"this release prepares only {FORMAT}")
        return self

    @property
    def snapshot_id(self) -> str:
        return f"{self.dataset}/{self.revision}/{self.preparation_version}/{self.scope}"


class DatasetRequest(Strict):
    """Input of the manual dataset state machine. Sources come from the catalog, never from URLs."""

    dataset: Literal["fraud", "social", "ecommerce"]
    scope: Literal["smoke", "full"]
    revision: str = "pinned"
    mode: Literal["download", "import", "prepare"] = "download"
    import_uri: str | None = Field(default=None, pattern=r"^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/.+$")
    preparation_version: str = FORMAT

    @model_validator(mode="after")
    def consistent(self):
        if self.mode == "import" and not self.import_uri:
            raise ValueError("mode=import requires import_uri (an S3 prefix holding the pinned files)")
        if self.mode != "import" and self.import_uri:
            raise ValueError("import_uri is accepted only with mode=import")
        return self

    def snapshot(self) -> SnapshotRef:
        revision = DATASETS[self.dataset]["revision"] if self.revision == "pinned" else self.revision
        return SnapshotRef(
            dataset=self.dataset,
            revision=revision,
            preparation_version=self.preparation_version,
            scope=self.scope,
        )


class ImageRef(Strict):
    repository: str = Field(min_length=1)
    digest: str = Field(pattern=DIGEST)
    tag: str = Field(pattern=r"^git-[0-9a-f]{40}$")

    @property
    def uri(self) -> str:
        return f"{self.repository}@{self.digest}"


class ReleaseManifest(Strict):
    version: Literal["release-v1"] = "release-v1"
    git_sha: str = Field(pattern=GIT_SHA)
    stack: str
    region: str = "us-east-1"
    images: dict[Literal["runtime", "ml"], ImageRef]
    job_definitions: dict[Literal["benchmark", "prepare"], str]
    control_function: str
    benchmark_compute: Literal["batch-fargate-spot"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    pipeline_execution_id: str | None = None

    @field_validator("job_definitions")
    @classmethod
    def pinned_revisions(cls, value):
        for name, arn in value.items():
            if not arn.startswith("arn:aws:batch:") or not arn.rsplit(":", 1)[-1].isdigit():
                raise ValueError(f"job definition {name} must be an ARN with an explicit revision")
        return value

    @field_validator("control_function")
    @classmethod
    def pinned_function(cls, value):
        if not value.startswith("arn:aws:lambda:") or not value.rsplit(":", 1)[-1].isdigit():
            raise ValueError("control function must be a version-qualified Lambda ARN")
        return value


class MatrixMetricsCacheRef(Strict):
    """A completed run's immutable matrix metrics, never serialized model state."""

    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("uri")
    @classmethod
    def campaign_evidence_only(cls, value):
        from continual_bench.infrastructure.object_store import split_uri

        _, key = split_uri(value)
        if not re.fullmatch(
            r"campaigns/[a-z0-9-]{1,40}/runs/[a-f0-9]{20}/attempts/[A-Za-z0-9_.-]+/matrix-metrics\.json\.gz",
            key,
        ):
            raise ValueError("matrix metric cache must name a completed campaign evidence file")
        return value


class ZiaMatrix(Strict):
    kind: Literal["zia"] = "zia"
    policies: tuple[ZiaPolicy, ...] = ZIA_POLICIES
    # Keep historical requests as one seed/full retention. New campaigns opt in
    # explicitly to three independent seeded units and metrics-only retention.
    seeds: tuple[int, ...] = Field(default=(42,), min_length=1)
    artifact_mode: Literal["full", "metrics_only"] = "full"
    iterative_training: dict = Field(default_factory=dict, exclude_if=lambda value: not value)
    metric_cache_sources: dict[int, MatrixMetricsCacheRef] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @field_validator("policies")
    @classmethod
    def unique(cls, value):
        if not value or len(set(value)) != len(value):
            raise ValueError("policies must be nonempty and unique")
        return value

    @field_validator("seeds")
    @classmethod
    def valid_seeds(cls, value):
        if len(set(value)) != len(value) or any(seed < 0 or seed >= 2**31 for seed in value):
            raise ValueError("seeds must be unique integers in [0, 2**31)")
        return value

    @field_validator("iterative_training")
    @classmethod
    def valid_training(cls, value):
        from continual_bench.config import IterativeTrainingConfig

        if "seed" in value:
            raise ValueError("set training randomness through matrix.seeds")
        IterativeTrainingConfig.model_validate(value)
        return value

    @model_validator(mode="after")
    def metric_cache_scope(self):
        if self.metric_cache_sources and (
            self.artifact_mode != "metrics_only" or not set(self.metric_cache_sources) <= set(self.seeds)
        ):
            raise ValueError("metric cache sources require metrics_only and matching matrix seeds")
        return self


class StreamMatrix(ZiaMatrix):
    """The same batch-prequential evaluator with a checksum-pinned prepared stream."""

    kind: Literal["stream"] = "stream"


class ExperimentMatrixRef(Strict):
    """A matrix the project already plans (`configs/matrices/*.yaml`), one job per case."""

    kind: Literal["experiment_matrix"] = "experiment_matrix"
    path: str = Field(pattern=r"^configs/matrices/[A-Za-z0-9_.-]+\.ya?ml$")


class ProjectSuite(Strict):
    kind: Literal["project_suite"] = "project_suite"
    samples: int = Field(default=3000, ge=300, le=100_000_000)
    dataset_examples: int = Field(default=100, ge=100, le=1000)
    epochs: int = Field(default=2, ge=1, le=20)
    label_delay: int = Field(default=3, ge=0, le=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class BenchmarkWorkload(Strict):
    """Generic workload matrix for any registered dataset, model, policy, detector, and metrics."""

    kind: Literal["benchmark"] = "benchmark"
    evaluator: Literal["binary", "typed", "epochs"] = "binary"
    datasets: tuple[str, ...] = Field(min_length=1)
    models: tuple[str, ...] = Field(default=("logistic",), min_length=1)
    policies: tuple[str, ...] = Field(default=("cara", "upf", "oracle", "drift"), min_length=1)
    detectors: tuple[str, ...] = Field(default=("adwin",), min_length=1)
    modes: tuple[Literal["streaming", "incremental", "offline"], ...] = ("streaming", "incremental")
    metrics: tuple[str, ...] = Field(default_factory=tuple)
    seeds: tuple[int, ...] = Field(default=(42,), min_length=1)
    samples: int | None = Field(default=None, ge=10, le=100_000_000)
    label_delay: int | None = Field(default=None, ge=0, le=10_000)
    base: dict = Field(default_factory=dict)
    dataset_params: dict = Field(default_factory=dict)
    model_params: dict = Field(default_factory=dict)
    policy_params: dict = Field(default_factory=dict)
    on_incompatible: Literal["skip", "error"] = "skip"


class CampaignRequest(Strict):
    campaign_id: str = Field(pattern=CAMPAIGN_ID)
    dataset: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    scope: Literal["smoke", "full"] | None = None
    snapshot: SnapshotRef | None = None
    release: str = Field(default="latest", pattern=r"^(latest|[0-9a-f]{40})$")
    matrix: ZiaMatrix | StreamMatrix | ExperimentMatrixRef | ProjectSuite | BenchmarkWorkload = Field(
        default_factory=ZiaMatrix, discriminator="kind"
    )
    source_digests: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    max_parallel: int | None = Field(default=None, ge=1, le=256, exclude_if=lambda value: value is None)
    max_attempts: int = Field(default=2, ge=1, le=5)
    queue_timeout_seconds: int = Field(default=3600, ge=600, le=86_400)
    run_timeout_seconds: int = Field(default=14400, ge=900, le=14 * 24 * 3600)
    tracking: str = Field(default="configs/base/tracking.yaml", pattern=r"^configs/[A-Za-z0-9_./-]+\.ya?ml$")
    notify: bool = True

    @model_validator(mode="before")
    @classmethod
    def snapshot_defaults(cls, value):
        if isinstance(value, dict) and isinstance(value.get("snapshot"), dict):
            value = {**value}
            value.setdefault("dataset", value["snapshot"].get("dataset"))
            value.setdefault("scope", value["snapshot"].get("scope"))
        return value

    @model_validator(mode="after")
    def consistent(self):
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", dataset) or not re.fullmatch(r"[0-9a-f]{64}", sha)
            for dataset, sha in self.source_digests.items()
        ):
            raise ValueError("source_digests requires dataset names and SHA-256 hex digests")
        if self.matrix.kind == "zia":
            if self.source_digests:
                raise ValueError("Zia campaigns take their data from the prepared snapshot")
            if self.snapshot is None:
                raise ValueError("Zia campaigns require a prepared snapshot")
            if (self.dataset, self.scope) != (self.snapshot.dataset, self.snapshot.scope):
                raise ValueError("dataset/scope differ from the snapshot")
        elif self.matrix.kind == "stream":
            if self.snapshot is not None or not self.dataset or not self.scope:
                raise ValueError("stream campaigns require dataset/scope and no Zia snapshot")
            if set(self.source_digests) != {self.dataset}:
                raise ValueError("stream campaigns require exactly the dataset prepared source digest")
        elif self.snapshot is not None:
            raise ValueError("matrix campaigns use source_digests for external data")
        return self


class RunUnit(Strict):
    """One independent logical execution with one final manifest."""

    run_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    campaign_id: str = Field(pattern=CAMPAIGN_ID)
    kind: Literal["zia", "matrix_case"]
    params: dict
    contract: dict
    contract_sha256: str
    job_name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    unit_key: str
    labels: dict[str, str] = Field(default_factory=dict)


def run_identity(campaign_id: str, contract: dict) -> tuple[str, str]:
    """Deterministic run_id from the campaign and the full execution contract."""
    contract_sha256 = digest(contract)
    return digest({"campaign_id": campaign_id, "contract_sha256": contract_sha256})[:20], contract_sha256


SCOPE_VALUES = SCOPES
