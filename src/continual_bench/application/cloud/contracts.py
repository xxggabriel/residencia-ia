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
OFFLINE_MATRIX_FAMILIES = ("river_abrupt", "river_moving", "river_regression")


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


class ZiaMatrix(Strict):
    kind: Literal["zia"] = "zia"
    policies: tuple[ZiaPolicy, ...] = ZIA_POLICIES

    @field_validator("policies")
    @classmethod
    def unique(cls, value):
        if not value or len(set(value)) != len(value):
            raise ValueError("policies must be nonempty and unique")
        return value


class ExperimentMatrixRef(Strict):
    """A matrix the project already plans (`configs/matrices/*.yaml`), one job per case."""

    kind: Literal["experiment_matrix"] = "experiment_matrix"
    path: str = Field(pattern=r"^configs/matrices/[A-Za-z0-9_.-]+\.ya?ml$")


class ProjectSuite(Strict):
    kind: Literal["project_suite"] = "project_suite"
    samples: int = Field(default=3000, ge=300, le=100_000)
    dataset_examples: int = Field(default=100, ge=100, le=1000)
    epochs: int = Field(default=2, ge=1, le=20)
    label_delay: int = Field(default=3, ge=0, le=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class CampaignRequest(Strict):
    campaign_id: str = Field(pattern=CAMPAIGN_ID)
    dataset: Literal["fraud", "social", "ecommerce"] | None = None
    scope: Literal["smoke", "full"] | None = None
    snapshot: SnapshotRef | None = None
    release: str = Field(default="latest", pattern=r"^(latest|[0-9a-f]{40})$")
    matrix: ZiaMatrix | ExperimentMatrixRef | ProjectSuite = Field(
        default_factory=ZiaMatrix, discriminator="kind"
    )
    source_digests: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    max_parallel: int = Field(default=4, ge=1, le=256)
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
        elif self.snapshot is not None:
            raise ValueError("matrix campaigns use source_digests for external data")
        return self


class RunUnit(Strict):
    """One independent execution: one Batch job, one run, one final manifest."""

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
