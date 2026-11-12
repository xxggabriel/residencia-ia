"""AWS SageMaker Experiments adapter using native boto3 with fail-safe CloudWatch stdout fallback."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

from continual_bench.config.iterative import IterativeTrainingConfig

from .session import client as make_client
from .session import error_code

REQUIRED_EPOCH_METRICS = (
    "train:log_loss",
    "train:accuracy",
    "validation:log_loss",
    "validation:accuracy",
    "validation:f1_macro",
    "epoch",
    "early_stopped",
    "best_epoch",
    "patience_consumed",
    "training_duration_seconds",
    "validation_duration_seconds",
    "peak_rss_mb",
    "best_validation_loss",
)

BATCH_PUT_METRICS_MAX_ITEMS = 10


def format_cloudwatch_metric_line(metrics: dict[str, Any]) -> str:
    """Format epoch metrics as `name=value;` pairs for CloudWatch Logs / SageMaker AMT."""
    ordered_keys = [k for k in REQUIRED_EPOCH_METRICS if k in metrics]
    ordered_keys.extend(k for k in metrics if k not in REQUIRED_EPOCH_METRICS)
    tokens = []
    for key in ordered_keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            tokens.append(f"{key}={int(value)};")
        elif isinstance(value, (int, float)) and math.isfinite(value):
            tokens.append(f"{key}={value};")
    return " ".join(tokens)


def format_trial_parameters(
    hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert hyperparameters into SageMaker TrialComponent ParameterValue structures."""
    base = IterativeTrainingConfig().to_hyperparameters_dict()
    if isinstance(hyperparameters, IterativeTrainingConfig):
        merged = {**base, **hyperparameters.to_hyperparameters_dict()}
    elif isinstance(hyperparameters, dict):
        cfg = IterativeTrainingConfig.from_params(hyperparameters)
        merged = {**base, **cfg.to_hyperparameters_dict(), **hyperparameters}
    else:
        merged = base

    formatted: dict[str, dict[str, Any]] = {}
    for key, value in merged.items():
        if value is None:
            continue
        if isinstance(value, bool):
            formatted[str(key)] = {"StringValue": str(value).lower()}
        elif isinstance(value, (int, float)) and math.isfinite(value):
            formatted[str(key)] = {"NumberValue": float(value)}
        else:
            formatted[str(key)] = {"StringValue": str(value)[:256]}
    return formatted


class SageMakerExperimentsTracker:
    """Decoupled boto3 adapter for AWS SageMaker Experiments with fail-safe resilience."""

    def __init__(
        self,
        *,
        dataset: str,
        policy: str,
        model_name: str,
        run_id: str,
        job_name: str | None = None,
        project: str = "continual-bench",
        stack: str | None = None,
        git_sha: str | None = None,
        client: Any | None = None,
        metrics_client: Any | None = None,
        logger: logging.Logger | None = None,
        emit_stdout: bool = True,
    ):
        self.project = project
        self.dataset = dataset
        self.policy = policy
        self.model_name = model_name
        self.run_id = run_id
        self.job_name = job_name or f"cb-{dataset}-{policy}-{run_id}"
        self.stack = stack or project
        self.git_sha = git_sha or os.environ.get("BENCH_GIT_REVISION", "uncommitted")
        self._client = client
        self._metrics_client = metrics_client
        self.logger = logger or logging.getLogger("continual_bench.aws.experiments")
        self.emit_stdout = emit_stdout
        self.recorded_epochs: list[dict[str, Any]] = []
        self.api_errors: list[str] = []
        self.experiment_arn: str | None = None
        self.trial_arn: str | None = None
        self.trial_component_arn: str | None = None

    @property
    def experiment_name(self) -> str:
        return f"{self.project}-zia-experiments"

    @property
    def trial_name(self) -> str:
        return f"{self.dataset}-{self.policy}-{self.model_name}-{self.run_id}"

    @property
    def trial_component_name(self) -> str:
        return f"{self.job_name}-training"

    @property
    def tags(self) -> list[dict[str, str]]:
        return [
            {"Key": "cb-dataset", "Value": str(self.dataset)},
            {"Key": "cb-policy", "Value": str(self.policy)},
            {"Key": "cb-stack", "Value": str(self.stack)},
            {"Key": "cb-git-sha", "Value": str(self.git_sha)},
        ]

    def operational_tags(self) -> list[dict[str, str]]:
        return self.tags

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.recorded_epochs

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client("sagemaker")
        return self._client

    @property
    def metrics_client(self) -> Any:
        if self._metrics_client is not None:
            return self._metrics_client
        if self._client is not None and hasattr(self._client, "batch_put_metrics"):
            return self._client
        if self._client is None:
            self._metrics_client = make_client("sagemaker-metrics")
            return self._metrics_client
        return None

    def _record_error(self, action: str, error: Exception) -> None:
        msg = f"{action} failed ({type(error).__name__}): {error}"
        self.api_errors.append(msg)
        self.logger.warning("SageMaker Experiments %s", msg)

    def ensure_experiment(self) -> str | None:
        try:
            response = self.client.describe_experiment(ExperimentName=self.experiment_name)
            self.experiment_arn = response["ExperimentArn"]
            return self.experiment_arn
        except Exception as error:
            if error_code(error) not in {"ValidationException", "ResourceNotFound"}:
                self._record_error("describe_experiment", error)
                return None
        try:
            response = self.client.create_experiment(
                ExperimentName=self.experiment_name,
                Description=f"Zia continual-learning experiments for {self.project}",
                Tags=self.tags,
            )
            self.experiment_arn = response["ExperimentArn"]
            return self.experiment_arn
        except Exception as error:
            if error_code(error) in {"ValidationException", "ConflictException", "ResourceInUse"}:
                try:
                    response = self.client.describe_experiment(ExperimentName=self.experiment_name)
                    self.experiment_arn = response["ExperimentArn"]
                    return self.experiment_arn
                except Exception as inner:
                    self._record_error("describe_experiment", inner)
                    return None
            self._record_error("create_experiment", error)
            return None

    def ensure_trial(self) -> str | None:
        try:
            response = self.client.describe_trial(TrialName=self.trial_name)
            self.trial_arn = response["TrialArn"]
            return self.trial_arn
        except Exception as error:
            if error_code(error) not in {"ValidationException", "ResourceNotFound"}:
                self._record_error("describe_trial", error)
                return None
        try:
            response = self.client.create_trial(
                TrialName=self.trial_name,
                ExperimentName=self.experiment_name,
                Tags=self.tags,
            )
            self.trial_arn = response["TrialArn"]
            return self.trial_arn
        except Exception as error:
            if error_code(error) in {"ValidationException", "ConflictException", "ResourceInUse"}:
                try:
                    response = self.client.describe_trial(TrialName=self.trial_name)
                    self.trial_arn = response["TrialArn"]
                    return self.trial_arn
                except Exception as inner:
                    self._record_error("describe_trial", inner)
                    return None
            self._record_error("create_trial", error)
            return None

    def ensure_trial_component(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
    ) -> str | None:
        parameters = format_trial_parameters(hyperparameters)
        try:
            response = self.client.describe_trial_component(
                TrialComponentName=self.trial_component_name
            )
            self.trial_component_arn = response["TrialComponentArn"]
            if hyperparameters is not None:
                self.client.update_trial_component(
                    TrialComponentName=self.trial_component_name,
                    Status={"PrimaryStatus": "InProgress"},
                    Parameters=parameters,
                )
            return self.trial_component_arn
        except Exception as error:
            if error_code(error) not in {"ValidationException", "ResourceNotFound"}:
                self._record_error("describe_trial_component", error)
                return None
        try:
            response = self.client.create_trial_component(
                TrialComponentName=self.trial_component_name,
                DisplayName=self.trial_component_name,
                Status={"PrimaryStatus": "InProgress"},
                Parameters=parameters,
                Tags=self.tags,
            )
            self.trial_component_arn = response["TrialComponentArn"]
            return self.trial_component_arn
        except Exception as error:
            if error_code(error) in {"ValidationException", "ConflictException", "ResourceInUse"}:
                try:
                    response = self.client.describe_trial_component(
                        TrialComponentName=self.trial_component_name
                    )
                    self.trial_component_arn = response["TrialComponentArn"]
                    return self.trial_component_arn
                except Exception as inner:
                    self._record_error("describe_trial_component", inner)
                    return None
            self._record_error("create_trial_component", error)
            return None

    def start(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
    ) -> dict[str, str | None]:
        """Initialize Experiment, Trial, and TrialComponent hierarchy in SageMaker Experiments."""
        exp_arn = self.ensure_experiment()
        trial_arn = self.ensure_trial() if exp_arn is not None else None
        tc_arn = self.ensure_trial_component(hyperparameters) if trial_arn is not None else None
        if trial_arn is not None and tc_arn is not None:
            try:
                self.client.associate_trial_component(
                    TrialComponentName=self.trial_component_name,
                    TrialName=self.trial_name,
                )
            except Exception as error:
                if error_code(error) not in {"ValidationException", "ConflictException", "ResourceInUse"}:
                    self._record_error("associate_trial_component", error)
        return {
            "experiment_name": self.experiment_name,
            "experiment_arn": exp_arn,
            "trial_name": self.trial_name,
            "trial_arn": trial_arn,
            "trial_component_name": self.trial_component_name,
            "trial_component_arn": tc_arn,
        }

    begin = start

    def initialize(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
    ) -> bool:
        result = self.start(hyperparameters)
        return bool(
            result["experiment_arn"] and result["trial_arn"] and result["trial_component_arn"]
        )

    def log_hyperparameters(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig,
    ) -> bool:
        """Update TrialComponent parameters in a fail-safe manner."""
        parameters = format_trial_parameters(hyperparameters)
        try:
            self.client.update_trial_component(
                TrialComponentName=self.trial_component_name,
                Parameters=parameters,
            )
            return True
        except Exception as error:
            self._record_error("update_trial_component", error)
            return False

    def add_tags(self, resource_arn: str) -> bool:
        """Attach operational tags to a SageMaker Experiment/Trial/TrialComponent ARN."""
        try:
            self.client.add_tags(ResourceArn=resource_arn, Tags=self.tags)
            return True
        except Exception as error:
            self._record_error("add_tags", error)
            return False

    def log_epoch_metrics(
        self,
        metrics: dict[str, float | int],
        *,
        step: int | None = None,
        timestamp: int | float | None = None,
    ) -> bool:
        """Emit epoch metrics to CloudWatch stdout and BatchPutMetrics (fail-safe)."""
        epoch_step = int(step if step is not None else metrics.get("epoch", len(self.recorded_epochs) + 1))
        ts = int(timestamp if timestamp is not None else time.time())
        record = {"epoch": epoch_step, **metrics}
        self.recorded_epochs.append(record)

        if self.emit_stdout:
            line = format_cloudwatch_metric_line(record)
            if line:
                print(line, flush=True)

        if self.trial_component_arn is None and self.api_errors:
            return False

        m_client = self.metrics_client
        if m_client is None:
            return False

        metric_items = []
        for name in REQUIRED_EPOCH_METRICS:
            if name in record and record[name] is not None:
                val = float(record[name])
                if math.isfinite(val):
                    metric_items.append(
                        {
                            "MetricName": name,
                            "Timestamp": ts,
                            "Step": epoch_step,
                            "Value": val,
                        }
                    )
        for name, raw_val in record.items():
            is_numeric = isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool)
            if name not in REQUIRED_EPOCH_METRICS and is_numeric:
                val = float(raw_val)
                if math.isfinite(val):
                    metric_items.append(
                        {
                            "MetricName": str(name),
                            "Timestamp": ts,
                            "Step": epoch_step,
                            "Value": val,
                        }
                    )

        success = True
        for idx in range(0, len(metric_items), BATCH_PUT_METRICS_MAX_ITEMS):
            chunk = metric_items[idx : idx + BATCH_PUT_METRICS_MAX_ITEMS]
            try:
                response = m_client.batch_put_metrics(
                    TrialComponentName=self.trial_component_name,
                    MetricData=chunk,
                )
                errors = (response or {}).get("Errors") or []
                if errors:
                    self.logger.warning(
                        "SageMaker Experiments BatchPutMetrics reported partial errors: %s", errors
                    )
            except Exception as error:
                success = False
                self._record_error("batch_put_metrics", error)
                break
        return success

    log_epoch = log_epoch_metrics

    def finish(
        self,
        status: str = "Completed",
        message: str | None = None,
        parameters: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        """Mark the TrialComponent as Completed/Failed/Stopped."""
        if self.trial_component_arn is None and self.api_errors:
            return False
        status_payload: dict[str, str] = {"PrimaryStatus": status}
        if message:
            status_payload["Message"] = str(message)[:1024]
        kwargs: dict[str, Any] = {
            "TrialComponentName": self.trial_component_name,
            "Status": status_payload,
        }
        if parameters is not None:
            kwargs["Parameters"] = parameters
        try:
            self.client.update_trial_component(**kwargs)
            return True
        except Exception as error:
            self._record_error("finish_trial_component", error)
            return False

    def finalize(self, summary: dict[str, Any] | None = None) -> bool:
        params = format_trial_parameters(summary) if summary else None
        msg = f"Completed: {summary}" if summary else "Completed"
        return self.finish("Completed", message=msg, parameters=params)


__all__ = [
    "BATCH_PUT_METRICS_MAX_ITEMS",
    "REQUIRED_EPOCH_METRICS",
    "SageMakerExperimentsTracker",
    "format_cloudwatch_metric_line",
    "format_trial_parameters",
]
