"""Recover training evidence from completed reports without replaying models."""

from collections import defaultdict
from math import isfinite

from .epoch_history import holdout_epoch_rows


def _metrics(values):
    return {
        key: value for key, value in values.items() if isinstance(value, (int, float)) and isfinite(value)
    }


def _fit(
    number,
    action,
    rows,
    units,
    *,
    seq_range=None,
    train_samples=None,
    validation_samples=None,
    presentations=None,
    summary=None,
    checkpoint=None,
    deployed_version=None,
):
    return {
        "fit_id": f"fit-{number:06d}",
        "action": action,
        "status": "complete",
        "block_index": None,
        "start_seq": seq_range[0] if seq_range else None,
        "end_seq": seq_range[1] if seq_range else None,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "training_presentations": presentations,
        "summary": summary or {},
        "checkpoint": checkpoint,
        "deployed_version": deployed_version,
        "steps": [{"step": index, "kind": units, "metrics": row} for index, row in enumerate(rows, 1)],
    }


def _epoch_rows(rows, metrics):
    if [row.get("epoch") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("retained epochs are not a complete, ordered sequence")
    return [metrics(row) for row in rows]


def _repeated_epochs(report):
    rows = report["epochs"]
    if not rows:
        return []

    counts = [row.get("train_presentations") for row in rows]
    return [
        _fit(
            1,
            "initial_fit",
            _epoch_rows(holdout_epoch_rows(rows), _metrics),
            "epoch",
            train_samples=counts[0] if all(count == counts[0] for count in counts) else None,
            presentations=sum(counts) if all(isinstance(count, int) for count in counts) else None,
            summary={"epochs_trained": len(rows)},
        )
    ]


def _iterative(report):
    by_update = defaultdict(list)
    for position, row in enumerate(report["epoch_history"], 1):
        update = row.get("update")
        if not isinstance(update, int) or update < 1:
            raise ValueError("iterative epoch is missing its update identity")
        if row.get("global_epoch", position) != position:
            raise ValueError("retained global epochs are not a complete, ordered sequence")
        by_update[update].append(row)
    updates = {row["update"]: row for row in report.get("updates", [])}

    def metrics(row):
        values = _metrics({k: v for k, v in row.items() if k not in ("update", "global_epoch")})
        if report.get("config", {}).get("eval_metric") == "validation:mae":
            # Old iterative reports copied their MAE selection metric into a log-loss alias.
            selection = values.pop("validation:log_loss", None)
            if selection is not None:
                values.setdefault("validation:mae", selection)
            if "best_validation_loss" in values:
                values["best_validation_mae"] = values.pop("best_validation_loss")
        return values

    fits = []
    for number in sorted(by_update):
        retained = updates.get(number, {})
        rows = _epoch_rows(by_update[number], metrics)
        if "epochs_run" in retained and retained["epochs_run"] != len(rows):
            raise ValueError("iterative update count differs from its retained epochs")
        summary = _metrics(
            {
                key: retained.get(key)
                for key in ("epochs_run", "best_epoch", "early_stopped", "patience_consumed", "seconds")
            }
        )
        train_samples = retained.get("train")
        fits.append(
            _fit(
                number,
                "initial_fit" if number == 1 else "continue",
                rows,
                "epoch",
                seq_range=retained.get("range"),
                train_samples=train_samples,
                validation_samples=retained.get("validation"),
                presentations=len(rows) * train_samples if isinstance(train_samples, int) else None,
                summary=summary,
            )
        )
    if any(number not in by_update and row.get("epochs_run") for number, row in updates.items()):
        raise ValueError("iterative update has no retained epoch history")
    return fits


def _native_training(report):
    result = report["training"]
    rows = result["history"]
    if not rows or result.get("stopped_by_signal"):
        return []
    metadata = report.get("metadata", {})
    splits = metadata.get("splits", {})
    checkpoint = None
    if metadata.get("state_sha256"):
        checkpoint = {
            "file": metadata.get("native_path", metadata.get("payload")),
            "sha256": metadata["state_sha256"],
        }
    return [
        _fit(
            1,
            "initial_fit",
            _epoch_rows(rows, _metrics),
            "epoch",
            seq_range=splits.get("train"),
            train_samples=result.get("train_rows"),
            validation_samples=result.get("validation_rows"),
            presentations=result.get("training_presentations"),
            checkpoint=checkpoint,
            deployed_version=0 if checkpoint else None,
            summary=_metrics(result),
        )
    ]


def _legacy_zia(report):
    fits = []
    for number, version in enumerate(report["model_versions"], 1):
        end = version.get("end_seq")
        if end is None or not version.get("action"):
            raise ValueError("retained Zia checkpoint lacks its training coordinates")
        start = version.get("start_seq")
        checkpoint = None
        if version.get("sha256"):
            checkpoint = {
                "file": version.get("file"),
                "sha256": version["sha256"],
                "bytes": version.get("bytes"),
            }
        values = _metrics(
            {
                "reported_fit_rows": version.get("train_samples"),
                "train_seconds": version.get("train_seconds"),
                "bytes": version.get("bytes"),
            }
        )
        fits.append(
            _fit(
                number,
                version["action"],
                [{"end_seq": end, **values}],
                "update",
                seq_range=(start, end),
                checkpoint=checkpoint,
                deployed_version=number - 1,
            )
        )
    return fits


def recover_training_history(report, directory=None):
    """Return fits and limitations based on retained evidence for recognized report families."""
    if report.get("complete") is False:
        return [], "incomplete", ["report is incomplete"]
    if "training_history" in report:
        return report["training_history"], "recorded", []
    if report.get("version") == "semana09-epochs-v1" and isinstance(report.get("epochs"), list):
        fits, scheme = _repeated_epochs(report), "repeated_epochs"
        limitations = [
            "the report retains frozen holdout metrics only and does not identify model checkpoints",
        ]
    elif isinstance(report.get("epoch_history"), list) and isinstance(report.get("updates"), list):
        fits, scheme = _iterative(report), "iterative_epochs"
        limitations = ["historical reports do not link individual fits to retained model states"]
    elif isinstance(report.get("training"), dict) and isinstance(report["training"].get("history"), list):
        fits, scheme = _native_training(report), "native_epochs"
        limitations = []
    elif report.get("protocol") == "batch_prequential_with_final_temporal_holdout" and isinstance(
        report.get("model_versions"), list
    ):
        fits, scheme = _legacy_zia(report), "zia_model_versions"
        limitations = [
            "checkpoint metadata retains update totals; epoch and minibatch histories are unavailable",
        ]
    elif (
        report.get("schema_version") in {1, "task_evaluation-typed-v1"}
        or report.get("protocol") == "temporal_holdout"
    ):
        from .stream_training_history import stream_training_history

        fits, scheme = stream_training_history(report, directory), "stream_updates"
        limitations = ["the report records update events, without additional epoch or minibatch coordinates"]
        if any(fit.get("checkpoint") is None for fit in fits):
            limitations.append("the retained evidence does not link every update to a checkpoint")
    else:
        return [], "unsupported", ["report has no recognized training evidence"]
    global_step = 0
    for fit in fits:
        for step in fit["steps"]:
            global_step += 1
            step["global_step"] = global_step
    if not fits:
        limitations.append("no completed fit with recorded steps was found")
    return fits, scheme, limitations
