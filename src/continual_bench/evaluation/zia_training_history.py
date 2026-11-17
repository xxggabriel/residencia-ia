"""Per-fit evidence for Zia, independent of the tracking service."""

import shutil

from continual_bench.evaluation.environment import encoded
from continual_bench.infrastructure.storage import atomic_write

HISTORY_FILE = "training-history.json"
OBSERVATION_FIELDS = {
    "observation_steps",
    "observation_epochs",
    "observed_step_kind",
    "observation_scopes",
    "observation_effort",
    "observation_resources",
}


class TrainingHistory:
    def __init__(self, output, stop_requested=None):
        self.output = output
        self.partial = output / "training-fits-partial"
        self.stop_requested = stop_requested
        self.fits = []
        self.global_step = 0

    def check_interrupted(self):
        if self.stop_requested is not None and self.stop_requested():
            raise InterruptedError("Zia evaluation was interrupted")

    def capture(
        self,
        result,
        *,
        action,
        block_index,
        start,
        timestamps,
        seconds,
        current_block_start=None,
        source_model_version=None,
    ):
        history = result["history"]
        summary = {
            key: value for key, value in result.items() if key != "history" and key not in OBSERVATION_FIELDS
        }
        kind = "minibatch" if action == "online" else "epoch"
        step_field = "minibatch_step" if kind == "minibatch" else "epoch"
        steps = []
        for step, metrics in enumerate(history, start=1):
            if metrics[step_field] != step:
                raise ValueError(f"invalid {kind} step sequence in the training summary")
            self.global_step += 1
            steps.append({"step": step, "global_step": self.global_step, "kind": kind, "metrics": metrics})
        interrupted = result["stopped_by_signal"]
        if not steps and not interrupted:
            raise ValueError("completed fit has no training steps")
        fit = {
            "fit_id": f"fit-{len(self.fits) + 1:06d}",
            "block_index": block_index,
            "action": action,
            "status": "interrupted" if interrupted else "complete",
            "start_seq": start,
            "end_seq": start + len(timestamps),
            "start_time_ms": int(timestamps[0]),
            "end_time_ms": int(timestamps[-1]),
            "train_samples": result["train_rows"],
            "validation_samples": result["validation_rows"],
            "training_presentations": result["training_presentations"],
            "train_seconds": seconds,
            "steps": steps,
            "summary": summary,
            "checkpoint": None,
            "deployed_version": None,
            "source_model_version": source_model_version,
            "source_model_reason": "fresh_initialization" if action not in ("continue", "online") else None,
            "promotion_status": "interrupted" if interrupted else "candidate",
            "fit_sequence": len(self.fits) + 1,
            **{key: result[key] for key in OBSERVATION_FIELDS if key in result},
        }
        epoch_coordinates = ("best_epoch", "last_epoch", "checkpoint_epoch")
        fit.update({key: result[key] for key in epoch_coordinates if key in result})
        effort = fit["observation_effort"] = dict(fit.get("observation_effort", {}))
        active_start = start if current_block_start is None else current_block_start
        replay_rows = min(result["train_rows"], active_start - start)
        effort["replay_presentations"] = replay_rows * (1 if kind == "minibatch" else len(steps))
        effort["examples_observed"] = len(timestamps)
        if kind == "minibatch":
            fit["epoch_availability"] = {
                "status": "unavailable",
                "reason": "online_minibatches_have_no_epoch",
            }
        if fit.get("observation_steps") and all(
            row["metrics"]["training/native_loss"] is None for row in fit["observation_steps"]
        ):
            reason = (
                "complete_tree_regularized_native_objective_not_exposed"
                if fit["observed_step_kind"].startswith("xgboost")
                else "native_objective_not_observed"
            )
            fit["native_loss_availability"] = {
                "status": "unavailable",
                "reason": reason,
            }
        self.fits.append(fit)
        if interrupted:
            self.save_partial(fit)
            raise InterruptedError("Zia fit was interrupted")
        return fit

    def save_partial(self, fit):
        atomic_write(self.partial / f"{fit['fit_id']}.json", encoded(fit))

    def save(self):
        atomic_write(self.output / HISTORY_FILE, encoded({"version": 1, "fits": self.fits}))
        if self.partial.exists():
            shutil.rmtree(self.partial)


def unique_train_rows(fits):
    """Union of training prefixes, excluding the validation suffix of each fit."""
    intervals = sorted((fit["start_seq"], fit["start_seq"] + fit["train_samples"]) for fit in fits)
    total = end = 0
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total
