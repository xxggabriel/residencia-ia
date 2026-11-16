"""Per-fit evidence for Zia, independent of the tracking service."""

import shutil

from continual_bench.evaluation.environment import encoded
from continual_bench.infrastructure.storage import atomic_write

HISTORY_FILE = "training-history.json"


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

    def capture(self, result, *, action, block_index, start, timestamps, seconds):
        history = result["history"]
        summary = {key: value for key, value in result.items() if key != "history"}
        kind = "minibatch" if action == "online" else "epoch"
        step_field = "minibatch_step" if kind == "minibatch" else "epoch"
        steps = []
        for step, metrics in enumerate(history, start=1):
            if metrics[step_field] != step:
                raise ValueError(f"invalid {kind} step sequence in the training summary")
            self.global_step += 1
            steps.append({"step": step, "global_step": self.global_step, "kind": kind,
                          "metrics": metrics})
        interrupted = result["stopped_by_signal"]
        if not steps and not interrupted:
            raise ValueError("completed fit has no training steps")
        fit = {
            "fit_id": f"fit-{len(self.fits) + 1:06d}",
            "block_index": block_index, "action": action,
            "status": "interrupted" if interrupted else "complete",
            "start_seq": start, "end_seq": start + len(timestamps),
            "start_time_ms": int(timestamps[0]), "end_time_ms": int(timestamps[-1]),
            "train_samples": result["train_rows"], "validation_samples": result["validation_rows"],
            "training_presentations": result["training_presentations"], "train_seconds": seconds,
            "steps": steps, "summary": summary, "checkpoint": None, "deployed_version": None,
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
