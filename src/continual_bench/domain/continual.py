"""Episode-level continual evaluation, independent of the learner's policy budget."""

import copy
import math
import time

METRICS = (
    "cl_average_accuracy",
    "cl_average_incremental_accuracy",
    "cl_forgetting",
    "cl_backward_transfer",
    "cl_intransigence",
    "cl_forward_transfer",
)


class ContinualEvaluation:
    """Consume released labels in sequence; task ends are exclusive source offsets.

    Probes are indexed by episode, including distinct episodes of a recurrent concept.
    All declared synthetic episode probes are diagnostic evaluations at every boundary;
    future probes never affect adaptation or the released-label training prefix.
    References use the same learner in one pass, rather than an optimized joint oracle.
    """

    def __init__(self, model_factory, probes, task_ends, reference_budget=None, references_enabled=True):
        if not task_ends or any(type(end) is not int or end <= 0 for end in task_ends):
            raise ValueError("task_ends must contain positive integer exclusive offsets")
        if any(left >= right for left, right in zip(task_ends, task_ends[1:])):
            raise ValueError("task_ends must be strictly increasing")
        if reference_budget is not None and (type(reference_budget) is not int or reference_budget < 0):
            raise ValueError("reference_budget must be a nonnegative integer or None")
        if any(type(task) is not int or not 0 <= task < len(task_ends) for task in probes):
            raise ValueError("probe keys must identify episode indices")
        self.model_factory = model_factory
        self.probes = copy.deepcopy(probes)
        self.task_ends = list(task_ends)
        self.reference_budget = reference_budget
        self.references_enabled = references_enabled
        self.examples = []
        self.next_seq = 0
        self.matrix = []
        self.reference_models = []
        self.probe_seconds = 0.0
        self.reference_train_seconds = 0.0

    def _accuracy(self, model, samples):
        started = time.perf_counter()
        try:
            frozen = copy.deepcopy(model)
            correct = 0
            for x, y in samples:
                probability = frozen.predict(copy.deepcopy(x))
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("probe probability must be finite and in [0,1]")
                correct += (probability >= 0.5) == bool(y)
            return correct / len(samples)
        finally:
            self.probe_seconds += time.perf_counter() - started

    def _reference(self, name, task, start, end):
        required = end - start
        metadata = {
            "name": name,
            "task": task,
            "seq_range": [start, end],
            "required_train_samples": required,
            "budget_per_model": self.reference_budget,
            "train_samples": 0,
            "seconds": 0.0,
            "available": False,
            "reason": None,
        }
        if not self.references_enabled:
            metadata["reason"] = "references_disabled"
        elif self.reference_budget is not None and required > self.reference_budget:
            metadata["reason"] = "reference_budget_exceeded"
        elif not self.probes.get(task):
            metadata["reason"] = "missing_probe"
        if metadata["reason"]:
            return None, metadata
        started = time.perf_counter()
        reference = self.model_factory()
        for x, y in self.examples[start:end]:
            reference.learn(copy.deepcopy(x), y)
        seconds = time.perf_counter() - started
        self.reference_train_seconds += seconds
        metadata.update(available=True, train_samples=required, seconds=seconds)
        self.reference_models.append({
            "name": name,
            "task": task,
            "model": reference,
            "seq_range": [start, end],
            "train_samples": required,
            "seconds": seconds,
        })
        return self._accuracy(reference, self.probes[task]), metadata

    def after_label(self, seq, x, y, model):
        """Call after a released label's policy update, including warmup labels.

        No source or pending queue is inspected. Delayed labels therefore remain
        invisible until the caller explicitly releases them through this method.
        """
        if type(seq) is not int or seq != self.next_seq:
            raise ValueError(f"expected released label seq {self.next_seq}, got {seq}")
        if seq >= self.task_ends[-1]:
            raise ValueError("released label is outside the declared episodes")
        self.examples.append((copy.deepcopy(x), bool(y)))
        self.next_seq += 1
        task = len(self.matrix)
        if self.next_seq != self.task_ends[task]:
            return
        accuracies = {
            str(episode): self._accuracy(model, self.probes[episode])
            if self.probes.get(episode) else None
            for episode in range(len(self.task_ends))
        }
        start = self.task_ends[task - 1] if task else 0
        single, single_info = self._reference("task_only", task, start, self.next_seq)
        joint, joint_info = self._reference("joint_prefix_one_pass", task, 0, self.next_seq)
        self.matrix.append({
            "task": task,
            "end_seq": seq,
            "accuracies": accuracies,
            "joint_accuracy": joint,
            "single_accuracy": single,
            "reference_joint": joint_info,
            "reference_single": single_info,
        })
        values, availability = self._metrics(task)
        self.matrix[-1]["metrics"] = values
        self.matrix[-1]["availability"] = availability

    def _metrics(self, task):
        row = self.matrix[task]
        values = dict.fromkeys(METRICS)
        reasons = dict.fromkeys(METRICS, "missing_probe")

        def assign(name, terms, reason="missing_probe"):
            if all(value is not None for value in terms):
                values[name] = sum(terms) / len(terms)
                reasons[name] = None
            else:
                reasons[name] = reason

        assign("cl_average_accuracy", [row["accuracies"][str(previous)] for previous in range(task + 1)])
        incremental = [previous["metrics"]["cl_average_accuracy"] for previous in self.matrix[:task]]
        assign("cl_average_incremental_accuracy", incremental + [values["cl_average_accuracy"]])
        if task:
            forgetting, backward = [], []
            for previous in range(task):
                current = row["accuracies"][str(previous)]
                history = [self.matrix[old]["accuracies"][str(previous)]
                           for old in range(task)]
                first = self.matrix[previous]["accuracies"][str(previous)]
                forgetting.append(max(history) - current
                                  if current is not None and all(v is not None for v in history) else None)
                backward.append(current - first if current is not None and first is not None else None)
            assign("cl_forgetting", forgetting)
            assign("cl_backward_transfer", backward)
            forward = []
            for previous in range(1, task + 1):
                learned = self.matrix[previous]["accuracies"][str(previous)]
                single = self.matrix[previous]["single_accuracy"]
                forward.append(learned - single if learned is not None and single is not None else None)
            missing_reference = next((self.matrix[j]["reference_single"]["reason"]
                                      for j in range(1, task + 1)
                                      if self.matrix[j]["single_accuracy"] is None), "missing_probe")
            assign("cl_forward_transfer", forward, missing_reference)
        else:
            for name in ("cl_forgetting", "cl_backward_transfer", "cl_forward_transfer"):
                reasons[name] = "insufficient_tasks"
        diagonal = row["accuracies"][str(task)]
        if row["joint_accuracy"] is not None and diagonal is not None:
            values["cl_intransigence"] = row["joint_accuracy"] - diagonal
            reasons["cl_intransigence"] = None
        else:
            reasons["cl_intransigence"] = row["reference_joint"]["reason"] or "missing_probe"
        return values, {name: {"available": values[name] is not None, "reason": reasons[name]}
                        for name in METRICS}

    def report(self):
        """Return JSON-compatible evidence; trained reference objects remain on the instance."""
        values = self.matrix[-1]["metrics"] if self.matrix else dict.fromkeys(METRICS)
        availability = self.matrix[-1]["availability"] if self.matrix else {
            name: {"available": False, "reason": "no_completed_tasks"} for name in METRICS
        }
        return copy.deepcopy({
            "schema_version": 1,
            "matrix": self.matrix,
            "series": {name: [{"step": row["task"], "value": row["metrics"][name]}
                              for row in self.matrix] for name in METRICS},
            "metrics": values,
            "availability": availability,
            "released_labels": self.next_seq,
            "completed_tasks": len(self.matrix),
            "probe_seconds": self.probe_seconds,
            "reference_train_seconds": self.reference_train_seconds,
            "overhead_seconds": self.probe_seconds + self.reference_train_seconds,
            "reference_train_samples": sum(item["train_samples"] for item in self.reference_models),
            "contract": {
                "task_ends": self.task_ends,
                "task_intervals": "half-open; episode indices distinguish recurring concepts",
                "matrix_scope": (
                    "all_declared_synthetic_episode_probes; future scores diagnostic only no adaptation"
                ),
                "average_accuracy_scope": "learned episodes only, j <= current task",
                "reference_budget_per_model": self.reference_budget,
                "references_enabled": self.references_enabled,
                "reference_kind": "same-learner task-only and joint-prefix one-pass operational baselines",
                "reference_budget_scope": "separate from main policy budget; no cumulative reference cap",
                "forward_transfer": "mean learned-task diagonal minus task-only accuracy, excluding task 0",
                "forgetting": "past sampled maximum excluding current row minus current accuracy; unclipped",
                "forgetting_max_scope": (
                    "all prior rows, including evaluations before the episode was learned"
                ),
                "formula_interpretation": (
                    "Wang 2302.00487v3 Eq.3 literal all-prior-row maximum; "
                    "extends the earlier triangular accuracy definition with diagnostic future probes"
                ),
                "overhead_scope": "probe deepcopy/inference plus reference construction/training",
            },
        })
