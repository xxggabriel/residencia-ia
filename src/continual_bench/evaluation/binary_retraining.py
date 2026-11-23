"""Cost-aware batch retraining (CARA, UPF, Oracle) for binary benchmarks.

Supports two roles:
1. Executing a binary run whose active policy is a batch policy (`cara`, `cara_cumulative`,
   `cara_periodic`, `upf`, `oracle`).
2. Computing the CARA/UPF/Oracle performance matrix and analysis metrics for any binary
   prequential run, even when it uses another policy (`drift`, `periodic`, `online`, `none`).
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from continual_bench.core.contracts import DecisionContext, Event
from continual_bench.core.metrics import Metrics
from continual_bench.core.observability import OnlineObservability
from continual_bench.retraining import PerformanceMatrix, SimResult, build_matrix, simulate
from continual_bench.retraining.analysis import (
    DEFAULT_RULES,
    analyze,
    deployed_metrics,
    json_parameters,
    retraining_reference_metrics,
)
from continual_bench.retraining.rules import Oracle

from .binary import BinaryRun, Outcome, Progress, continual_evaluation
from .environment import current_rss_bytes

MAX_RETRAINING_BLOCKS = 24


class FeatureCodec:
    """Lossless encoder/decoder between Event.x dicts and finite float64 arrays."""

    def __init__(self, events: list[Event]):
        keys: set[str] = set()
        for event in events:
            keys.update(event.x.keys())
        self.names: tuple[str, ...] = tuple(sorted(keys))
        self.cat_to_id: dict[str, dict[Any, float]] = {}
        self.id_to_cat: dict[str, dict[float, Any]] = {}
        for event in events:
            for name in self.names:
                val = event.x.get(name)
                if val is None or isinstance(val, (bool, int, float)):
                    continue
                mapping = self.cat_to_id.setdefault(name, {})
                if val not in mapping:
                    idx = float(len(mapping))
                    mapping[val] = idx
                    self.id_to_cat.setdefault(name, {})[idx] = val
        self.bool_cols = {
            name
            for name in self.names
            if any(isinstance(e.x.get(name), bool) for e in events)
            and not any(isinstance(e.x.get(name), float) for e in events)
        }

    def encode(self, events: list[Event]) -> tuple[np.ndarray, np.ndarray]:
        rows = len(events)
        cols = len(self.names)
        if cols == 0:
            return (
                np.zeros((rows, 1), dtype=np.float64),
                np.array([int(bool(e.y)) for e in events], dtype=np.int64),
            )
        x = np.empty((rows, cols), dtype=np.float64)
        for r, event in enumerate(events):
            for c, name in enumerate(self.names):
                val = event.x.get(name)
                if val is None:
                    x[r, c] = np.nan
                elif name in self.cat_to_id and val in self.cat_to_id[name]:
                    x[r, c] = self.cat_to_id[name][val]
                elif isinstance(val, bool):
                    x[r, c] = 1.0 if val else 0.0
                elif isinstance(val, (int, float)):
                    x[r, c] = float(val)
                else:
                    x[r, c] = np.nan
        y = np.array([int(bool(e.y)) for e in events], dtype=np.int64)
        return x, y

    def decode(self, x_arr: np.ndarray) -> list[dict[str, Any]]:
        if not self.names:
            return [{} for _ in range(len(x_arr))]
        out: list[dict[str, Any]] = []
        for row in x_arr:
            item: dict[str, Any] = {}
            for c, name in enumerate(self.names):
                val = float(row[c])
                if math.isnan(val):
                    continue
                if name in self.id_to_cat and val in self.id_to_cat[name]:
                    item[name] = self.id_to_cat[name][val]
                elif name in self.bool_cols:
                    item[name] = bool(val >= 0.5)
                else:
                    item[name] = val
            out.append(item)
        return out


class BinaryLearner:
    """Adapter from continual_bench binary models to the retraining Matrix Learner protocol."""

    def __init__(self, model_factory, codec: FeatureCodec, replay_window: int):
        self.model_factory = model_factory
        self.codec = codec
        self.replay_window = replay_window
        self.pending: deque[tuple[int, int, dict[str, Any], bool]] = deque()
        self.released: deque[tuple[int, int, dict[str, Any], bool]] = deque(maxlen=replay_window)
        self.training_records: list[dict[str, Any]] = []

    def fit(self, index: int, x: np.ndarray, y: np.ndarray, *context: np.ndarray) -> Any:
        rows = self.codec.decode(x)
        if len(context) != 3:
            raise ValueError("binary retraining requires availability, sequence and decision cutoff")
        available_at, sequences, cutoff_array = context
        cutoff = int(cutoff_array[0])
        self.pending.extend(
            (int(seq), int(available), feat, bool(target))
            for seq, available, feat, target in zip(sequences, available_at, rows, y, strict=True)
        )
        newly_released = 0
        while self.pending and self.pending[0][1] <= cutoff:
            self.released.append(self.pending.popleft())
            newly_released += 1
        model = self.model_factory()
        for _, _, feat, target in self.released:
            model.learn(feat, target)
        seqs = [seq for seq, _, _, _ in self.released]
        released_at = [available for _, available, _, _ in self.released]
        self.training_records.append(
            {
                "batch": index,
                "cutoff_timestamp": cutoff,
                "train_samples": len(self.released),
                "newly_released_samples": newly_released,
                "pending_samples": len(self.pending),
                "train_seq_range": [min(seqs), max(seqs) + 1] if seqs else None,
                "train_available_at_range": [min(released_at), max(released_at)] if released_at else None,
                "train_sequences_sha256": hashlib.sha256(
                    ",".join(str(seq) for seq in seqs).encode()
                ).hexdigest(),
                "replay_window": self.replay_window,
                "replay_selection": "last released rows of the observed prefix, each row admitted once",
                "available": bool(self.released),
                "reason": None if self.released else "no_released_training_labels",
            }
        )
        return model

    def losses(self, state: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows = self.codec.decode(x)
        preds = np.array([state.predict(feat) >= 0.5 for feat in rows], dtype=np.bool_)
        return preds != np.asarray(y, dtype=np.bool_)


def partition_blocks(
    events: list[Event],
    batch_size: int,
    *,
    max_blocks: int = MAX_RETRAINING_BLOCKS,
) -> list[list[Event]]:
    """Split events into at least 4 and at most ``max_blocks`` contiguous blocks."""
    n = len(events)
    if n < 4:
        return []
    size = max(int(batch_size), math.ceil(n / max_blocks))
    if n // size < 4:
        size = max(1, n // 4)
    blocks = [events[i : i + size] for i in range(0, n, size)]
    if len(blocks) >= 5 and len(blocks[-1]) < max(2, size // 4):
        blocks[-2].extend(blocks[-1])
        blocks.pop()
    return blocks if len(blocks) >= 4 else []


def compute_binary_retraining(
    events: list[Event],
    model_factory,
    *,
    batch_size: int,
    replay_window: int,
    seed: int = 0,
    upf_alpha: float = 1.0,
    n_alphas: int = 10,
    mc_samples: int = 100,
    delta: float = 0.95,
    exclude_seqs: frozenset[int] | set[int] = (),
) -> tuple[list[list[Event]], PerformanceMatrix, list[Any], dict[str, Any], dict[str, Any]] | None:
    """Build the binary performance matrix and run CARA/UPF/Oracle analysis."""
    if exclude_seqs:
        events = [event for event in events if event.seq not in exclude_seqs]
    for previous, event in zip(events, events[1:]):
        if event.timestamp <= previous.timestamp or event.available_at < previous.available_at:
            raise ValueError("binary retraining requires ordered timestamps and label availability")
    if any(event.available_at < event.timestamp for event in events):
        raise ValueError("a label cannot be available before its event timestamp")
    blocks = partition_blocks(events, batch_size)
    if not blocks:
        return None
    codec = FeatureCodec(events)
    learner = BinaryLearner(model_factory, codec, replay_window)
    available_at = [np.asarray([event.available_at for event in block], dtype=np.int64) for block in blocks]
    cutoffs = [block[-1].timestamp for block in blocks]
    encoded = [
        (
            *codec.encode(block),
            available_at[index],
            np.asarray([e.seq for e in block]),
            np.asarray([cutoffs[index]]),
        )
        for index, block in enumerate(blocks)
    ]
    offline = max(2, min(len(blocks) - 2, len(blocks) // 3))
    matrix, states = build_matrix(
        encoded,
        learner,
        offline=offline,
        seed=seed,
        loss_dtype=np.bool_,
        label_available_at=available_at,
        decision_cutoffs=cutoffs,
    )
    analysis = analyze(
        matrix,
        rules=DEFAULT_RULES,
        n_alphas=n_alphas,
        focus_alpha=upf_alpha,
        delta=delta,
        samples=mc_samples,
        seed=seed,
    )
    analysis["temporal_contract"] = matrix.to_dict()["temporal_contract"]
    analysis["temporal_contract"].update(
        {
            "clock": "source Event.timestamp and available_at, in the source manifest clock units",
            "eligibility": "available_at <= end-of-block timestamp",
            "deployment": (
                "after the decision block; predictions in the next block use its selected candidate"
            ),
            "replay": "at most replay_window most recently released non-reserved rows",
            "staleness_reference": (
                "sampled original model-index block, restricted to labels available at the decision cutoff; "
                "this reference population is distinct from the candidate replay training window"
            ),
            "empty_label_information": "unavailable, never substituted by zero error",
            "oracle": (
                "future scoring losses may choose the schedule; candidate fits retain the same causal cuts"
            ),
        }
    )
    focus_alpha = analysis["setup"]["focus_alpha"]
    cara_curve = (analysis["rules"].get("cara") or {}).get("curve") or []
    focus_pt = (
        min(cara_curve, key=lambda c: abs(c["alpha"] - focus_alpha))
        if cara_curve and focus_alpha is not None
        else (cara_curve[0] if cara_curve else {})
    )
    retrains_set = set(focus_pt.get("retrain_points") or [])
    deployed_idx, served_list = matrix.initial, []
    for j in range(matrix.offline, matrix.batches):
        served_list.append(deployed_idx)
        if j in retrains_set:
            deployed_idx = j
    analysis["deployed"] = {
        "policy": "cara",
        "alpha": focus_pt.get("alpha", focus_alpha),
        "retrain_points": list(focus_pt.get("retrain_points") or []),
        "served_by": served_list,
        "parameters": focus_pt.get("parameters") or {},
        "oracle_retrain_points": list(analysis.get("oracle_schedule_at_focus") or []),
        "model_rows": [record["train_samples"] for record in matrix.training_records],
        "source_block_rows": [int(r) for r in matrix.rows],
    }
    ref_metrics = retraining_reference_metrics(analysis, matrix, focus_alpha=upf_alpha)
    return blocks, matrix, states, analysis, ref_metrics


@dataclass
class BatchRetrainingState:
    evaluation_finished: bool = True
    evaluation_duration_seconds: float = 0.0
    report: dict | None = None
    model: Any = None
    observer: Any = None


def _simulation_of(policy, matrix: PerformanceMatrix) -> SimResult:
    deployed, served, errors = matrix.initial, [], []
    retrains = set(policy.retrains)
    for j in range(matrix.offline, matrix.batches):
        served.append(deployed)
        errors.append(float(matrix.err[deployed, j]))
        if j in retrains:
            deployed = j
    return SimResult(
        policy.kind,
        policy.alpha,
        list(policy.retrains),
        served,
        errors,
        policy.rule.parameters(),
        list(getattr(policy.rule, "forecasts", [])),
    )


def run_batch_retraining(run: BinaryRun) -> Outcome | Progress:
    """Execute a binary benchmark driven by a batch policy (CARA, UPF, or Oracle)."""
    if run.snapshot and getattr(run.snapshot.get("state"), "evaluation_finished", False):
        cached = run.snapshot["state"]
        return Outcome(cached.report, cached, cached.model, run.versions, cached.observer)

    config, settings = run.config, run.settings
    events = list(run.events())
    if run.stop_after is not None and run.stop_after < len(events):
        return Progress(run.stop_after)
    if len(events) < 4:
        raise ValueError("batch policies require at least 4 events")

    continual = continual_evaluation(
        run.source,
        run.model_factory,
        settings,
        events=events,
        warmup=config.warmup,
    )
    reserved = getattr(continual, "reserved_seqs", frozenset())

    from continual_bench.retraining.pricing import resolve_retrain_alpha

    alpha_info = resolve_retrain_alpha(
        config.policy.params,
        default_alpha=float(settings.get("upf_alpha", 1.0)),
        tracking=settings,
    )
    upf_alpha = float(alpha_info["alpha"])
    computed = compute_binary_retraining(
        events,
        run.model_factory,
        batch_size=config.batch_size,
        replay_window=config.replay_window,
        seed=config.dataset.seed,
        upf_alpha=upf_alpha,
        exclude_seqs=reserved,
    )
    if computed is None:
        raise ValueError("dataset too small to build a batch retraining performance matrix")
    blocks, matrix, states, analysis, ref_metrics = computed

    policy = run.policy_factory()
    policy.prepare(matrix)

    observer = OnlineObservability(
        settings,
        continual,
        current_rss_bytes,
        warmup=sum(len(b) for b in blocks[: matrix.offline]),
    )

    has_source_probes = hasattr(run.source, "probes")
    probes = run.source.probes() if has_source_probes else {}
    seen_concepts: set[int] = set()
    best_probe: dict[int, float] = {}
    retention_probes: list[dict[str, Any]] = []

    def _record_probes(observed_seq: int) -> None:
        if has_source_probes:
            active_probes, active_concepts = probes, sorted(seen_concepts)
        elif observer.continual is not None:
            active_probes = observer.continual.probes
            task_ends = observer.continual.task_ends
            active_concepts = [
                idx for idx in range(len(task_ends)) if observed_seq + 1 > (task_ends[idx - 1] if idx else 0)
            ]
        else:
            return
        frozen = copy.deepcopy(deployed_model)
        for concept in active_concepts:
            samples = active_probes.get(concept, [])
            if not samples:
                continue
            accuracy = sum((frozen.predict(dict(x)) >= 0.5) == y for x, y in samples) / len(samples)
            best_probe[concept] = max(best_probe.get(concept, 0.0), accuracy)
            retention_probes.append(
                {
                    "observed_seq": observed_seq,
                    "concept": concept,
                    "accuracy": accuracy,
                    "drop_from_best_probe": best_probe[concept] - accuracy,
                }
            )

    total = Metrics()
    windows: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    audit = hashlib.sha256(b"continual-bench-v1").hexdigest()

    # No model fitted on the first block may serve predictions inside that block.
    deployed_model = run.model_factory()
    labeled = 0
    predict_seconds = 0.0
    train_seconds = float(np.nansum(matrix.fit_seconds))

    ev_idx = 0
    pending_labels = deque()

    def _release_labels(cutoff: int) -> None:
        nonlocal labeled
        while pending_labels and pending_labels[0][0].available_at <= cutoff:
            event, probability, trace, evaluated, reserved_event = pending_labels.popleft()
            labeled += 1
            observer.after_label(
                event,
                probability,
                trace,
                cutoff,
                deployed_model,
                evaluated,
                near_future_evaluated=True if reserved_event else None,
            )

    def _advance_reserved(limit_seq: int) -> None:
        nonlocal ev_idx
        while ev_idx < len(events) and events[ev_idx].seq <= limit_seq:
            ev = events[ev_idx]
            ev_idx += 1
            if ev.seq in reserved:
                r_trace = observer.before_prediction(ev, deployed_model, labeled, 0.0)
                pending_labels.append((ev, 0.5, r_trace, False, True))

    for t, block in enumerate(blocks):
        current = Metrics()
        for event in block:
            _advance_reserved(event.seq - 1)
            if ev_idx < len(events) and events[ev_idx].seq == event.seq:
                ev_idx += 1
            p_start = time.perf_counter()
            prob = float(deployed_model.predict(dict(event.x)))
            p_sec = time.perf_counter() - p_start
            predict_seconds += p_sec
            trace = observer.before_prediction(event, deployed_model, labeled, p_sec)
            audit = hashlib.sha256(f"{audit}:{event.seq}:{prob:.12g}".encode()).hexdigest()
            is_online = t >= matrix.offline
            if is_online:
                total.add(event.y, prob)
                current.add(event.y, prob)
            pending_labels.append((event, prob, trace, is_online, False))
            if event.concept is not None:
                seen_concepts.add(event.concept)

        end_seq = block[-1].seq
        _release_labels(block[-1].timestamp)
        if t < matrix.offline:
            # During offline phase, advance to the latest offline block model so at t == offline-1
            # we hold matrix.initial.
            if t < len(states):
                deployed_model = copy.deepcopy(states[t])
            if t == matrix.offline - 1:
                version = run.versions.record(
                    deployed_model,
                    {
                        "action": "initial_fit",
                        "seq_range": [0, end_seq],
                        **matrix.training_records[t],
                    },
                )
                observer.recorded_model_version(version)
            if current.n:
                windows.append({"end_seq": end_seq, **current.values()})
            _record_probes(end_seq)
            continue

        window = {"end_seq": end_seq, **current.values()}
        windows.append(window)
        if run.progress is not None:
            run.progress(end_seq, window)

        decision = policy.decide(
            DecisionContext(labeled=labeled, window_size=len(block), drift=False, batch=t)
        )
        if decision.action == "reset_retrain" and t < len(states):
            previous = observer.before_update(deployed_model)
            deployed_model = copy.deepcopy(states[t])
            fit_sec = float(matrix.fit_seconds[t]) if math.isfinite(matrix.fit_seconds[t]) else 0.0
            training = matrix.training_records[t]
            update_record = {
                "seq": end_seq,
                "batch": t,
                "action": "reset_retrain",
                "reason": decision.reason,
                "samples": training["train_samples"],
                **training,
                "seconds": fit_sec,
                "promoted": True,
            }
            updates.append(update_record)
            candidates.append(update_record)
            observer.after_update(previous, update_record)
            observer.record_learning(
                end_seq,
                fit_sec,
                training["train_samples"],
                initial=False,
            )
            ver = run.versions.record(
                deployed_model,
                {
                    "action": "reset_retrain",
                    "seq_range": [int(block[0].seq), int(end_seq)],
                    "batch": t,
                    "seq": end_seq,
                    **training,
                },
            )
            observer.recorded_model_version(ver)
        _record_probes(end_seq)

    if events:
        _advance_reserved(events[-1].seq)
        _release_labels(max(event.available_at for event in events))
    observer.sample_memory()
    sim_result = _simulation_of(policy, matrix)
    oracle_sim = simulate(Oracle(), matrix, policy.alpha)
    rule_key = policy.kind if policy.kind in analysis["rules"] else "oracle"
    dep_metrics = deployed_metrics(analysis, rule_key, sim_result, oracle_sim)
    analysis["deployed"] = {
        "policy": policy.kind,
        "alpha": policy.alpha,
        "retrain_points": list(policy.retrains),
        "served_by": list(sim_result.deployed),
        "parameters": json_parameters(sim_result.parameters),
        "oracle_retrain_points": list(oracle_sim.retrains),
        "model_rows": [record["train_samples"] for record in matrix.training_records],
        "source_block_rows": [int(r) for r in matrix.rows],
        "is_retrospective_reference": bool(getattr(policy, "is_retrospective_reference", False)),
        "future_information_used": getattr(policy, "future_information_used", None),
    }
    analysis["fargate_pricing"] = alpha_info
    analysis["oracle_reference"] = {
        "is_retrospective_reference": bool(getattr(policy, "is_retrospective_reference", False)),
        "deployable_online": not bool(getattr(policy, "is_retrospective_reference", False)),
        "future_information_used": getattr(policy, "future_information_used", None),
    }

    summary = total.values()
    summary["train_seconds"] = train_seconds
    summary["prediction_seconds"] = predict_seconds
    summary["update_count"] = len(updates)
    summary["retrain_count"] = len(updates)
    summary["candidate_training_samples"] = sum(record["train_samples"] for record in matrix.training_records)
    summary["selected_retrain_samples"] = sum(update["train_samples"] for update in updates)
    summary.update(dep_metrics)
    summary.update(ref_metrics)

    compute_seconds = time.monotonic() - run.started
    report = {
        "schema_version": 1,
        "observed": len(events),
        "labeled": labeled,
        "metrics": summary,
        "windows": windows,
        "alarms": [],
        "updates": updates,
        "candidates": candidates,
        "promotion": {
            "gate": None,
            "trained": len(candidates),
            "promoted": len(candidates),
            "rejected": 0,
            "training_seconds": train_seconds,
        },
        "detector_signal": "batch_performance_matrix",
        "training": {
            "budget": config.training_budget,
            "used": sum(u["samples"] for u in updates),
            "blocked_decisions": 0,
            "excludes_warmup": True,
            "candidate_build_training_samples": summary["candidate_training_samples"],
            "candidate_build_training_seconds": train_seconds,
            "selected_retrain_samples": summary["selected_retrain_samples"],
            "candidate_training": matrix.training_records,
        },
        "retention_probes": retention_probes,
        "prediction_digest": audit,
        "compute_seconds": compute_seconds,
        "retraining": analysis,
    }
    state = BatchRetrainingState(
        evaluation_finished=True,
        evaluation_duration_seconds=compute_seconds,
        report=report,
        model=deployed_model,
        observer=observer,
    )
    if run.checkpoint_due is not None:
        run.versions.flush()
        run.checkpoint(state, force=True)
    return Outcome(report, state, deployed_model, run.versions, observer)
