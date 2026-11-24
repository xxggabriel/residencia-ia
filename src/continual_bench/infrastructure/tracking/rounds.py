"""One child run per completed fit, shared by the homelab and AWS publishers."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from .backends import _point, as_backend
from .reconcile import missing_metric_points

EVIDENCE_TAGS = ("cb-fit-evidence-sha256", "cb-fit-observations-sha256", "cb-fit-checkpoint-uri")


def publish_training_rounds(
    client,
    experiment_id,
    parent_run_id,
    report,
    *,
    now_ms=0,
    checkpoint_base_uri=None,
    include_artifact_uri=False,
):
    """Publish fits as child runs keyed by ``cb-fit-id``; replay is idempotent.

    Each child needs at most create + one log-batch (metrics, params and tags
    together) + terminate, and independent fits are published concurrently on
    remote stores. A child already tracking different evidence is rejected.
    Payloads are validated lazily. An invalid later fit can leave earlier
    children finished; idempotent retry reuses those children without duplicates.
    """
    from continual_bench.infrastructure.training_tracking import training_round_mlflow

    if not report.get("training_history"):
        return []
    backend = as_backend(client)
    existing = {}
    for run in backend.search_runs(experiment_id, f"tags.`mlflow.parentRunId` = '{parent_run_id}'"):
        if fit_id := run.tags.get("cb-fit-id"):
            if fit_id in existing:
                raise ValueError(f"duplicate MLflow child runs for {parent_run_id}/{fit_id}")
            existing[fit_id] = run
    fits = training_round_mlflow(parent_run_id, report, now_ms)

    def publish(fit):
        tags, params, metrics = fit
        fit_id = tags["cb-fit-id"]
        checkpoint_uri = None
        if checkpoint_base_uri and (checkpoint := params.get("checkpoint")):
            checkpoint_uri = f"{checkpoint_base_uri.rstrip('/')}/{checkpoint}"
            tags["cb-fit-checkpoint-uri"] = checkpoint_uri
        run = existing.get(fit_id)
        created = run is None
        if created:
            run = backend.create_run(experiment_id, tags, tags.get("mlflow.runName"))
        extending = False
        for key in EVIDENCE_TAGS:
            expected = tags.get(key)
            if expected is None:
                continue
            remote = run.tags.get(key)
            if not created and remote is not None and remote != expected:
                raise ValueError(f"MLflow child {fit_id} already tracks different training evidence")
            extending |= not created and remote is None
        if run.status != "FINISHED" or extending:
            missing = missing_metric_points(
                backend,
                run.run_id,
                [(p["key"], p["step"], p["value"]) for p in metrics],
                known_keys=set() if created else set(run.metric_keys),
                max_workers=1,
            )
            stamps = {(p["key"], p["step"]): p.get("timestamp", now_ms) for p in metrics}
            backend.log_batch(
                run.run_id,
                metrics=[
                    _point({"key": k, "value": v, "step": s, "timestamp": stamps.get((k, s), now_ms)})
                    for k, s, v in missing
                ],
                params=params,
                tags=tags,
            )
            backend.set_terminated(run.run_id, "FINISHED")
        receipt = {"fit_id": fit_id, "run_id": run.run_id, "metric_points": len(metrics)}
        if checkpoint_uri:
            receipt["checkpoint_uri"] = checkpoint_uri
        if include_artifact_uri and report.get("tracking_observations"):
            receipt["artifact_uri"] = run.artifact_uri
        return receipt

    workers = min(getattr(backend, "concurrency", 1) or 1, len(report["training_history"]))
    if workers > 1:
        with ThreadPoolExecutor(workers, thread_name_prefix="mlflow-fits") as pool:
            # Executor.map eagerly consumes its input on the supported Python
            # version. Bound submitted payloads to the publication slots, plus
            # the generator's temporary per-fit construction workspace.
            pending = deque(pool.submit(publish, fit) for fit in islice(fits, workers))
            receipts = []
            while pending:
                receipts.append(pending.popleft().result())
                for fit in islice(fits, 1):
                    pending.append(pool.submit(publish, fit))
                    del fit
            return receipts
    receipts = []
    for fit in fits:
        receipts.append(publish(fit))
        del fit
    return receipts
