"""Independent numerical recomputation tolerances must not weaken remote evidence checks."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from continual_bench.domain.metrics import Metrics


def auditor():
    path = Path(__file__).parents[1] / "scripts/verify_tracking_metrics.py"
    spec = importlib.util.spec_from_file_location("verify_tracking_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mae_recomputation_accepts_one_ulp_accumulation_difference_but_rejects_wrong_value():
    module = auditor()
    errors = [0.1] * 7
    sequential = 0.0
    for error in errors:
        sequential += error
    published = sum(errors) / len(errors)
    assert published != sequential / len(errors)
    assert abs(published - sequential / len(errors)) == math.ulp(published)
    metrics = Metrics()
    events, rows = [], []
    for seq in range(len(errors)):
        metrics.add(False, 0.1)
        event = {"seq": seq, "event_id": str(seq), "y": False, "timestamp": seq, "available_at": seq}
        events.append(event)
        rows.append({"seq": seq, "event_id": str(seq), "label": False, "probability": 0.1,
                     "timestamp": seq, "available_at": seq, "released_at": seq})
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    values = metrics.values()
    report = {"config": {"warmup": 0}, "observed": 7, "metrics": values,
              "windows": [{"end_seq": 6, **values}]}
    tracking = {
        "metrics": {**values, "mae_probability": published},
        "series": {key: [{"step": 6, "value": value}] for key, value in values.items()},
        "prediction_ledger": {"sha256": hashlib.sha256(raw).hexdigest(), "rows": 7},
    }
    assert module.check_predictions(report, tracking, raw, events, native=True) == (7, len(values))
    tracking["metrics"]["mae_probability"] = published + 1e-12
    with pytest.raises(ValueError, match="published probability MAE differs"):
        module.check_predictions(report, tracking, raw, events, native=True)


def test_remote_history_still_requires_exact_mae_value_despite_recomputation_tolerance():
    module = auditor()
    key = "tracking_v1/summary/mae_probability"
    tracking = {"version": "residency-metrics-v1", "metrics": {"mae_probability": 0.1},
                "series": {}, "settings": {}, "availability": {}}
    remote = math.nextafter(0.1, 1)
    run = SimpleNamespace(info=SimpleNamespace(run_id="fixture"), data=SimpleNamespace(metrics={key: remote}))

    class Client:
        def get_metric_history(self, run_id, name):
            return [SimpleNamespace(step=0, value=remote)]

    with pytest.raises(ValueError, match="remote history differs"):
        module.check_histories(Client(), run, tracking)
