"""Drift detector catalog. Detectors only raise alarms; policies decide updates."""

from continual_bench.core.registry import Registry, Spec

RIVER_DETECTORS = ("adwin", "ddm", "eddm", "hddm_a", "hddm_w", "page_hinkley", "kswin")

DETECTORS: Registry[Spec] = Registry("detector")
DETECTORS.add(Spec("none", "continual_bench.detectors.river:NullDetector"))
for _name in RIVER_DETECTORS:
    DETECTORS.add(Spec(_name, "continual_bench.detectors.river:RiverDetector", {"name": _name},
                       {"signal": "released_label_binary_prediction_error"}))
