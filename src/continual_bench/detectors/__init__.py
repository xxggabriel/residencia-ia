"""Drift detector catalog. Detectors only raise alarms; policies decide updates."""

from continual_bench.core.registry import Registry, Spec

RIVER_DETECTORS = ("adwin", "ddm", "eddm", "hddm_a", "hddm_w", "page_hinkley", "kswin")
SKMULTIFLOW_DETECTORS = ("adwin", "ddm", "kswin")

DETECTORS: Registry[Spec] = Registry("detector")
DETECTORS.add(Spec("none", "continual_bench.detectors.river:NullDetector"))
for _name in RIVER_DETECTORS:
    DETECTORS.add(
        Spec(
            _name,
            "continual_bench.detectors.river:RiverDetector",
            {"name": _name},
            {"signal": "released_label_binary_prediction_error", "backend": "river"},
        )
    )
for _name in SKMULTIFLOW_DETECTORS:
    DETECTORS.add(
        Spec(
            f"skmultiflow_{_name}",
            "continual_bench.detectors.skmultiflow:SkmultiflowDetector",
            {"name": _name},
            {"signal": "released_label_binary_prediction_error", "backend": "skmultiflow"},
        )
    )
DETECTORS.add(
    Spec(
        "fhddm",
        "continual_bench.detectors.fhddm:FHDDM",
        {},
        {"signal": "released_label_binary_prediction_error"},
    )
)


def make_detector(name: str, *, backend: str = "river", **params):
    """Instantiate a registered drift detector by name and backend (``'river'`` or ``'skmultiflow'``)."""
    if backend == "river":
        return DETECTORS.get(name).build(**params)
    if backend == "skmultiflow":
        if name in SKMULTIFLOW_DETECTORS:
            return DETECTORS.get(f"skmultiflow_{name}").build(**params)
        if name in ("none", "fhddm") or name.startswith("skmultiflow_"):
            return DETECTORS.get(name).build(**params)
        raise ValueError(
            f"detector {name!r} is not supported for backend='skmultiflow'; "
            f"available={sorted((*SKMULTIFLOW_DETECTORS, 'fhddm', 'none'))}"
        )
    raise ValueError(f"unknown detector backend: {backend!r}")
