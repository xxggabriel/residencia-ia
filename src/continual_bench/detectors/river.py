"""Drift detectors over the released-label binary prediction error."""

from river import drift


class NullDetector:
    def update(self, error: int) -> bool:
        return False


class RiverDetector:
    signal = "released_label_binary_prediction_error"

    def __init__(self, name: str, **params):
        constructors = {
            "adwin": drift.ADWIN, "ddm": drift.binary.DDM, "eddm": drift.binary.EDDM,
            "hddm_a": drift.binary.HDDMA, "hddm_w": drift.binary.HDDMW,
            "page_hinkley": drift.PageHinkley, "kswin": drift.KSWIN,
        }
        if name not in constructors:
            raise ValueError(f"unknown River detector: {name}")
        if name == "kswin":
            params.setdefault("seed", 42)
            window = params.get("window_size", 100)
            statistic = params.get("stat_size", 30)
            if (not isinstance(window, int) or not isinstance(statistic, int)
                    or not 0 < statistic <= window // 2):
                raise ValueError("KSWIN requires 0 < stat_size <= window_size / 2")
            if not 0 < params.get("alpha", 0.005) < 1:
                raise ValueError("KSWIN alpha must be in (0,1)")
        self.inner = constructors[name](**params)

    def update(self, error: int) -> bool:
        if error not in (0, 1):
            raise ValueError("detector expects binary prediction error, not raw features")
        self.inner.update(error)
        return bool(self.inner.drift_detected)
