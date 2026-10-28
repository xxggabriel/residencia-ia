import hashlib
import json
import numbers
from dataclasses import asdict

from river import drift, linear_model, optim, preprocessing
from river.datasets import synth

from continual_bench.domain.contracts import Event


class RiverModel:
    """Predict is read-only. Preprocessor is learned strictly inside learn()."""

    def __init__(self, learning_rate: float = 0.05):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        self.scaler = preprocessing.StandardScaler()
        self.classifier = linear_model.LogisticRegression(optimizer=optim.SGD(learning_rate))

    def encode(self, x: dict, *, learn: bool) -> dict:
        numeric = {k: float(v) for k, v in x.items() if isinstance(v, numbers.Number)}
        categorical = {f"{k}={v}": 1.0 for k, v in x.items() if not isinstance(v, numbers.Number)}
        if learn:
            self.scaler.learn_one(numeric)
        return {**self.scaler.transform_one(numeric), **categorical}

    def predict(self, x: dict) -> float:
        return float(self.classifier.predict_proba_one(self.encode(x, learn=False)).get(True, 0.5))

    def learn(self, x: dict, y: bool) -> None:
        self.classifier.learn_one(self.encode(x, learn=True), bool(y))


class NullDetector:
    def update(self, error: int) -> bool:
        return False


class RiverDetector:
    def __init__(self, name: str, **params):
        self.inner = {"adwin": drift.ADWIN, "ddm": drift.binary.DDM}[name](**params)

    def update(self, error: int) -> bool:
        self.inner.update(error)
        return bool(self.inner.drift_detected)


class SyntheticSource:
    """Three piecewise stationary concepts; independent frozen reference probe seed."""

    def __init__(self, name, samples, seed, label_delay, epoch):
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.functions = [0, 1, 0] if name == "agrawal" else [0, 1, 2]
        self.boundaries = [samples // 3, 2 * samples // 3]
        raw = json.dumps(["river-0.26.1-v1", name, samples, seed, label_delay, epoch])
        self.version = hashlib.sha256(raw.encode()).hexdigest()

    def generator(self, function, seed):
        constructor = {"sea": synth.SEA, "agrawal": synth.Agrawal, "stagger": synth.STAGGER}[self.name]
        keyword = "variant" if self.name == "sea" else "classification_function"
        return iter(constructor(seed=seed, **{keyword: function}))

    def events(self):
        generators = [self.generator(fn, self.seed + i) for i, fn in enumerate(self.functions)]
        for seq in range(self.samples):
            segment = sum(seq >= b for b in self.boundaries)
            x, y = next(generators[segment])
            # River synth can use integer keys; JSON/event transport contract uses strings.
            x = {str(k): v for k, v in x.items()}
            stamp = self.epoch + seq
            event_id = hashlib.sha256(f"{self.version}:{seq}".encode()).hexdigest()
            yield Event(event_id, seq, stamp, stamp + self.label_delay, x, bool(y), self.functions[segment])

    def probes(self, count=64):
        probes = {}
        for function in set(self.functions):
            generator = self.generator(function, self.seed + 1_000_000 + function)
            probes[function] = [
                ({str(k): v for k, v in x.items()}, bool(y))
                for x, y in (next(generator) for _ in range(count))
            ]
        return probes

    def manifest(self):
        return {
            "name": self.name,
            "version": self.version,
            "samples": self.samples,
            "seed": self.seed,
            "license": "BSD-3-Clause (River generator implementation)",
            "source": "https://github.com/online-ml/river/tree/0.26.1/river/datasets/synth",
            "timestamp_kind": "synthetic_epoch_plus_sequence_seconds",
            "changepoints": self.boundaries,
            "functions": self.functions,
            "label_delay_seconds": self.label_delay,
            "config": {
                "name": self.name,
                "samples": self.samples,
                "seed": self.seed,
                "label_delay": self.label_delay,
                "epoch": self.epoch,
            },
        }


def event_json(event):
    return json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))
