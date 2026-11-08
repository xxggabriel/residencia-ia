"""River generators: abrupt (SEA/Agrawal/STAGGER), moving and regression drift streams.

Each stream is bounded, deterministic per seed and versioned by its parameters.
Synthetic timestamps are simulated (`epoch + seq`), never real event time.
"""

import hashlib
import json
from itertools import islice

from river.datasets import synth

from continual_bench.core.contracts import Event, TaskSample

from .base import stable_identity, validate_bounds


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


class MovingSyntheticSource:
    """River-native continuously moving binary concepts; no discrete truth points."""

    def __init__(self, name, samples, seed, label_delay, epoch, *, n_features=10,
                 n_drift_features=2, mag_change=0.001, noise_percentage=0.05,
                 sigma=0.1, n_centroids=20, n_drift_centroids=10, change_speed=0.001):
        validate_bounds(samples, seed, label_delay, epoch)
        if name not in {"hyperplane", "rbf_drift"}:
            raise ValueError("unsupported moving synthetic source")
        if not 2 <= n_features <= 1000:
            raise ValueError("n_features must be between 2 and 1000")
        if name == "hyperplane":
            if not 1 <= n_drift_features <= n_features or not 0 < mag_change <= 1:
                raise ValueError("hyperplane requires moving features and positive mag_change")
            params = dict(seed=seed, n_features=n_features, n_drift_features=n_drift_features,
                          mag_change=mag_change, noise_percentage=noise_percentage, sigma=sigma)
            constructor = synth.Hyperplane
        else:
            if not 1 <= n_drift_centroids <= n_centroids <= 10_000 or not 0 < change_speed <= 1:
                raise ValueError("rbf_drift requires moving centroids and positive change_speed")
            params = dict(seed_model=seed, seed_sample=seed, n_classes=2, n_features=n_features,
                          n_centroids=n_centroids, n_drift_centroids=n_drift_centroids,
                          change_speed=change_speed)
            constructor = synth.RandomRBFDrift
        # Instantiate now so native parameter errors fail before a run is admitted.
        constructor(**params)
        self.constructor, self.params = constructor, params
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.version = ""
        self.version = stable_identity({k: v for k, v in self.manifest().items() if k != "version"})

    def events(self):
        for seq, (x, y) in enumerate(islice(self.constructor(**self.params), self.samples)):
            if y not in (0, 1):
                raise ValueError("binary generator emitted a non-binary target")
            timestamp = self.epoch + seq
            yield Event(stable_identity([self.version, seq]), seq, timestamp, timestamp + self.label_delay,
                        {str(k): v for k, v in x.items()}, bool(y), None)

    def manifest(self):
        return {"adapter": "river-moving-v2", "name": self.name, "version": self.version,
                "river_version": "0.26.1",
                "samples": self.samples, "seed": self.seed, "params": self.params,
                "task": "binary", "drift_kind": "continuously_moving_concept",
                "changepoints": None, "timestamp_kind": "simulated_ordinal_seconds",
                "epoch": self.epoch, "label_delay_seconds": self.label_delay,
                "license": "BSD-3-Clause (generator implementation)",
                "source": "https://github.com/online-ml/river/tree/0.26.1/river/datasets/synth"}


class FriedmanDriftSource:
    """Native regression samples; binary runner rejection prevents target collapse."""

    def __init__(self, samples, seed, label_delay=0, epoch=0, *, name="friedman_drift",
                 drift_type="gra", position=None, transition_window=0):
        validate_bounds(samples, seed, label_delay, epoch)
        count = 3 if drift_type == "lea" else 2
        position = tuple(position or (samples * i // (count + 1) for i in range(1, count + 1)))
        if (len(position) != count or tuple(sorted(set(position))) != position
                or position[0] < 1 or position[-1] >= samples):
            raise ValueError("drift positions must be strictly increasing inside the sample range")
        if transition_window < 0 or (drift_type == "gsg" and
                                   any(b - a <= transition_window for a, b in zip(position, position[1:]))):
            raise ValueError("transition windows must be nonnegative and nonoverlapping")
        self.params = dict(seed=seed, drift_type=drift_type, position=position,
                           transition_window=transition_window)
        synth.FriedmanDrift(**self.params)
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.version = ""
        self.version = stable_identity({k: v for k, v in self.manifest().items() if k != "version"})

    def samples_iter(self):
        for seq, (x, y) in enumerate(islice(synth.FriedmanDrift(**self.params), self.samples)):
            timestamp = self.epoch + seq
            identity = stable_identity([self.version, seq])
            yield TaskSample(identity, seq, timestamp, timestamp + self.label_delay,
                             {str(k): v for k, v in x.items()}, float(y))

    def events(self):
        raise ValueError("FriedmanDrift is regression; use samples_iter() and a regression evaluator")

    def manifest(self):
        # River uses a zero-based counter: the new regime starts at seq=position.
        return {"adapter": "river-friedman-v1", "name": self.name, "version": self.version,
                "river_version": "0.26.1",
                "task": "regression", "samples": self.samples, "params": self.params,
                "changepoints": list(self.params["position"]),
                "drift_kind": "concept", "epoch": self.epoch,
                "timestamp_kind": "simulated_ordinal_seconds", "label_delay_seconds": self.label_delay,
                "license": "BSD-3-Clause (generator implementation)",
                "source": "https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/friedman.py"}
