"""Single explicit entry point for all admitted Semana 09 dataset implementations."""

import os
from dataclasses import asdict

from .real_sources import RealSource
from .river import SyntheticSource
from .task_sources import FriedmanDriftSource, MovingSyntheticSource, TaskSample

DATASETS = (
    "sea", "agrawal", "stagger", "hyperplane", "rbf_drift", "friedman_drift",
    "electricity", "airlines", "insects", "gas", "split_mnist", "permuted_mnist",
    "split_cifar10", "split_cifar100", "core50", "stream51", "clear10", "clear100",
)


class EventSamples:
    def __init__(self, source):
        self.source = source

    def samples_iter(self):
        for event in self.source.events():
            yield TaskSample(event.event_id, event.seq, event.timestamp, event.available_at,
                             event.x, "1" if event.y else "0", task_id=event.concept)

    def manifest(self):
        return {**self.source.manifest(), "task": "multiclass", "labels": ["0", "1"],
                "target_conversion": "explicit binary boolean to class strings, no information loss"}


def load_dataset(name, cache, *, examples=100, seed=42, download=False, real_directory=None):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name}")
    if type(examples) is not int or not 10 <= examples <= 10_000:
        raise ValueError("examples must be an integer in [10,10000]")
    arguments = dict(samples=examples, seed=seed, label_delay=3, epoch=1_735_689_600)
    if name in {"sea", "agrawal", "stagger"}:
        return EventSamples(SyntheticSource(name=name, **arguments))
    if name in {"hyperplane", "rbf_drift"}:
        return EventSamples(MovingSyntheticSource(name=name, **arguments))
    if name == "friedman_drift":
        return FriedmanDriftSource(**arguments)
    if name in {"electricity", "airlines"}:
        return EventSamples(RealSource(name=name, directory=real_directory, **arguments))
    if name in {"insects", "gas"}:
        from .real_stream_datasets import prepare_real_stream

        return prepare_real_stream(name, cache, limit=examples, seed=seed, download=download)
    if name in {"split_cifar10", "split_cifar100"}:
        from .cifar_parquet import prepare_cifar_parquet

        return prepare_cifar_parquet(name, cache, limit=examples, seed=seed, download=download)
    if name in {"split_mnist", "permuted_mnist"}:
        from .image_datasets import prepare_image_dataset

        return prepare_image_dataset(name, cache, limit=examples, seed=seed, download=download)
    from .temporal_vision import prepare_temporal_dataset

    local = ({"train": os.environ["BENCH_STREAM51_ARCHIVE"]}
             if name == "stream51" and os.environ.get("BENCH_STREAM51_ARCHIVE") else None)
    return prepare_temporal_dataset(name, cache, limit=examples, seed=seed, download=download,
                                    local_archives=local)


def sample_dict(sample):
    """Canonical evidence retains original ID/task/split, never just a feature matrix."""
    return asdict(sample)
