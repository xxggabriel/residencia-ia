"""Continual-learning task protocols over prepared samples (split, permuted, temporal)."""

import random
from dataclasses import replace

from .base import stable_identity


def split_class_tasks(samples, class_order, classes_per_task=2):
    """Split MNIST/CIFAR protocol on prepared samples; preserves labels and splits."""
    class_order = tuple(str(c) for c in class_order)
    if (not class_order or len(set(class_order)) != len(class_order) or classes_per_task < 1
            or len(class_order) % classes_per_task):
        raise ValueError("class order must be unique and evenly divisible into tasks")
    mapping = {label: i // classes_per_task for i, label in enumerate(class_order)}
    seen = set()
    tasks = [[] for _ in range(len(class_order) // classes_per_task)]
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError("sample identity overlaps across protocol inputs")
        seen.add(sample.sample_id)
        if str(sample.y) not in mapping:
            raise ValueError("sample label is absent from declared class order")
        task = mapping[str(sample.y)]
        tasks[task].append(replace(sample, task_id=task))
    if any(not task for task in tasks):
        raise ValueError("every declared task must contain samples")
    return tasks


def permuted_tasks(samples, tasks, seed, feature_order):
    """Fixed permutation per task shared across train/test; identity task zero."""
    feature_order = tuple(feature_order)
    if tasks < 1 or tasks > 100 or not feature_order or len(set(feature_order)) != len(feature_order):
        raise ValueError("invalid permutation protocol")
    samples = tuple(samples)
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("duplicate sample identities")
    rng, output, permutations = random.Random(seed), [], []
    for task in range(tasks):
        permutation = list(feature_order)
        if task:
            rng.shuffle(permutation)
        permutations.append(permutation)
        transformed = []
        for sample in samples:
            if set(sample.x) != set(feature_order):
                raise ValueError("permutation feature schema mismatch")
            identity = stable_identity([sample.sample_id, task, permutation])
            transformed.append(replace(sample, sample_id=identity,
                                       task_id=task,
                                       x={k: sample.x[v] for k, v in zip(feature_order, permutation)}))
        output.append(transformed)
    return output, permutations


def temporal_tasks(samples, group_order):
    """Preserve official CORe50/CLEAR groups and source splits.

    The caller supplies the official session/bucket order and normalized records;
    this function never invents a split or shuffles a video sequence.
    """
    group_order = tuple(str(g) for g in group_order)
    if not group_order or len(set(group_order)) != len(group_order):
        raise ValueError("group order must contain unique groups")
    mapping = {group: i for i, group in enumerate(group_order)}
    output = [[] for _ in group_order]
    seen, groups_by_split = set(), {}
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError("sample identity overlaps across protocol inputs")
        seen.add(sample.sample_id)
        if sample.group not in mapping:
            raise ValueError("sample group is absent from official group order")
        key = (sample.group, sample.split)
        if sample.timestamp < groups_by_split.get(key, -1):
            raise ValueError("within-group temporal order decreased")
        groups_by_split[key] = sample.timestamp
        output[mapping[sample.group]].append(replace(sample, task_id=mapping[sample.group]))
    if any(not group for group in output):
        raise ValueError("every declared temporal group must contain samples")
    return output
