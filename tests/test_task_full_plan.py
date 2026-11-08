import importlib.util
import json
import sys
from pathlib import Path

import pytest

from continual_bench.adapters.dataset_loading import DATASETS
from continual_bench.application.config import Config
from continual_bench.application.epoch_benchmark import EpochConfig


def jobs():
    directory = Path(__file__).parents[1] / 'dags'
    for name in ('bench_jobs', 'task_jobs'):
        spec = importlib.util.spec_from_file_location(name, directory / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules['task_jobs']


def test_complete_plan_covers_catalog_models_modes_and_policy_extensions():
    module = jobs()
    cases = module.configurations({'samples': 3000, 'dataset_examples': 1000,
                                   'epochs': 2, 'seeds': [42], 'label_delay': 3}, 'a' * 64)
    epochs = [EpochConfig.model_validate(c['config']) for c in cases if c['kind'] == 'epochs']
    binary = [Config.model_validate(c['config']) for c in cases if c['kind'] == 'binary']
    assert {c.dataset for c in epochs} == set(DATASETS) and len(epochs) == 18
    assert {c.model.name for c in binary} == {'logistic', 'hoeffding_tree', 'knn'}
    assert {c.mode for c in binary} == {'streaming', 'incremental', 'offline'}
    assert {'performance', 'cost_heuristic', 'drift', 'online', 'none', 'periodic'} <= {
        c.policy.name for c in binary}
    assert {'adwin', 'ddm', 'eddm', 'hddm_a', 'hddm_w', 'kswin', 'page_hinkley'} <= {
        c.detector.name for c in binary}
    assert len(cases) <= 1024 and len({json.dumps(c, sort_keys=True) for c in cases}) == len(cases)
    with pytest.raises(ValueError, match='exactly one'):
        module.configurations({'seeds': [42, 43]}, 'a' * 64)


def test_polling_requires_publication_and_rejects_failed_jobs(monkeypatch):
    module = jobs()
    monkeypatch.setattr(module, 'request', lambda path, payload=None: {
        'run_key': 'fixture', 'status': 'succeeded', 'receipt': {}})
    with pytest.raises(RuntimeError, match='publication'):
        module.execute({'kind': 'binary', 'config': {}})
    monkeypatch.setattr(module, 'request', lambda path, payload=None: {
        'request_key': 'fixture', 'status': 'failed'})
    with pytest.raises(RuntimeError, match='failed'):
        module.execute({'kind': 'epochs', 'config': {}})
