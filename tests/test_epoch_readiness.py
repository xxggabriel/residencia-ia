import json

import pytest

from continual_bench.datasets.catalog import DATASETS
from continual_bench.orchestration import epoch_server


def test_worker_refuses_missing_partial_or_stale_dataset_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(epoch_server, 'SMOKE_ROOT', tmp_path)
    monkeypatch.setattr(epoch_server, 'environment', lambda: {'code_sha256': 'current'})
    monkeypatch.setattr(epoch_server, 'dataset_fingerprint', lambda: 'current-runtime')
    with pytest.raises(ValueError, match='absent'):
        epoch_server.dataset_readiness()
    path = tmp_path / 'dataset-smoke-complete.json'
    rows = [{'dataset': name, 'complete': True, 'unique_examples': 100, 'epochs': 2,
             'code_sha256': 'current', 'runtime_fingerprint': 'current-runtime'} for name in DATASETS]
    path.write_text(json.dumps({'complete': True, 'datasets': rows[:-1]}))
    with pytest.raises(ValueError, match='incomplete'):
        epoch_server.dataset_readiness()
    path.write_text(json.dumps({'complete': True, 'datasets': rows}))
    assert epoch_server.dataset_readiness()['datasets'] == 18
    rows[0]['code_sha256'] = 'old'
    path.write_text(json.dumps({'complete': True, 'datasets': rows}))
    with pytest.raises(ValueError, match='stale'):
        epoch_server.dataset_readiness()
