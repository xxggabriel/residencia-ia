import json

import pytest

from continual_bench.application.project_suite import project_configurations
from continual_bench.config import Config
from continual_bench.datasets.catalog import DATASETS
from continual_bench.evaluation.epochs import EpochConfig


def test_complete_plan_covers_catalog_models_modes_and_policy_extensions():
    cases = project_configurations(
        {"samples": 3000, "dataset_examples": 1000, "epochs": 2, "seeds": [42], "label_delay": 3},
        "a" * 64,
    )
    epochs = [EpochConfig.model_validate(c["config"]) for c in cases if c["kind"] == "epochs"]
    binary = [Config.model_validate(c["config"]) for c in cases if c["kind"] == "binary"]
    assert {c.dataset for c in epochs} == set(DATASETS) and len(epochs) == 17
    assert {c.model.name for c in binary} == {"logistic", "hoeffding_tree", "knn"}
    assert {c.mode for c in binary} == {"streaming", "incremental", "offline"}
    assert {"performance", "drift", "online", "none", "periodic"} <= {c.policy.name for c in binary}
    assert {"adwin", "ddm", "eddm", "hddm_a", "hddm_w", "kswin", "page_hinkley"} <= {
        c.detector.name for c in binary
    }
    assert len(cases) <= 1024 and len({json.dumps(c, sort_keys=True) for c in cases}) == len(cases)
    with pytest.raises(ValueError, match="exactly one"):
        project_configurations({"seeds": [42, 43]}, "a" * 64)
