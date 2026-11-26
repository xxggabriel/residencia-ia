"""A training child must expose its actual seed rather than only inheriting a parent link."""

import pytest

from continual_bench.infrastructure.training_tracking import training_round_mlflow


@pytest.mark.parametrize("seed", [42, 43, 44, None])
def test_training_child_preserves_seed_and_legacy_absence(seed):
    fit = {
        "fit_id": "fit-000001",
        "action": "candidate",
        "steps": [{"step": 1, "global_step": 1, "kind": "epoch", "metrics": {"loss": 0.1}}],
        "summary": {},
    }
    if seed is not None:
        fit["seed"] = seed
    tags, params, _ = next(training_round_mlflow("parent", {"training_history": [fit]}))
    if seed is None:
        assert "seed" not in params and "cb-seed" not in tags
    else:
        assert params["seed"] == tags["cb-seed"] == str(seed)
