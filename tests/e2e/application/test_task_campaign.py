import pytest

from continual_bench.application.campaign import Campaign, execute_campaign
from continual_bench.config import Config


def campaign():
    return Campaign(name="bounded", cases=(Config(name="first"), Config(name="second")))


def test_campaign_rejects_duplicate_cases():
    with pytest.raises(ValueError, match="duplicate"):
        Campaign(name="duplicate", cases=(Config(), Config()))


def test_campaign_failed_attempt_is_visible_and_explicit_resume_rechecks_cases(tmp_path):
    seen = []

    def execute(config, root, services):
        seen.append(config.name)
        if config.name == "second" and len(seen) == 2:
            raise ConnectionError("private URL must not enter journal")
        return {"complete": True, "run_key": config.identity(), "metrics": {"accuracy": 0.5}}

    cfg = campaign()
    with pytest.raises(ConnectionError):
        execute_campaign(cfg, tmp_path, executor=execute)
    journal = (tmp_path / cfg.identity() / "campaign.json").read_text()
    assert '"failed"' in journal and "private URL" not in journal
    result = execute_campaign(cfg, tmp_path, executor=execute)
    assert result["complete"]
    assert seen == ["first", "second", "first", "second"]
    with pytest.raises(ValueError, match="contract changed"):
        execute_campaign(cfg, tmp_path, services=True, executor=execute)


def test_campaign_never_accepts_uncertain_or_foreign_result(tmp_path):
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        execute_campaign(campaign(), tmp_path, executor=lambda *a, **k: {"complete": True, "run_key": "x"})
