"""Benchmark protocols and campaign orchestration."""

from continual_bench.protocols.base import BenchmarkProtocol, CampaignConfig, ProtocolResult
from continual_bench.protocols.campaign import load_campaign_config, run_campaign
from continual_bench.protocols.cara import CaraProtocol
from continual_bench.protocols.report import compare_with_published_reference
from continual_bench.protocols.upf import UpfProtocol

__all__ = [
    "BenchmarkProtocol",
    "CampaignConfig",
    "CaraProtocol",
    "ProtocolResult",
    "UpfProtocol",
    "compare_with_published_reference",
    "load_campaign_config",
    "run_campaign",
]
