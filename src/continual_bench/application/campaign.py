"""Bounded declarative campaigns; retries reuse the runner's checked checkpoints."""

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from continual_bench.adapters.local import atomic_write, run_lock

from .config import Config, Strict


class Campaign(Strict):
    version: Literal["semana09-v1"] = "semana09-v1"
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    max_seconds: int = Field(default=600, ge=10, le=86400)
    cases: tuple[Config, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def distinct_cases(self):
        keys = [case.identity() for case in self.cases]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate campaign cases")
        return self

    def identity(self):
        raw = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


def execute_campaign(campaign, root, *, services=False, zenml=False, executor=None):
    """Fail fast, retain completed cases, and explicitly resume on a subsequent call.

    No automatic retry of denied or uncertain external operations. Each individual
    runner verifies config/code/data/transport contracts before a resume. Even a
    completed case goes through that verification on a subsequent invocation.
    """
    if executor is None:
        if zenml:
            from continual_bench.adapters.zenml_pipeline import execute_pipeline

            executor = execute_pipeline
        else:
            from .runner import run

            executor = run
    directory = Path(root) / campaign.identity()
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "campaign.json"
    contract = {"campaign": campaign.model_dump(mode="json"), "services": services, "zenml": zenml}
    with run_lock(root, "campaign-" + campaign.identity()):
        state = json.loads(journal.read_text()) if journal.exists() else {
            "contract": contract, "cases": {}, "complete": False,
        }
        if state["contract"] != contract:
            raise ValueError("campaign execution contract changed")
        state["complete"] = False

        def save():
            atomic_write(journal, json.dumps(state, indent=2).encode())

        started = time.monotonic()
        for case in campaign.cases:
            key = case.identity()
            if time.monotonic() - started >= campaign.max_seconds:
                save()
                raise TimeoutError("campaign deadline reached before next case")
            state["cases"][key] = {"status": "running"}
            save()
            try:
                report = executor(case, str(directory / "runs"), services=services)
                if not report.get("complete") or report.get("run_key") != key:
                    raise ValueError("case returned an incomplete or mismatched report")
                if time.monotonic() - started > campaign.max_seconds:
                    raise TimeoutError("campaign deadline exceeded during case; artifacts retained")
            except Exception as error:
                # Store type only: third-party error strings can contain credential URLs.
                state["cases"][key] = {"status": "failed", "error_type": type(error).__name__}
                save()
                raise
            state["cases"][key] = {
                "status": "passed", "metrics": report["metrics"],
                "report": str(directory / "runs" / key / "report.json"),
            }
            save()
        state["complete"] = True
        save()
        return state
