"""Materialize a campaign request with its selected, immutable release."""

import json
import os
import re
import sys
from pathlib import Path

GIT_SHA = r"[0-9a-f]{40}"
CAMPAIGN_ID = r"[a-z0-9][a-z0-9-]{0,39}"


def prepare(campaign, *, release=None, campaign_id=None, dataset=None):
    request = campaign["request"]
    if campaign_id:
        request["campaign_id"] = campaign_id
    if not re.fullmatch(CAMPAIGN_ID, request["campaign_id"]):
        raise ValueError("campaign_id must contain 1–40 lowercase letters, digits or dashes")

    release = release or request.get("release", "latest")
    zia = request.get("matrix", {}).get("kind", "zia") == "zia"
    if release == "latest":
        if zia:
            raise ValueError(
                "Zia campaigns require RELEASE_SHA or request.release set to a published Git SHA"
            )
    elif not re.fullmatch(GIT_SHA, release):
        raise ValueError("release must be a 40-character lowercase Git SHA")
    request["release"] = release

    if dataset is not None:
        snapshot = request.get("snapshot") or {}
        source = dataset["request"]
        if zia and any(source.get(field) != snapshot.get(field) for field in ("dataset", "scope")):
            raise ValueError("dataset preparation does not match the campaign snapshot")
        dataset["release"] = release
    return request["campaign_id"]


def main():
    input_path, output_path, *dataset_paths = map(Path, sys.argv[1:])
    campaign = json.loads(input_path.read_text())
    dataset = json.loads(dataset_paths[0].read_text()) if dataset_paths else None
    try:
        campaign_id = prepare(
            campaign, release=os.getenv("RELEASE_SHA"), campaign_id=os.getenv("CAMPAIGN_ID"), dataset=dataset
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Invalid campaign input: {error}") from error
    output_path.write_text(json.dumps(campaign))
    if dataset is not None:
        dataset_paths[1].write_text(json.dumps(dataset))
    print(campaign_id)


if __name__ == "__main__":
    main()
