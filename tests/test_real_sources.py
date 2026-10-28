import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from continual_bench.adapters.real_sources import REAL_SOURCES, RealSource, iter_arff, validate_source

ELECTRICITY_HEADER = """@relation fixture
@attribute date numeric
@attribute day {1,2,3,4,5,6,7}
@attribute period numeric
@attribute nswprice numeric
@attribute nswdemand numeric
@attribute vicprice numeric
@attribute vicdemand numeric
@attribute transfer numeric
@attribute class {UP,DOWN}
@data
"""


@pytest.fixture
def admitted_fixture(tmp_path, monkeypatch):
    path = tmp_path / "electricity.arff"
    path.write_text(ELECTRICITY_HEADER + "0,2,0,.1,.2,.3,.4,.5,UP\n"
                    ".1,3,.5,.2,.3,.4,.5,.6,DOWN\n.2,4,1,.3,.4,.5,.6,.7,UP\n")
    spec = {**REAL_SOURCES["electricity"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 3, "bytes": path.stat().st_size}
    monkeypatch.setitem(REAL_SOURCES, "electricity", spec)
    return tmp_path, path


def source(directory, samples=2, seed=42):
    return RealSource("electricity", samples, seed, 3, 1000, directory=directory)


def test_real_prefix_is_reproducible_across_seeds_and_keeps_nominal_fields(admitted_fixture):
    directory, _ = admitted_fixture
    first, second = source(directory), source(directory, seed=99)
    events = list(first.events())
    assert events == list(second.events())
    assert first.version == second.version
    assert [e.y for e in events] == [True, False]
    assert [e.seq for e in events] == [0, 1]
    assert [e.timestamp for e in events] == [1000, 1001]
    assert all(e.available_at == e.timestamp + 3 and e.concept is None for e in events)
    assert events[0].x["day"] == "2"
    assert isinstance(events[0].x["date"], float)
    assert "class" not in events[0].x
    assert "timestamp" not in events[0].x
    assert source(directory, samples=3).version != first.version
    manifest = first.manifest()
    assert manifest["changepoints"] is None and manifest["seed_ignored"]
    assert manifest["raw_rows"] == 3 and manifest["selection"]["range"] == [0, 2]
    assert not manifest["real_event_timestamps_known"]
    assert not manifest["real_label_availability_known"]
    assert manifest["license"] == "Public"


def test_unprepared_or_modified_real_source_is_rejected(admitted_fixture, tmp_path):
    directory, path = admitted_fixture
    with pytest.raises(ValueError, match="exceeds"):
        source(directory, samples=4)
    path.write_text(path.read_text().replace(".7,UP", ".8,UP"))
    with pytest.raises(ValueError, match="SHA-256"):
        source(directory)
    with pytest.raises(ValueError, match="absent"):
        RealSource("airlines", 2, 42, 0, 1000, directory=tmp_path)


def test_source_admission_checks_full_cardinality(admitted_fixture, monkeypatch):
    directory, path = admitted_fixture
    assert validate_source(path, "electricity")["raw_rows"] == 3
    monkeypatch.setitem(REAL_SOURCES["electricity"], "rows", 4)
    with pytest.raises(ValueError, match="row count"):
        validate_source(path, "electricity")
    with pytest.raises(ValueError, match="fewer rows"):
        list(source(directory, samples=4).events())


@pytest.mark.parametrize("row,error", [
    ("?,2,0,.1,.2,.3,.4,.5,UP", "Missing"),
    ("nan,2,0,.1,.2,.3,.4,.5,UP", "Non-finite"),
    ("0,9,0,.1,.2,.3,.4,.5,UP", "categories"),
    ("0,2,0,.1,.2,.3,.4,.5,INVALID", "categories"),
    ("0,2,0,.1,.2,.3,.4,UP", "schema"),
])
def test_parser_rejects_missing_nonfinite_and_invalid_categories(tmp_path, row, error):
    path = tmp_path / "bad.arff"
    path.write_text(ELECTRICITY_HEADER + row + "\n")
    with pytest.raises(ValueError, match=error):
        list(iter_arff(path))


def test_airlines_parser_preserves_scheduled_departure_categories(tmp_path):
    path = tmp_path / "airlines.arff"
    path.write_text("""@relation fixture
@attribute Airline {CO,US}
@attribute Flight numeric
@attribute AirportFrom {'SFO','PHX'}
@attribute AirportTo {'IAH','CLT'}
@attribute DayOfWeek {3,4}
@attribute Time numeric
@attribute Length numeric
@attribute Delay {0,1}
@data
CO,269,SFO,IAH,3,15,205,1
US,1558,PHX,CLT,4,15,222,0
""")
    rows = list(iter_arff(path))
    assert rows[0]["AirportFrom"] == "SFO" and rows[0]["DayOfWeek"] == "3"
    assert rows[0]["Flight"] == 269.0 and rows[1]["Delay"] == "0"


def test_prep_copies_verified_source_atomically_and_rejects_bad_cache(admitted_fixture, tmp_path):
    directory, path = admitted_fixture
    module_path = Path(__file__).parents[1] / "scripts/prepare_real_sources.py"
    spec = importlib.util.spec_from_file_location("prepare_real_sources", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = directory / "admitted"
    receipt = module.prepare("electricity", destination, directory)
    assert receipt["raw_rows"] == 3
    assert (destination / path.name).read_bytes() == path.read_bytes()
    assert json.loads((destination / "electricity.admission.json").read_text())["raw_rows"] == 3
    (destination / path.name).write_text("corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare("electricity", destination, directory)
    assert not list(destination.glob(".pending-*"))


def test_lock_matches_packaged_source_pins():
    lock = json.loads((Path(__file__).parents[1] / "configs/real-sources.lock.json").read_text())
    assert lock["sources"] == REAL_SOURCES
