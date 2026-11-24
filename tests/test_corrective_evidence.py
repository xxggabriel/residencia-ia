import csv
import gc
import gzip
import io
import json
import weakref

import pytest

from scripts import export_corrective_evidence as evidence
from scripts.export_corrective_evidence import (
    export,
    full_source_coverage_valid,
    pack_large_tables,
    portable,
    replication_rows,
    sha256,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_campaign(root, *, missing=False):
    cases, statuses = [], {}
    for seed in (42, 43):
        for policy in ("online", "cara"):
            key = f"{policy}-{seed}"
            coordinates = {
                "dataset": "insects",
                "model": "logistic",
                "policy": policy,
                "detector": "none",
                "seed": seed,
                "mode": "incremental",
            }
            cases.append(
                {"case_key": key, "coordinates": coordinates, "config": {"artifact_mode": "metrics_only"}}
            )
            directory = root / "matrix" / "cases" / key / "binary" / key
            path = directory / "report.json"
            if missing and key == "cara-43":
                statuses[key] = {"status": "pending"}
                continue
            report = {
                "complete": True,
                "run_key": key,
                "config": {"artifact_mode": "metrics_only"},
                "metrics": {"accuracy": 0.6, "f1": 0.4, "f1_macro": 0.5},
                "tracking": {
                    "availability": {"missing": {"reason": "not_observed"}},
                    "metrics": {"missing": None, "zero": 0.0},
                },
                "environment": {"code_sha256": "code", "git_revision": "revision", "hostname": "private"},
                "dataset": {"sha256": "data", "cache": str(root / "inputs")},
                "tracking_uri": "http://name:password@192.168.100.27:5000/",
                "windows": [{"end_seq": i, "f1_macro": i / 400} for i in range(301)],
                "system_metrics": {
                    "samples": [{"timestamp_ms": i, "values": {"rss": i}} for i in range(201)]
                },
            }
            dump(path, report)
            # The exporter must never copy weights or the raw input, even if present beside metrics.
            (directory / "model.pkl").write_bytes(b"model-secret-bytes")
            (directory / "raw-input.csv").write_bytes(b"features,labels")
            rows = [
                {
                    "event_id": f"event-{i}",
                    "seq": i,
                    "label": i == 3,
                    "probability": 0.7 if i == 3 else 0.2,
                    "features": {"private": 123},
                }
                for i in range(1 if policy == "online" else 2, 5)
            ]
            (directory / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            statuses[key] = {"status": "passed", "report": path.relative_to(root).as_posix()}
    upf = {
        "dataset": "electricity",
        "protocol": "UPF-train-reference-v2",
        "trials_detail": [
            {
                "seed": seed,
                "curves": {
                    "cara": [
                        {
                            "alpha": 0.0,
                            "cost": cost,
                            "average_accuracy": 0.75,
                            "retrain_points": [7, 9],
                            "served_by": [6, 7],
                            "parameters": {"threshold": 0.25},
                        }
                    ]
                },
                "summary": {"cara": {"auc": cost}},
            }
            for seed, cost in enumerate((0.0, 2.0))
        ],
    }
    dump(root / "replications" / "upf.json", {"electricity": upf})
    dump(
        root / "campaign-plan.json",
        {"campaign_id": "fixture", "matrix": {"cases": cases}, "replications": {"jobs": [{"job_id": "upf"}]}},
    )
    dump(
        root / "campaign-status.json",
        {
            "complete": not missing,
            "matrix": statuses,
            "replications": {"upf": {"status": "passed", "output": "replications/upf.json"}},
        },
    )
    return root


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_export_pairs_actual_event_intersection_and_preserves_all_metric_points(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    output = tmp_path / "evidence"
    result = export(state, output, make_plots=False, require_complete=True)
    assert result["complete"] is True
    assert result["completed_cases"] == result["expected_cases"] == 4
    (cohort,) = result["paired_cohorts"]
    assert cohort["evaluated_common_events"] == 3
    assert cohort["all_expected_cases_available"] is True
    paired = read_csv(output / "paired-by-seed.csv")
    macro = [row for row in paired if row["metric"] == "f1_macro"]
    assert len(macro) == 4 and {float(row["value"]) for row in macro} == {1.0}
    assert {int(row["evaluated"]) for row in macro} == {3}
    aggregate = read_csv(output / "aggregate-metrics.csv")
    f1 = [
        row
        for row in aggregate
        if row["metric"] == "f1_macro" and row["scope"] == "paired_event_intersection"
    ]
    assert all(row["n"] == "2" and row["ddof"] == "1" and row["divisor"] == "1" for row in f1)
    metrics = read_csv(output / "metrics-long.csv")
    assert len([r for r in metrics if r["metric"] == "windows/f1_macro"]) == 4 * 301
    assert len([r for r in metrics if r["metric"] == "system_metrics/samples/values/rss"]) == 4 * 201
    nulls = [row for row in metrics if row["metric"] == "tracking/metrics/missing"]
    assert all(
        row["value"] == "" and row["available"] == "False" and row["reason"] == "not_observed"
        for row in nulls
    )
    zeros = [row for row in metrics if row["metric"] == "tracking/metrics/zero"]
    assert all(row["value"] == "0.0" and row["available"] == "True" for row in zeros)
    with gzip.open(output / "reports" / "online-42.json.gz", "rt") as stream:
        report = json.load(stream)
    assert "hostname" not in report["environment"]
    assert report["tracking_uri"] == "http://<tracking-endpoint>/"
    assert report["dataset"]["cache"] == "<campaign>/inputs"
    with gzip.open(output / "predictions" / "online-42.jsonl.gz", "rt") as stream:
        assert all("features" not in json.loads(line) for line in stream)
    assert not list(output.rglob("*.pkl")) and not list(output.rglob("raw-input.csv"))
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha256(output / name) == expected
    assert all(
        item["source_report_sha256"]
        == sha256(state / "matrix" / "cases" / item["case_key"] / "binary" / item["case_key"] / "report.json")
        for item in result["cases"]
    )


def test_export_repetition_aggregation_uses_population_ddof_and_keeps_schedule(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    output = tmp_path / "evidence"
    export(state, output, make_plots=False)
    rows = read_csv(output / "replication-aggregate.csv")
    (cost,) = [r for r in rows if r["metric"] == "cost"]
    assert float(cost["mean"]) == 1 and float(cost["std"]) == 1
    assert cost["n"] == "2" and cost["ddof"] == "0" and cost["divisor"] == "2"
    details = read_csv(output / "replication-by-seed.csv")
    assert json.loads(next(r for r in details if r["metric"] == "cost")["schedule"]) == [7, 9]
    assert not any(r["metric"] == "retrain_points" for r in details)


def test_missing_case_remains_visible_and_complete_export_is_rejected(tmp_path):
    state = fixture_campaign(tmp_path / "state", missing=True)
    with pytest.raises(ValueError, match="campaign is incomplete"):
        export(state, tmp_path / "complete", make_plots=False, require_complete=True)
    output = tmp_path / "partial"
    report = export(state, output, make_plots=False)
    assert report["complete"] is False and report["completed_cases"] == 3
    assert len(read_csv(output / "coverage.csv")) == 4
    assert report["paired_cohorts"][0]["missing_cases"] == ["cara-43"]
    assert report["paired_cohorts"][0]["all_expected_cases_available"] is False


def test_conflicting_labels_and_duplicate_event_ids_are_not_silently_accepted(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    path = state / "matrix/cases/cara-43/binary/cara-43/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["label"] = not rows[0]["label"]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="inconsistent labels"):
        export(state, tmp_path / "labels", make_plots=False)
    rows.append(rows[0])
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="duplicate prediction"):
        export(state, tmp_path / "duplicates", make_plots=False)


def test_portable_keeps_public_dataset_identifier_but_strips_secrets(tmp_path):
    assert portable("https://example.org/data?id=151", tmp_path) == "https://example.org/data?id=151"
    assert portable("https://user:pass@example.org/data?id=151&token=private", tmp_path) == (
        "https://example.org/data?id=151"
    )
    assert portable({"password": "secret", "hardware": "Intel", "n": 0}, tmp_path) == {
        "hardware": "Intel",
        "n": 0,
    }


def test_cara_curves_by_policy_use_kappa_and_keep_individual_seed():
    payload = {
        "dataset": "electricity",
        "protocol": "CARA-historical",
        "trials_detail": [
            {
                "seed": 4,
                "curves_by_policy": {
                    "cara": [{"kappa": 0.2, "strategy_cost": 0.3, "retrain_points": [0, 3]}]
                },
            }
        ],
    }
    (row,) = list(replication_rows(payload, "cara"))
    assert row["seed"] == 4 and row["parameter"] == "kappa" and row["parameter_value"] == 0.2
    assert row["metric"] == "strategy_cost" and row["value"] == 0.3


def test_expected_replication_and_prediction_ledger_cannot_be_omitted(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    status = json.loads((state / "campaign-status.json").read_text())
    status["replications"] = {}
    dump(state / "campaign-status.json", status)
    with pytest.raises(ValueError, match="campaign is incomplete"):
        export(state, tmp_path / "missing-job", make_plots=False, require_complete=True)
    fixture_campaign(state)
    (state / "matrix/cases/cara-42/binary/cara-42/predictions.jsonl").unlink()
    with pytest.raises(ValueError, match="campaign is incomplete"):
        export(state, tmp_path / "missing-ledger", make_plots=False, require_complete=True)


def test_export_is_deterministic_and_does_not_modify_source(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    before = {p.relative_to(state).as_posix(): sha256(p) for p in state.rglob("*") if p.is_file()}
    for name in ("first", "second"):
        export(state, tmp_path / name, make_plots=False)
    assert (tmp_path / "first/SHA256SUMS").read_bytes() == (tmp_path / "second/SHA256SUMS").read_bytes()
    assert before == {p.relative_to(state).as_posix(): sha256(p) for p in state.rglob("*") if p.is_file()}


def test_replication_retains_candidate_steps_and_nested_matrix_coordinates():
    payload = {
        "dataset": "electricity",
        "protocol": "UPF-train-reference-v2",
        "trials_detail": [{"seed": 2, "performance_matrix": {"err": [[None, 0.1], [0.3, 0.4]]}}],
        "candidate_tracking": [
            {
                "contract": {"seed": 2},
                "training_history": [{"fit_id": "fit-001", "steps": [{"step": 1, "metrics": {"loss": 0.6}}]}],
            }
        ],
    }
    rows = list(replication_rows(payload, "upf-electricity"))
    matrix = [r for r in rows if r["metric"] == "performance_matrix/err"]
    assert {tuple(json.loads(r["coordinate"])["list_indices"]) for r in matrix} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    (null,) = [r for r in matrix if not r["available"]]
    assert null["value"] is None
    (loss,) = [r for r in rows if r["point_type"] == "candidate" and r["metric"].endswith("/loss")]
    assert loss["seed"] == 2 and loss["value"] == 0.6
    assert json.loads(loss["coordinate"])["fit_id"] == "fit-001"


def test_large_table_compression_is_lossless_and_removes_only_owned_plaintext(tmp_path):
    content = b"metric,value\naccuracy,0.75\n"
    (tmp_path / "metrics-long.csv").write_bytes(content)
    unrelated = tmp_path / "unrelated.csv"
    unrelated.write_bytes(content)
    paths = pack_large_tables(tmp_path, threshold=1)
    assert paths == {"metrics-long.csv": "metrics-long.csv.gz"}
    assert not (tmp_path / "metrics-long.csv").exists()
    assert gzip.decompress((tmp_path / "metrics-long.csv.gz").read_bytes()) == content
    assert unrelated.read_bytes() == content


def corrective_three_seed_fixture(root):
    fixture_campaign(root)
    plan = json.loads((root / "campaign-plan.json").read_text())
    status = json.loads((root / "campaign-status.json").read_text())
    cara = [case for case in plan["matrix"]["cases"] if case["coordinates"]["policy"] == "cara"]
    third = json.loads(json.dumps(cara[1]))
    third["case_key"] = "cara-44"
    third["coordinates"]["seed"] = 44
    cara.append(third)
    source = root / status["matrix"]["cara-43"]["report"]
    target = root / "matrix/cases/cara-44/binary/cara-44/report.json"
    report = json.loads(source.read_text())
    report["run_key"] = "cara-44"
    dump(target, report)
    (target.parent / "predictions.jsonl").write_bytes((source.parent / "predictions.jsonl").read_bytes())
    status["matrix"]["cara-44"] = {"status": "passed", "report": target.relative_to(root).as_posix()}
    for case in cara:
        path = root / status["matrix"][case["case_key"]]["report"]
        report = json.loads(path.read_text())
        report["retraining"] = {"deployed": {"model_rows": [2, 3], "source_block_rows": [4, 5]}}
        dump(path, report)
    plan["matrix"]["cases"] = cara
    plan["exclusions"] = [
        {
            "family": "matrix",
            "policy": "online",
            "reason": "requested_for_next_round",
            "seeds": [42, 43, 44],
            "configurations": 3,
        }
    ]
    plan["replications"].update(seeds=[42, 43, 44], trials=3, full_source=True)
    result_path = root / "replications/upf.json"
    payload = json.loads(result_path.read_text())
    dataset = payload["electricity"]
    details = dataset["trials_detail"]
    details.append(json.loads(json.dumps(details[-1])))
    for index, detail in enumerate(details):
        detail["seed"] = 42 + index
        detail["curves"]["cara"][0]["cost"] = float(index * 2)
        detail["coverage"] = {
            "source_rows": 75,
            "selected_rows": 75,
            "full_source": True,
            "batch_rows": [5] * 15,
            "source_ranges": [[i * 5, (i + 1) * 5] for i in range(15)],
        }
    dataset.update(trials=3, seeds=[42, 43, 44], evaluation={"full_source": True})
    dump(result_path, payload)
    dump(root / "campaign-plan.json", plan)
    dump(root / "campaign-status.json", status)
    return root


def test_deliberate_exclusion_is_not_a_failed_case_and_three_seed_ddof_is_explicit(tmp_path):
    state = corrective_three_seed_fixture(tmp_path / "state")
    output = tmp_path / "evidence"
    result = export(state, output, make_plots=False, require_complete=True)
    assert result["expected_cases"] == result["completed_cases"] == 3
    assert len(read_csv(output / "coverage.csv")) == 3
    (exclusion,) = read_csv(output / "intentional-exclusions.csv")
    assert exclusion["policy"] == "online" and exclusion["status"] == "deliberately_excluded"
    assert json.loads(exclusion["seeds"]) == [42, 43, 44]
    assert result["seed_contract"]["matrix_observed_complete"] == [42, 43, 44]
    matrix = [
        r
        for r in read_csv(output / "aggregate-metrics.csv")
        if r["scope"] == "paired_event_intersection" and r["metric"] == "f1_macro"
    ]
    assert all(row["n"] == "3" and row["ddof"] == "1" and row["divisor"] == "2" for row in matrix)
    (cost,) = [r for r in read_csv(output / "replication-aggregate.csv") if r["metric"] == "cost"]
    assert cost["n"] == "3" and cost["ddof"] == "0" and cost["divisor"] == "3"
    counts = read_csv(output / "metrics-long.csv")
    assert {row["value"] for row in counts if row["metric"] == "retraining/deployed/model_rows"} == {"2", "3"}
    assert {row["value"] for row in counts if row["metric"] == "retraining/deployed/source_block_rows"} == {
        "4",
        "5",
    }


def test_deliberate_exclusion_cannot_overlap_an_executed_case(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    plan = json.loads((state / "campaign-plan.json").read_text())
    plan["exclusions"] = [{"family": "matrix", "policy": "online", "reason": "requested_for_next_round"}]
    dump(state / "campaign-plan.json", plan)
    with pytest.raises(ValueError, match="deliberately excluded configuration"):
        export(state, tmp_path / "evidence", make_plots=False)


@pytest.mark.parametrize("fault", ["seed", "duplicate_seed", "partial_source"])
def test_declared_seed_and_full_source_contracts_are_checked_before_completion(tmp_path, fault):
    state = corrective_three_seed_fixture(tmp_path / "state")
    path = state / "replications/upf.json"
    dataset = json.loads(path.read_text())
    if fault == "seed":
        dataset["electricity"]["trials_detail"][0]["seed"] = 0
    elif fault == "duplicate_seed":
        dataset["electricity"]["trials_detail"][0]["seed"] = 43
    else:
        dataset["electricity"]["trials_detail"][0]["coverage"]["selected_rows"] = 50
    dump(path, dataset)
    with pytest.raises(ValueError, match="campaign is incomplete"):
        export(state, tmp_path / "evidence", make_plots=False, require_complete=True)


def test_historical_table_uses_shared_configurations_and_preserves_separate_populations(tmp_path):
    state = corrective_three_seed_fixture(tmp_path / "state")
    plan = json.loads((state / "campaign-plan.json").read_text())
    for case in plan["matrix"]["cases"]:
        case["coordinates"]["dataset"] = "insects_abrupt_imbalanced"
    dump(state / "campaign-plan.json", plan)
    history = tmp_path / "history"
    history.mkdir()
    header = [
        "dataset",
        "modelo",
        "politica",
        "avaliados",
        "passo_inicial",
        "passo_final",
        "positivos",
        "f1",
        "ap",
        "acuracia",
        "recall",
        "precisao",
        "f1_dp",
        "ap_dp",
        "acuracia_dp",
        "recall_dp",
        "precisao_dp",
        "seeds",
    ]
    rows = [
        {
            **dict.fromkeys(header, 0.1),
            "dataset": "Abrupto",
            "modelo": "Logística",
            "politica": policy,
            "avaliados": 10,
            "passo_inicial": 15,
            "passo_final": 24,
            "seeds": 3,
        }
        for policy in ("CARA", "Online")
    ]
    with (history / "metricas-pareadas-medias.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, header)
        writer.writeheader()
        writer.writerows(rows)
    (history / "cobertura.csv").write_text(
        "dataset,modelo,politica,previstas,completas\n"
        "Abrupto,Logística,CARA,3,3\nAbrupto,Logística,Online,3,3\n"
    )
    before = {path.name: sha256(path) for path in history.iterdir()}
    output = tmp_path / "evidence"
    result = export(state, output, make_plots=False, require_complete=True, historical_insects=history)
    comparison = read_csv(output / "historical-comparison.csv")
    assert len(comparison) == 5 and {row["policy"] for row in comparison} == {"cara"}
    assert all(
        row["historical_evaluated_per_run"] == "10"
        and row["current_evaluated_per_run"] == "3"
        and row["historical_repetitions"] == row["current_repetitions"] == "3"
        for row in comparison
    )
    assert all(
        row["comparison_scope"] == "descriptive_only_protocols_and_event_populations_differ"
        for row in comparison
    )
    assert not any("delta" in name for name in comparison[0])
    metadata = result["historical_comparison"]
    assert metadata["historical_completed_units"] == metadata["historical_planned_units"] == 6
    (excluded,) = metadata["unmatched_historical_configurations"]
    assert excluded["policy"] == "online" and excluded["status"] == "deliberately_excluded"
    assert "não identifica um efeito causal" in (output / "historical-comparison.md").read_text()
    assert before == {path.name: sha256(path) for path in history.iterdir()}


def test_full_source_canonical_ranges_cover_all_fifteen_batches_without_a_dropped_tail():
    coverage = {
        "full_source": True,
        "source_rows": 75,
        "selected_rows": 75,
        "batch_rows": [5] * 15,
        "source_row_ranges": [
            {"batch": i, "start": 5 * i, "stop": 5 * (i + 1), "train_rows": 4, "holdout_rows": 1}
            for i in range(15)
        ],
        "offline_rows": 35,
        "online_rows": 40,
        "excluded_rows": 0,
        "all_source_rows_included": True,
    }
    assert full_source_coverage_valid(coverage)
    coverage["source_row_ranges"][7]["start"] += 1
    assert not full_source_coverage_valid(coverage)
    coverage["source_row_ranges"][7]["start"] -= 1
    coverage["source_row_ranges"][-1]["stop"] -= 1
    assert not full_source_coverage_valid(coverage)


def test_replication_candidate_and_analysis_rows_are_released_while_streaming(tmp_path, monkeypatch):
    state = fixture_campaign(tmp_path / "state")
    live = peak = consumed = 0

    class TrackedRow(dict):
        pass

    def released():
        nonlocal live
        live -= 1

    def many_rows(_payload, job):
        nonlocal live, peak, consumed
        yield {
            "job": job,
            "protocol": "fixture",
            "dataset": "electricity",
            "seed": 0,
            "policy": "cara",
            "parameter": "alpha",
            "parameter_value": 0.0,
            "point_type": "curve",
            "metric": "cost",
            "step": 0,
            "value": 1.5,
            "available": True,
            "coordinate": "{}",
        }
        for index in range(1000):
            row = TrackedRow(
                job=job,
                protocol="fixture",
                dataset="electricity",
                seed=0,
                policy="candidate_matrix",
                parameter="",
                parameter_value="",
                point_type="candidate" if index % 2 == 0 else "analysis",
                metric="training/loss",
                step=index,
                value=index / 1000,
                available=True,
                coordinate="{}",
            )
            live += 1
            peak = max(peak, live)
            weakref.finalize(row, released)
            yield row
            consumed += 1
            del row
            gc.collect()
            # CSV consumption must not retain the previously yielded point.
            # Materializing the generator with list.extend fails here.
            assert live == 0

    monkeypatch.setattr(evidence, "replication_rows", many_rows)
    output = tmp_path / "evidence"
    result = export(state, output, make_plots=False, require_complete=True)
    assert result["complete"] is True
    assert consumed == 1000 and live == 0 and peak == 1
    rows = read_csv(output / "replication-by-seed.csv")
    assert len(rows) == 1001
    assert [row["point_type"] for row in rows[1:]] == [
        "candidate" if index % 2 == 0 else "analysis" for index in range(1000)
    ]
    assert [float(row["value"]) for row in rows[1:]] == [index / 1000 for index in range(1000)]
    (aggregate,) = read_csv(output / "replication-aggregate.csv")
    assert aggregate["point_type"] == "curve" and float(aggregate["mean"]) == 1.5


def test_incremental_replication_csv_preserves_legacy_writer_bytes_and_payload_hash(tmp_path):
    state = fixture_campaign(tmp_path / "state")
    output = tmp_path / "evidence"
    result = export(state, output, make_plots=False, require_complete=True)
    actual = output / "replication-by-seed.csv"
    fields = next(csv.reader(io.StringIO(actual.read_text())))
    reference = io.StringIO(newline="")
    previous = csv.DictWriter(reference, fields, extrasaction="ignore")
    previous.writeheader()
    payload = json.loads((state / "replications/upf.json").read_text())
    previous.writerows(list(replication_rows(portable(payload, state), "upf")))
    assert actual.read_bytes() == reference.getvalue().encode()
    (job,) = result["replications"]
    assert job["source_sha256"] == sha256(state / "replications/upf.json")
    retained = gzip.decompress((output / job["artifact"]).read_bytes())
    assert json.loads(retained) == portable(payload, state)
