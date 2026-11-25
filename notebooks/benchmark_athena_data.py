"""Load the hash-bound Athena result tables used by the four-stream figures.

This module reads completed query results. It never submits a query or runs a
benchmark. The optional downloader only retrieves existing result CSV objects.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RELEASE = "f1092ee0232a6d16504b3a2211a037c41a02068a"
PROTOCOL = "ordered-stream-batch-prequential-metrics-v2"
DATASETS = (
    "electricity",
    "airlines",
    "insects_abrupt_imbalanced",
    "insects_incremental_abrupt_reoccurring_imbalanced",
)
POLICIES = (
    "frozen",
    "drift",
    "drift_adwin",
    "drift_ddm",
    "incremental",
    "periodic",
    "cara",
    "cara_cumulative",
    "cara_periodic",
    "upf",
    "oracle",
)
SEEDS = (42, 43, 44)
TEMPORAL_METRICS = ("accuracy", "macro_f1", "pr_auc", "error_rate")
BOUNDS = {
    "electricity": (45312, 13599, 40781, "9d0091b03ba5b3ba6bee24b0b10ec4d49099e1683e8b9b62848bc63db18be500"),
    "airlines": (539383, 161815, 485445, "3c80a172bd84b11f4d187e8d0c76c0c47913a475a7159f877d0c6153a745374a"),
    "insects_abrupt_imbalanced": (
        355275,
        106583,
        319748,
        "66d2ba7761322b43c39025216a30617fddda65ee10953da9e57acf16f11995c9",
    ),
    "insects_incremental_abrupt_reoccurring_imbalanced": (
        452044,
        135615,
        406840,
        "744469bf84e301c35a8e517ff095b9eb28aeec092c2e4c3e436436ccb8d80067",
    ),
}
# Digests bind decoded rows, not CSV quoting or the local filesystem location.
QUERIES = {
    "12-primary-common-metrics": (
        "a97023a9-f737-471f-b724-d4cf5e4da0f3",
        132,
        "13e8eb0ebd9a08bc70b60f7848f8ee2686f569cfa55efc6e7db86d0866c91730",
    ),
    "13b-main-resources": (
        "e931f10b-82ca-41b7-9f35-fee2b662b30e",
        6168,
        "eff515c3ace52cad0b40db766f260ff3cc908f950a060ab5cd5a55ce600993f9",
    ),
    "14b-all-metrics-summary": (
        "d5acb9d0-962b-440d-8050-3f66ecb12069",
        7707,
        "6a045ebabb875be56ccdbe651fcdffd85f385b225b09b3944d3030e46350779e",
    ),
    "16-window-curves": (
        "59db7383-f7d9-4030-9b68-c2ccef89ae1a",
        132,
        "c88e490f9785b728e291a882cc41e36dbbc3accc625abd225e2e64eed1748538",
    ),
}
EXPECTED = {(d, p, s) for d in DATASETS for p in POLICIES for s in SEEDS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(rows):
    return json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    require(math.isfinite(result), "Valor numérico não finito na fonte.")
    return result


def coordinate(row):
    return row["dataset"], row["policy"], int(row["seed"])


def source_rows(directory, stem, *, prefer_csv=False):
    query_id, expected_count, expected_sha = QUERIES[stem]
    directory = Path(directory)
    json_path = directory / f"{stem}-rows.json"
    csv_path = directory / f"{query_id}.csv"
    if json_path.is_file() and not prefer_csv:
        rows = json.loads(json_path.read_text())
    elif csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [
                {key: value if value != "" else None for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    else:
        raise FileNotFoundError(
            f"Fonte ausente: {json_path.name} ou {csv_path.name}. "
            "Configure BENCHMARK_ATHENA_DIR ou use baixar_resultados_existentes "
            "com credenciais AWS de leitura. Nenhum benchmark será iniciado."
        )
    require(isinstance(rows, list) and len(rows) == expected_count, f"Cobertura da consulta {stem} inválida.")
    require(
        hashlib.sha256(canonical(rows)).hexdigest() == expected_sha,
        f"Conteúdo da consulta {stem} difere do conjunto validado.",
    )
    return rows


def baixar_resultados_existentes(directory, *, bucket, region="us-east-1"):
    """Download four completed Athena CSVs through the user's existing AWS CLI credentials."""
    require(
        bucket and "/" not in bucket and ":" not in bucket, "Informe somente o nome do bucket de resultados."
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stem, (query_id, _, _) in QUERIES.items():
        target = directory / f"{query_id}.csv"
        if target.exists():
            source_rows(directory, stem, prefer_csv=True)
            continue
        pending = target.with_suffix(".csv.pending")
        try:
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"s3://{bucket}/athena-results/{query_id}.csv",
                    str(pending),
                    "--region",
                    region,
                    "--only-show-errors",
                    "--no-cli-pager",
                ],
                check=True,
                timeout=300,
            )
            pending.replace(target)
            source_rows(directory, stem, prefer_csv=True)
        except Exception:
            pending.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        seed = int(row["seed"])
        require(seed in SEEDS and seed not in groups.setdefault(key, {}), "Seed repetida ou desconhecida.")
        groups[key][seed] = row["value"]
    result = []
    for key, values in groups.items():
        require(set(values) == set(SEEDS), "A comparação exige as três seeds.")
        finite = [v for v in values.values() if v is not None]
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_seeds_present": 3,
                "n_defined": len(finite),
                "n_undefined": 3 - len(finite),
                "sd_ddof": 1,
                "mean": statistics.mean(finite) if finite else None,
                "sample_sd": statistics.stdev(finite) if len(finite) > 1 else None,
            }
        )
    return result


@dataclass
class AthenaEvidence:
    primary: pd.DataFrame
    quality: pd.DataFrame
    resource_values: pd.DataFrame
    resources: pd.DataFrame
    temporal_values: pd.DataFrame
    temporal: pd.DataFrame
    series: dict
    holdout: pd.DataFrame
    coverage: pd.DataFrame
    provenance: dict


def load_athena(directory):
    sources = {name: source_rows(directory, name) for name in QUERIES}
    primary = sources["12-primary-common-metrics"]
    cases, primary_long = {}, []
    metadata = {
        "dataset",
        "policy",
        "seed",
        "run_id",
        "metric_scope",
        "ranking_estimator",
        "n",
        "expected_n",
        "coverage_valid",
        "distinct_seq",
        "min_seq",
        "max_seq",
    }
    for row in primary:
        key = coordinate(row)
        require(key in EXPECTED and key not in cases, "Caso primário repetido ou não previsto.")
        cases[key] = row
        _, begin, end, _ = BOUNDS[key[0]]
        require(
            row["coverage_valid"] == "true" and row["metric_scope"] == "common_postcalibration",
            "A fonte não representa a coorte comum pós-calibração.",
        )
        require(row["ranking_estimator"] == "uniform_histogram_4096_bins", "Estimador de área diferente.")
        require(
            int(row["n"]) == int(row["expected_n"]) == int(row["distinct_seq"]) == end - begin,
            "Denominador primário incorreto.",
        )
        require(
            int(row["min_seq"]) == begin and int(row["max_seq"]) == end - 1, "Intervalo primário incorreto."
        )
        for metric, value in row.items():
            if metric not in metadata:
                primary_long.append(
                    {
                        "dataset": key[0],
                        "policy": key[1],
                        "seed": key[2],
                        "run_id": row["run_id"],
                        "metric": metric,
                        "value": number(value),
                    }
                )
    require(
        set(cases) == EXPECTED and len({row["run_id"] for row in primary}) == 132, "Matriz 4×11×3 incompleta."
    )

    resource_long, resource_keys = [], set()
    for row in sources["13b-main-resources"]:
        key = coordinate(row)
        require(key in cases and row["run_id"] == cases[key]["run_id"], "Recurso associado a outro caso.")
        total, _, _, dataset_sha = BOUNDS[key[0]]
        require(
            row["git_sha"] == RELEASE and row["protocol"] == PROTOCOL and row["model"] == "xgboost",
            "Revisão, modelo ou protocolo dos recursos divergente.",
        )
        require(
            int(row["source_rows"]) == total and row["dataset_sha256"] == dataset_sha,
            "Fonte integral ou hash do dataset divergente.",
        )
        require(row["identity_mismatch"] == row["nonfinite_fact"] == "0", "Identidade de recurso inválida.")
        fact = (*key, row["metric"])
        require(fact not in resource_keys, "Métrica de recurso duplicada.")
        resource_keys.add(fact)
        resource_long.append(
            {
                "dataset": key[0],
                "policy": key[1],
                "seed": key[2],
                "run_id": row["run_id"],
                "metric": row["metric"],
                "value": number(row["value"]),
                "source_fact_present": True,
            }
        )
    resource_metrics = sorted({row["metric"] for row in resource_long})
    require(len(resource_metrics) == 48, "O catálogo selecionado de recursos está incompleto.")
    for key, case in cases.items():
        for metric in resource_metrics:
            if (*key, metric) not in resource_keys:
                resource_long.append(
                    {
                        "dataset": key[0],
                        "policy": key[1],
                        "seed": key[2],
                        "run_id": case["run_id"],
                        "metric": metric,
                        "value": None,
                        "source_fact_present": False,
                    }
                )
    # The two plotted resources are defined for every case. Other resources retain their own applicability.
    for metric in ("actual_total_fit_seconds", "update_count"):
        require(
            {(r["dataset"], r["policy"], r["seed"]) for r in resource_long if r["metric"] == metric}
            == EXPECTED,
            "Recurso representado na figura sem cobertura das três seeds.",
        )

    series, temporal_long = {}, []
    for row in sources["16-window-curves"]:
        key = coordinate(row)
        require(
            key in cases and key not in series and row["run_id"] == cases[key]["run_id"],
            "Identidade temporal divergente ou duplicada.",
        )
        _, begin, end, dataset_sha = BOUNDS[key[0]]
        require(
            row["source_sha256"] == dataset_sha and row["temporal_coverage_valid"] == "true",
            "Fonte ou cobertura temporal inválida.",
        )
        require(
            all(
                row[k] == "0"
                for k in (
                    "windows_without_three_seeds",
                    "windows_without_all_33_runs",
                    "invalid_fact_windows",
                    "discontinuous_windows",
                )
            ),
            "Janelas não alinhadas.",
        )
        points = json.loads(row["window_series_json"])
        require(len(points) == 16 and int(row["covered_n"]) == end - begin, "Cobertura temporal incompleta.")
        previous, normalized = begin, []
        for window, point in enumerate(points, 1):
            first, last, n = (int(point[k]) for k in ("start_seq", "end_seq", "n"))
            require(first == previous and last - first == n and n > 0 and last <= end, "Janela descontínua.")
            previous = last
            values = {metric: number(point[metric]) for metric in TEMPORAL_METRICS}
            require(all(v is None or 0 <= v <= 1 for v in values.values()), "Escore temporal fora de 0..1.")
            normalized.append({"window": window, "start_seq": first, "end_seq": last, "n": n, **values})
            for metric, value in values.items():
                temporal_long.append(
                    {
                        "dataset": key[0],
                        "policy": key[1],
                        "seed": key[2],
                        "run_id": row["run_id"],
                        "window": window,
                        "start_seq": first,
                        "end_seq": last,
                        "n": n,
                        "metric": metric,
                        "value": value,
                    }
                )
        require(previous == end, "A última janela não termina no fim da coorte.")
        series[key] = normalized
    require(set(series) == EXPECTED, "Matriz temporal incompleta.")
    for dataset in DATASETS:
        for window in range(16):
            require(
                len(
                    {
                        tuple(series[d, p, s][window][k] for k in ("start_seq", "end_seq", "n"))
                        for d, p, s in EXPECTED
                        if d == dataset
                    }
                )
                == 1,
                "As 33 execuções não compartilham janelas.",
            )

    holdout_long = []
    reported = sources["14b-all-metrics-summary"]
    require(sum(int(r["n_facts"]) for r in reported) == 23121, "Catálogo de fatos incompleto.")
    for row in reported:
        require(
            row["metric_summary_valid"] == "true" and row["sd_ddof"] == "1", "Agregado analítico inválido."
        )
        values = [number(row[f"seed_{seed}_value"]) for seed in SEEDS]
        finite = [value for value in values if value is not None]
        require(len(finite) == int(row["n_defined"]), "Denominador de métrica incorreto.")
        if finite:
            require(
                math.isclose(statistics.mean(finite), number(row["mean"]), rel_tol=1e-10, abs_tol=1e-12),
                "Média analítica inconsistente.",
            )
        if len(finite) > 1:
            require(
                math.isclose(statistics.stdev(finite), number(row["sample_sd"]), rel_tol=1e-8, abs_tol=1e-10),
                "Desvio padrão analítico inconsistente.",
            )
        if row["partition"] == "holdout" and row["metric"] in ("macro_f1", "pr_auc", "n"):
            for seed, value in zip(SEEDS, values, strict=True):
                holdout_long.append(
                    {
                        "dataset": row["dataset"],
                        "policy": row["policy"],
                        "seed": seed,
                        "metric": row["metric"],
                        "value": value,
                    }
                )
    holdout = aggregate(holdout_long, ("dataset", "policy", "metric"))
    require(len(holdout) == 132, "Diagnóstico de holdout incompleto.")
    for row in holdout:
        if row["metric"] == "n":
            total, _, end, _ = BOUNDS[row["dataset"]]
            require(row["mean"] == total - end and row["n_defined"] == 3, "Denominador de holdout incorreto.")
    coverage = [
        {
            "dataset": d,
            "observacoes_fonte": BOUNDS[d][0],
            "politicas": 11,
            "seeds": 3,
            "unidades": 33,
            "inicio_coorte": BOUNDS[d][1],
            "fim_exclusivo_coorte": BOUNDS[d][2],
            "exemplos_comuns": BOUNDS[d][2] - BOUNDS[d][1],
        }
        for d in DATASETS
    ]
    return AthenaEvidence(
        pd.DataFrame(primary_long),
        pd.DataFrame(aggregate(primary_long, ("dataset", "policy", "metric"))),
        pd.DataFrame(resource_long),
        pd.DataFrame(aggregate(resource_long, ("dataset", "policy", "metric"))).rename(
            columns={"n_undefined": "n_missing_or_undefined"}
        ),
        pd.DataFrame(temporal_long),
        pd.DataFrame(aggregate(temporal_long, ("dataset", "policy", "window", "metric"))),
        series,
        pd.DataFrame(holdout),
        pd.DataFrame(coverage),
        {
            "release": RELEASE,
            "protocol": PROTOCOL,
            "cases": 132,
            "seeds": list(SEEDS),
            "queries": {
                name: {"query_id": spec[0], "rows": spec[1], "decoded_rows_sha256": spec[2]}
                for name, spec in QUERIES.items()
            },
            "native_mlflow_histories_requeried": False,
            "source": "Completed Athena SELECT results; semantic hashes checked before aggregation",
        },
    )


def load_tables(directory):
    """Read configurable scientific CSVs without inheriting the F109 audit.

    A ``benchmark-tables-v1`` manifest must bind each CSV by byte hash and
    declare the complete datasets, policies, seeds and cohort boundaries.
    This validates the supplied tables and their internal contract. It does
    not establish the authenticity of a different benchmark campaign.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "benchmark-tables-v1", "Manifesto de tabelas incompatível.")
    contract = manifest["contract"]
    datasets = contract["datasets"]
    policies = contract["policies"]
    seeds = contract["seeds"]
    require(isinstance(datasets, list) and datasets, "Datasets não declarados.")
    require(
        isinstance(policies, list) and policies and len(set(policies)) == len(policies),
        "Políticas inválidas.",
    )
    require(
        isinstance(seeds, list)
        and seeds
        and all(type(seed) is int for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "Seeds devem ser inteiros distintos declarados no contrato.",
    )
    names = [row["name"] for row in datasets]
    require(len(set(names)) == len(names), "Dataset repetido no contrato.")
    bounds = {row["name"]: row for row in datasets}
    for row in datasets:
        require(
            0 <= row["common_start"] < row["common_end"] <= row["holdout_start"] <= row["source_rows"],
            "Limites da coorte inválidos.",
        )
        digest = row["source_sha256"]
        require(
            isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "Hash da fonte de dados inválido.",
        )
    require(contract.get("sd_ddof") == 1, "A análise utiliza desvio-padrão amostral, ddof=1.")
    require(contract.get("protocol") and contract.get("model"), "Declare protocolo e modelo.")
    for field in ("primary_metrics", "resource_metrics", "temporal_metrics"):
        metrics = contract[field]
        require(
            isinstance(metrics, list) and metrics and len(metrics) == len(set(metrics)),
            f"Catálogo {field} vazio ou repetido.",
        )
    expected = {(d, p, s) for d in names for p in policies for s in seeds}

    def table(name):
        spec = manifest["tables"][name]
        path = (directory / spec["path"]).resolve()
        require(path.is_relative_to(directory) and path.is_file(), "Caminho de tabela inválido.")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == spec["sha256"], f"Hash da tabela {name} divergente.")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == spec["rows"], f"Contagem da tabela {name} divergente.")
        return rows

    cases = {}
    for row in table("cases"):
        key = coordinate(row)
        require(key in expected and key not in cases, "Caso repetido ou fora do contrato.")
        require(row["run_id"], "Caso sem identificador de execução.")
        for field in ("source_rows", "common_start", "common_end", "holdout_start"):
            require(int(row[field]) == bounds[key[0]][field], "Caso usa outros limites de dados.")
        require(row["source_sha256"] == bounds[key[0]]["source_sha256"], "Caso usa outro dataset.")
        for field in ("model", "protocol", "release"):
            if row.get(field):
                require(row[field] == contract.get(field), "Metadados do caso divergem do contrato.")
        cases[key] = row
    require(
        set(cases) == expected and len({r["run_id"] for r in cases.values()}) == len(expected),
        "Matriz de casos incompleta ou com identificadores repetidos.",
    )

    def facts(name, metrics):
        result, seen = [], set()
        for row in table(name):
            key = coordinate(row)
            require(key in cases and row["run_id"] == cases[key]["run_id"], "Identidade de fato divergente.")
            require(row["metric"] in metrics, "Métrica fora do contrato.")
            identity = (*key, row["metric"])
            require(identity not in seen, "Fato repetido.")
            seen.add(identity)
            value = number(row["value"])
            normalized = {**row, "seed": key[2], "value": value}
            if name == "primary":
                require(
                    int(row["cohort_n"]) == bounds[key[0]]["common_end"] - bounds[key[0]]["common_start"],
                    "Denominador da métrica primária diverge da coorte declarada.",
                )
                if row["metric"] in {"accuracy", "macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"}:
                    require(value is None or 0 <= value <= 1, "Métrica de qualidade fora de 0..1.")
            if name == "resource_values":
                flag = row.get("source_fact_present", "true").lower()
                require(flag in ("true", "false"), "Presença do fato de recurso inválida.")
                require(flag != "false" or value is None, "Fato ausente não pode conter um valor.")
                normalized["source_fact_present"] = flag == "true"
                if row["metric"] in {"actual_total_fit_seconds", "update_count"}:
                    require(value is None or value >= 0, "Tempo ou contagem de atualização negativa.")
                if row["metric"] == "update_count":
                    require(value is None or value == int(value), "Contagem de atualização não inteira.")
            result.append(normalized)
        require(
            seen == {(*key, metric) for key in expected for metric in metrics},
            f"A tabela {name} precisa explicitar cada métrica e NULL ausente da matriz.",
        )
        return result

    primary = facts("primary", contract["primary_metrics"])
    resources = facts("resource_values", contract["resource_metrics"])

    def aggregate_declared(rows, keys):
        groups = {}
        for row in rows:
            key = tuple(row[k] for k in keys)
            require(row["seed"] not in groups.setdefault(key, {}), "Seed repetida no agregado.")
            groups[key][row["seed"]] = row["value"]
        result = []
        for key, observations in groups.items():
            require(set(observations) == set(seeds), "Agregado sem todas as seeds declaradas.")
            finite = [value for value in observations.values() if value is not None]
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "n_seeds_present": len(seeds),
                    "n_defined": len(finite),
                    "n_missing_or_undefined": len(seeds) - len(finite),
                    "sd_ddof": 1,
                    "mean": statistics.mean(finite) if finite else None,
                    "sample_sd": statistics.stdev(finite) if len(finite) > 1 else None,
                }
            )
        return result

    temporal_metrics = contract["temporal_metrics"]
    window_count = contract["windows_per_run"]
    require(type(window_count) is int and window_count > 0, "Número de janelas inválido.")
    windows = {}
    for row in table("windows"):
        key, window = coordinate(row), int(row["window"])
        require(key in cases and row["run_id"] == cases[key]["run_id"], "Identidade temporal inválida.")
        require(1 <= window <= window_count and window not in windows.setdefault(key, {}), "Janela repetida.")
        start, end, n = (int(row[k]) for k in ("start_seq", "end_seq", "n"))
        require(start < end and end - start == n, "Intervalo temporal inválido.")
        values = {metric: number(row[metric]) for metric in temporal_metrics}
        require(
            all(value is None or 0 <= value <= 1 for value in values.values()),
            "Escore temporal fora de 0..1.",
        )
        windows[key][window] = {"window": window, "start_seq": start, "end_seq": end, "n": n, **values}
    require(set(windows) == expected, "Cobertura temporal incompleta.")
    series, temporal = {}, []
    for key in expected:
        require(set(windows[key]) == set(range(1, window_count + 1)), "Faltam janelas declaradas.")
        previous, points = bounds[key[0]]["common_start"], []
        for window in range(1, window_count + 1):
            point = windows[key][window]
            require(point["start_seq"] == previous, "Janelas descontínuas.")
            previous = point["end_seq"]
            points.append(point)
            for metric in temporal_metrics:
                temporal.append(
                    {
                        "dataset": key[0],
                        "policy": key[1],
                        "seed": key[2],
                        "run_id": cases[key]["run_id"],
                        "window": window,
                        "start_seq": point["start_seq"],
                        "end_seq": point["end_seq"],
                        "n": point["n"],
                        "metric": metric,
                        "value": point[metric],
                    }
                )
        require(previous == bounds[key[0]]["common_end"], "Janela final não coincide com a coorte.")
        series[key] = points
    for dataset in names:
        for window in range(window_count):
            require(
                len(
                    {
                        tuple(series[key][window][k] for k in ("start_seq", "end_seq", "n"))
                        for key in expected
                        if key[0] == dataset
                    }
                )
                == 1,
                "Janelas não alinhadas entre políticas/seeds.",
            )

    holdout = []
    holdout_metrics = {"macro_f1", "pr_auc", "n"}
    holdout_seen = set()
    reported_seen = set()
    for row in table("reported_metrics"):
        key = coordinate(row)
        identity = (*key, row["metric_scope"], row["metric"])
        require(
            key in expected and row["run_id"] == cases[key]["run_id"],
            "Métrica reportada com identidade divergente.",
        )
        require(identity not in reported_seen, "Métrica reportada repetida.")
        reported_seen.add(identity)
        value = number(row["value"])
        raw_presence, raw_availability = row.get("source_fact_present"), row.get("availability")
        require(
            isinstance(raw_presence, str) and isinstance(raw_availability, str),
            "Fato reportado precisa declarar source_fact_present e availability.",
        )
        presence, availability = raw_presence.strip().lower(), raw_availability.strip().lower()
        require(presence in {"true", "false"}, "Presença de fato reportado inválida.")
        present = presence == "true"
        if availability == "defined":
            require(present and value is not None, "Fato definido exige presença e valor numérico.")
        elif availability in {"undefined", "source_null"}:
            require(present and value is None, "Fato indefinido exige presença e NULL explícito.")
        elif availability in {"missing", "missing_or_undefined"}:
            require(not present and value is None, "Fato ausente exige ausência e NULL explícito.")
        else:
            raise ValueError("Disponibilidade de fato reportado desconhecida.")
        if row["metric_scope"] != "holdout" or row["metric"] not in holdout_metrics:
            continue
        holdout_seen.add((*key, row["metric"]))
        if row["metric"] in {"macro_f1", "pr_auc"}:
            require(value is None or 0 <= value <= 1, "Métrica de qualidade do holdout fora de 0..1.")
        if row["metric"] == "n" and value is not None:
            require(
                value == bounds[key[0]]["source_rows"] - bounds[key[0]]["holdout_start"],
                "Denominador do holdout divergente.",
            )
        holdout.append(
            {
                "dataset": key[0],
                "policy": key[1],
                "seed": key[2],
                "metric": row["metric"],
                "value": value,
            }
        )
    require(
        holdout_seen == {(*key, metric) for key in expected for metric in holdout_metrics},
        "O diagnóstico de holdout exige macro_f1, pr_auc e n para cada caso, com NULL explícito se ausente.",
    )
    coverage = [
        {
            "dataset": d,
            "observacoes_fonte": bounds[d]["source_rows"],
            "politicas": len(policies),
            "seeds": len(seeds),
            "unidades": len(policies) * len(seeds),
            "inicio_coorte": bounds[d]["common_start"],
            "fim_exclusivo_coorte": bounds[d]["common_end"],
            "exemplos_comuns": bounds[d]["common_end"] - bounds[d]["common_start"],
        }
        for d in names
    ]
    return AthenaEvidence(
        pd.DataFrame(primary),
        pd.DataFrame(aggregate_declared(primary, ("dataset", "policy", "metric"))),
        pd.DataFrame(resources),
        pd.DataFrame(aggregate_declared(resources, ("dataset", "policy", "metric"))),
        pd.DataFrame(temporal),
        pd.DataFrame(aggregate_declared(temporal, ("dataset", "policy", "window", "metric"))),
        series,
        pd.DataFrame(aggregate_declared(holdout, ("dataset", "policy", "metric"))),
        pd.DataFrame(coverage),
        {
            "source": "Scientific CSV tables with declared contract and verified hashes",
            "validation_level": "contract_schema_and_hashes_only",
            "release": contract.get("release"),
            "protocol": contract["protocol"],
            "model": contract["model"],
            "datasets": names,
            "dataset_metadata": datasets,
            "policies": policies,
            "seeds": seeds,
            "cases": len(expected),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "native_mlflow_histories_requeried": False,
            "F109_audit_inherited": False,
        },
    )
