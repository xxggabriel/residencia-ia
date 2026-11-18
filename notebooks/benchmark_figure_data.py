"""Evidence loaders for the Week 10 benchmark figures."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DATASET_LABELS = {
    "insects_abrupt_imbalanced": "Abrupto",
    "insects_incremental_abrupt_reoccurring_imbalanced": "Recorrente",
}
MODEL_LABELS = {"hoeffding_tree": "Hoeffding", "logistic": "Logística"}
POLICY_LABELS = {
    "none": "Congelado",
    "cara": "CARA",
    "upf": "UPF",
    "oracle": "Oracle",
    "drift": "ADWIN",
    "online": "Online",
}
RUN_KEY = ["dataset", "modelo", "politica", "seed"]
QUALITY = "observations_v1/quality/prequential/"
COST = "observations_v1/resources/cost_total"
UPDATES = "observations_v1/effort/updates"
ALARMS = "observations_v1/drift/alarm_count"
RETENTION = "observations_v1/retention/absolute_retention"
WINDOW_F1 = "observations_v1/windows/prequential/macro_f1"


@dataclass
class InsectsEvidence:
    seed: int | tuple[int, ...]
    captured_at: str | None
    runs: pd.DataFrame
    pending: pd.DataFrame
    paired: pd.DataFrame
    regimes: pd.DataFrame
    detector: pd.DataFrame
    windows: pd.DataFrame
    manifests: dict
    drift_artifacts: dict

    @property
    def capture_label(self):
        if self.captured_at is None:
            return "data da captura não informada"
        date = datetime.fromisoformat(self.captured_at).astimezone(ZoneInfo("America/Sao_Paulo"))
        return date.strftime("captura de %d/%m/%Y às %Hh%M BRT")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _manifests_by_dataset(runs, manifests):
    result = {}
    for dataset, group in runs.groupby("dataset"):
        first, *others = [manifests[run] for run in group.run]
        keys = ("variant", "raw_sha256", "target_transformation", "samples", "changepoints")
        for other in others:
            if any(first[key] != other[key] for key in keys):
                raise ValueError(
                    f"Manifestos incompatíveis em {dataset}. Use a mesma versão e ordem do fluxo."
                )
        result[dataset] = first
    return result


def _select_seed(runs, pending, seed):
    runs = runs.loc[runs.seed.eq(seed)].copy()
    pending = pending.loc[pending.seed.eq(seed)].copy()
    if runs.empty:
        raise ValueError(f"Nenhum run completo para a seed {seed}.")
    duplicates = runs.loc[runs.duplicated(RUN_KEY, keep=False), ["run", *RUN_KEY]]
    if not duplicates.empty:
        raise ValueError("Selecione um run canônico por combinação:\n" + duplicates.to_string(index=False))
    # A new successful reexecution fills the same slot as its earlier attempt.
    pending = pending.merge(runs[RUN_KEY], on=RUN_KEY, how="left", indicator=True)
    return runs, pending.loc[pending._merge.eq("left_only")].drop(columns="_merge")


def included_policies(model):
    policies = ("Congelado", "CARA", "UPF", "Oracle", "ADWIN", "Online")
    return policies[:-1] if model == "Hoeffding" else policies


def from_snapshot(folder: Path, seed: int) -> InsectsEvidence:
    """Read the archived report inputs without contacting MLflow."""
    data = folder / "dados"
    drift = read_json(data / f"drift-seed{seed}.json")
    if drift["seed"] != seed:
        raise ValueError("A seed do arquivo de drift diverge da seed solicitada.")
    all_runs = pd.read_csv(data / "runs.csv")
    all_runs = all_runs.loc[~(all_runs.modelo.eq("Hoeffding") & all_runs.politica.eq("Online"))]
    runs, pending = _select_seed(all_runs, all_runs.iloc[:0].copy(), seed)
    run_ids = set(runs.run)

    def selected_csv(name):
        rows = pd.read_csv(data / f"{name}-seed{seed}.csv")
        return rows.loc[rows.run.isin(run_ids)].copy()

    paired = selected_csv("metricas-pareadas")
    if set(paired.run) != run_ids:
        raise ValueError(f"O CSV pareado não cobre os runs selecionados da seed {seed}.")
    detector = selected_csv("detector")
    windows = selected_csv("janelas-f1")
    if set(windows.run) != run_ids or windows.duplicated(["run", "passo"]).any():
        raise ValueError("O histórico de F1 tem runs ausentes ou janelas duplicadas.")
    if set(detector.run) != set(drift["drift_artifacts"]):
        raise ValueError("Os artefatos de drift não correspondem aos runs do detector.")
    return InsectsEvidence(
        seed,
        drift["captured_at"],
        runs,
        pending,
        paired,
        selected_csv("regimes"),
        detector,
        windows,
        drift["manifests"],
        drift["drift_artifacts"],
    )


def _ledger(path, expected_rows):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.json as paj

    schema = pa.schema(
        [
            ("seq", pa.int64()),
            ("event_id", pa.string()),
            ("label", pa.bool_()),
            ("probability", pa.float64()),
        ]
    )
    table = paj.read_json(
        path,
        read_options=paj.ReadOptions(block_size=2**22),
        parse_options=paj.ParseOptions(explicit_schema=schema, unexpected_field_behavior="ignore"),
    )
    if len(table) != expected_rows or not len(table) or any(col.null_count for col in table.columns):
        raise ValueError(f"Cardinalidade ou campos do ledger incompatíveis em {path}.")
    events = table["event_id"]
    if pc.count_distinct(events).as_py() != len(table) or pc.any(pc.equal(events, "")).as_py():
        raise ValueError(f"Identificadores de evento vazios ou repetidos em {path}.")
    seq, label, score = (table[col].to_numpy() for col in ("seq", "label", "probability"))
    if (np.diff(seq) <= 0).any() or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError(f"Ordem de eventos ou probabilidades inválidas em {path}.")
    return seq, events, label, score


def _f1(labels, scores):
    from sklearn.metrics import f1_score

    return f1_score(labels, scores >= 0.5, labels=[False, True], average="macro", zero_division=0)


def _pair_dataset(group, ledgers, manifest):
    import pyarrow.compute as pc

    common = ledgers[group.run.iloc[0]][0]
    for run in group.run.iloc[1:]:
        common = np.intersect1d(common, ledgers[run][0], assume_unique=True)
    if not common.size:
        raise ValueError(f"Sem eventos em comum para {group.dataset.iloc[0]}.")

    bounds = [0, *manifest["changepoints"], manifest["samples"]]
    cuts = np.searchsorted(common, bounds)
    reference = None
    paired, regimes = [], []
    for row in group.itertuples():
        seq, events, labels, scores = ledgers[row.run]
        index = np.searchsorted(seq, common)
        ids, y, pred = pc.take(events, index), labels[index], scores[index]
        if reference is None:
            reference = (ids, y)
        elif not ids.equals(reference[0]) or not np.array_equal(y, reference[1]):
            raise ValueError(f"IDs ou rótulos divergentes nas mesmas posições em {row.run}.")
        identity = dict(run=row.run, dataset=row.dataset, modelo=row.modelo, politica=row.politica)
        paired.append(
            {
                **identity,
                "avaliados": len(common),
                "passo_inicial": int(common[0]),
                "passo_final": int(common[-1]),
                "f1": _f1(y, pred),
            }
        )
        for regime, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:], strict=True), 1):
            if stop > start:
                regimes.append(
                    {
                        **identity,
                        "regime": regime,
                        "avaliados": int(stop - start),
                        "positivos": int(y[start:stop].sum()),
                        "f1": _f1(y[start:stop], pred[start:stop]),
                    }
                )
    return paired, regimes


def from_mlflow(client, run_ids, seed: int, *, inventory_seeds=None) -> InsectsEvidence:
    """Read selected run metadata and artifacts, without changing the tracking server."""
    complete, pending = [], []
    metrics_by_run = {}
    inventory_seeds = inventory_seeds or {}
    for run_id in dict.fromkeys(run_ids):
        run = client.get_run(run_id)
        tags = run.data.tags
        declared_seed = tags.get("cb-observation-seed") or run.data.params.get("seed")
        listed_seed = inventory_seeds.get(run_id)
        if declared_seed is None and listed_seed is None:
            raise ValueError(f"Run {run_id} sem seed no MLflow. Inclua o run e sua seed no inventário.")
        selected_seed = int(declared_seed if declared_seed is not None else listed_seed)
        if listed_seed is not None and selected_seed != int(listed_seed):
            raise ValueError(f"Seed do inventário diverge do MLflow no run {run_id}.")
        if selected_seed != seed:
            continue
        row = dict(
            run=run_id,
            dataset=DATASET_LABELS[tags["cb-dataset"]],
            modelo=MODEL_LABELS[tags["cb-model"]],
            politica=POLICY_LABELS[tags["cb-policy"]],
            seed=selected_seed,
        )
        if row["politica"] not in included_policies(row["modelo"]):
            continue
        if run.info.status == "FINISHED" and tags.get("benchmark_complete") == "true":
            metrics_by_run[run_id] = run.data.metrics
            complete.append(row)
        else:
            pending.append(row)

    fields = ["run", *RUN_KEY]
    runs, pending = _select_seed(
        pd.DataFrame(complete, columns=fields), pd.DataFrame(pending, columns=fields), seed
    )
    runs["custo_registrado"] = runs.run.map(lambda r: metrics_by_run[r].get(COST, np.nan))
    runs["atualizacoes"] = runs.run.map(lambda r: int(metrics_by_run[r][UPDATES]))
    runs["retencao"] = runs.run.map(lambda r: metrics_by_run[r].get(RETENTION, np.nan))
    sources, drift = {}, {}
    paired, regimes, detector, windows = [], [], [], []
    for dataset, group in runs.groupby("dataset", sort=False):
        manifests, ledgers = {}, {}
        for row in group.itertuples():
            with TemporaryDirectory(prefix="insects-figures-") as run_dir:

                def download(path):
                    return Path(client.download_artifacts(row.run, path, run_dir))

                m = read_json(download("manifest.json"))
                manifests[row.run] = m
                metrics = metrics_by_run[row.run]
                ledger = _ledger(download("predictions.jsonl"), int(metrics[QUALITY + "evaluated_examples"]))
                if ledger[0][0] < 0 or ledger[0][-1] >= int(m["samples"]):
                    raise ValueError(f"Ledger fora dos limites do fluxo em {row.run}.")
                if not np.isclose(
                    _f1(ledger[2], ledger[3]), metrics[QUALITY + "macro_f1"], atol=1e-9, rtol=0
                ):
                    raise ValueError(f"F1 do ledger não confere com o MLflow em {row.run}.")
                ledgers[row.run] = ledger
                windows.extend(
                    dict(
                        run=row.run,
                        dataset=dataset,
                        modelo=row.modelo,
                        politica=row.politica,
                        passo=metric.step,
                        f1=metric.value,
                    )
                    for metric in client.get_metric_history(row.run, WINDOW_F1)
                )
                if row.politica == "ADWIN":
                    drift[row.run] = {
                        "detection_timeline": read_json(download("data/drift/timeline.json")),
                        "recovery": read_json(download("data/drift/recovery_delay.json")),
                    }
                    detector.append(
                        dict(
                            run=row.run, dataset=row.dataset, modelo=row.modelo, alarmes=int(metrics[ALARMS])
                        )
                    )
        sources.update(_manifests_by_dataset(group, manifests))
        aligned, by_regime = _pair_dataset(group, ledgers, sources[dataset])
        paired.extend(aligned)
        regimes.extend(by_regime)
    return InsectsEvidence(
        seed,
        datetime.now(timezone.utc).isoformat(),
        runs,
        pending,
        pd.DataFrame(paired),
        pd.DataFrame(regimes),
        pd.DataFrame(detector, columns=["run", "dataset", "modelo", "alarmes"]),
        pd.DataFrame(windows, columns=["run", "dataset", "modelo", "politica", "passo", "f1"]),
        sources,
        drift,
    )


def from_campaign(folder: Path) -> InsectsEvidence:
    """Load audited per-seed inputs and aggregate each completed configuration."""
    data = folder / "dados"
    drift = read_json(data / "drift-por-seed.json")
    key = ["dataset", "modelo", "politica"]
    runs = pd.read_csv(data / "runs.csv")
    runs = runs.loc[~(runs.modelo.eq("Hoeffding") & runs.politica.eq("Online"))].copy()
    seeds = tuple(sorted(runs.seed.unique()))
    if (
        not len(seeds)
        or runs.duplicated(RUN_KEY).any()
        or not runs.groupby(key).seed.nunique().eq(len(seeds)).all()
    ):
        raise ValueError("Cada configuração concluída deve ter um run por seed.")

    def read(name):
        frame = pd.read_csv(data / f"{name}-por-seed.csv")
        frame = frame.loc[frame.run.isin(runs.run)].copy()
        return frame

    paired = read("metricas-pareadas")
    regimes = read("regimes")
    windows = read("janelas-f1")
    if set(paired.run) != set(runs.run) or paired.run.duplicated().any():
        raise ValueError("Os dados pareados devem cobrir cada run uma vez.")
    if (
        not paired.groupby("dataset")[["avaliados", "passo_inicial", "passo_final", "positivos"]]
        .nunique()
        .eq(1)
        .all()
        .all()
    ):
        raise ValueError("Os runs de cada fluxo devem usar o mesmo recorte pareado.")
    if not regimes.groupby(["dataset", "regime"])[["avaliados", "positivos"]].nunique().eq(1).all().all():
        raise ValueError("Os runs devem usar os mesmos eventos por regime.")
    if windows.duplicated(["run", "passo"]).any():
        raise ValueError("Cada run deve ter uma única métrica por janela.")

    def means(frame, fields):
        return frame.groupby(key, as_index=False)[fields].mean()

    campaign_runs = means(runs, ["custo_registrado", "atualizacoes", "retencao"])
    campaign_pairs = means(paired, ["avaliados", "positivos", "f1", "ap", "acuracia", "recall", "precisao"])
    campaign_pairs["f1_dp"] = paired.groupby(key).f1.std().to_numpy()
    for frame in (campaign_runs, campaign_pairs):
        frame["run"] = frame[key].agg("|".join, axis=1)

    window_means = windows.groupby([*key, "passo"], as_index=False).f1.agg(f1="mean", dp="std", n="size")
    window_means = window_means.loc[window_means.n.eq(len(seeds))]
    if set(map(tuple, window_means[key].drop_duplicates().to_numpy())) != set(
        map(tuple, runs.loc[runs.modelo.eq("Logística"), key].drop_duplicates().to_numpy())
    ):
        raise ValueError("Faltam janelas comuns às seeds para uma política da regressão logística.")

    detector = runs.loc[
        runs.politica.eq("ADWIN"), ["run", "dataset", "modelo", "seed", "atualizacoes"]
    ].copy()
    if set(detector.run) != set(drift["drift_artifacts"]):
        raise ValueError("Os artefatos de drift devem cobrir todos os runs ADWIN.")
    return InsectsEvidence(
        seeds,
        None,
        campaign_runs,
        runs.iloc[:0],
        campaign_pairs,
        regimes.groupby([*key, "regime"], as_index=False)[["avaliados", "positivos", "f1"]].mean(),
        detector,
        window_means,
        drift["manifests"],
        drift["drift_artifacts"],
    )
