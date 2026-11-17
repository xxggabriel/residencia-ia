"""Publish the additive observations without changing existing MLflow coordinates."""

import hashlib
import json
import math
from pathlib import Path

from .retraining_pages import HTML_REVISION, PAGE_PATHS
from .storage import atomic_write

ARTIFACT_ROOT = "tracking-observations"
PREFIX = "observations_v1"


def scalar_points(values, prefix, step):
    for name, value in values.items():
        if type(value) in (int, float) and math.isfinite(value):
            yield f"{prefix}/{name}", step, value


def parent_observation_points(report):
    if not (data := report.get("tracking_observations")):
        return
    for scope, values in data["scopes"].items():
        yield from scalar_points(values, f"{PREFIX}/quality/{scope}", 0)
    for section in ("effort", "resources", "drift", "retention"):
        yield from scalar_points(data[section], f"{PREFIX}/{section}", 0)
    for row in data["windows"]:
        yield from scalar_points(row["quality"], f"{PREFIX}/windows/{row['scope']}", row["step"])
    for row in data["retention"].get("series") or ():
        yield from scalar_points(
            {key: value for key, value in row.items() if key != "step"},
            f"{PREFIX}/retention/checkpoints",
            row["step"],
        )


def fit_observation_points(fit, *, include_metadata=True):
    for axis in ("step", "epoch"):
        rows = fit.get(f"observation_{axis}s")
        if rows is None:
            continue
        path = f"{PREFIX}/{axis}"
        for index, row in enumerate(rows, 1):
            if type(row[axis]) is not int or row[axis] != index:
                raise ValueError(f"invalid {axis} sequence for fit {fit['fit_id']}")
            yield from scalar_points(row["metrics"], path, index)
            yield from scalar_points(
                {key: row.get(key) for key in ("first_step", "last_step")},
                path,
                index,
            )
            for scope, values in row.get("scopes", {}).items():
                yield from scalar_points(values, f"{path}/{scope}", index)
    if include_metadata:
        for section in ("effort", "resources"):
            yield from scalar_points(fit.get(f"observation_{section}", {}), f"{PREFIX}/{section}", 0)
        for scope, values in fit.get("observation_scopes", {}).items():
            yield from scalar_points(values, f"{PREFIX}/quality/{scope}", 0)


def observation_tags(report):
    if not (data := report.get("tracking_observations")):
        return {}
    tags = {"cb-observation-version": data["version"], "cb-observation-task": data["task"]}
    for key in ("dataset", "dataset_version", "model", "policy", "seed", "environment", "protocol"):
        value = data["lineage"].get(key)
        if isinstance(value, dict):
            value = next(
                (value[k] for k in ("name", "family", "execution", "platform") if value.get(k)), None
            )
        if type(value) in (str, int, float):
            tags[f"cb-observation-{key.replace('_', '-')}"] = str(value)[:500]
    return tags


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()


def _fit_source_checksum(directory):
    history = Path(directory) / "training-history.json"
    if history.is_file():
        with history.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    return None


def _bind_plot_manifest(path, bindings, fit_id=None):
    if not path.is_file():
        return
    document = json.loads(path.read_bytes())
    parent_id = bindings["parent_mlflow_run_id"]
    children = bindings["children"]
    document["parent_run_id"] = parent_id
    if fit_id is not None:
        document["run_id"] = children[fit_id]
        fit_prefix = f"fits/{fit_id}/"
        document["artifacts"] = [path.removeprefix(fit_prefix) for path in document["artifacts"]]
    for entry in document["figures"]:
        fit = entry.get("fit_id")
        entry["parent_run_id"] = parent_id
        entry["run_id"] = children.get(fit) if fit else parent_id
        prefix = f"fits/{fit}/" if fit else ""
        if entry["path"]:
            entry["artifact_path"] = f"{ARTIFACT_ROOT}/" + entry["path"].removeprefix(prefix)
        source = entry.get("source") or {}
        if fit and isinstance(source.get("artifact"), str):
            source["artifact"] = source["artifact"].removeprefix(prefix)
    atomic_write(path, _json_bytes(document))


def write_observation_evidence(directory, report, parent_run_id=None, children=()):
    if not (data := report.get("tracking_observations")):
        return None
    from .training_figures import write_observation_figures

    root = Path(directory) / ARTIFACT_ROOT
    payload = _json_bytes(data)
    manifest_path = root / "plots-manifest.json"
    signature = {
        "observations_sha256": hashlib.sha256(payload).hexdigest(),
        "training_history_sha256": _fit_source_checksum(directory),
        "html_revision": HTML_REVISION,
    }
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else {}
    if manifest.get("source_checksums") != signature or not (root / "observations.json").is_file():
        # The signature is committed after the plots and canonical data.
        manifest = write_observation_figures(root, report)
        atomic_write(root / "observations.json", payload)
        manifest["source_checksums"] = signature
        for fit in report.get("training_history", ()):
            summary = fit.get("summary") or {}
            missing = {}
            reasons = {"step": "step_statistics_not_observed", "epoch": "no_native_epoch_axis"}
            for axis, default in reasons.items():
                if not fit.get(f"observation_{axis}s"):
                    reason = summary.get(f"{axis}s_unavailable_reason", default)
                    missing[axis] = {"value": None, "reason": reason}
            fit_root = root / "fits" / fit["fit_id"]
            if fit_root.is_dir():
                atomic_write(fit_root / "availability.json", _json_bytes(missing))
        atomic_write(manifest_path, _json_bytes(manifest))
    if parent_run_id is not None:
        bindings = {
            "version": data["version"],
            "parent_mlflow_run_id": parent_run_id,
            "children": {child["fit_id"]: child["run_id"] for child in children},
        }
        atomic_write(root / "mlflow-lineage.json", _json_bytes(bindings))
        _bind_plot_manifest(manifest_path, bindings)
        lineage = _json_bytes(bindings).replace(b"<", rb"\u003c")
        for relative in PAGE_PATHS:
            page = root / relative
            if page.is_file():
                marker = b'<script type="application/json" id="run-lineage">'
                start, separator, tail = page.read_bytes().partition(marker)
                if separator:
                    suffix = tail.split(b"</script>", 1)[1]
                    atomic_write(page, start + separator + lineage + b"</script>" + suffix)
        for child in children:
            fit_root = root / "fits" / child["fit_id"]
            if fit_root.is_dir():
                _bind_plot_manifest(fit_root / "plots-manifest.json", bindings, child["fit_id"])
                atomic_write(
                    fit_root / "mlflow-lineage.json",
                    _json_bytes(
                        {
                            "version": data["version"],
                            "parent_mlflow_run_id": parent_run_id,
                            "child_mlflow_run_id": child["run_id"],
                            "fit_id": child["fit_id"],
                        }
                    ),
                )
    return root


def publish_sdk_observations(client, run_id, directory, report, children):
    from mlflow.entities import RunTag

    from .metric_tracking import log_metric_points

    root = write_observation_evidence(directory, report, run_id, children)
    if root is None:
        return None
    log_metric_points(client, run_id, parent_observation_points(report))
    client.log_batch(
        run_id, tags=[RunTag(key, value) for key, value in observation_tags(report).items()], synchronous=True
    )
    for path in sorted(root.iterdir()):
        if path.is_file():
            client.log_artifact(run_id, str(path), artifact_path=ARTIFACT_ROOT)
        elif path.name != "fits":
            client.log_artifacts(run_id, str(path), artifact_path=f"{ARTIFACT_ROOT}/{path.name}")
    for child in children:
        folder = root / "fits" / child["fit_id"]
        if folder.is_dir():
            client.log_artifacts(child["run_id"], str(folder), artifact_path=ARTIFACT_ROOT)
    return f"{ARTIFACT_ROOT}/plots-manifest.json"
