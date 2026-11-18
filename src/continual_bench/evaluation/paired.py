"""Paired common-event comparison over event-ID-tagged prediction ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from continual_bench.evaluation.stream_observations import BINARY_QUALITY


def _row_label(row: dict[str, Any]) -> bool:
    if "y" in row:
        return bool(row["y"])
    if "label" in row:
        return bool(row["label"])
    raise ValueError(f"prediction row missing label/y for event_id={row.get('event_id')!r}")


def _row_predicted_label(row: dict[str, Any]) -> bool:
    if "probability" in row and row["probability"] is not None:
        return float(row["probability"]) >= 0.5
    if "prediction" in row and row["prediction"] is not None:
        return bool(row["prediction"])
    raise ValueError(f"prediction row missing prediction/probability for event_id={row.get('event_id')!r}")


def _normalize_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    label = _row_label(row)
    if "probability" in row and row["probability"] is not None:
        probability = float(row["probability"])
    else:
        probability = 1.0 if _row_predicted_label(row) else 0.0
    normalized = dict(row)
    normalized["label"] = label
    normalized["probability"] = probability
    return normalized


def load_prediction_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an event-ID-tagged ``predictions.jsonl`` ledger."""
    ledger_path = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    prev_seq: int | None = None
    for line_no, raw_line in enumerate(ledger_path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"{ledger_path}:{line_no}: missing non-empty event_id")
        if event_id in seen_ids:
            raise ValueError(f"{ledger_path}:{line_no}: duplicate event_id {event_id!r}")
        seen_ids.add(event_id)
        seq = row.get("seq")
        if type(seq) is not int:
            raise ValueError(f"{ledger_path}:{line_no}: seq must be an integer")
        if prev_seq is not None and seq <= prev_seq:
            raise ValueError(f"{ledger_path}:{line_no}: non-increasing seq {seq} <= {prev_seq}")
        prev_seq = seq
        _row_label(row)
        rows.append(row)
    return rows


def verify_and_align_ledgers(
    ledgers: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Verify common event IDs, relative order, and ground-truth labels across runs."""
    if len(ledgers) < 2:
        raise ValueError("paired comparison requires at least two runs")
    names = list(ledgers.keys())
    by_name_index: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    for name, rows in ledgers.items():
        index: dict[str, tuple[int, dict[str, Any]]] = {}
        for pos, row in enumerate(rows):
            event_id = row.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(f"run {name!r} has row without event_id")
            if event_id in index:
                raise ValueError(f"run {name!r} has duplicate event_id {event_id!r}")
            index[event_id] = (pos, row)
        by_name_index[name] = index

    common_sets = [set(index.keys()) for index in by_name_index.values()]
    common_ids_set = set.intersection(*common_sets)
    if not common_ids_set:
        raise ValueError("no common event_ids across runs")

    ref_name = names[0]
    common_event_ids = [row["event_id"] for row in ledgers[ref_name] if row["event_id"] in common_ids_set]

    for name in names[1:]:
        candidate_order = [row["event_id"] for row in ledgers[name] if row["event_id"] in common_ids_set]
        if candidate_order != common_event_ids:
            raise ValueError(f"common event_id order mismatch between {ref_name!r} and {name!r}")

    for event_id in common_event_ids:
        ref_label = _row_label(by_name_index[ref_name][event_id][1])
        for name in names[1:]:
            cand_label = _row_label(by_name_index[name][event_id][1])
            if cand_label != ref_label:
                raise ValueError(f"label mismatch on event_id={event_id!r} between {ref_name!r} and {name!r}")

    aligned = {name: [by_name_index[name][event_id][1] for event_id in common_event_ids] for name in names}
    return common_event_ids, aligned


def compare_paired_ledgers(
    ledgers: dict[str, list[dict[str, Any]]],
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    """Compute paired quality metrics and McNemar contingency counts on common events."""
    common_ids, aligned = verify_and_align_ledgers(ledgers)
    names = list(aligned.keys())
    ref_name = reference or names[0]
    if ref_name not in aligned:
        raise ValueError(f"unknown reference run {ref_name!r}")

    normalized = {name: [_normalize_ledger_row(row) for row in rows] for name, rows in aligned.items()}
    metrics_by_run = {name: BINARY_QUALITY.collect(rows) for name, rows in normalized.items()}
    ref_rows = normalized[ref_name]
    ref_acc = metrics_by_run[ref_name]["accuracy"]

    pairwise: dict[str, Any] = {}
    for name in names:
        if name == ref_name:
            continue
        cand_rows = normalized[name]
        both_correct = ref_only = cand_only = both_wrong = 0
        for r_row, c_row in zip(ref_rows, cand_rows, strict=True):
            y = _row_label(r_row)
            r_ok = _row_predicted_label(r_row) == y
            c_ok = _row_predicted_label(c_row) == y
            if r_ok and c_ok:
                both_correct += 1
            elif r_ok and not c_ok:
                ref_only += 1
            elif not r_ok and c_ok:
                cand_only += 1
            else:
                both_wrong += 1
        discordant = ref_only + cand_only
        mcnemar_chi2 = ((abs(ref_only - cand_only) - 1) ** 2) / discordant if discordant > 0 else None
        cand_acc = metrics_by_run[name]["accuracy"]
        pairwise[name] = {
            "reference": ref_name,
            "common_events": len(common_ids),
            "accuracy_delta": cand_acc - ref_acc,
            "error_rate_delta": (1.0 - cand_acc) - (1.0 - ref_acc),
            "contingency": {
                "both_correct": both_correct,
                "reference_only_correct": ref_only,
                "candidate_only_correct": cand_only,
                "both_wrong": both_wrong,
            },
            "mcnemar_chi2_cc": mcnemar_chi2,
            "mcnemar_unavailable_reason": None if mcnemar_chi2 is not None else "no_discordant_pairs",
        }

    return {
        "common_events": len(common_ids),
        "first_event_id": common_ids[0],
        "last_event_id": common_ids[-1],
        "reference": ref_name,
        "runs": metrics_by_run,
        "pairwise": pairwise,
    }


def compare_run_directories(
    run_dirs: dict[str, str | Path],
    *,
    reference: str | None = None,
    verify_digest: bool = True,
) -> dict[str, Any]:
    """Load ``predictions.jsonl`` from each run directory, verify digests, and compare."""
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for name, directory in run_dirs.items():
        root = Path(directory)
        ledger_path = root / "predictions.jsonl"
        if verify_digest and (root / "report.json").is_file():
            report = json.loads((root / "report.json").read_text())
            expected = ((report.get("tracking") or {}).get("prediction_ledger") or {}).get("sha256")
            if expected is not None:
                actual = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"prediction_ledger sha256 mismatch for {name!r}")
        ledgers[name] = load_prediction_ledger(ledger_path)
    return compare_paired_ledgers(ledgers, reference=reference)
