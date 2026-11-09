"""Point-in-time feature materialization and temporal-leakage checks for the Feature Store path.

The Zia social features are computed by `prepare()` from information strictly before each
edge's date. The offline Feature Store holds the same values with `event_time` as the record
time; materializing a training snapshot must pick, per entity, the latest record whose time
is not after the prediction instant, and must reproduce the prepared features exactly.
"""

import bisect


class TemporalLeakage(ValueError):
    pass


def point_in_time(requests, records, *, entity="entity_id", time_key="event_time"):
    """As-of join: for each (entity, time) request, the latest record with record time ≤ time."""
    index = {}
    for record in sorted(records, key=lambda r: (r[entity], r[time_key])):
        index.setdefault(record[entity], ([], []))
        times, rows = index[record[entity]]
        times.append(record[time_key])
        rows.append(record)
    joined = []
    for request in requests:
        times, rows = index.get(request[entity], ([], []))
        position = bisect.bisect_right(times, request[time_key]) - 1
        match = rows[position] if position >= 0 else None
        if match is not None and match[time_key] > request[time_key]:
            raise TemporalLeakage("as-of join returned a future record")
        joined.append({**request, "features": match})
    return joined


def assert_no_leakage(rows, *, prediction_time="event_time", feature_time="feature_time"):
    future = [
        row for row in rows if row.get(feature_time) is not None and row[feature_time] > row[prediction_time]
    ]
    if future:
        raise TemporalLeakage(f"{len(future)} rows use features recorded after the prediction time")
    return len(rows)


def parity(prepared_rows, materialized_rows, features, *, key="event_id", tolerance=0.0) -> dict:
    """Prepared Parquet vs Feature Store materialization: same keys, same values."""
    prepared = {row[key]: row for row in prepared_rows}
    materialized = {row[key]: row for row in materialized_rows}
    missing = sorted(set(prepared) - set(materialized))
    extra = sorted(set(materialized) - set(prepared))
    mismatches = []
    for identifier in sorted(set(prepared) & set(materialized)):
        for name in features:
            a, b = float(prepared[identifier][name]), float(materialized[identifier][name])
            if abs(a - b) > tolerance:
                mismatches.append({key: identifier, "feature": name, "prepared": a, "materialized": b})
    return {
        "rows": len(prepared),
        "missing": missing[:20],
        "extra": extra[:20],
        "mismatches": mismatches[:20],
        "missing_count": len(missing),
        "extra_count": len(extra),
        "mismatch_count": len(mismatches),
        "ok": not missing and not extra and not mismatches,
    }
