"""Unified stream dataset loader supporting CARA, UPF, and custom protocols."""

from __future__ import annotations

import csv
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.sparse import hstack, issparse
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder

from .arff import iter_arff

_STREAM_SOURCES: dict[str, Callable[..., Any]] = {}

_ALIASES: dict[str, tuple[str, ...]] = {
    "airlines": ("airlines", "airplanes"),
    "airplanes": ("airplanes", "airlines"),
    "electricity": ("electricity", "elecNormNew"),
}

_ELECTRICITY_FEATURES = ("period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer")
_AIRLINES_NOMINAL = ("Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek")
_AIRLINES_NUMERIC = ("Time", "Length")


@dataclass
class BatchSplit:
    """Single chronological batch split into train, test, and query subsets."""

    batch_index: int
    X_train: Any
    y_train: np.ndarray
    X_test: Any
    y_test: np.ndarray
    X_query: Any
    y_query: np.ndarray
    is_offline: bool


class StreamDatasetSpec(BaseModel):
    """Specification for loading and splitting a continual stream dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    protocol: Literal["cara", "upf", "custom"] = "cara"
    source_path: str | None = None
    batch_size: int | None = None
    num_batches: int | None = None
    offline_batches: int = 25
    online_batches: int = 75
    split_ratio: float = 0.9
    query_ratio: float = 0.1
    use_test_as_queries: bool | None = None
    causal_preprocessing: bool = False
    staleness_features: bool = True
    seed: int = 42


def register_stream_source(name: str, loader_fn: Callable[..., Any]) -> None:
    """Register a custom stream dataset loader under ``name``."""
    _STREAM_SOURCES[name] = loader_fn


def _candidate_names(name: str) -> tuple[str, ...]:
    base = Path(name).stem if Path(name).suffix else name
    return _ALIASES.get(base.lower(), (base,))


def resolve_dataset_path(
    name: str,
    source_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve dataset file across explicit path, env var, cache, and local directories."""
    extensions = (".arff", ".csv", ".parquet", ".npz", ".npy")
    names = _candidate_names(name)

    if source_path is not None:
        explicit = Path(source_path).expanduser()
        if explicit.is_file():
            return explicit
        if explicit.is_dir():
            for cand in names:
                for ext in extensions:
                    p = explicit / f"{cand}{ext}"
                    if p.is_file():
                        return p
        raise FileNotFoundError(f"Dataset source_path not found: {source_path}")

    search_dirs: list[Path] = []
    if cache_dir is not None:
        search_dirs.append(Path(cache_dir).expanduser())
    env_dir = os.environ.get("BENCH_DATASETS")
    if env_dir:
        search_dirs.append(Path(env_dir).expanduser())
    search_dirs.extend(
        [
            Path.home() / ".cache" / "continual-bench" / "datasets",
            Path("state/datasets"),
            Path("datasets"),
        ]
    )

    searched: list[str] = []
    for directory in search_dirs:
        if not directory.exists():
            searched.append(str(directory))
            continue
        for cand in names:
            direct = directory / cand
            if direct.is_file():
                return direct
            for ext in extensions:
                for path in (directory / f"{cand}{ext}", directory / cand / f"{cand}{ext}"):
                    searched.append(str(path))
                    if path.is_file():
                        return path

    raise FileNotFoundError(
        f"Could not locate dataset {name!r}; searched: {', '.join(searched)}"
    )


def _canonical_name(name: str) -> str:
    lower = name.lower()
    if lower in ("airplanes", "airlines"):
        return "airlines"
    if lower in ("electricity", "elecnormnew"):
        return "electricity"
    return lower


def _parse_binary_label(raw: Any) -> int:
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    text = str(raw).strip().strip("'\"").upper()
    if text in ("UP", "1", "TRUE", "YES", "1.0"):
        return 1
    if text in ("DOWN", "0", "FALSE", "NO", "0.0"):
        return 0
    return int(float(text))


def _load_electricity_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".arff":
        rows, labels = [], []
        started = False
        has_header_cols = False
        col_indices: list[int] | None = None
        label_idx = -1
        attr_names: list[str] = []
        with path.open(encoding="utf-8", newline="") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line or line.startswith("%"):
                    continue
                lower = line.lower()
                if not started:
                    if lower.startswith("@attribute "):
                        parts = line.split(None, 2)
                        if len(parts) >= 2:
                            attr_names.append(parts[1].strip("'\""))
                    elif lower == "@data":
                        started = True
                        if all(f in attr_names for f in _ELECTRICITY_FEATURES):
                            has_header_cols = True
                            col_indices = [attr_names.index(f) for f in _ELECTRICITY_FEATURES]
                            label_idx = attr_names.index("class") if "class" in attr_names else -1
                    continue
                fields = [v.strip().strip("'\"") for v in line.split(",")]
                if has_header_cols and col_indices is not None:
                    rows.append([float(fields[i]) for i in col_indices])
                    labels.append(_parse_binary_label(fields[label_idx]))
                else:
                    # Standard OpenML 151 order: date, day, period..transfer, class
                    feat_slice = fields[2:8] if len(fields) >= 9 else fields[:6]
                    rows.append([float(v) for v in feat_slice])
                    labels.append(_parse_binary_label(fields[-1]))
        return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)

    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            first = next(reader)
            lower_first = [c.strip().lower() for c in first]
            if all(f in lower_first for f in _ELECTRICITY_FEATURES):
                col_indices = [lower_first.index(f) for f in _ELECTRICITY_FEATURES]
                label_idx = lower_first.index("class") if "class" in lower_first else -1
                data_rows = reader
                rows, labels = [], []
                for fields in data_rows:
                    if not fields:
                        continue
                    rows.append([float(fields[i]) for i in col_indices])
                    labels.append(_parse_binary_label(fields[label_idx]))
                return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)

            # Either header without standard names or no header
            rows, labels = [], []
            try:
                float(first[2] if len(first) >= 9 else first[0])
                all_rows = [first, *reader]
            except (ValueError, IndexError):
                all_rows = list(reader)
            for fields in all_rows:
                if not fields:
                    continue
                feat_slice = fields[2:8] if len(fields) >= 9 else fields[:6]
                rows.append([float(v) for v in feat_slice])
                labels.append(_parse_binary_label(fields[-1]))
            return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)

    return _load_generic_raw(path)


def _normalize_cat_str(val: Any) -> str:
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val).strip()


def _load_airlines_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nominal_str_matrix [N, 5], numeric_matrix [N, 2], labels [N])."""
    suffix = path.suffix.lower()
    if suffix == ".arff":
        nominals: list[list[str]] = []
        numerics: list[list[float]] = []
        labels: list[int] = []
        for row in iter_arff(path):
            nominals.append([_normalize_cat_str(row[c]) for c in _AIRLINES_NOMINAL])
            numerics.append([float(row[c]) for c in _AIRLINES_NUMERIC])
            labels.append(_parse_binary_label(row["Delay"]))
        return (
            np.asarray(nominals, dtype=object),
            np.asarray(numerics, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
        )

    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = [c.strip() for c in next(reader)]
            nom_idx = [header.index(c) for c in _AIRLINES_NOMINAL]
            num_idx = [header.index(c) for c in _AIRLINES_NUMERIC]
            label_idx = header.index("Delay") if "Delay" in header else len(header) - 1
            nominals, numerics, labels = [], [], []
            for row in reader:
                if not row:
                    continue
                nominals.append([_normalize_cat_str(row[i]) for i in nom_idx])
                numerics.append([float(row[i]) for i in num_idx])
                labels.append(_parse_binary_label(row[label_idx]))
        return (
            np.asarray(nominals, dtype=object),
            np.asarray(numerics, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
        )

    raise ValueError(f"Unsupported airlines file format: {path.suffix}")


def _load_generic_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            x_key = "X" if "X" in data else "x"
            y_key = "y" if "y" in data else "Y"
            return np.asarray(data[x_key]), np.asarray(data[y_key], dtype=np.int64)
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr[:, :-1], dtype=np.float64), np.asarray(arr[:, -1], dtype=np.int64)
    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return _dataframe_to_xy(df)
    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        return _dataframe_to_xy(df)
    if suffix == ".arff":
        rows = list(iter_arff(path))
        if not rows:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.int64)
        keys = list(rows[0].keys())
        target_key = keys[-1]
        feat_keys = keys[:-1]
        codes: dict[str, dict[str, int]] = {}
        x_rows, y_rows = [], []
        for row in rows:
            vals = []
            for k in feat_keys:
                v = row[k]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                else:
                    mapping = codes.setdefault(k, {})
                    vals.append(float(mapping.setdefault(str(v), len(mapping))))
            x_rows.append(vals)
            y_rows.append(_parse_binary_label(row[target_key]))
        return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)
    raise ValueError(f"Unsupported dataset file format: {path.suffix}")


def _dataframe_to_xy(df: Any) -> tuple[np.ndarray, np.ndarray]:
    target_col = next(
        (c for c in ("label", "target", "y", "class", "Delay") if c in df.columns),
        df.columns[-1],
    )
    y_series = df[target_col]
    x_df = df.drop(columns=[target_col])
    for col in x_df.columns:
        if x_df[col].dtype == object or str(x_df[col].dtype) == "category":
            x_df[col] = x_df[col].astype("category").cat.codes
    x = x_df.to_numpy(dtype=np.float64)
    if y_series.dtype == object or str(y_series.dtype) == "category":
        y = np.asarray([_parse_binary_label(v) for v in y_series], dtype=np.int64)
    else:
        y = y_series.to_numpy(dtype=np.int64)
    return x, y


def _effective_protocol_params(spec: StreamDatasetSpec) -> tuple[int, int, float, float]:
    """Return (offline_batches, total_batches, split_ratio, query_ratio) applying protocol defaults."""
    fields_set = spec.model_fields_set
    if spec.protocol == "upf":
        offline = spec.offline_batches if "offline_batches" in fields_set else 7
        if spec.num_batches is not None:
            total = spec.num_batches
        elif "online_batches" in fields_set or "offline_batches" in fields_set:
            online = spec.online_batches if "online_batches" in fields_set else 8
            total = offline + online
        else:
            total = 15
        split_ratio = spec.split_ratio if "split_ratio" in fields_set else 0.8
        query_ratio = spec.query_ratio if "query_ratio" in fields_set else 0.1
        return offline, total, split_ratio, query_ratio

    offline = spec.offline_batches
    if spec.num_batches is not None:
        total = spec.num_batches
    else:
        total = spec.offline_batches + spec.online_batches
    return offline, total, spec.split_ratio, spec.query_ratio


def _partition_into_batches(
    X: Any,
    y: np.ndarray,
    total_batches: int,
    batch_size: int | None,
    protocol: str,
    seed: int,
) -> list[tuple[Any, np.ndarray]]:
    n_rows = X.shape[0]
    if protocol == "upf" or batch_size is not None:
        size = batch_size if batch_size is not None else max(1, n_rows // total_batches)
        needed = total_batches * size
        if needed > n_rows:
            raise ValueError(
                f"Stream has {n_rows} rows, fewer than {total_batches} * {size} = {needed}"
            )
        max_start = n_rows - needed
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return [
            (X[start + i * size : start + (i + 1) * size], y[start + i * size : start + (i + 1) * size])
            for i in range(total_batches)
        ]

    # CARA / equal contiguous partitioning across the full stream
    indices = np.array_split(np.arange(n_rows), total_batches)
    return [(X[idx], y[idx]) for idx in indices]


def _resolve_use_test_as_queries(
    spec: StreamDatasetSpec,
    split_ratio: float,
    query_ratio: float,
) -> bool:
    """Resolve whether queries are drawn directly from each batch's held-out test split."""
    if spec.use_test_as_queries is not None:
        return bool(spec.use_test_as_queries)
    if spec.protocol == "cara":
        if "query_ratio" in spec.model_fields_set and not math.isclose(
            query_ratio, 1.0 - split_ratio, abs_tol=1e-9
        ):
            return False
        return True
    return False


def _make_batch_splits(
    raw_batches: list[tuple[Any, np.ndarray]],
    offline_batches: int,
    split_ratio: float,
    query_ratio: float,
    seed: int,
    *,
    use_test_as_queries: bool = False,
    protocol: str = "custom",
) -> list[BatchSplit]:
    splits: list[BatchSplit] = []
    for i, (xb, yb) in enumerate(raw_batches):
        n = xb.shape[0]
        rng = np.random.default_rng(seed + i)
        perm = rng.permutation(n)
        cut = int(round(n * split_ratio))
        cut = max(1, min(n - 1, cut)) if n >= 2 else n
        train_idx = perm[:cut]
        test_idx = perm[cut:]

        if use_test_as_queries:
            query_idx = test_idx
        elif query_ratio > 0.0 and n > 0:
            n_query = max(1, int(round(n * query_ratio)))
            if protocol == "cara" and n_query <= test_idx.shape[0]:
                query_idx = rng.choice(test_idx, size=n_query, replace=False)
            else:
                n_query = min(n, n_query)
                query_idx = rng.choice(n, size=n_query, replace=False)
        else:
            query_idx = np.empty((0,), dtype=np.int64)

        splits.append(
            BatchSplit(
                batch_index=i,
                X_train=xb[train_idx],
                y_train=yb[train_idx],
                X_test=xb[test_idx],
                y_test=yb[test_idx],
                X_query=xb[query_idx],
                y_query=yb[query_idx],
                is_offline=(i < offline_batches),
            )
        )
    return splits


def _apply_scaler(scaler_cls: Any, X: Any, offline_rows: int, causal: bool) -> Any:
    scaler = scaler_cls()
    if causal and 0 < offline_rows < X.shape[0]:
        scaler.fit(X[:offline_rows])
        return scaler.transform(X)
    return scaler.fit_transform(X)


def load_stream_batches(
    spec: StreamDatasetSpec,
    cache_dir: Path | str | None = None,
) -> list[BatchSplit]:
    """Load a stream dataset and return chronological ``BatchSplit`` batches."""
    canonical = _canonical_name(spec.name)
    offline_batches, total_batches, split_ratio, query_ratio = _effective_protocol_params(spec)
    use_test_as_queries = _resolve_use_test_as_queries(spec, split_ratio, query_ratio)

    if spec.name in _STREAM_SOURCES or canonical in _STREAM_SOURCES:
        loader = _STREAM_SOURCES.get(spec.name) or _STREAM_SOURCES[canonical]
        sig = inspect.signature(loader)
        if len(sig.parameters) >= 2:
            result = loader(spec, cache_dir)
        elif len(sig.parameters) == 1:
            result = loader(spec)
        else:
            result = loader()
        if isinstance(result, list) and result and isinstance(result[0], BatchSplit):
            return result
        if isinstance(result, list) and result and isinstance(result[0], tuple):
            return _make_batch_splits(
                result,
                offline_batches,
                split_ratio,
                query_ratio,
                spec.seed,
                use_test_as_queries=use_test_as_queries,
                protocol=spec.protocol,
            )
        if isinstance(result, tuple) and len(result) == 2:
            X_raw, y_raw = result
            raw_batches = _partition_into_batches(
                X_raw,
                np.asarray(y_raw, dtype=np.int64),
                total_batches,
                spec.batch_size,
                spec.protocol,
                spec.seed,
            )
            return _make_batch_splits(
                raw_batches,
                offline_batches,
                split_ratio,
                query_ratio,
                spec.seed,
                use_test_as_queries=use_test_as_queries,
                protocol=spec.protocol,
            )
        raise ValueError(f"Custom stream loader for {spec.name!r} returned unsupported type {type(result)}")

    path = resolve_dataset_path(spec.name, source_path=spec.source_path, cache_dir=cache_dir)

    if canonical == "electricity":
        X, y = _load_electricity_raw(path)
        if spec.protocol == "cara":
            offline_rows = int(X.shape[0] * (offline_batches / max(1, total_batches)))
            X = _apply_scaler(MinMaxScaler, X, offline_rows, spec.causal_preprocessing)
            raw_batches = _partition_into_batches(
                X, y, total_batches, spec.batch_size, "cara", spec.seed
            )
        elif spec.protocol == "upf":
            batch_size = spec.batch_size or 2000
            raw_batches = _partition_into_batches(
                X, y, total_batches, batch_size, "upf", spec.seed
            )
        else:
            raw_batches = _partition_into_batches(
                X, y, total_batches, spec.batch_size, "custom", spec.seed
            )
        return _make_batch_splits(
            raw_batches,
            offline_batches,
            split_ratio,
            query_ratio,
            spec.seed,
            use_test_as_queries=use_test_as_queries,
            protocol=spec.protocol,
        )

    if canonical == "airlines":
        nominals, numerics, y = _load_airlines_columns(path)
        if spec.protocol == "cara":
            offline_rows = int(numerics.shape[0] * (offline_batches / max(1, total_batches)))
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            if spec.causal_preprocessing and 0 < offline_rows < numerics.shape[0]:
                encoder.fit(nominals[:offline_rows])
                encoded_nom = encoder.transform(nominals)
            else:
                encoded_nom = encoder.fit_transform(nominals)
            X = hstack([encoded_nom, numerics]).tocsr()
            X = _apply_scaler(MaxAbsScaler, X, offline_rows, spec.causal_preprocessing)
            raw_batches = _partition_into_batches(
                X, y, total_batches, spec.batch_size, "cara", spec.seed
            )
        elif spec.protocol == "upf":
            # 7 features: integer-encoded nominals (5) + numeric (2), float32
            encoded = np.empty((nominals.shape[0], 7), dtype=np.float32)
            for col_idx in range(nominals.shape[1]):
                mapping: dict[str, int] = {}
                encoded[:, col_idx] = [
                    float(mapping.setdefault(val, len(mapping))) for val in nominals[:, col_idx]
                ]
            encoded[:, 5:7] = numerics.astype(np.float32)
            batch_size = spec.batch_size or 3000
            raw_batches = _partition_into_batches(
                encoded, y, total_batches, batch_size, "upf", spec.seed
            )
        else:
            encoded = np.empty((nominals.shape[0], 7), dtype=np.float32)
            for col_idx in range(nominals.shape[1]):
                mapping = {}
                encoded[:, col_idx] = [
                    float(mapping.setdefault(val, len(mapping))) for val in nominals[:, col_idx]
                ]
            encoded[:, 5:7] = numerics.astype(np.float32)
            raw_batches = _partition_into_batches(
                encoded, y, total_batches, spec.batch_size, "custom", spec.seed
            )
        return _make_batch_splits(
            raw_batches,
            offline_batches,
            split_ratio,
            query_ratio,
            spec.seed,
            use_test_as_queries=use_test_as_queries,
            protocol=spec.protocol,
        )

    # Generic custom / CSV / Parquet / NumPy / ARFF
    X, y = _load_generic_raw(path)
    if spec.protocol == "cara" and not issparse(X):
        offline_rows = int(X.shape[0] * (offline_batches / max(1, total_batches)))
        X = _apply_scaler(MinMaxScaler, X, offline_rows, spec.causal_preprocessing)
    raw_batches = _partition_into_batches(
        X, y, total_batches, spec.batch_size, spec.protocol, spec.seed
    )
    return _make_batch_splits(
        raw_batches,
        offline_batches,
        split_ratio,
        query_ratio,
        spec.seed,
        use_test_as_queries=use_test_as_queries,
        protocol=spec.protocol,
    )
