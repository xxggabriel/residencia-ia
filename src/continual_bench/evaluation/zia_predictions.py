"""Write individual Zia predictions in bounded, checksummed Parquet parts."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from continual_bench.datasets.zia import sha256_file
from continual_bench.evaluation.environment import encoded
from continual_bench.evaluation.zia_metrics import decide
from continual_bench.infrastructure.storage import atomic_write

INDEX = "predictions-manifest.json"
BLOCKS_PER_PART = 16


class PredictionArchive:
    def __init__(self, directory, classes, dataset_sha256, origin, fading_factor=0.99, threshold=0.5):
        self.directory = Path(directory)
        self.classes = classes
        self.threshold = 0.5 if threshold is None else threshold
        self.schema = pa.schema([
            ("seq", pa.uint32()), ("event_time_ms", pa.int64()), ("label", pa.uint8()),
            ("prediction", pa.uint8()), ("holdout", pa.bool_()), ("model_end_seq", pa.uint32()),
            *((f"p_{i}", pa.float64()) for i in range(len(classes))),
        ])
        self.writer = None
        self.parts = []
        self.blocks = self.rows = self.part_rows = 0
        self.fading_factor = fading_factor
        self.fading_state = [0.0]
        self.fading_weight = 0.0
        self.fading = []
        self.temp = TemporaryDirectory(dir=self.directory, prefix=".predictions-")
        self.staging = Path(self.temp.name)
        self.manifest = {"version": "zia-predictions-v1", "origin": origin, "classes": classes,
                         "dataset_sha256": dataset_sha256, "fading_factor": fading_factor}
        if len(classes) == 2:
            self.manifest["decision_threshold"] = self.threshold

    def __enter__(self):
        return self

    def _close_part(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.parts.append({"file": self.path.name, "rows": self.part_rows,
                               "bytes": self.path.stat().st_size, "sha256": sha256_file(self.path)})

    def add(self, start, times, labels, probabilities, holdout, model_end):
        if self.blocks % BLOCKS_PER_PART == 0:
            self._close_part()
            self.path = self.staging / f"part-{len(self.parts):04d}.parquet"
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
            self.part_rows = 0
        n = len(labels)
        predicted = decide(probabilities, self.threshold).astype(np.uint8)
        self.writer.write_batch(pa.RecordBatch.from_arrays([
            pa.array(np.arange(start, start + n, dtype=np.uint32)), pa.array(times), pa.array(labels),
            pa.array(predicted), pa.array(np.full(n, holdout)), pa.array(np.full(n, model_end, np.uint32)),
            *(pa.array(probabilities[:, i], type=pa.float64()) for i in range(len(self.classes))),
        ], schema=self.schema))
        self.blocks += 1
        self.rows += n
        self.part_rows += n
        if not holdout:
            factor = self.fading_factor
            values, self.fading_state = lfilter([1], [1, -factor], predicted == labels, zi=self.fading_state)
            self.fading_weight = factor**n * self.fading_weight + (1 - factor**n) / (1 - factor)
            self.fading.append({"step": start + n, "value": float(values[-1] / self.fading_weight)})
        return int(np.count_nonzero(predicted != labels))

    def __exit__(self, exc_type, _error, _traceback):
        try:
            self._close_part()
            if exc_type is None:
                target = self.directory / "predictions"
                if target.exists():
                    shutil.rmtree(target)
                self.staging.replace(target)
                atomic_write(self.directory / INDEX, encoded({
                    **self.manifest, "rows": self.rows, "parts": self.parts, "fading_accuracy": self.fading,
                }))
        finally:
            self.temp.cleanup()


def prediction_manifest(directory, report):
    path = Path(directory) / INDEX
    if not path.is_file():
        return None
    value = json.loads(path.read_bytes())
    expected = report["metrics"]["n"] + report["holdout"]["n"]
    if (value["dataset_sha256"] != report["contract"]["input"]["sha256"] or value["rows"] != expected
            or sum(part["rows"] for part in value["parts"]) != expected):
        raise ValueError("prediction archive differs from the completed Zia report")
    return value
