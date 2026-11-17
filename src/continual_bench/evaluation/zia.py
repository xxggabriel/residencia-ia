"""Compatibility entry point: the benchmark is dataset-agnostic in ``stream_benchmark``.

``python -m continual_bench.evaluation.zia`` and ``from continual_bench.evaluation.zia
import run`` keep working for existing campaigns, images and scripts.
"""

from continual_bench.evaluation.stream_benchmark import *  # noqa: F403
from continual_bench.evaluation.stream_benchmark import (  # noqa: F401
    ADWIN_CLOCK,
    ADWIN_DELTA,
    BLOCK_ROWS,
    CHECKPOINT_VERSION,
    DDM_PARAMS,
    HOLDOUT_FRACTION,
    POLICIES,
    REPLAY_BLOCKS,
    StreamState,
    _run_cli,
    batches,
    chronological_blocks,
    drift_alarms,
    main,
    parquet_batches,
    run,
)

if __name__ == "__main__":
    main()
