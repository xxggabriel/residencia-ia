"""Pure-Python 1:1 equivalents of scikit-multiflow==0.5.3 drift detectors and unified adapter."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
from scipy import stats


class _ListItem:
    """Bucket row in ADWIN's exponential histogram linked list."""

    def __init__(
        self,
        max_buckets: int,
        next_node: _ListItem | None = None,
        previous_node: _ListItem | None = None,
    ) -> None:
        self.max_buckets = max_buckets
        self.bucket_size_row = 0
        self.bucket_total = np.zeros(max_buckets + 1, dtype=np.float64)
        self.bucket_variance = np.zeros(max_buckets + 1, dtype=np.float64)
        self.next = next_node
        self.previous = previous_node
        if next_node is not None:
            next_node.previous = self
        if previous_node is not None:
            previous_node.next = self

    def insert_bucket(self, value: float, variance: float) -> None:
        k = self.bucket_size_row
        self.bucket_total[k] = value
        self.bucket_variance[k] = variance
        self.bucket_size_row += 1

    def compress_buckets_row(self, num_deleted: int) -> None:
        for k in range(num_deleted, self.max_buckets + 1):
            self.bucket_total[k - num_deleted] = self.bucket_total[k]
            self.bucket_variance[k - num_deleted] = self.bucket_variance[k]
        for k in range(1, num_deleted + 1):
            idx = self.max_buckets - k + 1
            self.bucket_total[idx] = 0.0
            self.bucket_variance[idx] = 0.0
        self.bucket_size_row -= num_deleted


class _BucketList:
    """Doubly-linked list of bucket rows ordered from newest (head) to oldest (tail)."""

    def __init__(self, max_buckets: int) -> None:
        self.max_buckets = max_buckets
        self.count = 0
        self.head: _ListItem | None = None
        self.tail: _ListItem | None = None
        self.add_to_head()

    def add_to_head(self) -> None:
        self.head = _ListItem(self.max_buckets, self.head, None)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self) -> None:
        self.tail = _ListItem(self.max_buckets, None, self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self) -> None:
        if self.tail is None:
            return
        self.tail = self.tail.previous
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.count -= 1


class ADWIN:
    """Exact 1:1 pure-Python implementation of ``skmultiflow.drift_detection.ADWIN`` (0.5.3)."""

    MAX_BUCKETS = 5

    def __init__(self, delta: float = 0.002) -> None:
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delta = float(delta)
        self.mint_clock = 32
        self.mint_min_window_longitude = 10
        self.mint_min_window_length = 5
        self.reset()

    def reset(self) -> None:
        """Reset bucket list and window statistics."""
        self.last_bucket_row = 0
        self.list_row_buckets = _BucketList(self.MAX_BUCKETS)
        self.mint_time = 0
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.bucket_number = 0
        self.bucket_num_max = 0
        self.n_detections = 0
        self.detect = 0
        self.detect_twice = 0
        self.bln_bucket_deleted = False
        self.in_concept_change = False

    def detected_change(self) -> bool:
        """Return whether drift was detected on the most recent observation."""
        return bool(self.in_concept_change)

    def add_element(self, value: float) -> None:
        """Feed one numeric observation and update the adaptive window."""
        self.set_input(value)

    def set_input(self, value: float) -> None:
        """Feed one numeric observation (alias matching scikit-multiflow ADWIN)."""
        val = float(value)
        self.in_concept_change = False
        self._insert_element(val)
        self.bln_bucket_deleted = False
        self.mint_time += 1

        if self.mint_time % self.mint_clock == 0 and self.width > self.mint_min_window_longitude:
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = False
                bln_exit = False
                n0 = 0
                n1 = self.width
                u0 = 0.0
                u1 = self.total
                cursor = self.list_row_buckets.tail
                i = self.last_bucket_row
                while not bln_exit and cursor is not None:
                    for k in range(cursor.bucket_size_row - 1):
                        if i == 0 and k == cursor.bucket_size_row - 1:
                            bln_exit = True
                            break
                        bucket_sz = 1 << i
                        n0 += bucket_sz
                        n1 -= bucket_sz
                        u0 += cursor.bucket_total[k]
                        u1 -= cursor.bucket_total[k]
                        if n1 >= self.mint_min_window_length and n0 >= self.mint_min_window_length:
                            if self._bln_cut_expression(n0, n1, u0, u1):
                                bln_exit = True
                                bln_reduce_width = True
                                self.bln_bucket_deleted = True
                                if self.detect == 0:
                                    self.detect = self.mint_time
                                elif self.detect_twice == 0:
                                    self.detect_twice = self.mint_time
                                self.in_concept_change = True
                                self.n_detections += 1
                                if self.width > 0:
                                    n0 -= self._delete_element()
                                    break
                    cursor = cursor.previous
                    i -= 1

    def _insert_element(self, value: float) -> None:
        self.width += 1
        assert self.list_row_buckets.head is not None
        self.list_row_buckets.head.insert_bucket(value, 0.0)
        self.bucket_number += 1
        if self.bucket_number > self.bucket_num_max:
            self.bucket_num_max = self.bucket_number

        inc_variance = 0.0
        if self.width > 1:
            inc_variance = (
                (self.width - 1) * (value - self.total / (self.width - 1)) ** 2 / self.width
            )
        self.variance += inc_variance
        self.total += value
        self._compress_buckets()

    def _compress_buckets(self) -> None:
        cursor = self.list_row_buckets.head
        i = 0
        while cursor is not None:
            k = cursor.bucket_size_row
            if k == self.MAX_BUCKETS + 1:
                next_node = cursor.next
                if next_node is None:
                    self.list_row_buckets.add_to_tail()
                    next_node = cursor.next
                    self.last_bucket_row += 1
                assert next_node is not None
                n1 = float(1 << i)
                n2 = float(1 << i)
                u1 = cursor.bucket_total[0] / n1
                u2 = cursor.bucket_total[1] / n2
                inc_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(
                    cursor.bucket_total[0] + cursor.bucket_total[1],
                    cursor.bucket_variance[1] + inc_variance,
                )
                self.bucket_number -= 1
                cursor.compress_buckets_row(2)
                if next_node.bucket_size_row <= self.MAX_BUCKETS:
                    break
            else:
                break
            cursor = cursor.next
            i += 1

    def _delete_element(self) -> int:
        node = self.list_row_buckets.tail
        assert node is not None
        n1 = 1 << self.last_bucket_row
        self.width -= n1
        self.total -= node.bucket_total[0]
        if self.width > 0:
            u1 = node.bucket_total[0] / n1
            inc_variance = node.bucket_variance[0] + n1 * self.width * (
                u1 - self.total / self.width
            ) ** 2 / (n1 + self.width)
            self.variance = max(0.0, self.variance - inc_variance)
        else:
            self.variance = 0.0
        node.compress_buckets_row(1)
        self.bucket_number -= 1
        if node.bucket_size_row == 0:
            self.list_row_buckets.remove_from_tail()
            self.last_bucket_row -= 1
        return n1

    def _bln_cut_expression(self, n0: int, n1: int, u0: float, u1: float) -> bool:
        n = float(self.width)
        abs_value = abs((u0 / n0) - (u1 / n1))
        dd = float(np.log(2.0 * np.log(n) / self.delta))
        min_len = self.mint_min_window_length
        m = 1.0 / (n0 - min_len + 1) + 1.0 / (n1 - min_len + 1)
        v = max(0.0, self.variance / n)
        epsilon = math.sqrt(2.0 * m * v * dd) + (2.0 / 3.0) * dd * m
        return abs_value > epsilon


class DDM:
    """Exact 1:1 pure-Python implementation of ``skmultiflow.drift_detection.DDM`` (0.5.3)."""

    def __init__(
        self,
        min_num_instances: int = 30,
        warning_level: float = 2.0,
        out_control_level: float = 3.0,
    ) -> None:
        self.min_instances = int(min_num_instances)
        self.warning_level = float(warning_level)
        self.out_control_level = float(out_control_level)
        self.reset()

    def reset(self) -> None:
        """Reset error statistics and tracked minimum thresholds."""
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.delay = 0

    def add_element(self, prediction: float) -> None:
        """Update DDM statistics with a binary prediction error (0 or 1)."""
        if self.in_concept_change:
            self.reset()

        pred = float(prediction)
        self.miss_prob = self.miss_prob + (pred - self.miss_prob) / float(self.sample_count)
        self.miss_std = math.sqrt(
            max(0.0, self.miss_prob * (1.0 - self.miss_prob)) / float(self.sample_count)
        )
        self.sample_count += 1
        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.in_concept_change = True
        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.in_warning_zone = True
        else:
            self.in_warning_zone = False

    def detected_change(self) -> bool:
        """Return whether drift was detected on the most recent observation."""
        return bool(self.in_concept_change)

    def detected_warning_zone(self) -> bool:
        """Return whether the detector is currently in the warning zone."""
        return bool(self.in_warning_zone)


class KSWIN:
    """Exact 1:1 pure-Python implementation of ``skmultiflow.drift_detection.KSWIN`` (0.5.3)."""

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        data: Any = None,
        seed: int | None = None,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if stat_size <= 0 or stat_size > window_size:
            raise ValueError(f"stat_size must be in (0, window_size], got {stat_size}")

        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.stat_size = int(stat_size)
        self.data = data
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.p_value = 0.0
        self.n = 0
        self.change_detected = False
        self.window = (
            np.asarray(data, dtype=float).ravel().copy()
            if data is not None
            else np.array([], dtype=float)
        )

    def reset(self) -> None:
        """Reset sliding window and KS statistics."""
        self.p_value = 0.0
        self.n = 0
        self.change_detected = False
        self.window = np.array([], dtype=float)

    def add_element(self, input_value: float) -> None:
        """Update sliding window and run Kolmogorov-Smirnov two-sample test."""
        self.change_detected = False
        self.n += 1
        current_length = self.window.shape[0]
        if current_length >= self.window_size:
            self.window = np.delete(self.window, 0)
            history = self.window[: -self.stat_size]
            if history.shape[0] > 0:
                rng = self._rng if self._rng is not None else np.random
                rnd_window = rng.choice(history, self.stat_size)
                st, self.p_value = stats.ks_2samp(
                    rnd_window, self.window[-self.stat_size :], mode="exact"
                )
                if self.p_value <= self.alpha and st > 0.1:
                    self.change_detected = True
                    self.window = self.window[-self.stat_size :]
                else:
                    self.change_detected = False
        else:
            self.change_detected = False

        self.window = np.concatenate([self.window, [float(input_value)]])

    def detected_change(self) -> bool:
        """Return whether drift was detected on the most recent observation."""
        return bool(self.change_detected)


def _resolve_skmultiflow_class(name: str) -> tuple[type[Any], bool]:
    """Return ``(detector_cls, is_native)`` preferring native ``skmultiflow`` if installed."""
    pure_map: dict[str, type[Any]] = {
        "adwin": ADWIN,
        "ddm": DDM,
        "kswin": KSWIN,
    }
    if name not in pure_map:
        raise ValueError(f"unknown skmultiflow detector: {name}")
    try:
        from skmultiflow.drift_detection import ADWIN as NativeADWIN
        from skmultiflow.drift_detection import DDM as NativeDDM
        from skmultiflow.drift_detection import KSWIN as NativeKSWIN

        native_map: dict[str, type[Any]] = {
            "adwin": NativeADWIN,
            "ddm": NativeDDM,
            "kswin": NativeKSWIN,
        }
        return native_map[name], True
    except ImportError:
        return pure_map[name], False


class SkmultiflowDetector:
    """Adapter exposing ``scikit-multiflow==0.5.3`` detectors under the common Strategy interface."""

    signal = "released_label_binary_prediction_error"

    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self._params = dict(params)
        cls, is_native = _resolve_skmultiflow_class(name)
        if is_native and name == "kswin" and "seed" in params:
            sig = inspect.signature(cls)
            if "seed" not in sig.parameters:
                cls = KSWIN
                is_native = False
        self._is_native = is_native
        self.inner = cls(**params)

    @property
    def drift_detected(self) -> bool:
        """Whether the underlying detector signaled drift on the last update."""
        return bool(self.inner.detected_change())

    def update(self, error: int) -> bool:
        """Feed a binary prediction error (0 or 1) and return whether drift was detected."""
        if error not in (0, 1):
            raise ValueError("detector expects binary prediction error, not raw features")
        self.inner.add_element(int(error))
        return bool(self.inner.detected_change())

    def reset(self) -> None:
        """Reset internal detector state cleanly."""
        self.inner.reset()

    def warm_start(self, values: np.ndarray | list[int]) -> None:
        """Reset and pre-populate the detector without returning triggers."""
        self.reset()
        arr = np.asarray(values, dtype=np.int64).ravel() if len(values) else np.array([], dtype=np.int64)
        for v in arr:
            if int(v) not in (0, 1):
                raise ValueError("detector expects binary prediction error, not raw features")
        if self.name == "kswin":
            if arr.size > 0:
                window_size = getattr(self.inner, "window_size", arr.size)
                self.inner.window = arr[-window_size:].astype(float).copy()
            return
        for v in arr:
            self.inner.add_element(int(v))
