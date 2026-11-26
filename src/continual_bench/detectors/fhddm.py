"""Fast Hoeffding Drift Detection Method (FHDDM, Pesaranghader & Viktor 2016)."""

from __future__ import annotations

import math
from collections import deque


class FHDDM:
    """Sliding-window Hoeffding bound drift detector over binary prediction errors.

    Maintains a sliding window of size ``window_size`` over binary prediction
    outcomes (``0`` for correct, ``1`` for error) and signals drift when
    ``p_max - p_current > epsilon`` where ``p = 1 - err`` and
    ``epsilon = sqrt(ln(1 / delta) / (2 * window_size))``.
    """

    signal = "released_label_binary_prediction_error"

    def __init__(self, window_size: int = 25, delta: float = 0.05) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.window_size = int(window_size)
        self.delta = float(delta)
        self.epsilon = math.sqrt(math.log(1.0 / self.delta) / (2.0 * self.window_size))
        self._window: deque[float] = deque(maxlen=self.window_size)
        self._sum: float = 0.0
        self.p_max: float = 0.0
        self.p_current: float = 0.0
        self.drift_detected: bool = False

    @property
    def err_min(self) -> float:
        return 1.0 - self.p_max

    @property
    def err_current(self) -> float:
        return 1.0 - self.p_current

    def reset(self) -> None:
        """Reset sliding window and tracked statistics."""
        self._window.clear()
        self._sum = 0.0
        self.p_max = 0.0
        self.p_current = 0.0
        self.drift_detected = False

    def warm_start(self, values) -> None:
        """Reset and pre-populate the sliding window without raising false initial drift."""
        self.reset()
        for val in values:
            v = float(val)
            if math.isnan(v) or not (0.0 <= v <= 1.0):
                raise ValueError("FHDDM expects binary prediction error in [0, 1]")
            acc = 1.0 - v
            if len(self._window) == self.window_size:
                self._sum -= self._window[0]
            self._window.append(acc)
            self._sum += acc
        if self._window:
            self.p_current = self._sum / len(self._window)
            if len(self._window) == self.window_size:
                self.p_max = self.p_current
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Update with a binary prediction error (0=correct, 1=error) and return drift flag."""
        v = float(value)
        if math.isnan(v) or not (0.0 <= v <= 1.0):
            raise ValueError("FHDDM expects binary prediction error in [0, 1]")

        acc = 1.0 - v
        if len(self._window) == self.window_size:
            self._sum -= self._window[0]
        self._window.append(acc)
        self._sum += acc
        self.drift_detected = False

        if len(self._window) < self.window_size:
            self.p_current = self._sum / len(self._window)
            return False

        self.p_current = self._sum / self.window_size
        if self.p_current > self.p_max:
            self.p_max = self.p_current

        if (self.p_max - self.p_current) > self.epsilon:
            self.drift_detected = True
            self._window.clear()
            self._sum = 0.0
            self.p_max = 0.0
            return True
        return False
