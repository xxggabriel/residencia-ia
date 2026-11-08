"""Background host/process sampling during an evaluation attempt.

Names follow MLflow's system-metrics convention; the publisher logs them as
`system/<name>`, which MLflow shows in the run's "System metrics" tab. Sampling
is read-only and never influences predictions, policies or report digests.
psutil is optional: without it only the Linux process RSS is recorded and the
report states why the other series are unavailable.
"""

import os
import threading
import time
from pathlib import Path

from .environment import current_rss_bytes

MEGABYTE = 1024**2
DEFAULT_INTERVAL = float(os.environ.get("BENCH_SYSTEM_METRICS_INTERVAL", "0.5"))
MAX_SAMPLES = 3600


class SystemMetricsSampler:
    """Context manager: samples at start, every `interval` seconds, and at stop.

    When MAX_SAMPLES is reached, every other sample is dropped and the interval
    doubles, so memory stays bounded for arbitrarily long attempts.
    """

    def __init__(self, disk_path=".", interval: float = DEFAULT_INTERVAL):
        if not interval > 0:
            raise ValueError("system metrics interval must be positive")
        self.disk_path = str(Path(disk_path))
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        try:
            import psutil
        except ImportError:
            psutil = None
        self._psutil = psutil
        self._process = psutil.Process() if psutil else None
        self._network_start = None
        self._gpus = self._gpu_handles()

    @staticmethod
    def _gpu_handles():
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:  # No NVIDIA library or driver: GPU series are simply absent.
            return None, []

    def _measure(self, cpu: bool = True) -> dict:
        values = {}
        if self._psutil is not None:
            psutil = self._psutil
            memory = psutil.virtual_memory()
            if cpu:  # A CPU percentage needs an elapsed interval; the priming read has none.
                values["cpu_utilization_percentage"] = psutil.cpu_percent(interval=None)
            values["system_memory_usage_megabytes"] = memory.used / MEGABYTE
            values["system_memory_usage_percentage"] = memory.percent
            try:
                disk = psutil.disk_usage(self.disk_path)
                values["disk_usage_percentage"] = disk.percent
                values["disk_usage_megabytes"] = disk.used / MEGABYTE
                values["disk_available_megabytes"] = disk.free / MEGABYTE
            except OSError:
                pass
            network = psutil.net_io_counters()
            if network is not None:
                if self._network_start is None:
                    self._network_start = network
                values["network_receive_megabytes"] = (
                    network.bytes_recv - self._network_start.bytes_recv
                ) / MEGABYTE
                values["network_transmit_megabytes"] = (
                    network.bytes_sent - self._network_start.bytes_sent
                ) / MEGABYTE
            with self._process.oneshot():
                if cpu:
                    values["process_cpu_utilization_percentage"] = self._process.cpu_percent(interval=None)
                values["process_memory_rss_megabytes"] = self._process.memory_info().rss / MEGABYTE
                values["process_threads"] = self._process.num_threads()
        else:
            rss = current_rss_bytes()
            if rss is not None:
                values["process_memory_rss_megabytes"] = rss / MEGABYTE
        pynvml, handles = self._gpus
        for index, handle in enumerate(handles):
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                values[f"gpu_{index}_utilization_percentage"] = utilization.gpu
                values[f"gpu_{index}_memory_usage_megabytes"] = memory.used / MEGABYTE
                values[f"gpu_{index}_memory_usage_percentage"] = 100 * memory.used / memory.total
            except Exception:
                continue
        return values

    def sample(self, cpu: bool = True) -> None:
        values = self._measure(cpu)
        with self._lock:
            self.samples.append(
                {
                    "timestamp_ms": int(time.time() * 1000),
                    "elapsed_seconds": time.monotonic() - self._started,
                    "values": values,
                }
            )
            if len(self.samples) >= MAX_SAMPLES:
                self.samples = self.samples[::2]
                self.interval *= 2

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self._started = time.monotonic()
        if self._psutil is not None:  # Prime cumulative CPU counters; the first reading is 0 otherwise.
            self._psutil.cpu_percent(interval=None)
            self._process.cpu_percent(interval=None)
        self.sample(cpu=False)
        self._thread = threading.Thread(target=self._run, name="system-metrics", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        return False

    def report(self) -> dict:
        with self._lock:
            samples = list(self.samples)
        names = sorted({name for sample in samples for name in sample["values"]})
        summary = {}
        for name in names:
            series = [s["values"][name] for s in samples if name in s["values"]]
            summary[name] = {"mean": sum(series) / len(series), "max": max(series), "min": min(series)}
        return {
            "version": "system-metrics-v1",
            "naming": "MLflow system metrics; published as system/<name>",
            "scope": ("evaluation attempt in this process, including metric derivation; "
                      "excludes plots and publication"),
            "interval_seconds": self.interval,
            "collector": "psutil" if self._psutil is not None else "procfs_rss_only",
            "unavailable_reason": None if self._psutil is not None else "psutil is not installed",
            "gpus": len(self._gpus[1]),
            "metrics": names,
            "summary": summary,
            "samples": samples,
        }
