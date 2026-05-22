"""
Performance metrics collection and timing utilities.
"""

import time
from typing import Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    """
    Collect and track performance metrics for IMPRESS system.
    """
    ttft_ms: float = 0.0
    io_time_ms: float = 0.0
    gpu_cache_hits: int = 0
    gpu_cache_misses: int = 0
    cpu_cache_hits: int = 0
    cpu_cache_misses: int = 0
    disk_reads: int = 0
    tokens_loaded: int = 0
    tokens_skipped: int = 0

    @property
    def gpu_cache_hit_rate(self) -> float:
        """Calculate GPU cache hit rate."""
        total = self.gpu_cache_hits + self.gpu_cache_misses
        return self.gpu_cache_hits / total if total > 0 else 0.0

    @property
    def cpu_cache_hit_rate(self) -> float:
        """Calculate CPU cache hit rate."""
        total = self.cpu_cache_hits + self.cpu_cache_misses
        return self.cpu_cache_hits / total if total > 0 else 0.0

    @property
    def total_io_time_ms(self) -> float:
        """Total I/O time including all storage tiers."""
        return self.io_time_ms

    def reset(self):
        """Reset all metrics to zero."""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, 0)

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'ttft_ms': self.ttft_ms,
            'io_time_ms': self.io_time_ms,
            'gpu_cache_hit_rate': self.gpu_cache_hit_rate,
            'cpu_cache_hit_rate': self.cpu_cache_hit_rate,
            'disk_reads': self.disk_reads,
            'tokens_loaded': self.tokens_loaded,
            'tokens_skipped': self.tokens_skipped,
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"TTFT: {self.ttft_ms:.2f}ms | "
            f"I/O: {self.io_time_ms:.2f}ms | "
            f"GPU Hit Rate: {self.gpu_cache_hit_rate:.2%} | "
            f"CPU Hit Rate: {self.cpu_cache_hit_rate:.2%} | "
            f"Tokens Loaded: {self.tokens_loaded} | "
            f"Tokens Skipped: {self.tokens_skipped}"
        )


@contextmanager
def Timer(metrics: MetricsCollector, metric_name: str):
    """
    Context manager for timing operations and updating metrics.

    Usage:
        with Timer(metrics, 'io_time_ms'):
            # perform I/O operation
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        setattr(metrics, metric_name, getattr(metrics, metric_name, 0.0) + elapsed_ms)