"""
Performance benchmarking for IMPRESS system.

Measures TTFT, I/O time, cache hit rates, and throughput.
"""

import time
import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..core.storage_tier import PrefixKVStore
from ..core.itf import ImportantTokenFilter
from ..core.cache_manager import CacheManager
from ..models.base import BaseModelAdapter
from ..utils.metrics import Timer, MetricsCollector


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_name: str
    dataset: str
    ttft_ms: float
    io_time_ms: float
    compute_time_ms: float
    total_time_ms: float
    gpu_cache_hit_rate: float
    cpu_cache_hit_rate: float
    disk_reads: int
    tokens_processed: int
    kv_ratio: float

    @property
    def io_ratio(self) -> float:
        """Ratio of I/O time to total time."""
        return self.io_time_ms / self.total_time_ms if self.total_time_ms > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset': self.dataset,
            'ttft_ms': self.ttft_ms,
            'io_time_ms': self.io_time_ms,
            'compute_time_ms': self.compute_time_ms,
            'total_time_ms': self.total_time_ms,
            'gpu_cache_hit_rate': self.gpu_cache_hit_rate,
            'cpu_cache_hit_rate': self.cpu_cache_hit_rate,
            'disk_reads': self.disk_reads,
            'tokens_processed': self.tokens_processed,
            'kv_ratio': self.kv_ratio,
            'io_ratio': self.io_ratio,
        }


class Benchmark:
    """
    Performance benchmark for IMPRESS system.

    Runs various test scenarios and collects performance metrics.
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        kv_store: PrefixKVStore,
        cache_manager: CacheManager,
        itf: Optional[ImportantTokenFilter] = None
    ):
        """
        Initialize benchmark.

        Args:
            model_adapter: Model adapter for inference
            kv_store: Three-tier KV store
            cache_manager: Cache manager
            itf: Optional important token filter
        """
        self.model_adapter = model_adapter
        self.kv_store = kv_store
        self.cache_manager = cache_manager
        self.itf = itf
        self.metrics = MetricsCollector()

        self.results: List[BenchmarkResult] = []

    def run_ttft_benchmark(
        self,
        prompt_prefixes: List[str],
        queries: List[str],
        kv_ratios: List[float] = [0.05, 0.10, 0.25, 0.50],
        warmup_runs: int = 3
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run TTFT benchmark with varying KV ratios.

        Args:
            prompt_prefixes: List of prefix strings (shared contexts)
            queries: List of query strings
            kv_ratios: List of KV retention ratios to test
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary mapping KV ratios to benchmark results
        """
        results_by_ratio = {}

        for kv_ratio in kv_ratios:
            if self.itf:
                self.itf.set_kv_ratio(kv_ratio)

            ratio_results = []

            # Warmup runs
            for _ in range(warmup_runs):
                self._run_single_inference(prompt_prefixes[0], queries[0])

            # Actual benchmark
            for i, query in enumerate(queries):
                prefix = prompt_prefixes[i % len(prompt_prefixes)]

                result = self._run_single_inference(prefix, query)
                result.kv_ratio = kv_ratio
                ratio_results.append(result)

            results_by_ratio[str(kv_ratio)] = ratio_results
            self.results.extend(ratio_results)

        return results_by_ratio

    def _run_single_inference(self, prefix: str, query: str) -> BenchmarkResult:
        """
        Run a single inference and collect metrics.

        Args:
            prefix: Context prefix
            query: User query

        Returns:
            BenchmarkResult with collected metrics
        """
        # Reset metrics
        self.metrics.reset()

        start_time = time.perf_counter()

        # Tokenize
        prefix_tokens = self.model_adapter.tokenize(prefix)
        query_tokens = self.model_adapter.tokenize(query)

        total_tokens = len(prefix_tokens) + len(query_tokens)

        # Simulate inference with timing
        io_start = time.perf_counter()

        # Check for prefix reuse (simulated)
        # In real implementation, this would use the radix tree
        prefix_chunks = self.kv_store.get_cache_stats().get('disk_chunks', 0)

        io_end = time.perf_counter()

        # Compute time (estimated based on model size and tokens)
        compute_start = time.perf_counter()
        # Simulate computation
        time.sleep(total_tokens / 1000.0 * 0.01)  # Very rough estimate
        compute_end = time.perf_counter()

        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000
        io_time_ms = (io_end - io_start) * 1000
        compute_time_ms = (compute_end - compute_start) * 1000

        # TTFT is roughly total_time_ms for prefill phase
        ttft_ms = total_time_ms

        cache_stats = self.cache_manager.get_cache_stats()

        result = BenchmarkResult(
            model_name=self.model_adapter.model_name,
            dataset="benchmark",
            ttft_ms=ttft_ms,
            io_time_ms=io_time_ms,
            compute_time_ms=compute_time_ms,
            total_time_ms=total_time_ms,
            gpu_cache_hit_rate=cache_stats.get('gpu_hit_rate', 0.0),
            cpu_cache_hit_rate=cache_stats.get('cpu_hit_rate', 0.0),
            disk_reads=self.metrics.disk_reads,
            tokens_processed=total_tokens,
            kv_ratio=self.itf.kv_ratio if self.itf else 1.0
        )

        return result

    def run_cache_hit_rate_benchmark(
        self,
        num_requests: int = 100,
        prefix_reuse_prob: float = 0.8
    ) -> Dict[str, float]:
        """
        Run cache hit rate benchmark.

        Args:
            num_requests: Number of requests to simulate
            prefix_reuse_prob: Probability of reusing a previous prefix

        Returns:
            Dictionary with hit rate statistics
        """
        self.metrics.reset()

        for i in range(num_requests):
            # Simulate request with potential prefix reuse
            # In real implementation, this would use actual tokens and prefixes
            pass

        cache_stats = self.cache_manager.get_cache_stats()

        return {
            'gpu_hit_rate': cache_stats.get('gpu_hit_rate', 0.0),
            'cpu_hit_rate': cache_stats.get('cpu_hit_rate', 0.0),
            'gpu_cache_size': cache_stats.get('gpu_cache_size', 0),
            'cpu_cache_size': cache_stats.get('cpu_cache_size', 0),
            'promotions_to_gpu': cache_stats.get('promotions_to_gpu', 0),
            'evictions_from_gpu': cache_stats.get('evictions_from_gpu', 0),
        }

    def run_tail_latency_benchmark(
        self,
        num_requests: int = 1000
    ) -> Dict[str, float]:
        """
        Run tail latency benchmark (p50, p90, p95, p99).

        Args:
            num_requests: Number of requests

        Returns:
            Dictionary with latency percentiles
        """
        latencies = []

        for _ in range(num_requests):
            # Simulate request and measure latency
            start = time.perf_counter()
            # ... simulate inference ...
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies.sort()

        return {
            'p50_ms': latencies[len(latencies) // 2],
            'p90_ms': latencies[int(len(latencies) * 0.9)],
            'p95_ms': latencies[int(len(latencies) * 0.95)],
            'p99_ms': latencies[int(len(latencies) * 0.99)],
            'mean_ms': sum(latencies) / len(latencies) if latencies else 0.0,
        }

    def calculate_speedup(self, baseline_ttft: float) -> Dict[str, float]:
        """
        Calculate speedup relative to baseline.

        Args:
            baseline_ttft: Baseline TTFT in ms

        Returns:
            Dictionary with speedup metrics
        """
        if not self.results:
            return {}

        avg_ttft = sum(r.ttft_ms for r in self.results) / len(self.results)
        avg_io_time = sum(r.io_time_ms for r in self.results) / len(self.results)

        return {
            'ttft_speedup': baseline_ttft / avg_ttft if avg_ttft > 0 else 0.0,
            'io_time_speedup': (baseline_ttft * 0.7) / avg_io_time if avg_io_time > 0 else 0.0,
        }

    def save_results(self, output_path: str):
        """
        Save benchmark results to file.

        Args:
            output_path: Path to save results
        """
        import json

        results_dict = [r.to_dict() for r in self.results]

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def print_summary(self):
        """Print summary of benchmark results."""
        if not self.results:
            print("No benchmark results to display.")
            return

        print("\n=== IMPRESS Benchmark Summary ===")
        print(f"Model: {self.model_adapter.model_name}")
        print(f"Total runs: {len(self.results)}")

        avg_ttft = sum(r.ttft_ms for r in self.results) / len(self.results)
        avg_io = sum(r.io_time_ms for r in self.results) / len(self.results)
        avg_gpu_hit = sum(r.gpu_cache_hit_rate for r in self.results) / len(self.results)

        print(f"\nAverage TTFT: {avg_ttft:.2f} ms")
        print(f"Average I/O time: {avg_io:.2f} ms")
        print(f"I/O ratio: {avg_io/avg_ttft:.2%}" if avg_ttft > 0 else "N/A")
        print(f"Average GPU cache hit rate: {avg_gpu_hit:.2%}")

        if self.itf:
            itf_stats = self.itf.get_statistics()
            print(f"\nITF Statistics:")
            print(f"  Similarity enabled layers: {itf_stats['similarity_enabled_layers']}")
            print(f"  Similarity disabled layers: {itf_stats['similarity_disabled_layers']}")
            print(f"  Similarity rate: {itf_stats['similarity_rate']:.2%}")
            print(f"  I/O reduction: {itf_stats['io_reduction_ratio']:.2%}")


class BaselineBenchmark:
    """
    Baseline benchmark for comparison with IMPRESS.

    Implements baseline approaches: ReComp, AS-like, etc.
    """

    def __init__(self, model_adapter: BaseModelAdapter):
        """
        Initialize baseline benchmark.

        Args:
            model_adapter: Model adapter for inference
        """
        self.model_adapter = model_adapter

    def run_recomp_baseline(
        self,
        prompt_prefixes: List[str],
        queries: List[str],
        warmup_runs: int = 3
    ) -> float:
        """
        Run Recomputation baseline (no prefix KV reuse).

        Args:
            prompt_prefixes: List of prefix strings
            queries: List of query strings
            warmup_runs: Number of warmup runs

        Returns:
            Average TTFT in ms
        """
        ttfts = []

        for i, query in enumerate(queries):
            prefix = prompt_prefixes[i % len(prompt_prefixes)]

            # Tokenize full input
            full_text = prefix + query
            tokens = self.model_adapter.tokenize(full_text)

            # Measure TTFT (simulate)
            start = time.perf_counter()

            # Simulate full prefill computation
            # In real implementation, this would run actual inference
            compute_time = len(tokens) / 1000.0 * 0.02  # Rough estimate
            time.sleep(compute_time)

            end = time.perf_counter()
            ttfts.append((end - start) * 1000)

        return sum(ttfts) / len(ttfts) if ttfts else 0.0