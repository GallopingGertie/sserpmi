"""
Score-based Cache Management for IMPRESS.

Implements dual-tier cache (GPU + CPU) with score-based replacement policy.
Cache score = access_freq × importance_ratio, ensuring high-value chunks
are preferentially cached in GPU.
"""

from typing import Dict, List, Optional, Tuple
import heapq
import time
from dataclasses import dataclass, field

from .storage_tier import StorageTier, ChunkData, PrefixKVStore
from ..metadata.chunk_metadata import ChunkMetadata, ChunkMetadataManager
from ..utils.metrics import Timer, MetricsCollector


@dataclass(order=True)
class ScoredChunk:
    """
    Chunk with associated score for heap-based cache management.

    Lower score = lower priority for keeping in cache.
    """
    score: float
    chunk_id: str
    last_update: float = field(default_factory=time.time)

    def __eq__(self, other):
        return self.chunk_id == other.chunk_id


class CacheManager:
    """
    Manages GPU and CPU caches using IMPRESS score-based policy.

    Key features:
    1. Dual-tier cache (GPU for high-score chunks, CPU for medium)
    2. Score calculation: score = access_freq × importance_ratio
    3. Min-heap based replacement for O(log n) eviction
    4. Dynamic importance ratio updates (exponential moving average)
    5. Non-redundant caches between GPU and CPU
    """

    DECAY_FACTOR = 0.7  # EMA decay factor for importance ratio

    def __init__(
        self,
        kv_store: PrefixKVStore,
        metadata_manager: ChunkMetadataManager,
        metrics: Optional[MetricsCollector] = None
    ):
        """
        Initialize cache manager.

        Args:
            kv_store: Three-tier KV store
            metadata_manager: Chunk metadata manager
            metrics: Optional metrics collector
        """
        self.kv_store = kv_store
        self.metadata_manager = metadata_manager
        self.metrics = metrics

        # Min-heaps for cache management
        self.gpu_heap: List[ScoredChunk] = []  # Lower score = first to evict
        self.cpu_heap: List[ScoredChunk] = []

        # Tracking which chunks are where
        self.gpu_chunk_ids: Set[str] = set()
        self.cpu_chunk_ids: Set[str] = set()

        # Heap index mapping for O(1) lookup
        self.gpu_heap_index: Dict[str, int] = {}
        self.cpu_heap_index: Dict[str, int] = {}

        # Statistics
        self.gpu_hits = 0
        self.gpu_misses = 0
        self.cpu_hits = 0
        self.cpu_misses = 0
        self.evictions_from_gpu = 0
        self.evictions_from_cpu = 0
        self.promotions_to_gpu = 0

    def access_chunk(self, chunk_id: str, important_ratio: float = 0.5) -> Optional[ChunkData]:
        """
        Access a chunk, managing cache tiers as needed.

        Args:
            chunk_id: Chunk identifier
            important_ratio: Observed importance ratio for this access

        Returns:
            ChunkData if found, None otherwise
        """
        # Update metadata with new importance observation
        self.metadata_manager.record_access(chunk_id, important_ratio)

        # Update access frequencies in metadata manager
        if self.metadata_manager.total_accesses > 0:
            self._update_all_access_freqs()

        # Check GPU cache first
        if chunk_id in self.gpu_chunk_ids:
            self.gpu_hits += 1
            self._update_chunk_score(chunk_id, StorageTier.GPU)
            return self.kv_store.gpu_cache.get(chunk_id)

        self.gpu_misses += 1

        # Check CPU cache
        if chunk_id in self.cpu_chunk_ids:
            self.cpu_hits += 1
            chunk = self.kv_store.cpu_cache.get(chunk_id)

            # Check if should be promoted to GPU
            if self._should_promote_to_gpu(chunk_id):
                self._promote_chunk(chunk_id)

            self._update_chunk_score(chunk_id, StorageTier.CPU)
            return chunk

        self.cpu_misses += 1

        # Load from disk
        with Timer(self.metrics, 'io_time_ms') if self.metrics else nullcontext():
            return self._load_chunk(chunk_id)

    def _load_chunk(self, chunk_id: str) -> Optional[ChunkData]:
        """
        Load a chunk from disk into cache hierarchy.

        Args:
            chunk_id: Chunk identifier

        Returns:
            ChunkData if loaded successfully, None otherwise
        """
        # First load to CPU
        disk_chunk = self.kv_store.disk_storage.read_chunk(chunk_id)
        if disk_chunk is None:
            return None

        # Store in CPU cache
        cpu_chunk = self.kv_store._load_to_cpu(disk_chunk)
        self.kv_store.cpu_cache[chunk_id] = cpu_chunk
        self.cpu_chunk_ids.add(chunk_id)

        # Add to CPU heap
        self._add_to_heap(chunk_id, StorageTier.CPU)

        # Update metadata
        self.metadata_manager.update_storage_tier(chunk_id, StorageTier.CPU.value)

        # Check if should be in GPU instead
        if self._should_promote_to_gpu(chunk_id):
            self._promote_chunk(chunk_id)
            return self.kv_store.gpu_cache.get(chunk_id)

        return cpu_chunk

    def _should_promote_to_gpu(self, chunk_id: str) -> bool:
        """
        Determine if a chunk should be promoted to GPU cache.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if should promote, False otherwise
        """
        chunk_meta = self.metadata_manager.get_chunk(chunk_id)
        if chunk_meta is None:
            return False

        # Check if GPU cache is full
        if len(self.kv_store.gpu_cache) < self.kv_store.max_gpu_chunks:
            return True

        # Compare with lowest-scored chunk in GPU
        if not self.gpu_heap:
            return False

        lowest_gpu = self.gpu_heap[0]
        chunk_score = chunk_meta.score

        # Promote if score is higher than lowest in GPU
        return chunk_score > lowest_gpu.score

    def _promote_chunk(self, chunk_id: str):
        """
        Promote a chunk from CPU to GPU cache.

        Args:
            chunk_id: Chunk identifier
        """
        if chunk_id in self.gpu_chunk_ids:
            return

        # Remove from CPU heap and set
        if chunk_id in self.cpu_chunk_ids:
            self._remove_from_heap(chunk_id, StorageTier.CPU)
            self.cpu_chunk_ids.remove(chunk_id)

        # Check if need to evict from GPU
        if len(self.kv_store.gpu_cache) >= self.kv_store.max_gpu_chunks:
            self._evict_from_gpu()

        # Move chunk to GPU
        cpu_chunk = self.kv_store.cpu_cache.pop(chunk_id)
        gpu_chunk = self.kv_store._promote_to_gpu(cpu_chunk)
        self.kv_store.gpu_cache[chunk_id] = gpu_chunk

        # Add to GPU heap
        self._add_to_heap(chunk_id, StorageTier.GPU)

        # Update metadata
        self.metadata_manager.update_storage_tier(chunk_id, StorageTier.GPU.value)

        self.promotions_to_gpu += 1

    def _evict_from_gpu(self):
        """Evict lowest-scored chunk from GPU cache."""
        if not self.gpu_heap:
            return

        # Get lowest-scored chunk
        lowest = heapq.heappop(self.gpu_heap)
        self.gpu_heap_index.pop(lowest.chunk_id, None)

        # Remove from GPU
        chunk_id = lowest.chunk_id
        if chunk_id in self.gpu_chunk_ids:
            self.gpu_chunk_ids.remove(chunk_id)

        # Move to CPU
        gpu_chunk = self.kv_store.gpu_cache.pop(chunk_id)
        if gpu_chunk:
            cpu_chunk = self.kv_store._load_to_cpu(gpu_chunk)
            self.kv_store.cpu_cache[chunk_id] = cpu_chunk
            self.cpu_chunk_ids.add(chunk_id)

            # Add to CPU heap
            self._add_to_heap(chunk_id, StorageTier.CPU)

            # Update metadata
            self.metadata_manager.update_storage_tier(chunk_id, StorageTier.CPU.value)

        self.evictions_from_gpu += 1

    def _evict_from_cpu(self):
        """Evict lowest-scored chunk from CPU cache to disk."""
        if not self.cpu_heap:
            return

        # Get lowest-scored chunk
        lowest = heapq.heappop(self.cpu_heap)
        self.cpu_heap_index.pop(lowest.chunk_id, None)

        # Remove from CPU
        chunk_id = lowest.chunk_id
        if chunk_id in self.cpu_chunk_ids:
            self.cpu_chunk_ids.remove(chunk_id)
            self.kv_store.cpu_cache.pop(chunk_id, None)

            # Update metadata (chunk remains on disk)
            self.metadata_manager.update_storage_tier(chunk_id, StorageTier.DISK.value)

        self.evictions_from_cpu += 1

    def _add_to_heap(self, chunk_id: str, tier: StorageTier):
        """
        Add a chunk to the appropriate heap.

        Args:
            chunk_id: Chunk identifier
            tier: Storage tier
        """
        chunk_meta = self.metadata_manager.get_chunk(chunk_id)
        if chunk_meta is None:
            return

        scored_chunk = ScoredChunk(score=chunk_meta.score, chunk_id=chunk_id)

        if tier == StorageTier.GPU:
            heapq.heappush(self.gpu_heap, scored_chunk)
            self.gpu_heap_index[chunk_id] = len(self.gpu_heap) - 1
        else:
            heapq.heappush(self.cpu_heap, scored_chunk)
            self.cpu_heap_index[chunk_id] = len(self.cpu_heap) - 1

    def _remove_from_heap(self, chunk_id: str, tier: StorageTier):
        """
        Remove a chunk from the appropriate heap.

        Args:
            chunk_id: Chunk identifier
            tier: Storage tier
        """
        if tier == StorageTier.GPU:
            if chunk_id in self.gpu_heap_index:
                self.gpu_heap_index.pop(chunk_id)
        else:
            if chunk_id in self.cpu_heap_index:
                self.cpu_heap_index.pop(chunk_id)

    def _update_chunk_score(self, chunk_id: str, tier: StorageTier):
        """
        Update a chunk's score in the heap.

        Args:
            chunk_id: Chunk identifier
            tier: Storage tier
        """
        # Score is updated in metadata_manager
        # Heap is lazily updated during eviction/promotion
        pass

    def _update_all_access_freqs(self):
        """Update access frequencies for all chunks."""
        for chunk_meta in self.metadata_manager.metadata.values():
            chunk_meta.update_access_freq(self.metadata_manager.total_accesses)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_gpu_accesses = self.gpu_hits + self.gpu_misses
        total_cpu_accesses = self.cpu_hits + self.cpu_misses

        return {
            'gpu_hit_rate': self.gpu_hits / total_gpu_accesses if total_gpu_accesses > 0 else 0.0,
            'cpu_hit_rate': self.cpu_hits / total_cpu_accesses if total_cpu_accesses > 0 else 0.0,
            'gpu_cache_size': len(self.gpu_chunk_ids),
            'cpu_cache_size': len(self.cpu_chunk_ids),
            'gpu_cache_capacity': self.kv_store.max_gpu_chunks,
            'cpu_cache_capacity': self.kv_store.max_cpu_chunks,
            'evictions_from_gpu': self.evictions_from_gpu,
            'evictions_from_cpu': self.evictions_from_cpu,
            'promotions_to_gpu': self.promotions_to_gpu,
        }

    def get_heap_stats(self) -> Dict:
        """Get heap statistics for debugging."""
        return {
            'gpu_heap_size': len(self.gpu_heap),
            'cpu_heap_size': len(self.cpu_heap),
            'lowest_gpu_score': self.gpu_heap[0].score if self.gpu_heap else 0.0,
            'lowest_cpu_score': self.cpu_heap[0].score if self.cpu_heap else 0.0,
        }

    def clear(self):
        """Clear all cache state."""
        self.gpu_heap.clear()
        self.cpu_heap.clear()
        self.gpu_chunk_ids.clear()
        self.cpu_chunk_ids.clear()
        self.gpu_heap_index.clear()
        self.cpu_heap_index.clear()
        self.gpu_hits = 0
        self.gpu_misses = 0
        self.cpu_hits = 0
        self.cpu_misses = 0
        self.evictions_from_gpu = 0
        self.evictions_from_cpu = 0
        self.promotions_to_gpu = 0


class nullcontext:
    """Null context manager for when metrics is None."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass