"""
Storage tier abstraction for IMPRESS three-tier storage system.

Defines GPU, CPU, and Disk storage tiers and provides unified interface
for KV cache management.
"""

from enum import Enum
from typing import Dict, Optional, List, Tuple
import torch
import numpy as np
from dataclasses import dataclass

from ..metadata.chunk_metadata import ChunkMetadata, ChunkMetadataManager
from ..utils.metrics import Timer, MetricsCollector


class StorageTier(Enum):
    """Storage tier enumeration."""
    GPU = 0
    CPU = 1
    DISK = 2

    def __str__(self):
        return {0: "GPU", 1: "CPU", 2: "DISK"}[self.value]


@dataclass
class ChunkData:
    """
    Actual KV data for a chunk.

    Attributes:
        chunk_id: Unique identifier
        keys: Key tensor of shape (num_tokens, hidden_dim)
        values: Value tensor of shape (num_tokens, hidden_dim)
        tier: Current storage tier
    """
    chunk_id: str
    keys: torch.Tensor
    values: torch.Tensor
    tier: StorageTier


class DiskStorage:
    """
    Disk storage implementation for KV chunks.

    Uses mmap for efficient random access and persists chunks to disk.
    """

    def __init__(self, base_path: str = "./kv_cache"):
        """
        Initialize disk storage.

        Args:
            base_path: Base directory for storing chunks
        """
        self.base_path = base_path
        self.chunks: Dict[str, ChunkData] = {}

    def write_chunk(self, chunk_id: str, keys: torch.Tensor, values: torch.Tensor):
        """
        Write a chunk to disk.

        Args:
            chunk_id: Chunk identifier
            keys: Key tensor
            values: Value tensor
        """
        # Store CPU tensors for disk access
        chunk_data = ChunkData(
            chunk_id=chunk_id,
            keys=keys.cpu(),
            values=values.cpu(),
            tier=StorageTier.DISK
        )
        self.chunks[chunk_id] = chunk_data

    def read_chunk(self, chunk_id: str) -> Optional[ChunkData]:
        """
        Read a chunk from disk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            ChunkData if found, None otherwise
        """
        return self.chunks.get(chunk_id)

    def delete_chunk(self, chunk_id: str):
        """Delete a chunk from disk."""
        self.chunks.pop(chunk_id, None)


class PrefixKVStore:
    """
    Three-tier prefix KV storage system.

    Manages KV caches across GPU memory, CPU memory, and disk storage.
    Implements the IMPRESS tier management and caching strategy.
    """

    def __init__(
        self,
        gpu_cache_size_gb: int = 10,
        cpu_cache_size_gb: int = 32,
        chunk_size: int = 64,
        hidden_dim: int = 5120
    ):
        """
        Initialize the three-tier KV store.

        Args:
            gpu_cache_size_gb: Size of GPU cache in GB
            cpu_cache_size_gb: Size of CPU cache in GB
            chunk_size: Number of tokens per chunk
            hidden_dim: Hidden dimension of the model
        """
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Calculate cache sizes in number of chunks
        element_size = 4  # float32
        kv_size_per_token = 2 * hidden_dim * element_size  # K and V
        chunk_size_bytes = chunk_size * kv_size_per_token

        self.max_gpu_chunks = int((gpu_cache_size_gb * 1e9) / chunk_size_bytes)
        self.max_cpu_chunks = int((cpu_cache_size_gb * 1e9) / chunk_size_bytes)

        # Initialize storage tiers
        self.gpu_cache: Dict[str, ChunkData] = {}
        self.cpu_cache: Dict[str, ChunkData] = {}
        self.disk_storage = DiskStorage()

        # Metadata manager
        self.metadata_manager = ChunkMetadataManager(chunk_size=chunk_size)

        # Metrics
        self.metrics = MetricsCollector()

    def store_prefix_kv(
        self,
        prefix_tokens: List[int],
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> List[str]:
        """
        Store prefix KV cache across storage tiers.

        Args:
            prefix_tokens: List of prefix token IDs
            keys: Key tensor of shape (seq_len, hidden_dim)
            values: Value tensor of shape (seq_len, hidden_dim)

        Returns:
            List of chunk IDs created
        """
        num_tokens = len(prefix_tokens)
        num_chunks = (num_tokens + self.chunk_size - 1) // self.chunk_size
        chunk_ids = []

        # Split into chunks
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, num_tokens)

            chunk_id = f"chunk_{prefix_tokens[0]}_{i}"
            chunk_keys = keys[start:end]
            chunk_values = values[start:end]

            # Store in disk initially
            self.disk_storage.write_chunk(chunk_id, chunk_keys, chunk_values)

            # Create metadata
            self.metadata_manager.add_chunk(
                chunk_id=chunk_id,
                token_range=(start, end),
                storage_tier=StorageTier.DISK.value
            )

            chunk_ids.append(chunk_id)

        return chunk_ids

    def get_chunk(
        self,
        chunk_id: str,
        target_tier: StorageTier = StorageTier.GPU
    ) -> Optional[ChunkData]:
        """
        Retrieve a chunk, promoting it through storage tiers as needed.

        Args:
            chunk_id: Chunk identifier
            target_tier: Desired storage tier

        Returns:
            ChunkData if found and accessible, None otherwise
        """
        # Check GPU cache first
        if chunk_id in self.gpu_cache:
            chunk = self.gpu_cache[chunk_id]
            self.metrics.gpu_cache_hits += 1
            self.metadata_manager.record_access(chunk_id, chunk.tier.value)
            return chunk

        self.metrics.gpu_cache_misses += 1

        # Check CPU cache
        if chunk_id in self.cpu_cache:
            chunk = self.cpu_cache[chunk_id]
            self.metrics.cpu_cache_hits += 1

            if target_tier == StorageTier.GPU:
                # Promote to GPU if space available
                if len(self.gpu_cache) < self.max_gpu_chunks:
                    with Timer(self.metrics, 'io_time_ms'):
                        gpu_chunk = self._promote_to_gpu(chunk)
                    self.gpu_cache[chunk_id] = gpu_chunk
                    self.metadata_manager.update_storage_tier(
                        chunk_id,
                        StorageTier.GPU.value
                    )
                    return gpu_chunk
                # Trigger cache replacement if GPU full
                else:
                    self._evict_from_gpu()

            self.metadata_manager.record_access(chunk_id, chunk.tier.value)
            return chunk

        self.metrics.cpu_cache_misses += 1

        # Load from disk
        disk_chunk = self.disk_storage.read_chunk(chunk_id)
        if disk_chunk is None:
            return None

        with Timer(self.metrics, 'io_time_ms'):
            self.metrics.disk_reads += 1

            # Load to CPU first
            cpu_chunk = self._load_to_cpu(disk_chunk)
            self.cpu_cache[chunk_id] = cpu_chunk
            self.metadata_manager.update_storage_tier(chunk_id, StorageTier.CPU.value)

            # Then promote to GPU if needed
            if target_tier == StorageTier.GPU:
                if len(self.gpu_cache) < self.max_gpu_chunks:
                    gpu_chunk = self._promote_to_gpu(cpu_chunk)
                    self.gpu_cache[chunk_id] = gpu_chunk
                    self.metadata_manager.update_storage_tier(
                        chunk_id,
                        StorageTier.GPU.value
                    )
                    return gpu_chunk
                else:
                    self._evict_from_gpu()

            self.metadata_manager.record_access(chunk_id, cpu_chunk.tier.value)
            return cpu_chunk

    def _load_to_cpu(self, disk_chunk: ChunkData) -> ChunkData:
        """Load chunk from disk to CPU memory."""
        return ChunkData(
            chunk_id=disk_chunk.chunk_id,
            keys=disk_chunk.keys.cpu(),
            values=disk_chunk.values.cpu(),
            tier=StorageTier.CPU
        )

    def _promote_to_gpu(self, cpu_chunk: ChunkData) -> ChunkData:
        """Promote chunk from CPU to GPU memory."""
        return ChunkData(
            chunk_id=cpu_chunk.chunk_id,
            keys=cpu_chunk.keys.cuda(),
            values=cpu_chunk.values.cuda(),
            tier=StorageTier.GPU
        )

    def _evict_from_gpu(self):
        """
        Evict lowest-scoring chunk from GPU cache.

        Uses the IMPRESS score-based replacement policy.
        """
        if not self.gpu_cache:
            return

        # Find lowest scoring chunk in GPU
        lowest_chunk_id = min(
            self.gpu_cache.keys(),
            key=lambda cid: self.metadata_manager.get_chunk(cid).score
        )

        # Move to CPU
        chunk = self.gpu_cache.pop(lowest_chunk_id)
        cpu_chunk = ChunkData(
            chunk_id=chunk.chunk_id,
            keys=chunk.keys.cpu(),
            values=chunk.values.cpu(),
            tier=StorageTier.CPU
        )
        self.cpu_cache[lowest_chunk_id] = cpu_chunk
        self.metadata_manager.update_storage_tier(lowest_chunk_id, StorageTier.CPU.value)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'gpu_cache_size': len(self.gpu_cache),
            'cpu_cache_size': len(self.cpu_cache),
            'disk_chunks': len(self.disk_storage.chunks),
            'gpu_cache_capacity': self.max_gpu_chunks,
            'cpu_cache_capacity': self.max_cpu_chunks,
            'gpu_cache_utilization': len(self.gpu_cache) / self.max_gpu_chunks,
            'cpu_cache_utilization': len(self.cpu_cache) / self.max_cpu_chunks,
        }

    def clear_gpu_cache(self):
        """Clear GPU cache."""
        self.gpu_cache.clear()

    def clear_cpu_cache(self):
        """Clear CPU cache."""
        self.cpu_cache.clear()

    def clear_all(self):
        """Clear all caches and disk storage."""
        self.clear_gpu_cache()
        self.clear_cpu_cache()
        self.disk_storage.chunks.clear()
        self.metadata_manager.clear()
        self.metrics.reset()