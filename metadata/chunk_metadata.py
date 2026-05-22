"""
Chunk metadata management for IMPRESS.

Defines metadata structures for KV chunks including importance scores
and token mappings for KV reordering.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ChunkMetadata:
    """
    Metadata for a KV chunk.

    Attributes:
        chunk_id: Unique identifier for this chunk
        token_range: (start_idx, end_idx) of tokens in this chunk
        important_ratio: Proportion of important tokens (moving average)
        access_count: Number of times this chunk was accessed
        access_freq: Access frequency relative to other chunks
        token_mapping: Mapping from original to reordered positions
        storage_tier: Where this chunk is stored (0=GPU, 1=CPU, 2=Disk)
        last_access_time: Timestamp of last access
    """
    chunk_id: str
    token_range: tuple[int, int]
    important_ratio: float = 0.5
    access_count: int = 0
    access_freq: float = 1.0
    token_mapping: Dict[int, int] = field(default_factory=dict)
    storage_tier: int = 2  # Default to disk
    last_access_time: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """
        Calculate the importance score for this chunk.

        Score = access_freq × important_ratio

        Higher score = more likely to be cached in GPU.
        """
        return self.access_freq * self.important_ratio

    def update_important_ratio(self, new_ratio: float, decay: float = 0.7):
        """
        Update important ratio using exponential moving average.

        Args:
            new_ratio: New observed importance ratio
            decay: Decay factor for EMA (default 0.7)
        """
        self.important_ratio = decay * self.important_ratio + (1 - decay) * new_ratio

    def record_access(self):
        """Record an access to this chunk."""
        self.access_count += 1
        self.last_access_time = time.time()

    def update_access_freq(self, total_accesses: int):
        """
        Update relative access frequency.

        Args:
            total_accesses: Total number of accesses across all chunks
        """
        self.access_freq = self.access_count / total_accesses if total_accesses > 0 else 0.0

    def get_reordered_position(self, original_pos: int) -> Optional[int]:
        """Get reordered position from original position."""
        return self.token_mapping.get(original_pos, original_pos)


@dataclass
class ChunkInfo:
    """
    Information about chunk location and status.

    Used for quick lookup without loading chunk data.
    """
    chunk_id: str
    storage_tier: int  # 0=GPU, 1=CPU, 2=Disk
    size_bytes: int
    num_tokens: int
    kv_shape: tuple[int, int]  # (seq_len, hidden_dim)

    def is_in_gpu(self) -> bool:
        """Check if chunk is in GPU memory."""
        return self.storage_tier == 0

    def is_in_cpu(self) -> bool:
        """Check if chunk is in CPU memory."""
        return self.storage_tier == 1

    def is_on_disk(self) -> bool:
        """Check if chunk is on disk."""
        return self.storage_tier == 2


class ChunkMetadataManager:
    """
    Manager for all chunk metadata.

    Provides centralized access to chunk information and scores.
    """

    def __init__(self, chunk_size: int = 64):
        """
        Initialize chunk metadata manager.

        Args:
            chunk_size: Number of tokens per chunk
        """
        self.chunk_size = chunk_size
        self.metadata: Dict[str, ChunkMetadata] = {}
        self.total_accesses = 0

    def add_chunk(
        self,
        chunk_id: str,
        token_range: tuple[int, int],
        storage_tier: int = 2
    ) -> ChunkMetadata:
        """
        Add a new chunk with metadata.

        Args:
            chunk_id: Unique chunk identifier
            token_range: (start_idx, end_idx) of tokens
            storage_tier: Initial storage location

        Returns:
            Created ChunkMetadata object
        """
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            token_range=token_range,
            storage_tier=storage_tier
        )
        self.metadata[chunk_id] = metadata
        return metadata

    def get_chunk(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get metadata for a chunk."""
        return self.metadata.get(chunk_id)

    def record_access(self, chunk_id: str, important_ratio: float):
        """
        Record access to a chunk and update its metadata.

        Args:
            chunk_id: Chunk identifier
            important_ratio: Observed importance ratio for this access
        """
        if chunk_id in self.metadata:
            chunk = self.metadata[chunk_id]
            chunk.record_access()
            chunk.update_important_ratio(important_ratio)
            self.total_accesses += 1

            # Update all access frequencies periodically
            if self.total_accesses % 100 == 0:
                self._update_all_access_freqs()

    def _update_all_access_freqs(self):
        """Update access frequency for all chunks."""
        for chunk in self.metadata.values():
            chunk.update_access_freq(self.total_accesses)

    def get_top_scoring_chunks(self, n: int) -> List[ChunkMetadata]:
        """
        Get the n highest-scoring chunks.

        Args:
            n: Number of chunks to return

        Returns:
            List of ChunkMetadata sorted by score descending
        """
        sorted_chunks = sorted(
            self.metadata.values(),
            key=lambda x: x.score,
            reverse=True
        )
        return sorted_chunks[:n]

    def update_storage_tier(self, chunk_id: str, new_tier: int):
        """
        Update the storage tier for a chunk.

        Args:
            chunk_id: Chunk identifier
            new_tier: New storage tier (0=GPU, 1=CPU, 2=Disk)
        """
        if chunk_id in self.metadata:
            self.metadata[chunk_id].storage_tier = new_tier

    def update_token_mapping(self, chunk_id: str, mapping: Dict[int, int]):
        """
        Update token mapping for a chunk after KV reordering.

        Args:
            chunk_id: Chunk identifier
            mapping: New token position mapping
        """
        if chunk_id in self.metadata:
            self.metadata[chunk_id].token_mapping = mapping.copy()

    def clear(self):
        """Clear all metadata."""
        self.metadata.clear()
        self.total_accesses = 0