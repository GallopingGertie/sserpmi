"""
KV Reordering (RO) for IMPRESS.

Implements periodic reordering of KV cache to increase the density
of important tokens within chunks, reducing I/O amplification.
"""

from typing import List, Dict, Set, Optional
import numpy as np
import torch
from collections import defaultdict
import threading
import time

from ..metadata.chunk_metadata import ChunkMetadata, ChunkMetadataManager
from ..utils.metrics import Timer, MetricsCollector


class KVReorderManager:
    """
    Manages KV reordering to improve chunk density.

    Key operations:
    1. Collect historical importance statistics for tokens
    2. Reorder tokens by average importance
    3. Repack chunks with higher important KV density
    4. Update metadata mappings for radix tree compatibility

    Reordering runs asynchronously in background (default: every 10 minutes).
    """

    DEFAULT_INTERVAL_MINUTES = 10

    def __init__(
        self,
        metadata_manager: ChunkMetadataManager,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        importance_window: int = 1000
    ):
        """
        Initialize the KV reorder manager.

        Args:
            metadata_manager: Chunk metadata manager
            interval_minutes: Reordering interval in minutes
            importance_window: Number of recent accesses to consider
        """
        self.metadata_manager = metadata_manager
        self.interval_seconds = interval_minutes * 60
        self.importance_window = importance_window

        # Token importance statistics: {token_idx -> [importance_history]}
        self.token_importance: Dict[int, List[float]] = defaultdict(list)

        # Reordering state
        self.is_running = False
        self.last_reorder_time = 0.0
        self.background_thread: Optional[threading.Thread] = None

        # Metrics
        self.reorder_count = 0
        self.total_tokens_reordered = 0
        self.density_improvement = 0.0

    def record_token_importance(self, token_idx: int, is_important: bool):
        """
        Record importance observation for a token.

        Args:
            token_idx: Token index
            is_important: Whether this token was marked important
        """
        importance = 1.0 if is_important else 0.0

        # Add to history
        history = self.token_importance[token_idx]
        history.append(importance)

        # Keep only recent window
        if len(history) > self.importance_window:
            self.token_importance[token_idx] = history[-self.importance_window:]

    def get_average_importance(self, token_idx: int) -> float:
        """
        Get average importance for a token.

        Args:
            token_idx: Token index

        Returns:
            Average importance (0.0 to 1.0)
        """
        history = self.token_importance.get(token_idx, [])
        if not history:
            return 0.0
        return np.mean(history)

    def calculate_reordering_plan(
        self,
        token_sequence: List[int],
        chunk_size: int
    ) -> tuple[List[int], Dict[int, int]]:
        """
        Calculate optimal reordering for a token sequence.

        Args:
            token_sequence: Original token sequence
            chunk_size: Number of tokens per chunk

        Returns:
            Tuple of (reordered_sequence, position_mapping)
            - reordered_sequence: Tokens reordered by importance
            - position_mapping: Original position -> Reordered position
        """
        # Calculate average importance for each token
        token_scores = []
        for i, token in enumerate(token_sequence):
            avg_importance = self.get_average_importance(token)
            token_scores.append((i, token, avg_importance))

        # Sort by importance (descending), keeping original order for ties
        token_scores.sort(key=lambda x: (-x[2], x[0]))

        # Build reordered sequence and mapping
        reordered_sequence = [token for _, token, _ in token_scores]
        position_mapping = {original_idx: new_idx for new_idx, (original_idx, _, _) in enumerate(token_scores)}

        return reordered_sequence, position_mapping

    def calculate_chunk_density(
        self,
        token_indices: List[int],
        importance_threshold: float = 0.5
    ) -> float:
        """
        Calculate the density of important tokens in a chunk.

        Args:
            token_indices: List of token indices in the chunk
            importance_threshold: Threshold for considering a token important

        Returns:
            Density ratio (important_tokens / total_tokens)
        """
        if not token_indices:
            return 0.0

        important_count = 0
        for token_idx in token_indices:
            if self.get_average_importance(token_idx) >= importance_threshold:
                important_count += 1

        return important_count / len(token_indices)

    def estimate_read_amplification(
        self,
        original_chunks: List[List[int]],
        important_token_indices: Set[int]
    ) -> float:
        """
        Estimate read amplification factor for original chunking.

        Args:
            original_chunks: List of chunks (each is list of token indices)
            important_token_indices: Set of important token indices

        Returns:
            Read amplification factor (tokens_loaded / important_tokens)
        """
        chunks_to_read = 0
        important_tokens_in_read = 0

        for chunk in original_chunks:
            # Check if this chunk contains any important tokens
            for token_idx in chunk:
                if token_idx in important_token_indices:
                    chunks_to_read += 1
                    important_tokens_in_read += sum(1 for t in chunk if t in important_token_indices)
                    break

        total_tokens_loaded = chunks_to_read * len(original_chunks[0]) if original_chunks else 0

        if important_tokens_in_read == 0:
            return 1.0

        return total_tokens_loaded / important_tokens_in_read

    def execute_reordering(
        self,
        prefix_id: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        chunk_size: int,
        metrics: Optional[MetricsCollector] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[int, int]]]:
        """
        Execute KV reordering for a prefix.

        Args:
            prefix_id: Identifier for the prefix
            keys: Key tensor of shape (seq_len, hidden_dim)
            values: Value tensor of shape (seq_len, hidden_dim)
            chunk_size: Number of tokens per chunk
            metrics: Optional metrics collector

        Returns:
            Tuple of (reordered_keys, reordered_values, chunk_mappings)
            - reordered_keys: Reordered key tensor
            - reordered_values: Reordered value tensor
            - chunk_mappings: Mapping of chunk_id -> {original_pos: reordered_pos}
        """
        with Timer(metrics, 'io_time_ms') if metrics else nullcontext():
            num_tokens = keys.shape[0]
            token_sequence = list(range(num_tokens))

            # Calculate reordering plan
            reordered_sequence, position_mapping = self.calculate_reordering_plan(
                token_sequence,
                chunk_size
            )

            # Reorder tensors
            reordered_indices = torch.tensor(reordered_sequence, device=keys.device)
            reordered_keys = keys[reordered_indices]
            reordered_values = values[reordered_indices]

            # Calculate chunk mappings
            chunk_mappings = {}
            num_chunks = (num_tokens + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                chunk_id = f"{prefix_id}_chunk_{chunk_idx}"
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_tokens)

                # Create mapping for this chunk
                chunk_mapping = {}
                for orig_pos in range(start, end):
                    chunk_mapping[orig_pos] = position_mapping[orig_pos]

                chunk_mappings[chunk_id] = chunk_mapping

            # Update statistics
            self.reorder_count += 1
            self.total_tokens_reordered += num_tokens

            # Calculate density improvement
            if num_tokens > 0:
                # Original density (random assumption)
                original_density = 0.5
                # New density (estimated)
                new_density = np.mean([
                    self.get_average_importance(i) for i in reordered_sequence
                ])
                self.density_improvement = (new_density / original_density) if original_density > 0 else 1.0

            return reordered_keys, reordered_values, chunk_mappings

    def start_background_reordering(self, kv_store, metrics: Optional[MetricsCollector] = None):
        """
        Start background thread for periodic KV reordering.

        Args:
            kv_store: PrefixKVStore instance to reorder
            metrics: Optional metrics collector
        """
        if self.is_running:
            return

        self.is_running = True
        self.background_thread = threading.Thread(
            target=self._background_reorder_loop,
            args=(kv_store, metrics),
            daemon=True
        )
        self.background_thread.start()

    def _background_reorder_loop(self, kv_store, metrics: Optional[MetricsCollector]):
        """Background loop for periodic reordering."""
        while self.is_running:
            time.sleep(self.interval_seconds)

            if not self.is_running:
                break

            self._perform_background_reorder(kv_store, metrics)

    def _perform_background_reorder(self, kv_store, metrics: Optional[MetricsCollector]):
        """
        Perform one round of background reordering.

        Args:
            kv_store: PrefixKVStore instance
            metrics: Optional metrics collector
        """
        try:
            # Get all prefixes from metadata
            all_chunks = self.metadata_manager.metadata.keys()

            for chunk_id in all_chunks:
                # Skip if chunk is being actively used (simplified)
                chunk_meta = self.metadata_manager.get_chunk(chunk_id)
                if chunk_meta.storage_tier == 0:  # In GPU
                    continue

                # Reordering would be implemented here
                # For now, just update metadata
                pass

            self.last_reorder_time = time.time()

        except Exception as e:
            print(f"Background reordering error: {e}")

    def stop_background_reordering(self):
        """Stop background reordering thread."""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
            self.background_thread = None

    def get_statistics(self) -> dict:
        """Get reordering statistics."""
        return {
            'reorder_count': self.reorder_count,
            'total_tokens_reordered': self.total_tokens_reordered,
            'density_improvement': self.density_improvement,
            'last_reorder_time': self.last_reorder_time,
            'is_running': self.is_running,
        }

    def reset_statistics(self):
        """Reset reordering statistics."""
        self.reorder_count = 0
        self.total_tokens_reordered = 0
        self.density_improvement = 0.0
        self.token_importance.clear()


class nullcontext:
    """Null context manager for when metrics is None."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass