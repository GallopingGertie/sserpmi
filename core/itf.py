"""
Similarity-Guided Important Token Identification (ITF) for IMPRESS.

Implements the core algorithm that identifies important tokens by loading
only keys from a subset of probe heads, leveraging the observation that
important token indices are highly similar across heads.
"""

from typing import List, Set, Optional, Tuple
import torch
import numpy as np

from ..utils.similarity import (
    jaccard_similarity,
    calculate_head_similarity,
    calculate_similarity_threshold
)
from ..utils.metrics import Timer, MetricsCollector


class ImportantTokenFilter:
    """
    Identifies important tokens using similarity-guided algorithm.

    Loads only keys from probe heads to reduce I/O overhead,
    then uses similarity to approximate important tokens for all heads.

    Parameters from paper:
    - PROBE_HEAD_COUNT: 3 (number of heads used as probes)
    - ALPHA: 0.6 (adjustment factor for similarity threshold)
    - DEFAULT_KV_RATIO: 0.25 (default percentage of KVs to retain)
    """

    PROBE_HEAD_COUNT = 3
    ALPHA = 0.6
    DEFAULT_KV_RATIO = 0.25

    def __init__(
        self,
        probe_head_count: int = PROBE_HEAD_COUNT,
        alpha: float = ALPHA,
        kv_ratio: float = DEFAULT_KV_RATIO
    ):
        """
        Initialize the important token filter.

        Args:
            probe_head_count: Number of probe heads to use (default: 3)
            alpha: Adjustment factor for similarity threshold (default: 0.6)
            kv_ratio: Ratio of KVs to retain (default: 0.25 = 25%)
        """
        self.probe_head_count = probe_head_count
        self.alpha = alpha
        self.kv_ratio = kv_ratio

        # Statistics for analysis
        self.similarity_enabled_layers = 0
        self.similarity_disabled_layers = 0
        self.tokens_loaded_total = 0
        self.tokens_skipped_total = 0

    def identify_important_tokens(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        layer_idx: int,
        total_heads: int,
        metrics: Optional[MetricsCollector] = None
    ) -> Tuple[Set[int], bool]:
        """
        Identify important token indices for a given query.

        Args:
            query: Query tensor of shape (hidden_dim,)
            keys: Key tensor of shape (num_heads, seq_len, head_dim)
            layer_idx: Current transformer layer index
            total_heads: Total number of attention heads
            metrics: Optional metrics collector

        Returns:
            Tuple of (important_token_indices, similarity_enabled)
            - important_token_indices: Set of important token indices
            - similarity_enabled: Whether similarity-guided mode was used
        """
        num_tokens = keys.shape[1]
        num_important = max(1, int(num_tokens * self.kv_ratio))

        # Select probe heads (first N heads as suggested by paper)
        probe_indices = list(range(min(self.probe_head_count, total_heads)))

        # Step 1: Load only probe head keys to GPU
        probe_keys = keys[probe_indices]  # Shape: (probe_count, seq_len, head_dim)

        with Timer(metrics, 'io_time_ms') if metrics else nullcontext():
            self.tokens_loaded_total += num_tokens * self.probe_head_count

        # Step 2: Calculate attention weights for probe heads
        probe_attention = self._calculate_attention(query, probe_keys)

        # Step 3: Identify important tokens for each probe head
        probe_important_sets = []
        for head_idx in range(len(probe_indices)):
            attention_weights = probe_attention[head_idx]
            top_k_indices = torch.topk(
                attention_weights,
                min(num_important, len(attention_weights))
            ).indices
            probe_important_sets.append(set(top_k_indices.cpu().numpy()))

        # Step 4: Calculate similarity among probe heads
        similarity = calculate_head_similarity(probe_important_sets)

        # Step 5: Calculate threshold and decide mode
        threshold = calculate_similarity_threshold(
            total_tokens=num_tokens,
            important_tokens=num_important,
            alpha=self.alpha
        )

        if similarity >= threshold:
            # Similarity-guided mode: use probe results
            self.similarity_enabled_layers += 1

            # Vote on common important tokens
            common_tokens = set.intersection(*probe_important_sets)

            # If no intersection, use union
            if not common_tokens:
                common_tokens = set.union(*probe_important_sets)

            # Trim to desired number if needed
            if len(common_tokens) > num_important:
                common_tokens = set(list(common_tokens)[:num_important])

            # Estimate tokens saved (would have loaded all heads)
            self.tokens_skipped_total += num_tokens * (total_heads - self.probe_head_count)

            return common_tokens, True
        else:
            # Fallback mode: disable similarity for this layer
            self.similarity_disabled_layers += 1

            # Load all keys and compute attention
            all_attention = self._calculate_attention(query, keys)

            # Find important tokens across all heads
            # Average attention across heads
            avg_attention = torch.mean(all_attention, dim=0)
            top_k_indices = torch.topk(
                avg_attention,
                min(num_important, len(avg_attention))
            ).indices

            important_tokens = set(top_k_indices.cpu().numpy())
            return important_tokens, False

    def _calculate_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention scores between query and keys.

        Args:
            query: Query tensor of shape (head_dim,)
            keys: Key tensor of shape (num_heads, seq_len, head_dim)

        Returns:
            Attention weights of shape (num_heads, seq_len)
        """
        # Expand query to match key dimensions
        num_heads, seq_len, head_dim = keys.shape
        query_expanded = query.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, head_dim)

        # Calculate attention scores (simplified, no masking)
        # In full implementation, would include scaling and softmax
        scores = torch.einsum('bhd,bshd->bhs', query_expanded, keys).squeeze(0)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores.squeeze(0), dim=-1)

        return attention_weights

    def get_statistics(self) -> dict:
        """Get statistics about ITF performance."""
        total_layers = self.similarity_enabled_layers + self.similarity_disabled_layers
        similarity_rate = (
            self.similarity_enabled_layers / total_layers
            if total_layers > 0 else 0.0
        )

        return {
            'similarity_enabled_layers': self.similarity_enabled_layers,
            'similarity_disabled_layers': self.similarity_disabled_layers,
            'total_layers': total_layers,
            'similarity_rate': similarity_rate,
            'tokens_loaded': self.tokens_loaded_total,
            'tokens_skipped': self.tokens_skipped_total,
            'io_reduction_ratio': (
                self.tokens_skipped_total / (self.tokens_loaded_total + self.tokens_skipped_total)
                if (self.tokens_loaded_total + self.tokens_skipped_total) > 0 else 0.0
            )
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        self.similarity_enabled_layers = 0
        self.similarity_disabled_layers = 0
        self.tokens_loaded_total = 0
        self.tokens_skipped_total = 0

    def set_kv_ratio(self, ratio: float):
        """
        Update the KV retention ratio.

        Args:
            ratio: New ratio (between 0 and 1)
        """
        self.kv_ratio = max(0.0, min(1.0, ratio))


class nullcontext:
    """Null context manager for when metrics is None."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass