"""
Similarity calculation utilities for IMPRESS.

Implements Jaccard similarity and head similarity calculations
used in the similarity-guided important token identification.
"""

from typing import Set, List
import numpy as np


def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    """
    Calculate Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First set of token indices
        set_b: Second set of token indices

    Returns:
        Jaccard similarity value between 0 and 1
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def calculate_head_similarity(head_indices: List[Set[int]]) -> float:
    """
    Calculate average pairwise Jaccard similarity among multiple heads.

    Args:
        head_indices: List of important token index sets for each head

    Returns:
        Average Jaccard similarity across all head pairs
    """
    n_heads = len(head_indices)
    if n_heads < 2:
        return 1.0

    similarities = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            sim = jaccard_similarity(head_indices[i], head_indices[j])
            similarities.append(sim)

    return np.mean(similarities) if similarities else 1.0


def calculate_expected_similarity(total_tokens: int, important_tokens: int) -> float:
    """
    Calculate expected Jaccard similarity for random selection.

    Based on the formula: E(Jaccard) = (k/n) / (2 - k/n)

    where n = total_tokens, k = important_tokens

    Args:
        total_tokens: Total number of tokens
        important_tokens: Number of important tokens to select

    Returns:
        Expected similarity for random selection
    """
    if total_tokens == 0:
        return 0.0

    ratio = important_tokens / total_tokens
    expected_jaccard = ratio / (2 - ratio) if ratio < 2 else 1.0
    return expected_jaccard


def calculate_similarity_threshold(
    total_tokens: int,
    important_tokens: int,
    alpha: float = 0.6
) -> float:
    """
    Calculate similarity threshold using alpha adjustment.

    Threshold = alpha * E(Jaccard)

    Args:
        total_tokens: Total number of tokens
        important_tokens: Number of important tokens to select
        alpha: Adjustment factor (default 0.6 from paper)

    Returns:
        Similarity threshold value
    """
    expected = calculate_expected_similarity(total_tokens, important_tokens)
    return alpha * expected