"""
Basic tests for IMPRESS core functionality.

These tests verify the key components work as expected.
"""

import torch
import numpy as np
from impress.utils.similarity import (
    jaccard_similarity,
    calculate_head_similarity,
    calculate_similarity_threshold,
)


def test_jaccard_similarity():
    """Test Jaccard similarity calculation."""
    set_a = {1, 2, 3, 4}
    set_b = {2, 3, 4, 5}

    sim = jaccard_similarity(set_a, set_b)
    assert sim == 3 / 5, f"Expected 0.6, got {sim}"
    print("✓ Jaccard similarity test passed")


def test_jaccard_identical_sets():
    """Test Jaccard similarity with identical sets."""
    set_a = {1, 2, 3}
    set_b = {1, 2, 3}

    sim = jaccard_similarity(set_a, set_b)
    assert sim == 1.0, f"Expected 1.0, got {sim}"
    print("✓ Identical sets test passed")


def test_jaccard_disjoint_sets():
    """Test Jaccard similarity with disjoint sets."""
    set_a = {1, 2, 3}
    set_b = {4, 5, 6}

    sim = jaccard_similarity(set_a, set_b)
    assert sim == 0.0, f"Expected 0.0, got {sim}"
    print("✓ Disjoint sets test passed")


def test_calculate_head_similarity():
    """Test head similarity calculation."""
    head_indices = [
        {1, 2, 3, 4},
        {1, 2, 3, 5},
        {1, 2, 4, 6},
    ]

    similarity = calculate_head_similarity(head_indices)
    assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
    print(f"✓ Head similarity test passed (similarity: {similarity:.4f})")


def test_calculate_expected_similarity():
    """Test expected similarity calculation."""
    total_tokens = 100
    important_tokens = 25

    expected = calculate_expected_similarity(total_tokens, important_tokens)
    assert 0 <= expected <= 1, f"Expected similarity should be between 0 and 1"

    # Check formula: E(Jaccard) = (k/n) / (2 - k/n)
    ratio = important_tokens / total_tokens
    expected_calc = ratio / (2 - ratio)
    assert abs(expected - expected_calc) < 1e-6, "Expected similarity formula mismatch"

    print(f"✓ Expected similarity test passed (expected: {expected:.4f})")


def test_similarity_threshold():
    """Test similarity threshold calculation."""
    total_tokens = 100
    important_tokens = 25
    alpha = 0.6

    threshold = calculate_similarity_threshold(total_tokens, important_tokens, alpha)
    expected = 0.6 * (0.25 / (2 - 0.25))  # alpha * E(Jaccard)

    assert abs(threshold - expected) < 1e-6, f"Expected {expected}, got {threshold}"
    print(f"✓ Similarity threshold test passed (threshold: {threshold:.4f})")


def test_radix_tree():
    """Test Radix Tree prefix lookup."""
    from impress.metadata.radix_tree import RadixTree

    tree = RadixTree()

    # Insert some prefixes
    tree.insert_prefix([1, 2, 3], ["chunk_0"])
    tree.insert_prefix([1, 2, 3, 4], ["chunk_0", "chunk_1"])
    tree.insert_prefix([1, 5], ["chunk_2"])

    # Test longest common prefix lookup
    matched_tokens, node = tree.find_longest_common_prefix([1, 2, 3, 4, 5])

    assert matched_tokens == [1, 2, 3, 4], f"Expected [1, 2, 3, 4], got {matched_tokens}"
    assert node is not None, "Should find matching node"
    assert node.kv_chunks == ["chunk_0", "chunk_1"], "Should have correct chunks"

    print("✓ Radix Tree test passed")


def test_chunk_metadata():
    """Test chunk metadata and scoring."""
    from impress.metadata.chunk_metadata import ChunkMetadata

    metadata = ChunkMetadata(
        chunk_id="test_chunk",
        token_range=(0, 64),
        important_ratio=0.75,
        access_count=10,
        access_freq=0.5,
    )

    score = metadata.score
    expected_score = 0.5 * 0.75  # access_freq * important_ratio
    assert abs(score - expected_score) < 1e-6, f"Expected {expected_score}, got {score}"

    # Test important ratio update (EMA)
    initial_ratio = metadata.important_ratio
    metadata.update_important_ratio(1.0)  # All tokens important

    # Should move towards 1.0 but not immediately
    assert 0 < metadata.important_ratio <= 1.0, "Ratio should be between 0 and 1"
    assert metadata.important_ratio > initial_ratio, "Ratio should increase"

    print("✓ Chunk metadata test passed")


def test_storage_tier_enum():
    """Test StorageTier enumeration."""
    from impress.core.storage_tier import StorageTier

    assert StorageTier.GPU.value == 0
    assert StorageTier.CPU.value == 1
    assert StorageTier.DISK.value == 2

    assert str(StorageTier.GPU) == "GPU"
    assert str(StorageTier.CPU) == "CPU"
    assert str(StorageTier.DISK) == "DISK"

    print("✓ StorageTier enum test passed")


def run_all_tests():
    """Run all tests."""
    print("=== Running IMPRESS Core Functionality Tests ===\n")

    try:
        test_jaccard_similarity()
        test_jaccard_identical_sets()
        test_jaccard_disjoint_sets()
        test_calculate_head_similarity()
        test_calculate_expected_similarity()
        test_similarity_threshold()
        test_radix_tree()
        test_chunk_metadata()
        test_storage_tier_enum()

        print("\n=== All tests passed! ===")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)