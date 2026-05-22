"""
Example script demonstrating IMPRESS usage.

This script shows how to:
1. Initialize IMPRESS components
2. Run inference with prefix KV reuse
3. Benchmark performance
"""

import torch
from impress import (
    create_model_adapter,
    ImportantTokenFilter,
    KVReorderManager,
    CacheManager,
    PrefixKVStore,
    Benchmark,
    AccuracyEvaluator,
)


def main():
    """Run IMPRESS demonstration."""

    print("=== IMPRESS Demonstration ===\n")

    # Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # or "mistralai/Mistral-7B-v0.1"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Initialize model adapter
    print(f"Loading model: {MODEL_NAME}")
    model_adapter = create_model_adapter(MODEL_NAME, DEVICE)
    print(f"Model loaded with {model_adapter.get_num_layers()} layers")

    # Step 2: Initialize IMPRESS components
    hidden_dim = model_adapter.get_hidden_dim()

    kv_store = PrefixKVStore(
        gpu_cache_size_gb=10,
        cpu_cache_size_gb=32,
        chunk_size=64,
        hidden_dim=hidden_dim,
    )

    itf = ImportantTokenFilter(
        probe_head_count=3,
        alpha=0.6,
        kv_ratio=0.25,
    )

    kv_reorder_manager = KVReorderManager(
        metadata_manager=kv_store.metadata_manager,
        interval_minutes=10,
    )

    cache_manager = CacheManager(
        kv_store=kv_store,
        metadata_manager=kv_store.metadata_manager,
    )

    print("IMPRESS components initialized")

    # Step 3: Run benchmark
    print("\n--- Running Benchmark ---")

    benchmark = Benchmark(
        model_adapter=model_adapter,
        kv_store=kv_store,
        cache_manager=cache_manager,
        itf=itf,
    )

    # Sample prefixes (system prompts)
    prefixes = [
        "You are a helpful assistant. ",
        "Answer the following question concisely. ",
        "Think step by step. ",
    ]

    # Sample queries
    queries = [
        "What is the capital of France?",
        "Explain quantum computing.",
        "How does photosynthesis work?",
    ]

    # Run benchmark with different KV ratios
    results = benchmark.run_ttft_benchmark(
        prompt_prefixes=prefixes,
        queries=queries,
        kv_ratios=[0.10, 0.25, 0.50],
    )

    benchmark.print_summary()

    # Step 4: Evaluate accuracy
    print("\n--- Evaluating Accuracy ---")

    evaluator = AccuracyEvaluator(model_adapter=model_adapter)

    accuracy_results = evaluator.evaluate(
        kv_ratios=[0.10, 0.25, 0.50],
        few_shot_examples=5,
    )

    evaluator.print_summary()

    # Step 5: Show statistics
    print("\n--- Component Statistics ---")

    print(f"\nITF Statistics:")
    itf_stats = itf.get_statistics()
    print(f"  Similarity enabled layers: {itf_stats['similarity_enabled_layers']}")
    print(f"  Similarity disabled layers: {itf_stats['similarity_disabled_layers']}")
    print(f"  Similarity rate: {itf_stats['similarity_rate']:.2%}")
    print(f"  I/O reduction: {itf_stats['io_reduction_ratio']:.2%}")

    print(f"\nCache Statistics:")
    cache_stats = cache_manager.get_cache_stats()
    print(f"  GPU cache hit rate: {cache_stats['gpu_hit_rate']:.2%}")
    print(f"  CPU cache hit rate: {cache_stats['cpu_hit_rate']:.2%}")
    print(f"  GPU cache size: {cache_stats['gpu_cache_size']}/{cache_stats['gpu_cache_capacity']}")
    print(f"  CPU cache size: {cache_stats['cpu_cache_size']}/{cache_stats['cpu_cache_capacity']}")

    print(f"\nStorage Statistics:")
    storage_stats = kv_store.get_cache_stats()
    print(f"  GPU cache utilization: {storage_stats['gpu_cache_utilization']:.2%}")
    print(f"  CPU cache utilization: {storage_stats['cpu_cache_utilization']:.2%}")
    print(f"  Total disk chunks: {storage_stats['disk_chunks']}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()