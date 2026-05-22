# IMPRESS

**An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**

This is a reimplementation of the IMPRESS system from [USENIX FAST 2025](https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress).

## Overview

IMPRESS addresses the Time-to-First-Token (TTFT) bottleneck in LLM inference when using long contexts. By intelligently managing KV caches across three storage tiers (GPU, CPU, disk), IMPRESS achieves:

- **Up to 2.8× reduction in TTFT**
- **Up to 3.8× reduction in I/O time**
- **< 0.2% accuracy drop**

## Key Components

### 1. Important Token Filter (ITF)
Identifies important tokens using only keys from a subset of "probe heads", leveraging the observation that important token indices are highly similar across attention heads.

### 2. KV Reordering (RO)
Periodically reorders KV cache by importance to increase the density of important tokens within chunks, reducing I/O amplification.

### 3. Score-based Cache Management
Implements dual-tier cache (GPU + CPU) with score-based replacement: `score = access_freq × importance_ratio`, ensuring high-value chunks are preferentially cached.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from impress import (
    create_model_adapter,
    ImportantTokenFilter,
    PrefixKVStore,
    CacheManager,
    Benchmark,
)

# Initialize model
model = create_model_adapter("meta-llama/Llama-2-7b-hf", "cuda")

# Initialize IMPRESS components
kv_store = PrefixKVStore(gpu_cache_size_gb=10, cpu_cache_size_gb=32)
itf = ImportantTokenFilter(kv_ratio=0.25)
cache_manager = CacheManager(kv_store, kv_store.metadata_manager)

# Run benchmark
benchmark = Benchmark(model, kv_store, cache_manager, itf)
results = benchmark.run_ttft_benchmark(prefixes, queries)
benchmark.print_summary()
```

See `scripts/demo.py` for a complete example.

## Project Structure

```
impress/
├── core/                   # Core IMPRESS components
│   ├── itf.py             # Important Token Filter
│   ├── kv_reorder.py      # KV Reordering
│   ├── cache_manager.py   # Cache Management
│   └── storage_tier.py    # Three-tier Storage
├── metadata/              # Metadata management
│   ├── radix_tree.py      # Prefix indexing
│   └── chunk_metadata.py  # Chunk metadata
├── utils/                 # Utilities
│   ├── similarity.py      # Similarity calculations
│   └── metrics.py         # Performance metrics
├── models/                # Model adapters
│   ├── base.py            # Base adapter
│   ├── llama.py           # Llama adapter
│   └── mistral.py         # Mistral adapter
├── evaluation/            # Evaluation modules
│   ├── benchmark.py       # Performance benchmarking
│   └── accuracy.py        # Accuracy evaluation
└── scripts/               # Running scripts
    └── demo.py            # Demo script
```

## Supported Models

- **Llama**: Llama-2, Llama-3
- **Mistral**: Mistral-7B, Mixtral

## Configuration

Key parameters from the paper:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `probe_head_count` | 3 | Number of heads used as probes |
| `alpha` | 0.6 | Similarity threshold adjustment |
| `kv_ratio` | 0.25 | Percentage of KVs to retain |
| `chunk_size` | 64 | Tokens per chunk |

## Evaluation

Run the demo script to see IMPRESS in action:

```bash
python scripts/demo.py
```

## Reference

**Paper**: IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference
**Conference**: USENIX FAST 2025
**Authors**: Weijian Chen, Shuibing He, Haoyang Qu, Ruidong Zhang, Siling Yang, Ping Chen, Yi Zheng, Baoxing Huai, Gang Chen

## License

This implementation is for research purposes. Please cite the original paper if you use this code.

## Status

This is a **core functionality verification** implementation. The key components (ITF, KV Reordering, Cache Management) are implemented but integration with actual inference frameworks like FlexGen requires additional work.