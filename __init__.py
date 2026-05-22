"""
IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System

Implementation of the USENIX FAST 2025 paper for reducing TTFT in LLM inference
through intelligent KV cache management.

Key Components:
- ITF (Important Token Filter): Similarity-guided token identification
- KV Reordering: Improves chunk density by reordering by importance
- Score-based Cache Management: Dual-tier cache with score-based eviction
"""

from .core import (
    ImportantTokenFilter,
    KVReorderManager,
    CacheManager,
    PrefixKVStore,
    StorageTier,
)
from .metadata import RadixTree, ChunkMetadata, ChunkMetadataManager
from .models import create_model_adapter, BaseModelAdapter, LlamaAdapter, MistralAdapter
from .evaluation import Benchmark, AccuracyEvaluator

__version__ = "0.1.0"
__author__ = "IMPRESS Implementation Team"

__all__ = [
    # Core components
    'ImportantTokenFilter',
    'KVReorderManager',
    'CacheManager',
    'PrefixKVStore',
    'StorageTier',
    # Metadata
    'RadixTree',
    'ChunkMetadata',
    'ChunkMetadataManager',
    # Models
    'create_model_adapter',
    'BaseModelAdapter',
    'LlamaAdapter',
    'MistralAdapter',
    # Evaluation
    'Benchmark',
    'AccuracyEvaluator',
]