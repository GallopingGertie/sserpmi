"""
Metadata management for IMPRESS system.

Provides prefix indexing and chunk metadata management.
"""

from .radix_tree import RadixTreeNode, RadixTree
from .chunk_metadata import ChunkMetadata, ChunkInfo

__all__ = [
    'RadixTreeNode',
    'RadixTree',
    'ChunkMetadata',
    'ChunkInfo',
]