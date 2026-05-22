"""
Radix Tree implementation for prefix indexing in IMPRESS.

Supports longest common prefix query and maintains token mappings
for compatibility with KV reordering.
"""

from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class RadixTreeNode:
    """
    Node in the Radix Tree for prefix indexing.

    Attributes:
        token: The token value at this node
        children: Child nodes keyed by token
        kv_chunks: List of chunk IDs containing KV for this prefix
        token_mapping: Mapping from original position to reordered position
        access_count: Number of times this prefix was accessed
    """
    token: int
    children: Dict[int, 'RadixTreeNode'] = field(default_factory=dict)
    kv_chunks: List[str] = field(default_factory=list)
    token_mapping: Dict[int, int] = field(default_factory=dict)
    access_count: int = 0

    def add_chunk(self, chunk_id: str):
        """Add a chunk ID to this node."""
        if chunk_id not in self.kv_chunks:
            self.kv_chunks.append(chunk_id)

    def get_reordered_position(self, original_pos: int) -> Optional[int]:
        """Get reordered position from original position."""
        return self.token_mapping.get(original_pos)

    def update_mapping(self, new_mapping: Dict[int, int]):
        """Update token mapping after KV reordering."""
        self.token_mapping = new_mapping.copy()


class RadixTree:
    """
    Radix Tree for efficient prefix lookup and KV reuse.

    Enables quick search for reusable prefix KVs by finding
    the longest common prefix among all previous requests.
    """

    def __init__(self):
        """Initialize an empty Radix Tree."""
        self.root = RadixTreeNode(token=-1)  # Dummy root
        self.total_prefixes = 0

    def insert_prefix(self, prefix_tokens: List[int], chunk_ids: List[str]) -> RadixTreeNode:
        """
        Insert a prefix sequence into the Radix Tree.

        Args:
            prefix_tokens: List of token values in the prefix
            chunk_ids: List of chunk IDs containing KV for this prefix

        Returns:
            The leaf node representing this prefix
        """
        node = self.root

        for i, token in enumerate(prefix_tokens):
            if token not in node.children:
                node.children[token] = RadixTreeNode(token=token)
            node = node.children[token]

        # Add chunk IDs to leaf node
        for chunk_id in chunk_ids:
            node.add_chunk(chunk_id)

        self.total_prefixes += 1
        return node

    def find_longest_common_prefix(
        self,
        query_tokens: List[int]
    ) -> tuple[List[int], Optional[RadixTreeNode]]:
        """
        Find the longest common prefix between query and stored prefixes.

        Args:
            query_tokens: List of query tokens

        Returns:
            Tuple of (matched_prefix_tokens, matching_node)
            - matched_prefix_tokens: List of matched token values
            - matching_node: The node representing the LCP, or None if no match
        """
        node = self.root
        matched_tokens = []

        for token in query_tokens:
            if token in node.children:
                matched_tokens.append(token)
                node = node.children[token]
            else:
                break

        return matched_tokens, node if matched_tokens else None

    def get_node_for_prefix(self, prefix_tokens: List[int]) -> Optional[RadixTreeNode]:
        """
        Get the node corresponding to a specific prefix.

        Args:
            prefix_tokens: List of prefix tokens

        Returns:
            The node if prefix exists, None otherwise
        """
        node = self.root

        for token in prefix_tokens:
            if token in node.children:
                node = node.children[token]
            else:
                return None

        return node

    def record_access(self, prefix_tokens: List[int]):
        """
        Record an access to a prefix for cache scoring.

        Args:
            prefix_tokens: List of prefix tokens accessed
        """
        node = self.root

        for token in prefix_tokens:
            if token in node.children:
                node = node.children[token]
                node.access_count += 1
            else:
                break

    def get_total_prefixes(self) -> int:
        """Get total number of prefixes stored."""
        return self.total_prefixes

    def clear(self):
        """Clear all prefixes from the tree."""
        self.root = RadixTreeNode(token=-1)
        self.total_prefixes = 0