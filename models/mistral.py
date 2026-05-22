"""
Mistral model adapter for IMPRESS.

Provides integration with Mistral architecture models.
Mistral uses Grouped Query Attention (GQA) which differs from standard attention.
"""

from typing import List, Optional, Tuple
import torch
from .base import BaseModelAdapter


class MistralAdapter(BaseModelAdapter):
    """
    Adapter for Mistral architecture models.

    Mistral uses Grouped Query Attention (GQA) where:
    - num_key_value_heads may be less than num_attention_heads
    - This affects how probe heads are selected for ITF
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize Mistral adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        super().__init__(model_name, device)

        # Verify it's a Mistral model
        if "mistral" not in model_name.lower():
            print(f"Warning: Model '{model_name}' may not be a Mistral model")

        # Cache GQA parameters
        self._num_kv_heads = None

    def get_attention_layers(self) -> List[torch.nn.Module]:
        """
        Get all attention layers from the model.

        Returns:
            List of Mistral attention layer modules
        """
        layers = []
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                layers.append(layer.self_attn)
        return layers

    def get_num_heads(self, layer_idx: int) -> int:
        """
        Get number of attention heads (query heads) for a given layer.

        Args:
            layer_idx: Layer index

        Returns:
            Number of attention heads
        """
        return self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else 32

    def get_num_kv_heads(self, layer_idx: int) -> int:
        """
        Get number of key-value heads for GQA.

        In Mistral's GQA, num_kv_heads may be less than num_heads.

        Args:
            layer_idx: Layer index

        Returns:
            Number of key-value heads
        """
        if self._num_kv_heads is not None:
            return self._num_kv_heads

        if hasattr(self.config, 'num_key_value_heads'):
            self._num_kv_heads = self.config.num_key_value_heads
        else:
            # Fallback to num_heads if GQA not used
            self._num_kv_heads = self.get_num_heads(layer_idx)

        return self._num_kv_heads

    def get_head_dim(self, layer_idx: int) -> int:
        """
        Get dimension of each attention head.

        Args:
            layer_idx: Layer index

        Returns:
            Head dimension
        """
        if hasattr(self.config, 'head_dim'):
            return self.config.head_dim

        hidden_size = self.get_hidden_dim()
        num_heads = self.get_num_heads(layer_idx)
        return hidden_size // num_heads

    def get_kv_cache(
        self,
        layer_idx: int,
        head_idx: Optional[int] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get KV cache for a specific layer and head.

        Args:
            layer_idx: Layer index
            head_idx: Head index (None for all heads)

        Returns:
            Tuple of (keys, values) tensors or None
        """
        # Note: Actual KV cache access depends on the inference framework
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx]
            return None

        return None

    def get_num_layers(self) -> int:
        """
        Get total number of transformer layers.

        Returns:
            Number of layers
        """
        if hasattr(self.config, 'num_hidden_layers'):
            return self.config.num_hidden_layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        return 32  # Default fallback

    def get_probe_head_indices(self, layer_idx: int, num_probes: int = 3) -> List[int]:
        """
        Get indices of probe heads for ITF.

        For Mistral with GQA, we need to select heads carefully:
        - Use KV head indices (not query head indices)
        - Each KV head serves multiple query heads

        Args:
            layer_idx: Layer index
            num_probes: Number of probe heads to select

        Returns:
            List of KV head indices
        """
        num_kv_heads = self.get_num_kv_heads(layer_idx)
        return list(range(min(num_probes, num_kv_heads)))

    def get_kv_projection(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project hidden states to K and V tensors.

        Args:
            layer_idx: Layer index
            hidden_states: Hidden states tensor

        Returns:
            Tuple of (keys, values) tensors
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn

            # Mistral uses v_proj and k_proj
            if hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
                keys = attn.k_proj(hidden_states)
                values = attn.v_proj(hidden_states)

                # Reshape for GQA
                num_kv_heads = self.get_num_kv_heads(layer_idx)
                head_dim = self.get_head_dim(layer_idx)

                # keys shape: (batch, seq, num_kv_heads * head_dim)
                # reshape to: (batch, seq, num_kv_heads, head_dim)
                keys = keys.view(keys.shape[0], keys.shape[1], num_kv_heads, head_dim)
                values = values.view(values.shape[0], values.shape[1], num_kv_heads, head_dim)

                return keys, values

        return None, None

    def get_model_config(self) -> dict:
        """
        Get model configuration parameters.

        Returns:
            Dictionary of model configuration
        """
        return {
            'model_type': 'mistral',
            'num_layers': self.get_num_layers(),
            'hidden_size': self.get_hidden_dim(),
            'num_heads': self.get_num_heads(0),
            'num_kv_heads': self.get_num_kv_heads(0),
            'head_dim': self.get_head_dim(0),
            'vocab_size': self.config.vocab_size if hasattr(self.config, 'vocab_size') else 32000,
            'max_position_embeddings': (
                self.config.max_position_embeddings
                if hasattr(self.config, 'max_position_embeddings')
                else 32768
            ),
            'sliding_window': (
                self.config.sliding_window
                if hasattr(self.config, 'sliding_window')
                else None
            ),
        }