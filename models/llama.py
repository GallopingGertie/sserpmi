"""
Llama model adapter for IMPRESS.

Provides integration with Llama architecture models (Llama-2, Llama-3, etc.).
"""

from typing import List, Optional, Tuple
import torch
from .base import BaseModelAdapter


class LlamaAdapter(BaseModelAdapter):
    """
    Adapter for Llama architecture models.

    Llama uses standard transformer attention with RoPE (Rotary Position Embedding).
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize Llama adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        super().__init__(model_name, device)

        # Verify it's a Llama model
        if "llama" not in model_name.lower():
            print(f"Warning: Model '{model_name}' may not be a Llama model")

    def get_attention_layers(self) -> List[torch.nn.Module]:
        """
        Get all attention layers from the model.

        Returns:
            List of Llama attention layer modules
        """
        layers = []
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                layers.append(layer.self_attn)
        return layers

    def get_num_heads(self, layer_idx: int) -> int:
        """
        Get number of attention heads for a given layer.

        Args:
            layer_idx: Layer index

        Returns:
            Number of attention heads
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer.self_attn, 'num_heads'):
                return layer.self_attn.num_heads
            elif hasattr(layer.self_attn, 'n_head'):
                return layer.self_attn.n_head
            elif hasattr(layer.self_attn, 'num_key_value_heads'):
                return layer.self_attn.num_key_value_heads

        return self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else 32

    def get_head_dim(self, layer_idx: int) -> int:
        """
        Get dimension of each attention head.

        Args:
            layer_idx: Layer index

        Returns:
            Head dimension
        """
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
        # This is a simplified interface that would be integrated with FlexGen or similar
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx]
            # In actual implementation, this would access the KV cache
            # from the inference engine
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

        For Llama, we use the first N heads as suggested in the paper.

        Args:
            layer_idx: Layer index
            num_probes: Number of probe heads to select

        Returns:
            List of head indices
        """
        num_heads = self.get_num_heads(layer_idx)
        return list(range(min(num_probes, num_heads)))

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

            # Get projection matrices
            if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
                keys = attn.k_proj(hidden_states)
                values = attn.v_proj(hidden_states)
                return keys, values

        return None, None

    def get_model_config(self) -> dict:
        """
        Get model configuration parameters.

        Returns:
            Dictionary of model configuration
        """
        return {
            'model_type': 'llama',
            'num_layers': self.get_num_layers(),
            'hidden_size': self.get_hidden_dim(),
            'num_heads': self.get_num_heads(0),
            'head_dim': self.get_head_dim(0),
            'vocab_size': self.config.vocab_size if hasattr(self.config, 'vocab_size') else 32000,
            'max_position_embeddings': (
                self.config.max_position_embeddings
                if hasattr(self.config, 'max_position_embeddings')
                else 2048
            ),
        }