"""
Model adapters for IMPRESS integration with various LLM architectures.

Provides base adapter and specific implementations for Llama and Mistral models.
"""

from typing import Dict, List, Optional, Tuple
import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM

from .llama import LlamaAdapter
from .mistral import MistralAdapter


class BaseModelAdapter(ABC):
    """
    Base adapter for LLM model integration with IMPRESS.

    Defines the interface that all model adapters must implement.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the model adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        self.model_name = model_name
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        # Model configuration
        self.config = self.model.config

    @abstractmethod
    def get_attention_layers(self) -> List[torch.nn.Module]:
        """
        Get all attention layers from the model.

        Returns:
            List of attention layer modules
        """
        pass

    @abstractmethod
    def get_num_heads(self, layer_idx: int) -> int:
        """
        Get number of attention heads for a given layer.

        Args:
            layer_idx: Layer index

        Returns:
            Number of attention heads
        """
        pass

    @abstractmethod
    def get_head_dim(self, layer_idx: int) -> int:
        """
        Get dimension of each attention head.

        Args:
            layer_idx: Layer index

        Returns:
            Head dimension
        """
        pass

    @abstractmethod
    def get_kv_cache(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        """
        Get KV cache for a specific layer and head.

        Args:
            layer_idx: Layer index
            head_idx: Head index

        Returns:
            KV cache tensor or None
        """
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """
        Get total number of transformer layers.

        Returns:
            Number of layers
        """
        pass

    def get_hidden_dim(self) -> int:
        """
        Get model hidden dimension.

        Returns:
            Hidden dimension
        """
        return self.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize input text.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, return_tensors="pt")[0].tolist()

    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens)


def create_model_adapter(model_name: str, device: str = "cuda") -> BaseModelAdapter:
    """
    Factory function to create appropriate model adapter.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on

    Returns:
        Appropriate model adapter instance

    Raises:
        ValueError: If model is not supported
    """
    model_name_lower = model_name.lower()

    if "llama" in model_name_lower:
        return LlamaAdapter(model_name, device)
    elif "mistral" in model_name_lower:
        return MistralAdapter(model_name, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")