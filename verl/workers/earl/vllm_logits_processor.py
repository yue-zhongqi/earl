import torch
import torch.nn as nn

from verl.workers.earl.utils import ToolConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from typing import Optional, Union
from omegaconf import DictConfig


class EarlLogitsProcessor(LogitsProcessor):
    # factory method to create an instance of EarlLogitsProcessor from a vLLM LogitsProcessor
    @staticmethod
    def from_vllm_logits_processor(
        logits_processor: LogitsProcessor,
        tool_config: DictConfig,
        earl_head: Optional[nn.Module] = None
    ):
        """
        Convert a vLLM LogitsProcessor to an EARL LogitsProcessor.
        This is a placeholder method and should be implemented based on the specific requirements of EARL.
        """
        # Implement conversion logic here
        vocab_size = logits_processor.vocab_size
        org_vocab_size = logits_processor.org_vocab_size
        scale = logits_processor.scale
        logits_as_input = logits_processor.logits_as_input
        soft_cap = logits_processor.soft_cap
        return EarlLogitsProcessor(
            vocab_size,
            org_vocab_size,
            scale,
            logits_as_input,
            soft_cap,
            tool_config=tool_config,
            earl_head=earl_head
        )

    def __init__(self,
            vocab_size: int,
            org_vocab_size: Optional[int] = None,
            scale: float = 1.0,
            logits_as_input: bool = False,
            soft_cap: Optional[float] = None,
            tool_config: Optional[ToolConfig] = None,
            earl_head: Optional[nn.Module] = None,
        ):
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__(vocab_size, org_vocab_size, scale, logits_as_input, soft_cap)
        self.tool_config = tool_config
        self.earl_head = earl_head

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Apply the logits processor to the hidden states.
        This method should be overridden in subclasses to implement specific processing logic.
        """
        # Implement EARL-specific logits processing logic here
        # For now, we just call the parent method
        logits = super().forward(lm_head, hidden_states, sampling_metadata, embedding_bias)
        earl_logits = self.earl_head(hidden_states) if self.earl_head else None
        # concat logits with EARL logits if needed
        if earl_logits is not None:
            logits = torch.cat((logits, earl_logits), dim=-1)
        return logits