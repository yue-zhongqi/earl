import torch
from vllm.v1.sample.sampler import Sampler, _SAMPLING_EPS
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from verl.utils.device import get_device_id
from verl.workers.earl.utils import ToolConfig, get_allowed_token_ids_mask
from verl.workers.earl.tools import create_tool


class EarlSampler(Sampler):
    """
    Factory method to create a custom sampler for EARL that extends the vLLM Sampler.
    """
    @staticmethod
    def from_sampler(sampler, tool_config, tools):
        """
        Convert a vLLM Sampler to an EARL Sampler.
        This is a placeholder method and should be implemented based on the specific requirements of EARL.
        """
        # Implement conversion logic here
        return EarlSampler(
            tool_config=tool_config,
            tools=tools
        )
    
    def __init__(self, tool_config, tools):
        """
        Initialize the EARL Sampler with EARL-specific configurations.
        
        Args:
            tool_config: Configuration specific to EARL.
        """
        super().__init__()
        self.tool_config = tool_config
        self.org_vocab_size = tool_config.org_vocab_size
        self.use_tool = True
        self.tools = tools
        self.starting_mode = tool_config.starting_mode

    def turn_on_tool(self):
        """
        Enable the use of tools in the EARL Sampler.
        """
        self.use_tool = True

    def turn_off_tool(self):
        """
        Disable the use of tools in the EARL Sampler.
        """
        self.use_tool = False

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Apply the sampler to the logits.
        This method should be overridden in subclasses to implement specific sampling logic.
        """
        # Implement EARL-specific sampling logic here
        # For now, we just call the parent method
        # print(f"Sampling device {get_device_id()}: {sampling_metadata.output_token_ids[0]}.")
        return super().forward(logits, sampling_metadata)
    
    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = super().apply_allowed_token_ids(logits, sampling_metadata)
        if self.use_tool:
            mask = ~get_allowed_token_ids_mask(
                self.tool_config, self.tools, sampling_metadata, self.starting_mode
            )
            logits.masked_fill_(mask.to(logits.device), float("-inf"))
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            # print(f"Applying min_p: {sampling_metadata.min_p}")
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        # print(f"Applying top_k: {sampling_metadata.top_k}, top_p: {sampling_metadata.top_p}")
        # sampling_metadata.top_k = 1
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled