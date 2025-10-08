from copy import deepcopy
import torch
import torch.distributed
from verl.utils.debug import GPUMemoryLogger
from verl import DataProto
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from verl.third_party.vllm import vllm_version
from verl.workers.earl.vllm_scheduler import EarlVLLMScheduler
from tensordict import TensorDict
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, logger, _pre_process_inputs, _repeat_interleave
from verl.workers.earl.vllm_sampler import EarlSampler
from verl.workers.earl.utils import ToolConfig, find_active_non_earl_tool_name, get_output_seq_from_token_id, find_completed_non_earl_tool_name
from verl.workers.earl.tools import create_tool
from vllm.lora.request import LoRARequest
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from contextlib import contextmanager

import re
import gc
import time
import weakref
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import get_pp_group, graph_capture
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LayerBlockType, LazyLoader, cdiv,
                        check_use_alibi, is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin


_SAMPLING_EPS = 1e-5


class EarlRequestTracker:
    """
    A tracker for tool-actions in vllm requests.
    """

    def __init__(self, tool_config, tools, tokenizer, cleanup=False):
        self.running = {}
        self.cleanup = cleanup
        self.tool_config = tool_config
        self.tools = tools
        self.tokenizer = tokenizer

    def _record_tool_calls(self, prev_seq_in):
        if prev_seq_in is None or len(prev_seq_in) == 0:
            return []
        tool_calls = []
        for token in prev_seq_in:
            if token in self.tool_config.tool_action_ids:
                tool_name = self.tool_config.tool_action_id_to_name[token]
                tool_calls.append(tool_name)
        return tool_calls

    def init_requests(self, counter, bsz, rollout_n, prev_seq=None, interaction_kwargs=None):
        if prev_seq is not None and interaction_kwargs is not None:
            assert len(prev_seq) == len(interaction_kwargs), "prev_seq and interaction_kwargs must have the same length, either bsz or bsz * rollout_n (e.g., 001122 if bsz=3, rollout_n=2)."
        if interaction_kwargs is not None:
            if len(interaction_kwargs) == bsz:
                def index(i, j):
                    return i
            elif len(interaction_kwargs) == bsz * rollout_n:
                def index(i, j):
                    return i * rollout_n + j
            else:
                raise ValueError("interaction_kwargs must have length bsz or bsz * rollout_n.")
            
        for i in range(bsz):
            for j in range(rollout_n):
                req_id = f"{counter + i}" if rollout_n == 1 else f"{j}_{counter + i}"
                assert req_id not in self.running, f"Request {req_id} already exists in running."
                self.running[req_id] = {
                    'chosen_ids': [],
                    'last_write_idx': -1,
                    'desc_ids': [],
                    'mode': 'default',
                    'tool_idx': -1,
                    'ending': False,
                    'input_ids': [],
                    'response': [],
                    'seq_mask': [],
                    'earl_action_mask': [],
                    'tool_mask': [],
                    'insertable_mask': [],
                }
                if interaction_kwargs is not None:
                    kwarg = interaction_kwargs[index(i, j)]
                    kwarg.pop('state', None)    # remove state if exists
                    prev_seq_in = [] if prev_seq is None else prev_seq[index(i, j)]
                    for tool in self.tools.values():
                        kwarg = tool.init_state(kwarg, prev_seq_in, self.tool_config)
                    kwarg['tool_call_names'] = self._record_tool_calls(prev_seq_in)
                    self.running[req_id]['interaction_kwargs'] = kwarg

    def update_request(self, req_id, chosen_action, logprobs):
        if self.tool_config.is_sft_mode=="default":
            # breakpoint()
            assert len(chosen_action) == 1, 'Chosen action must be a single token ID.'
            desc_ids = chosen_action
            chosen_action = chosen_action[0]
            assert req_id in self.running, f"Request {req_id} not found in running. Perhaps you did not init_requests?"

            req = self.running[req_id]
            interaction_kwargs = req.get('interaction_kwargs', None)
            new_logprobs = logprobs
            # 1. compute tool_action_mask
            if req['mode'] == 'default' or len(req['chosen_ids']) == 0:
                mask = torch.zeros(self.tool_config.total_size, dtype=torch.bool)
                mask[self.tool_config.tool_action_ids_relative] = True
            else:
                tool = self.tools[req['mode']]
                mask = tool.get_allowed_token_ids_mask(req['chosen_ids'][1:])   # [1:] to remove the enter tool action
                mask = mask[self.tool_config.org_vocab_size:]    # mask is only for earl actions
            
            mode_when_action = req['mode']

            # check for non-earl tools
            non_earl_tool_name, _ = find_active_non_earl_tool_name(self.tool_config, req['input_ids'])
            inside_non_earl_tool = (non_earl_tool_name is not None)
            # cpo-insertable
            insertable = (chosen_action < self.tool_config.org_vocab_size) and (not inside_non_earl_tool)

            # 2. update mode
            if chosen_action in self.tool_config.tool_action_ids:
                tool_name = self.tool_config.tool_action_id_to_name[chosen_action]
                req['mode'] = tool_name
                req['tool_idx'] = self.tool_config.tool_action_id_to_idx[chosen_action]
                if interaction_kwargs is not None:
                    interaction_kwargs['tool_call_names'].append(tool_name)

            # 3. update tool-specific fields
            if req['mode'] != 'default':
                desc_ids, exiting = get_output_seq_from_token_id(
                    self.tools[req['mode']], req['chosen_ids'], chosen_action, interaction_kwargs
                )
                if chosen_action in self.tool_config.exit_action_ids or exiting:
                    # breakpoint()
                    # exiting
                    results = self.tools[req['mode']].execute(req['chosen_ids'], chosen_action, interaction_kwargs)
                    output_template = self.tool_config.tool_name_to_output_template[req['mode']]
                    results = output_template % results
                    results_ids = self.tokenizer(results)['input_ids']
                    desc_ids += results_ids
                    req['ending'] = True    # should set back to default in GPU runner update
                else:
                    results_ids = []

                req['last_write_idx'] = len(req['chosen_ids'])
                req['chosen_ids'].append(chosen_action)
                req['chosen_ids'] += results_ids
                req['desc_ids'].append(desc_ids)
                if new_logprobs:
                    new_logprobs[0][0] = desc_ids
                    new_logprobs[1][0] += [0] * (len(desc_ids) - 1)
                    new_logprobs[2].extend([0] * (len(desc_ids) - 1))

            tool_name, tool_call_seq = find_completed_non_earl_tool_name(
                self.tool_config, req['input_ids'] + [chosen_action]
            )

            if tool_name is not None:
                # 4. record output for non-earl tool calls
                tool = self.tools[tool_name]
                results = tool.execute(output_seq=tool_call_seq, exit_token=None, interaction_kwargs=interaction_kwargs)
                results_formated = self.tool_config.tool_name_to_output_template[tool_name] % results
                results_ids = self.tokenizer(results_formated)['input_ids']
                desc_ids += results_ids
                if new_logprobs:
                    new_logprobs[0][0] = desc_ids
                    new_logprobs[1][0] += [0] * (len(desc_ids) - 1)
                    new_logprobs[2].extend([0] * (len(desc_ids) - 1))
                req['earl_action_mask'] += [mask] * len(desc_ids)
                req['input_ids'] += desc_ids
                req['response'] += desc_ids  # for response_earl
                req['seq_mask'] += ([True] + [False] * (len(desc_ids) - 1))
                req['tool_mask'] += [False] * len(desc_ids)
                req['insertable_mask'] += [False] * len(desc_ids)
            else:
                # 4. record output otherwise
                if len(req['input_ids']) == 0 and self.tool_config.starting_mode != 'default':
                    seq_mask = False
                else:
                    seq_mask = True
                req['earl_action_mask'] += [mask] * len(desc_ids)
                req['input_ids'] += desc_ids
                req['response'] += [chosen_action] * len(desc_ids)  # for response_earl
                req['seq_mask'] += ([seq_mask] + [False] * (len(desc_ids) - 1))
                req['tool_mask'] += [mode_when_action != 'default'] * len(desc_ids)
                req['insertable_mask'] += [insertable] * len(desc_ids)
            return desc_ids, new_logprobs
        else:
            assert(self.tool_config.is_sft_mode,"sft")
            # breakpoint()
            assert len(chosen_action) == 1, 'Chosen action must be a single token ID.'
            desc_ids = chosen_action
            import pdb
            # pdb.set_trace()
            chosen_action = chosen_action[0]
            if req_id not in self.running:
                self.running[req_id] = {
                    'chosen_ids': [],
                    'last_write_idx': -1,
                    'desc_ids': [],
                    'mode': 'default',
                    'tool_idx': -1,
                    'ending': False,
                    'input_ids': [],
                    'response': [],
                    'seq_mask': [],
                    'earl_action_mask': [],
                    'tool_mask': [],
                    'insertable_mask': [],
                }

            req = self.running[req_id]
            new_logprobs = logprobs

            # 1. compute tool_action_mask
            if req['mode'] == 'default' or len(req['chosen_ids']) == 0:
                mask = torch.zeros(self.tool_config.total_size, dtype=torch.bool)
                mask[self.tool_config.tool_action_ids_relative] = True
            else:
                tool = self.tools[req['mode']]
                mask = tool.get_allowed_token_ids_mask(req['chosen_ids'])
                mask = mask[self.tool_config.org_vocab_size:]    # mask is only for earl actions
            
            mode_when_action = req['mode']

            # check for non-earl tools

            non_earl_tool_name, _ = find_active_non_earl_tool_name(self.tool_config, req['input_ids'])
            inside_non_earl_tool = (non_earl_tool_name is not None)
            # cpo-insertable
            insertable = (chosen_action < self.tool_config.org_vocab_size) and (not inside_non_earl_tool)

            # 2. update mode
            if chosen_action in self.tool_config.tool_action_ids:
                tool_name = self.tool_config.tool_action_id_to_name[chosen_action]
                req['mode'] = tool_name
                req['tool_idx'] = self.tool_config.tool_action_id_to_idx[chosen_action]

            # 3. update tool-specific fields
            if req['mode'] != 'default':
                desc_ids = get_output_seq_from_token_id(
                    self.tool_config, chosen_action
                )
                if chosen_action in self.tool_config.exit_action_ids:
                    # exiting
                    results = self.tools[req['mode']].execute(req['chosen_ids'], chosen_action)
                    output_template = self.tool_config.tool_name_to_output_template[req['mode']]
                    results = output_template % results
                    results_ids = self.tokenizer(results)['input_ids']
                    desc_ids += results_ids
                    req['ending'] = True    # should set back to default in GPU runner update
                else:
                    results_ids = []

                req['last_write_idx'] = len(req['chosen_ids'])
                req['chosen_ids'].append(chosen_action)
                req['chosen_ids'] += results_ids
                req['desc_ids'].append(desc_ids)
                if new_logprobs:
                    new_logprobs[0][0] = desc_ids
                    new_logprobs[1][0] += [0] * (len(desc_ids) - 1)
                    new_logprobs[2].extend([0] * (len(desc_ids) - 1))

            tool_name, tool_call_seq = find_completed_non_earl_tool_name(
                self.tool_config, req['input_ids'] + [chosen_action]
            )
            if tool_name is not None:
                # breakpoint()
                # 4. record output for non-earl tool calls
                tool = self.tools[tool_name]
                results = tool.execute(tool_call_seq)
                results_formated = self.tool_config.tool_name_to_output_template[tool_name] % results
                results_ids = self.tokenizer(results_formated)['input_ids']
                desc_ids += results_ids
                if new_logprobs:
                    new_logprobs[0][0] = desc_ids
                    new_logprobs[1][0] += [0] * (len(desc_ids) - 1)
                    new_logprobs[2].extend([0] * (len(desc_ids) - 1))
                req['earl_action_mask'] += [mask] * len(desc_ids)
                req['input_ids'] += desc_ids
                req['response'] += desc_ids  # for response_earl
                req['seq_mask'] += ([True] + [False] * (len(desc_ids) - 1))
                req['tool_mask'] += [False] * len(desc_ids)
                req['insertable_mask'] += [False] * len(desc_ids)
            else:
                # 4. record output otherwise
                if len(req['input_ids']) == 0 and self.tool_config.starting_mode != 'default':
                    seq_mask = False
                else:
                    seq_mask = True
                req['earl_action_mask'] += [mask] * len(desc_ids)
                req['input_ids'] += desc_ids
                req['response'] += [chosen_action] * len(desc_ids)  # for response_earl
                req['seq_mask'] += ([seq_mask] + [False] * (len(desc_ids) - 1))
                req['tool_mask'] += [mode_when_action != 'default'] * len(desc_ids)
                req['insertable_mask'] += [insertable] * len(desc_ids)
            return desc_ids, new_logprobs


    def close_request(self, req_id):
        if self.cleanup and req_id in self.running:
            del self.running[req_id]

    def clean(self):
        self.running = {}

    def set_default_mode(self, req_id):
        self.running[req_id].update({
            'chosen_ids': [],
            'last_write_idx': -1,
            'desc_ids': [],
            'mode': 'default',
            'tool_idx': -1,
            'ending': False,
        })


def pad_and_stack_masks(mask_list, pad_val, max_length):
    response_length = max(len(sub_list) for sub_list in mask_list)
    target_length = max_length if max_length is not None and max_length > response_length else response_length
    paddedd_masks = []
    for masks in mask_list:
        n_i = masks.shape[0]
        if n_i == target_length:
            paddedd_masks.append(masks)
        else:
            paddings = torch.full((target_length - n_i, masks.shape[1]), pad_val, dtype=masks.dtype)
            padded = torch.cat([masks, paddings], dim=0)
            paddedd_masks.append(padded)
    return torch.stack(paddedd_masks, dim=0)

# cut extra tokens, e.g., due to tool output
def cut_request_output(req, max_length: int):
    mod_req = req.copy()
    mod_req['earl_action_mask'] = req['earl_action_mask'][:max_length]
    mod_req['input_ids'] = req['input_ids'][:max_length]
    mod_req['response'] = req['response'][:max_length]
    mod_req['seq_mask'] = req['seq_mask'][:max_length]
    mod_req['tool_mask'] = req['tool_mask'][:max_length]
    mod_req['insertable_mask'] = req['insertable_mask'][:max_length]
    return mod_req

class EarlRollout(vLLMRollout):
    """
    Earl rollout class that extends the vLLMRollout for EARL-specific configurations.
    
    This class is used to handle the rollout process in a distributed manner
    while incorporating EARL-specific configurations and behaviors.
    """
    def __init__(self, model_path: str, config: DictConfig, tool_config:ToolConfig, tokenizer, model_hf_config, **kwargs):
        """Overwrite default vLLM rollout. Create vllm inference engine with EARL-specific configurations.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        self.config = config
        self.tool_config = tool_config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        engine_kwargs['scheduler_cls'] = EarlVLLMScheduler
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # create tools
        self._build_tools()
        # create request tracker
        self.earl_request_tracker = EarlRequestTracker(
            tool_config, self.tools, tokenizer, cleanup=False
        )
        # add tool config to scheduler
        scheduler = self.inference_engine.llm_engine.engine_core.engine_core.scheduler
        scheduler.set_tool_config(tool_config, self.tools, self.earl_request_tracker, tokenizer)
        # recreate earl-specific sampler
        sampler = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.sampler
        self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.sampler = \
            EarlSampler.from_sampler(sampler, tool_config, self.tools)
        
        # for monkey patching model runner, such that scheduler output (desc) is
        # translated back to action chosen for correct sampling metadata
        def custom_update(self, scheduler_output):
            # breakpoint()
            for req_id in scheduler_output.finished_req_ids:
                self.requests.pop(req_id, None)
                self.encoder_cache.pop(req_id, None)
            # Remove the finished requests from the persistent batch.
            # NOTE(woosuk): There could be an edge case where finished_req_ids and
            # scheduled_req_ids overlap. This happens when a request is aborted and
            # then resubmitted with the same ID. In this case, we treat them as two
            # distinct requests - clearing the cached states for the first request
            # and handling the second as a new request.
            removed_req_indices: list[int] = []
            for req_id in scheduler_output.finished_req_ids:
                req_index = self.input_batch.remove_request(req_id)
                if req_index is not None:
                    removed_req_indices.append(req_index)

            # Free the cached encoder outputs.
            for req_id, input_id in scheduler_output.free_encoder_input_ids:
                encoder_outputs = self.encoder_cache.get(req_id)
                if encoder_outputs is not None:
                    encoder_outputs.pop(input_id, None)
                    if not encoder_outputs:
                        self.encoder_cache.pop(req_id, None)

            # Remove the unscheduled requests from the persistent batch.
            # NOTE(woosuk): The unscheduled requests are either preempted requests
            # or running requests that are not scheduled in this step. We remove
            # them from the persistent batch but keep their cached states since
            # they will be scheduled again sometime in the future.
            scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
            cached_req_ids = self.input_batch.req_id_to_index.keys()
            unscheduled_req_ids = cached_req_ids - scheduled_req_ids
            # NOTE(woosuk): The persistent batch optimization assumes that
            # consecutive batches contain mostly the same requests. If batches
            # have low request overlap (e.g., alternating between two distinct
            # sets of requests), this optimization becomes very inefficient.
            for req_id in unscheduled_req_ids:
                req_index = self.input_batch.remove_request(req_id)
                assert req_index is not None
                removed_req_indices.append(req_index)

            req_ids_to_add: list[str] = []
            # Add new requests to the cached states.
            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                sampling_params = new_req_data.sampling_params
                if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
                else:
                    generator = None

                self.requests[req_id] = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=new_req_data.prompt_token_ids,
                    mm_inputs=new_req_data.mm_inputs,
                    mm_positions=new_req_data.mm_positions,
                    sampling_params=sampling_params,
                    generator=generator,
                    block_ids=new_req_data.block_ids,
                    num_computed_tokens=new_req_data.num_computed_tokens,
                    output_token_ids=[],
                    lora_request=new_req_data.lora_request,
                )

                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                if self.uses_mrope:
                    image_grid_thw = []
                    video_grid_thw = []
                    second_per_grid_ts = []
                    audio_feature_lengths = []
                    use_audio_in_video = False
                    for mm_input in self.requests[req_id].mm_inputs:
                        if mm_input.get("image_grid_thw") is not None:
                            image_grid_thw.extend(
                                mm_input["image_grid_thw"].tolist())
                        if mm_input.get("video_grid_thw") is not None:
                            video_grid_thw.extend(
                                mm_input["video_grid_thw"].tolist())
                        if mm_input.get("second_per_grid_ts") is not None:
                            second_per_grid_ts.extend(
                                mm_input["second_per_grid_ts"])
                        if mm_input.get("audio_feature_lengths") is not None:
                            audio_feature_lengths.extend(
                                mm_input["audio_feature_lengths"])
                        if mm_input.get("use_audio_in_video") is True:
                            use_audio_in_video = True

                    hf_config = self.model_config.hf_config

                    self.requests[req_id].mrope_positions, \
                        self.requests[req_id].mrope_position_delta = \
                        MRotaryEmbedding.get_input_positions_tensor(
                            self.requests[req_id].prompt_token_ids,
                            hf_config=hf_config,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            second_per_grid_ts=second_per_grid_ts,
                            audio_feature_lengths=audio_feature_lengths,
                            use_audio_in_video=use_audio_in_video,
                        )

                req_ids_to_add.append(req_id)

            # Update the states of the running/resumed requests.
            for req_data in scheduler_output.scheduled_cached_reqs:
                req_id = req_data.req_id
                req_state = self.requests[req_id]
                # breakpoint()
                # Update the cached states.
                num_computed_tokens = req_data.num_computed_tokens
                req_state.num_computed_tokens = num_computed_tokens
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec decode tokens.
                num_new_tokens = (num_computed_tokens +
                                len(req_data.new_token_ids) -
                                req_state.num_tokens)
                
                new_tokens = req_data.new_token_ids[-num_new_tokens:] if num_new_tokens > 0 else []
                if new_tokens:
                    assert req_id in self.earl_request_tracker.running, \
                        f"Request {req_id} not found in earl request tracker."
                    tracker = self.earl_request_tracker.running[req_id]
                    if tracker['mode'] != 'default':
                        idx = tracker['last_write_idx'] 
                        # if int(req_id) % 31 == 0:
                        #     print(f"Runner id {req_id}: {new_tokens} -> {tracker['chosen_ids'][idx:]}")
                        # from the "translated" new tokens (by scheduler) to the actual action token
                        new_tokens = tracker['chosen_ids'][idx:]
                    if tracker['ending']:
                        self.earl_request_tracker.set_default_mode(req_id)

                if len(new_tokens) == 1:
                    req_state.output_token_ids.append(
                        new_tokens[-1]
                    )
                elif len(new_tokens) > 0:
                    req_state.output_token_ids.extend(
                        new_tokens
                    )
                
                # Update the block IDs.
                if not req_data.resumed_from_preemption:
                    # Append the new blocks to the existing block IDs.
                    req_state.block_ids.extend(req_data.new_block_ids)
                else:
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = req_data.new_block_ids

                req_index = self.input_batch.req_id_to_index.get(req_id)
                if req_index is None:
                    # The request is not in the persistent batch.
                    # The request was either preempted and resumed later, or was not
                    # scheduled in the previous step and needs to be added again.
                    req_ids_to_add.append(req_id)
                    continue

                # Update the persistent batch.
                self.input_batch.num_computed_tokens_cpu[req_index] = (
                    num_computed_tokens)
                self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                        req_index)
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(req_data.new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = req_data.new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ())
                if spec_token_ids:
                    start_index = end_token_index
                    end_token_index += len(spec_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
                self.input_batch.num_tokens[req_index] = end_token_index

            # Check if the batch has changed. If not, we can skip copying the
            # sampling metadata from CPU to GPU.
            batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

            # Add the new or resumed requests to the persistent batch.
            # The smaller empty indices are filled first.
            removed_req_indices.sort(reverse=True)
            for req_id in req_ids_to_add:
                req_state = self.requests[req_id]
                if removed_req_indices:
                    # Fill the empty index.
                    req_index = removed_req_indices.pop()
                else:
                    # Append to the end.
                    req_index = None
                self.input_batch.add_request(req_state, req_index)

            # Condense the batched states if there are empty indices.
            if removed_req_indices:
                self.input_batch.condense(removed_req_indices)

            # Some attention backends (namely MLA) may want to separate requests
            # based on if the attention computation will be compute-bound or
            # memory-bound. This gives them a hook to do that.
            batch_reordered = self.attn_metadata_builder.reorder_batch(
                self.input_batch, scheduler_output)

            if batch_changed or batch_reordered:
                self.input_batch.refresh_sampling_metadata()
        
        # monkey patch model runner
        from types import MethodType
        model_to_patch = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        model_to_patch._update_states = MethodType(custom_update, model_to_patch)
        model_to_patch.tool_config = tool_config
        model_to_patch.earl_request_tracker = self.earl_request_tracker

        # FOR DEBUGGING: monkey patch model forward
        # def custom_forward(
        #     self,
        #     input_ids: torch.Tensor,
        #     positions: torch.Tensor,
        #     intermediate_tensors = None,
        #     inputs_embeds = None,
        # ):
        #     pattern = [16429, 29952, 25]
        #     # Find all matches
        #     indices = []
        #     for i in range(len(input_ids) - len(pattern) + 1):
        #         if torch.equal(input_ids[i:i+len(pattern)], torch.tensor(pattern, device=input_ids.device)):
        #             indices.append(i)
        #     # Print result
        #     if len(indices) > 0:
        #         print("Correct pattern found at indices:", indices)

        #     hidden_states = self.model(input_ids, positions, intermediate_tensors,
        #                             inputs_embeds)
        #     return hidden_states
        # model_to_patch = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        # model_to_patch.forward = MethodType(custom_forward, model_to_patch)

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        # print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    def _build_tools(self):
        """
        Initialize tools based on the provided tool configuration.
        This method should be implemented to set up the tools used in EARL.
        """
        # Implement tool initialization logic here
        self.tools = {}
        for tool_name in self.tool_config.tool_name_to_action_id.keys():
            self.tools[tool_name] = create_tool(tool_name, self.tool_config)
    
    def _SFT_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        # breakpoint()
        with self.update_sampling_params(**kwargs):
            self.earl_request_tracker.clean()   # clean up prev requests
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = []
            response_earl = []
            input = []
            seq_mask = []
            rollout_log_probs = []
            earl_action_mask = []
            tool_mask = []
            insertable_mask = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    if len(output.outputs) == 1:
                        req_id = f"{output.request_id}"
                    else:
                        # breakpoint()
                        req_id = f"{sample_id}_{output.request_id}"
                    assert req_id in self.earl_request_tracker.running
                    # import pdb
                    # pdb.set_trace()
                    req = self.earl_request_tracker.running[req_id]
                    req = cut_request_output(req, self.config.response_length)  # cut extra tokens, e.g., due to tool output
                    # extract info from output
                    response_earl.append(req['response'])   # req[response] records the action chosen by the model, including tool actions
                    input.append(req['input_ids'])  # input_ids record the description of the tool action and natural language tokens.
                    response.append(req['input_ids'])   # this is for computing reward, hence using input_ids as the description of action
                    assert len(req['input_ids']) == len(req['response']), \
                        f"input_ids and response should have the same length, got {len(req['input_ids'])} and {len(req['response'])}"
                    assert output.outputs[sample_id].token_ids == req['input_ids']
                    seq_mask.append(req['seq_mask'])
                    tool_mask.append(req['tool_mask'])
                    insertable_mask.append(req['insertable_mask'])
                    seq_action_masks = torch.stack(req['earl_action_mask'], dim=0)
                    #earl_action_mask.append(req['earl_action_mask'])  # earl_action_mask records the action mask for each token in the response
                    earl_action_mask.append(seq_action_masks)
                    if self.config.calculate_log_probs:
                        sample_idx, req_id = req_id.split('_')
                        sample_idx = int(sample_idx)
                        # for extracting logprobs
                        relevant_output = output.outputs[sample_id]
                        logprobs = relevant_output.logprobs
                        response_ids = relevant_output.token_ids
                        curr_log_prob = []
                        for i, logprob in enumerate(logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)
            # breakpoint()
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            response_earl = pad_2d_list_to_length(response_earl, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            input = pad_2d_list_to_length(input, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            seq_mask = pad_2d_list_to_length(seq_mask, 0, max_length=self.config.response_length).to(idx.device)
            tool_mask = pad_2d_list_to_length(tool_mask, 0, max_length=self.config.response_length).to(idx.device)
            insertable_mask = pad_2d_list_to_length(insertable_mask, 0, max_length=self.config.response_length).to(idx.device)
            earl_action_mask = pad_and_stack_masks(earl_action_mask, True, max_length=self.config.response_length).to(idx.device)

            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)
                if "expr" in non_tensor_batch.keys():
                    non_tensor_batch["expr"] = _repeat_interleave(non_tensor_batch["expr"], self.sampling_params.n)

            seq = torch.cat([idx, input], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # breakpoint()
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "response_earl": response_earl,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seq_mask": seq_mask,
                "tool_mask": tool_mask,
                "insertable_mask": insertable_mask,
                "earl_action_mask": earl_action_mask,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        # breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    def _default_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            self.earl_request_tracker.clean()   # clean up prev requests
            self.earl_request_tracker.init_requests(
                counter=self.inference_engine.request_counter.counter,
                bsz=batch_size,
                rollout_n=self.sampling_params.n,
                prev_seq=None,
                interaction_kwargs=non_tensor_batch.get("interaction_kwargs", None)
            )
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )
            # breakpoint()
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = []
            response_earl = []
            input = []
            seq_mask = []
            rollout_log_probs = []
            earl_action_mask = []
            tool_mask = []
            insertable_mask = []
            interaction_kwargs_list = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    if len(output.outputs) == 1:
                        req_id = f"{output.request_id}"
                    else:
                        req_id = f"{sample_id}_{output.request_id}"
                    assert req_id in self.earl_request_tracker.running
                    req = self.earl_request_tracker.running[req_id]
                    req = cut_request_output(req, self.config.response_length)  # cut extra tokens, e.g., due to tool output
                    # extract info from output
                    response_earl.append(req['response'])   # req[response] records the action chosen by the model, including tool actions
                    input.append(req['input_ids'])  # input_ids record the description of the tool action and natural language tokens.
                    response.append(req['input_ids'])   # this is for computing reward, hence using input_ids as the description of action
                    assert len(req['input_ids']) == len(req['response']), \
                        f"input_ids and response should have the same length, got {len(req['input_ids'])} and {len(req['response'])}"
                    assert output.outputs[sample_id].token_ids == req['input_ids']
                    seq_mask.append(req['seq_mask'])
                    tool_mask.append(req['tool_mask'])
                    insertable_mask.append(req['insertable_mask'])
                    seq_action_masks = torch.stack(req['earl_action_mask'], dim=0)
                    #earl_action_mask.append(req['earl_action_mask'])  # earl_action_mask records the action mask for each token in the response
                    earl_action_mask.append(seq_action_masks)
                    if 'interaction_kwargs' in req:
                        interaction_kwargs_list.append(req.get('interaction_kwargs', None))
                    if self.config.calculate_log_probs:
                        sample_idx, req_id = req_id.split('_')
                        sample_idx = int(sample_idx)
                        # for extracting logprobs
                        relevant_output = output.outputs[sample_id]
                        logprobs = relevant_output.logprobs
                        response_ids = relevant_output.token_ids
                        curr_log_prob = []
                        for i, logprob in enumerate(logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)
            # breakpoint()
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            response_earl = pad_2d_list_to_length(response_earl, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            input = pad_2d_list_to_length(input, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            seq_mask = pad_2d_list_to_length(seq_mask, 0, max_length=self.config.response_length).to(idx.device)
            tool_mask = pad_2d_list_to_length(tool_mask, 0, max_length=self.config.response_length).to(idx.device)
            insertable_mask = pad_2d_list_to_length(insertable_mask, 0, max_length=self.config.response_length).to(idx.device)
            earl_action_mask = pad_and_stack_masks(earl_action_mask, True, max_length=self.config.response_length).to(idx.device)

            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)
                if "expr" in non_tensor_batch.keys():
                    non_tensor_batch["expr"] = _repeat_interleave(non_tensor_batch["expr"], self.sampling_params.n)
            if "interaction_kwargs" in non_tensor_batch.keys():
                if len(interaction_kwargs_list) > 0:
                    assert len(interaction_kwargs_list) == len(non_tensor_batch["interaction_kwargs"]), \
                        f"interaction_kwargs_list length {len(interaction_kwargs_list)} does not match expected length {len(non_tensor_batch['interaction_kwargs']) * self.sampling_params.n}"
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(interaction_kwargs_list, 1)
            seq = torch.cat([idx, input], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # breakpoint()
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "response_earl": response_earl,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seq_mask": seq_mask,
                "tool_mask": tool_mask,
                "insertable_mask": insertable_mask,
                "earl_action_mask": earl_action_mask,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        # breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.tool_config.is_sft_mode=="default":
            return self._default_generate_sequences(prompts,**kwargs)
        else:
            return self._SFT_generate_sequences(prompts,**kwargs)


    def _default_compute_gt_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch

        # fill in the request
        # TODO: now hard-coded to use calculator and specific prompt format!!!
        # breakpoint()
        self.earl_request_tracker.clean()   # clean up prev requests
        self.earl_request_tracker.init_requests(
            counter=0,
            bsz=batch_size,
            rollout_n=1,
            prev_seq=None,
            interaction_kwargs=non_tensor_batch.get("interaction_kwargs", None)
        )
        for i, prompt in enumerate(idx):
            # breakpoint()
            gt = non_tensor_batch["expr"][i].replace(' ', '')
            self.earl_request_tracker.update_request(
                req_id = str(i),
                chosen_action = [self.tool_config.tool_action_ids[0]],
                logprobs=None
            )
            for char in gt:
                token_id = None
                for key, val in self.tool_config.id_to_str.items():
                    if char in val:
                        token_id = key
                        break
                assert token_id is not None, f"Character {char} not found in tool config id_to_str mapping."
                self.earl_request_tracker.update_request(
                    req_id = str(i),
                    chosen_action = [token_id],
                    logprobs=None
                )
            # =
            self.earl_request_tracker.update_request(
                req_id = str(i),
                chosen_action = [self.tool_config.exit_action_ids[0]],
                logprobs=None
            )
            self.earl_request_tracker.set_default_mode(str(i))
            # .
            # self.earl_request_tracker.update_request(
            #     req_id = str(i),
            #     chosen_action = [13],
            #     logprobs=None
            # )
            # <|im_end|>
            self.earl_request_tracker.update_request(
                req_id = str(i),
                chosen_action = [151645],
                logprobs=None
            )
        # breakpoint()
        response = []
        response_earl = []
        input = []
        seq_mask = []
        earl_action_mask = []
        tool_mask = []
        for i in range(batch_size):
            req_id = str(i)
            req = self.earl_request_tracker.running[req_id]
            req = cut_request_output(req, self.config.response_length)  # cut extra tokens, e.g., due to tool output
            # extract info from output
            response_earl.append(req['response'])   # req[response] records the action chosen by the model, including tool actions
            input.append(req['input_ids'])  # input_ids record the description of the tool action and natural language tokens.
            response.append(req['input_ids'])   # this is for computing reward, hence using input_ids as the description of action
            assert len(req['input_ids']) == len(req['response']), \
                f"input_ids and response should have the same length, got {len(req['input_ids'])} and {len(req['response'])}"
            seq_mask.append(req['seq_mask'])
            tool_mask.append(req['tool_mask'])
            seq_action_masks = torch.stack(req['earl_action_mask'], dim=0)
            #earl_action_mask.append(req['earl_action_mask'])  # earl_action_mask records the action mask for each token in the response
            earl_action_mask.append(seq_action_masks)

        # compute corresponding values
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        response_earl = pad_2d_list_to_length(response_earl, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        input = pad_2d_list_to_length(input, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        seq_mask = pad_2d_list_to_length(seq_mask, 0, max_length=self.config.response_length).to(idx.device)
        tool_mask = pad_2d_list_to_length(tool_mask, 0, max_length=self.config.response_length).to(idx.device)
        earl_action_mask = pad_and_stack_masks(earl_action_mask, True, max_length=self.config.response_length).to(idx.device)

        seq = torch.cat([idx, input], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "response_earl": response_earl,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seq_mask": seq_mask,
                "tool_mask": tool_mask,
                "earl_action_mask": earl_action_mask,
            },
            batch_size=batch_size,
        )
        # breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    def _SFT_compute_gt_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        # fill in the request
        # TODO: now hard-coded to use calculator and specific prompt format!!!
        # breakpoint()
        import pdb
        # pdb.set_trace()
        self.earl_request_tracker.clean()   # clean up prev requests
        import re
        for i, prompt in enumerate(idx):
            # breakpoint()
            # pdb.set_trace()
            gt = non_tensor_batch["expr"][i]
            if '<answer>' in gt and '</answer>' in gt:
                ''' countdown sft '''
            # gt = " First, let's consider the numbers we have: 71, 5, 49, and 9. We need to find a combination of these numbers using basic arithmetic operations to get 165.\nLet's start by trying to combine the larger numbers first. \nIf we multiply 71 by 5, we get:\n<calculator>71 * 5</calculator>\n<result>355</result>\n355 is too large, so let's try another combination.\nIf we multiply 49 by 5, we get:\n<calculator>49 * 5</calculator>\n<result>245</result>\n245 is still too large, so let's try another combination.\nIf we multiply 71 by 9, we get:\n<calculator>71 * 9</calculator>\n<result>639</result>\n639 is too large, so let's try another combination.\nIf we multiply 49 by 9, we get:\n<calculator>49 * 9</calculator>\n<result>441</result>\n441 is too large, so let's try another combination.\nIf we add 71 and 49, we get:\n<calculator>71 + 49</calculator>\n<result>120</result>\n120 is closer to 165, so let's see if we can use the remaining numbers (5 and 9) to reach 165.\nIf we add 5 and 9, we get:\n<calculator>5 + 9</calculator>\n<result>14</result>\nIf we add 120 and 14, we get:\n<calculator>120 + 14</calculator>\n<result>134</result>\n134 is still not 165, so let's try another combination.\nIf we subtract 5 from 71, we get:\n<calculator>71 - 5</calculator>\n<result>66</result>\nIf we add 66 and 49, we get:\n<calculator>66 + 49</calculator>\n<result>115</result>\n115 is still not 165, so let's try another combination.\nIf we multiply 5 by 9, we get:\n<calculator>5 * 9</calculator>\n<result>45</result>\nIf we add 45 to 120, we get:\n<calculator>120 + 45</calculator>\n<result>165</result>\nSo, the equation that equals 165 is:\n71 + 49 + (5 * 9)</think>\n<answer>71 + 49 + (5 * 9)</answer>"
                expr = gt.split('<answer>')[1].split('</answer>')[0].strip()
                format_expr = ''
                for char in expr.replace(' ', ''):
                    if char in ['+','-','*','/', '(', ')']: # convert to correct subword of token id
                        format_expr += ' ' + char
                    else:
                        format_expr += char
                gt = gt.replace(expr,format_expr)
                temp_str = ''
                for item in gt.replace('\n\n',' \n').replace('<answer>',' <answer> ').replace('</answer>',' </answer> ').replace('</think>', ' </think> ').split('<result>'):
                    item_list = item.split('</result>')
                    if len(item_list) == 1:
                        temp_str += item_list[0]
                    elif len(item_list) == 2:
                        temp_str += item_list[1]
                    else:
                        for chk in item_list:
                            if not chk.strip().isdigit():
                                temp_str += chk
                temp_str = temp_str.replace('\n\n',' \n').replace('\n','')
                for item in re.split(rf'({re.escape("<calculator>")})', temp_str):
                    if "</calculator>" in item:
                        ans_chunk_list = re.split(rf'({re.escape("</calculator>")})', item)
                        for j in range(len(ans_chunk_list)):
                            ans_chunk = ans_chunk_list[j]
                            if j == 0:
                                inter_expr = ans_chunk.replace(' ', '').replace('\n','')
                                # 71*5
                                # if 'So' in inter_expr:
                                #     pdb.set_trace()
                                for char in inter_expr:
                                    token_id = None
                                    for key, val in self.tool_config.tool_action_id_to_non_earl_actions[-1].items():
                                        if char in key:
                                            token_id = val
                                            break
                                    assert token_id is not None, f"Character {char} not found in tool config id_to_str mapping."
                                    self.earl_request_tracker.update_request(
                                        req_id = str(i),
                                        chosen_action = [token_id],
                                        logprobs=None
                                    )
                            elif j == 1:
                                # </calculator>
                                for val in self.tool_config.tool_action_id_to_exit_seq[-1]:
                                    self.earl_request_tracker.update_request(
                                        req_id = str(i),
                                        chosen_action = [val],
                                        logprobs=None
                                    )
                                self.earl_request_tracker.set_default_mode(str(i))
                            else:
                                # remianing text such as 355 is too large, so let's try another combination.
                                # If we multiply 49 by 5, we get:
                                self.earl_request_tracker.set_default_mode(str(i))
                                remain_tokens = tokenizer.encode(ans_chunk)
                                for remain_token_id in remain_tokens:
                                    self.earl_request_tracker.update_request(
                                        req_id = str(i),
                                        chosen_action = [remain_token_id],
                                        logprobs=None
                                    )
                    elif item == "<calculator>" :
                        # ' calculate '
                        for val in self.tool_config.id_to_seq[-1]:
                            self.earl_request_tracker.update_request(
                                req_id = str(i),
                                chosen_action = [val],
                                logprobs=None
                            )
                    else:
                        # First, let's consider the numbers we have: 71, 5, 49, and 9. We need to find a combination of these numbers using basic arithmetic operations to get 165.
                        # Let's start by trying to combine the larger numbers first. 
                        # If we multiply 71 by 5, we get:
                        description_tokens = tokenizer.encode(item)
                        for description_token_id in description_tokens:
                            self.earl_request_tracker.update_request(
                                req_id = str(i),
                                chosen_action = [description_token_id],
                                logprobs=None
                            )
                self.earl_request_tracker.update_request(
                    req_id = str(i),
                    chosen_action = [151645],
                    logprobs=None
                )

                    # expr = gt.split('<answer>')[1].split('</answer>')[0].strip()
                    # temp_str = ''
                    # for item in gt.replace('\n\n',' \n').replace('<calculator>',' <calculator> ').replace('</calculator>',' </calculator> ').replace('</think>', ' </think> ').replace('<think>\n',' ').replace('<answer>',' <answer> ').replace('</answer>',' </answer> ').split('<result>'):
                    #     item_list = item.split('</result>')
                    #     if len(item_list) == 1:
                    #         temp_str += item_list[0]
                    #     elif len(item_list) == 2:
                    #         temp_str += item_list[1]
                    #     else:
                    #         for chk in item_list:
                    #             if not chk.strip().isdigit():
                    #                 temp_str += chk

                    # temp_str = temp_str.replace('\n\n',' \n')
                    # temp_str = ' ' + ' '.join(temp_str.replace('\n', ' \n ').split())
                    # format_str = ''
                    # for item in re.split(f'({re.escape(" <calculator> ")})', temp_str):
                    #     if ' </calculator> ' in item:
                    #         temp_list = re.split(f'({re.escape(" </calculator> ")})', item)
                    #         for j in range(len(temp_list)):
                    #             temp = temp_list[j]
                    #             if j == 0: 
                    #                 for char in temp.replace(' ', ''):
                    #                     # if char != ',': # <calculator> 119,566 * 264 </calculator>\
                    #                     if char in ['+','-','*','/', '(', ')']: # convert to correct subword of token id
                    #                         format_str += ' ' + char
                    #                     else:
                    #                         format_str += char
                    #             else:
                    #                 format_str += temp
                    #     else:
                    #         format_str += item
                    # gt_list = re.split(f'({re.escape(expr)})', format_str)
                    # # print(gt_list)
                    # for gt_element in gt_list:
                    #     if gt_element == expr:
                    #         gt_element_inputs = gt_element.replace(' ', '')
                    #         for char in gt_element_inputs:
                    #             token_id = None
                    #             for key, val in self.tool_config.tool_action_id_to_non_earl_actions[-1].items():
                    #                 if char in key:
                    #                     token_id = val
                    #                     break
                    #             assert token_id is not None, f"Character {char} not found in tool config id_to_str mapping."
                    #             self.earl_request_tracker.update_request(
                    #                 req_id = str(i),
                    #                 chosen_action = [token_id],
                    #                 logprobs=None
                    #             )
                    #     else:
                    #         # pdb.set_trace()
                    #         gt_element = gt_element.replace('+',' +').replace('-',' -').replace('*',' *').replace('/',' /').replace( '(', ' (').replace( ')', ' )')
                    #         gt_element_input_token_ids = tokenizer.encode(gt_element)
                    #         for token_id in gt_element_input_token_ids:
                    #             self.earl_request_tracker.update_request(
                    #                 req_id = str(i),
                    #                 chosen_action = [token_id],
                    #                 logprobs=None
                    #             )
                    # self.earl_request_tracker.update_request(
                    #     req_id = str(i),
                    #     chosen_action = [151645],
                    #     logprobs=None
                    # ) 

            else:
                ''' arithmetic sft '''
                ans = '\n' + non_tensor_batch["expr"][i].split('\n')[1]
                expr = non_tensor_batch["expr"][i].split('\n')[0]
                expr = expr.replace(' ', '')
                format_expr = ''
                for char in expr:
                    if char in ['+','-','*','/', '(', ')']:
                        format_expr += ' ' + char
                    else:
                        format_expr += char
                gt_expr = " <calculator> " + format_expr + " </calculator>"
                gt_expr_input_token_ids = tokenizer.encode(gt_expr)
                for token_id in gt_expr_input_token_ids:
                    self.earl_request_tracker.update_request(
                        req_id = str(i),
                        chosen_action = [token_id],
                        logprobs=None
                    )
                ans_token_ids = tokenizer.encode(ans)
                for token_id in ans_token_ids:
                    self.earl_request_tracker.update_request(
                        req_id = str(i),
                        chosen_action = [token_id],
                        logprobs=None
                    )
                self.earl_request_tracker.update_request(
                    req_id = str(i),
                    chosen_action = [151645],
                    logprobs=None
                )    
            # self.earl_request_tracker.set_default_mode(str(i))
            # self.earl_request_tracker.update_request(
            #     req_id = str(i),
            #     chosen_action = [13],
            #     logprobs=None
            # )

            # gt = non_tensor_batch["extra_info"]["answer"][i].replace(' ', '')
            # self.earl_request_tracker.update_request(
            #     req_id = str(i),
            #     chosen_action = [self.tool_config.tool_action_ids[0]],
            #     logprobs=None
            # )
            
            # for char in gt:
            #     token_id = None
            #     for key, val in self.tool_config.id_to_str.items():
            #         if char in val:
            #             token_id = key
            #             break
            #     assert token_id is not None, f"Character {char} not found in tool config id_to_str mapping."
            #     self.earl_request_tracker.update_request(
            #         req_id = str(i),
            #         chosen_action = [token_id],
            #         logprobs=None
            #     )
            # pdb.set_trace()
            
            # pdb.set_trace()
            # self.earl_request_tracker.update_request(
            #     req_id = str(i),
            #     chosen_action = [self.tool_config.exit_action_ids[0]],
            #     logprobs=None
            # )
            
            # self.earl_request_tracker.set_default_mode(str(i))

            # self.earl_request_tracker.update_request(
            #     req_id = str(i),
            #     chosen_action = [151645],
            #     logprobs=None
            # )

        # breakpoint()
        # pdb.set_trace()
        response = []
        response_earl = []
        input = []
        seq_mask = []
        earl_action_mask = []
        tool_mask = []
        for i in range(batch_size):
            req_id = str(i)
            req = self.earl_request_tracker.running[req_id]
            req = cut_request_output(req, self.config.response_length)  # cut extra tokens, e.g., due to tool output
            # extract info from output
            response_earl.append(req['response'])   # req[response] records the action chosen by the model, including tool actions
            input.append(req['input_ids'])  # input_ids record the description of the tool action and natural language tokens.
            response.append(req['input_ids'])   # this is for computing reward, hence using input_ids as the description of action
            assert len(req['input_ids']) == len(req['response']), \
                f"input_ids and response should have the same length, got {len(req['input_ids'])} and {len(req['response'])}"
            seq_mask.append(req['seq_mask'])
            tool_mask.append(req['tool_mask'])
            seq_action_masks = torch.stack(req['earl_action_mask'], dim=0)
            #earl_action_mask.append(req['earl_action_mask'])  # earl_action_mask records the action mask for each token in the response
            earl_action_mask.append(seq_action_masks)

        # compute corresponding values
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        response_earl = pad_2d_list_to_length(response_earl, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        input = pad_2d_list_to_length(input, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        seq_mask = pad_2d_list_to_length(seq_mask, 0, max_length=self.config.response_length).to(idx.device)
        tool_mask = pad_2d_list_to_length(tool_mask, 0, max_length=self.config.response_length).to(idx.device)
        earl_action_mask = pad_and_stack_masks(earl_action_mask, True, max_length=self.config.response_length).to(idx.device)

        seq = torch.cat([idx, input], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "response_earl": response_earl,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seq_mask": seq_mask,
                "tool_mask": tool_mask,
                "earl_action_mask": earl_action_mask,
            },
            batch_size=batch_size,
        )
        # breakpoint()
        
        # print(tokenizer.batch_decode(batch['prompts']))
        # print(tokenizer.batch_decode(batch["responses"]))
        # print(tokenizer.batch_decode(batch["input_ids"]))
        # print(tokenizer.batch_decode(batch["response_earl"]))
        # pdb.set_trace()
        # tokenizer.batch_decode(batch['prompts'])
        # tokenizer.batch_decode(batch["responses"][0][:26])
        # tokenizer.batch_decode(batch[])
        # tokenizer.batch_decode(batch[])
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    @GPUMemoryLogger(role="gt rollout", logger=logger)
    @torch.no_grad()
    def compute_gt_sequences(self, prompts: DataProto)-> DataProto:
        if self.tool_config.is_sft_mode=="default":
            return self._default_compute_gt_sequences(prompts)
        else:
            assert(self.tool_config.is_sft_mode,"sft")
            return self._SFT_compute_gt_sequences(prompts)

    
    def _default_rewrite_sequences(self, prompts: DataProto, mask: torch.Tensor) -> DataProto:
        """
        Rewrite sequences using the vLLM inference engine.
        This method is similar to `generate_sequences`, but it is used for rewriting
        existing sequences rather than generating new ones.
        """
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        non_tensor_batch = prompts.non_tensor_batch
        ori = prompts[~mask]
        prompts = prompts[mask]

        if mask.sum() > 0:
            # Need rewriting

            # TODO: fix to the first trainable tools. Maybe enable multiple tool call position?
            prompts.batch['call_tool_pos'] = prompts.batch['call_tool_pos'].squeeze(1) + 1
            starting_tool_id = prompts.batch['call_tool_id'][0][0].item()
            starting_mode = self.tool_config.tool_action_id_to_name[starting_tool_id]

            prompt_length = prompts.batch['input_ids'].size(1) - prompts.batch['responses'].size(1)
            idx = prompts.batch["input_ids"][:, :prompt_length]  # (bs, prompt_length)
            # left-padded attention_mask
            attention_mask = prompts.batch["attention_mask"][:, :prompt_length]
            position_ids = prompts.batch["position_ids"][:, :prompt_length]

            # used to construct attention_mask
            eos_token_id = prompts.meta_info["eos_token_id"]
            batch_size = idx.size(0)

            old_response_lengths = [
                s["attention_mask"][:s["call_tool_pos"]].sum().item() - s["attention_mask"][:prompt_length].sum(dim=-1).item()
                for s in prompts.batch
            ]

            def get_raw_prompt_ids(input_ids, attn_mask, pos):
                return input_ids[:pos][attn_mask[:pos].bool()].tolist()

            vllm_inputs = [
                {"prompt_token_ids": get_raw_prompt_ids(
                    prompts.batch['input_ids'][i],
                    prompts.batch['attention_mask'][i],
                    prompts.batch['call_tool_pos'][i])
                } for i in range(batch_size)
            ]
            # breakpoint()

            # ensure the type of `prompt_token_ids` passed to vllm is list[int]
            # https://github.com/volcengine/verl/pull/772
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")
        
            kwargs = {
                "n": 1,
            }

            lora_requests = None
            if self.lora_kwargs:
                lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    lora_int_id = lora_int_ids[0]
                    lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

            # users can customize different sampling_params at different run
            # breakpoint()
            rewrite_interaction_kwargs = non_tensor_batch.get("interaction_kwargs", None)
            if rewrite_interaction_kwargs is not None:
                rewrite_interaction_kwargs = rewrite_interaction_kwargs[mask.bool().cpu().tolist()]
            # prev_seq = [prompts.batch['response_earl'][i][:prev].tolist() for i, prev in enumerate(old_response_lengths)]
            # breakpoint()
            prev_seq = [prompts.batch['response_earl'][i][:prev][prompts.batch['seq_mask'][i][:prev].bool()].cpu().tolist() for i, prev in enumerate(old_response_lengths)]
            with self.update_sampling_params(**kwargs):
                with self.update_vllm_sampler(starting_mode=starting_mode):
                    # try:
                    self.earl_request_tracker.clean()   # clean up prev requests
                    # breakpoint()
                    self.earl_request_tracker.init_requests(
                        counter=self.inference_engine.request_counter.counter,
                        bsz=batch_size,
                        rollout_n=self.sampling_params.n,
                        prev_seq=prev_seq,
                        interaction_kwargs=rewrite_interaction_kwargs
                    )
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                    # except Exception as e:
                    #     print(e)
                    #     breakpoint()

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
                response = []
                response_earl = []
                input = []
                seq_mask = []
                earl_action_mask = []
                tool_mask = []
                interaction_kwargs_list = []
                for i, output in enumerate(outputs):
                    for sample_id in range(len(output.outputs)):
                        if len(output.outputs) == 1:
                            req_id = f"{output.request_id}"
                        else:
                            # breakpoint()
                            req_id = f"{sample_id}_{output.request_id}"
                        assert req_id in self.earl_request_tracker.running
                        sample = prompts.batch[i]
                        prev = old_response_lengths[i]
                        req = self.earl_request_tracker.running[req_id]
                        req = cut_request_output(req, self.config.response_length - prev)  # cut extra tokens, e.g., due to tool output
                        if 'interaction_kwargs' in req:
                            interaction_kwargs_list.append(req.get('interaction_kwargs', None))
                        # extract info from output
                        response_earl.append(
                            sample['response_earl'][:prev].tolist() + req['response']
                        )   # req[response] records the action chosen by the model, including tool actions
                        input.append(
                            sample['responses'][:prev].tolist() + req['input_ids']
                        )  # input_ids record the description of the tool action and natural language tokens.
                        response.append(
                            sample['responses'][:prev].tolist() + req['input_ids']
                        )   # this is for computing reward, hence using input_ids as the description of action
                        assert len(req['input_ids']) == len(req['response']), \
                            f"input_ids and response should have the same length, got {len(req['input_ids'])} and {len(req['response'])}"
                        seq_mask.append(
                            sample['seq_mask'][:prev].bool().tolist() + req['seq_mask']
                        )
                        tool_mask.append(
                            sample['tool_mask'][:prev].bool().tolist() + req['tool_mask']
                        )
                        seq_action_masks = torch.cat(
                            (
                                sample['earl_action_mask'][:prev].cpu().clone(),
                                torch.stack(req['earl_action_mask'], dim=0)
                            ), dim=0
                        )
                        earl_action_mask.append(seq_action_masks)
                # breakpoint()
                # pad list of tensor of variable lengths -> tensor with fixed length
                response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                response_earl = pad_2d_list_to_length(response_earl, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                input = pad_2d_list_to_length(input, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                seq_mask = pad_2d_list_to_length(seq_mask, 0, max_length=self.config.response_length).to(idx.device)
                tool_mask = pad_2d_list_to_length(tool_mask, 0, max_length=self.config.response_length).to(idx.device)
                earl_action_mask = pad_and_stack_masks(earl_action_mask, True, max_length=self.config.response_length).to(idx.device)
                seq = torch.cat([idx, input], dim=-1)
            
            # breakpoint()
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        else:
            idx = None
            response = None
            seq = None
            response_earl = None
            attention_mask = None
            position_ids = None
            seq_mask = None
            tool_mask = None
            earl_action_mask = None
            interaction_kwargs_list = None

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # breakpoint()
        
        def combine(t1, t2, mask):
            """
            Combine two tensors based on a mask.
            If mask is True, take from t1; otherwise, take from t2.
            """
            if mask.sum() == 0:
                return t2
            new_t = torch.zeros(torch.cat((t1,t2), dim=0).shape, dtype=t2.dtype, device=t1.device)
            new_t[mask] = t1.to(t2.dtype)
            new_t[~mask] = t2
            return new_t

        batch = TensorDict(
            {
                "prompts": combine(idx, ori.batch["prompts"], mask),
                "responses": combine(response, ori.batch["responses"], mask),
                "input_ids": combine(seq, ori.batch["input_ids"], mask),
                "response_earl": combine(response_earl, ori.batch["response_earl"], mask),
                "attention_mask": combine(attention_mask, ori.batch["attention_mask"], mask),
                "position_ids": combine(position_ids, ori.batch["position_ids"], mask),
                "seq_mask": combine(seq_mask, ori.batch["seq_mask"], mask),
                "tool_mask": combine(tool_mask, ori.batch["tool_mask"], mask),
                "earl_action_mask": combine(earl_action_mask, ori.batch["earl_action_mask"], mask),
                "rewrite_mask": mask,
            },
            batch_size=len(mask),
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        # breakpoint()
        non_tensor_batch['interaction_kwargs'][mask.bool().cpu().numpy()] = np.array(interaction_kwargs_list)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    def _SFT_rewrite_sequences(self, prompts: DataProto, mask: torch.Tensor) -> DataProto:
        """
        Rewrite sequences using the vLLM inference engine.
        This method is similar to `generate_sequences`, but it is used for rewriting
        existing sequences rather than generating new ones.
        """
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        non_tensor_batch = prompts.non_tensor_batch
        ori = prompts[~mask]
        prompts = prompts[mask]

        if mask.sum() > 0:
            # Need rewriting

            # TODO: fix to the first trainable tools. Maybe enable multiple tool call position?
            prompts.batch['call_tool_pos'] = prompts.batch['call_tool_pos'].squeeze(1) + 1
            starting_tool_id = prompts.batch['call_tool_id'][0][0].item()
            starting_mode = self.tool_config.tool_action_id_to_name[starting_tool_id]

            prompt_length = prompts.batch['input_ids'].size(1) - prompts.batch['responses'].size(1)
            idx = prompts.batch["input_ids"][:, :prompt_length]  # (bs, prompt_length)
            # left-padded attention_mask
            attention_mask = prompts.batch["attention_mask"][:, :prompt_length]
            position_ids = prompts.batch["position_ids"][:, :prompt_length]

            # used to construct attention_mask
            eos_token_id = prompts.meta_info["eos_token_id"]
            batch_size = idx.size(0)

            old_response_lengths = [
                s["attention_mask"][:s["call_tool_pos"]].sum().item() - s["attention_mask"][:prompt_length].sum(dim=-1).item()
                for s in prompts.batch
            ]

            def get_raw_prompt_ids(input_ids, attn_mask, pos):
                return input_ids[:pos][attn_mask[:pos].bool()].tolist()

            vllm_inputs = [
                {"prompt_token_ids": get_raw_prompt_ids(
                    prompts.batch['input_ids'][i],
                    prompts.batch['attention_mask'][i],
                    prompts.batch['call_tool_pos'][i])
                } for i in range(batch_size)
            ]
            # breakpoint()

            # ensure the type of `prompt_token_ids` passed to vllm is list[int]
            # https://github.com/volcengine/verl/pull/772
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")
        
            kwargs = {
                "n": 1,
            }

            lora_requests = None
            if self.lora_kwargs:
                lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    lora_int_id = lora_int_ids[0]
                    lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

            # users can customize different sampling_params at different run
            # breakpoint()
            with self.update_sampling_params(**kwargs):
                with self.update_vllm_sampler(starting_mode=starting_mode):
                    # try:
                    self.earl_request_tracker.clean()   # clean up prev requests
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                    # except Exception as e:
                    #     print(e)
                    #     breakpoint()

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
                response = []
                response_earl = []
                input = []
                seq_mask = []
                rollout_log_probs = []
                earl_action_mask = []
                tool_mask = []
                for i, output in enumerate(outputs):
                    for sample_id in range(len(output.outputs)):
                        if len(output.outputs) == 1:
                            req_id = f"{output.request_id}"
                        else:
                            # breakpoint()
                            req_id = f"{sample_id}_{output.request_id}"
                        assert req_id in self.earl_request_tracker.running
                        sample = prompts.batch[i]
                        prev = old_response_lengths[i]
                        req = self.earl_request_tracker.running[req_id]
                        req = cut_request_output(req, self.config.response_length - prev)  # cut extra tokens, e.g., due to tool output
                        # extract info from output
                        response_earl.append(
                            sample['response_earl'][:prev].tolist() + req['response']
                        )   # req[response] records the action chosen by the model, including tool actions
                        input.append(
                            sample['responses'][:prev].tolist() + req['input_ids']
                        )  # input_ids record the description of the tool action and natural language tokens.
                        response.append(
                            sample['responses'][:prev].tolist() + req['input_ids']
                        )   # this is for computing reward, hence using input_ids as the description of action
                        assert len(req['input_ids']) == len(req['response']), \
                            f"input_ids and response should have the same length, got {len(req['input_ids'])} and {len(req['response'])}"
                        seq_mask.append(
                            sample['seq_mask'][:prev].bool().tolist() + req['seq_mask']
                        )
                        tool_mask.append(
                            sample['tool_mask'][:prev].bool().tolist() + req['tool_mask']
                        )
                        seq_action_masks = torch.cat(
                            (
                                sample['earl_action_mask'][:prev].cpu().clone(),
                                torch.stack(req['earl_action_mask'], dim=0)
                            ), dim=0
                        )
                        earl_action_mask.append(seq_action_masks)
                # pad list of tensor of variable lengths -> tensor with fixed length
                response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                response_earl = pad_2d_list_to_length(response_earl, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                input = pad_2d_list_to_length(input, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                seq_mask = pad_2d_list_to_length(seq_mask, 0, max_length=self.config.response_length).to(idx.device)
                tool_mask = pad_2d_list_to_length(tool_mask, 0, max_length=self.config.response_length).to(idx.device)
                earl_action_mask = pad_and_stack_masks(earl_action_mask, True, max_length=self.config.response_length).to(idx.device)
                seq = torch.cat([idx, input], dim=-1)
            
            # breakpoint()
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        else:
            idx = None
            response = None
            seq = None
            response_earl = None
            attention_mask = None
            position_ids = None
            seq_mask = None
            tool_mask = None
            earl_action_mask = None

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # breakpoint()
        
        def combine(t1, t2, mask):
            """
            Combine two tensors based on a mask.
            If mask is True, take from t1; otherwise, take from t2.
            """
            if mask.sum() == 0:
                return t2
            new_t = torch.zeros(torch.cat((t1,t2), dim=0).shape, dtype=t2.dtype, device=t1.device)
            new_t[mask] = t1.to(t2.dtype)
            new_t[~mask] = t2
            return new_t

        batch = TensorDict(
            {
                "prompts": combine(idx, ori.batch["prompts"], mask),
                "responses": combine(response, ori.batch["responses"], mask),
                "input_ids": combine(seq, ori.batch["input_ids"], mask),
                "response_earl": combine(response_earl, ori.batch["response_earl"], mask),
                "attention_mask": combine(attention_mask, ori.batch["attention_mask"], mask),
                "position_ids": combine(position_ids, ori.batch["position_ids"], mask),
                "seq_mask": combine(seq_mask, ori.batch["seq_mask"], mask),
                "tool_mask": combine(tool_mask, ori.batch["tool_mask"], mask),
                "earl_action_mask": combine(earl_action_mask, ori.batch["earl_action_mask"], mask),
                "rewrite_mask": mask,
            },
            batch_size=len(mask),
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        # breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @GPUMemoryLogger(role="vllm rewrite rollout", logger=logger)
    @torch.no_grad()
    def rewrite_sequences(self, prompts: DataProto, mask: torch.Tensor) -> DataProto:
        if self.tool_config.is_sft_mode=="default":
            return self._default_rewrite_sequences(prompts,mask)
        else:
            assert(self.tool_config.is_sft_mode,"sft")
            return self._SFT_rewrite_sequences(prompts,mask)

    @contextmanager
    def update_vllm_sampler(self, starting_mode: str = "default"):
        # update sampling params
        sampler = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.sampler
        old_starting_mode = sampler.starting_mode
        sampler.starting_mode = starting_mode
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        sampler.starting_mode = old_starting_mode