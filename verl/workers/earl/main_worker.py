from verl.workers.fsdp_workers import ActorRolloutRefWorker, device_name, logger, get_sharding_strategy
from verl.workers.earl.vllm_logits_processor import EarlLogitsProcessor
import os
import warnings
from dataclasses import asdict
from typing import Optional, Union
import itertools

import psutil
import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_outputs import CausalLMOutputWithPast
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.workers.earl.earl_functional import top_k_top_p_sample
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.debug.performance import reduce_timing
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from verl.workers.earl.utils import build_tool_config, print_rollout_results
from verl.workers.earl.models import create_earl_head
from verl.workers.earl.earl_actor import EarlActor

from contextlib import contextmanager

class EarlActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    Actor worker for the Earl model runner.
    This worker is used to run the model in a distributed manner.
    """
    def __init__(self, *args, **kwargs):
        tool_config = kwargs.pop('tool_config', None)
        self.tool_config = tool_config
        super().__init__(*args, **kwargs)
        self.earl_config = self.config.earl
        self.org_vocab_size = 0
        self.model_config = self.earl_config.model
    
    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        self.torch_dtype = torch_dtype

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            if role == 'actor':
                # create earl head
                self.org_vocab_size = actor_module.lm_head.weight.shape[0]
                self.tool_config = build_tool_config(
                    self.org_vocab_size,
                    self.earl_config.training.starting_mode,
                    self.tool_config,
                    self.tokenizer
                )
                if self.tool_config.total_size > 0:
                    earl_head = create_earl_head(
                        actor_module, self.tool_config, self.model_config, torch_dtype
                    )
                    actor_module.earl_head = earl_head

                    # patch forward function
                    def custom_forward(
                        self,
                        input_ids: Optional[torch.LongTensor] = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_values = None,
                        inputs_embeds: Optional[torch.FloatTensor] = None,
                        labels: Optional[torch.LongTensor] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        cache_position: Optional[torch.LongTensor] = None,
                        logits_to_keep: Union[int, torch.Tensor] = 0,
                        **kwargs,
                    ):
                        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                        output_hidden_states = (
                            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                        )
                        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            cache_position=cache_position,
                            **kwargs,
                        )

                        hidden_states = outputs.last_hidden_state
                        
                        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                        logits = self.lm_head(hidden_states[:, slice_indices, :])
                        extra_logits = self.earl_head(hidden_states[:, slice_indices, :])
                        loss = None
                        logits = torch.cat([logits, extra_logits], dim=-1)
                        if labels is not None:
                            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                        return CausalLMOutputWithPast(
                            loss=loss,
                            logits=logits,
                            past_key_values=outputs.past_key_values,
                            hidden_states=outputs.hidden_states,
                            attentions=outputs.attentions,
                        )
                    
                    from types import MethodType
                    model_to_patch = actor_module
                    model_to_patch.forward = MethodType(custom_forward, model_to_patch)


            # freeze base model if needed
            freeze_base_model = self.model_config.get('freeze_base_model', False)
            if role == 'actor' and freeze_base_model:
                for name, param in actor_module.named_parameters():
                    if 'earl_head' not in name:
                        param.requires_grad = False

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {"task_type": TaskType.CAUSAL_LM, "r": self.config.model.lora_rank, "lora_alpha": self.config.model.lora_alpha, "target_modules": convert_to_regular_types(self.config.model.target_modules), "bias": "none"}
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get("lora_rank", 0) > 0)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=freeze_base_model,  # when this is false, does not support partial param finetuning
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.config.actor.fsdp_config.forward_prefetch,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
            # get trainable parameters
            trainable_params = itertools.chain(
                (p for p in actor_module_fsdp.parameters() if p.requires_grad),
            )
            actor_optimizer = optim.AdamW(
                trainable_params,
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = EarlActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    def _build_rollout(self, trust_remote_code=False):
        assert self.tool_config is not None, "Tool config must be built (in _build_model_optimizer for actor) before building rollout"
        from torch.distributed.device_mesh import init_device_mesh
        from verl.workers.rollout.vllm_rollout import vllm_mode
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        assert rollout_name == 'vllm' and vllm_mode == 'spmd' and self.config.rollout.mode == "sync", "Only sync vllm spmd rollout is supported for earl"

        from verl.workers.earl.vllm_rollout import EarlRollout
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
        # lora_kwargs = {}

        rollout = EarlRollout(
            model_path=local_path,
            config=self.config.rollout,
            tool_config=self.tool_config,
            tokenizer=self.tokenizer, 
            model_hf_config=self.actor_model_config,
            device_mesh=rollout_device_mesh,
            trust_remote_code=trust_remote_code,
            **lora_kwargs
        )
        log_gpu_memory_usage(f"After building EaRL {rollout_name} rollout", logger=logger)
        full_params = torch.distributed.get_world_size() == 1
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            full_params=full_params,
            device_mesh=rollout_device_mesh,
            offload_param=self._is_offload_param,
            load_format=self.config.rollout.load_format,
            layered_summon=self.config.rollout.get("layered_summon", False),
        )
        if self.tool_config.total_size > 0:
            # add earl head to rollout model
            rollout_model = rollout_sharding_manager.model_runner.model
            temp_config = self.model_config.copy()
            temp_config.init_from_base = False  # this prevents error in model creation; weight is synced with actor by rollout_sharding_manager
            earl_head = create_earl_head(
                rollout_model, self.tool_config, temp_config, rollout_model.lm_head.weight.dtype
            )
            rollout_model.earl_head = earl_head
            # set VLLM logits processor to allow additional actions
            original_logits_processor = rollout_model.logits_processor
            rollout_sharding_manager.model_runner.model.logits_processor = \
                EarlLogitsProcessor.from_vllm_logits_processor(
                    original_logits_processor,
                    self.tool_config,
                    rollout_model.earl_head
                )
        # set VLLM sampler to EaRL sampler
        # set VLLM GPU runner to EaRL GPU runner
        log_gpu_memory_usage("After building sharding manager", logger=logger)
        return rollout, rollout_sharding_manager

    # TODO: Implement the save and load checkpoint methods
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
    
    @contextmanager
    def update_finetune_context(self, load_optimizer: bool = False, load_extra: bool = False):
        old_contents = self.checkpoint_manager.checkpoint_load_contents
        new_contents = ["model"]
        if load_optimizer:
            new_contents.append("optimizer")
        if load_extra:
            new_contents.append("extra")
        self.checkpoint_manager.checkpoint_load_contents = new_contents
        yield
        self.checkpoint_manager.checkpoint_load_contents = old_contents

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_finetune_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor or (not self._is_actor and self._is_rollout), f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got {self._is_actor} and {self._is_rollout}"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        with self.update_finetune_context(load_optimizer=False, load_extra=False):
            self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=False)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="blue")
    def compute_log_prob(self, data: DataProto):
        return super().compute_log_prob(data)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="blue")
    def compute_tool_scores(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # Support all hardwares
        from contextlib import nullcontext
        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_device_id())
        tool_name = data.meta_info.pop("call_tool_name", None)
        data.meta_info["micro_batch_size"] = self.config.earl.training.tool_score_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = True    # for efficiency
        data.meta_info["temperature"] = 1.
        data.meta_info["temperature_tool"] = self.config.earl.training.tool_score_temperature
        top_k = self.config.earl.training.tool_score_top_k
        top_p = self.config.earl.training.tool_score_top_p
        data.meta_info['org_vocab_size'] = self.tool_config.org_vocab_size
        output = {}
        all_pos = []
        all_tool_ids = []
        
        # get seq
        action_id = self.tool_config.tool_name_to_action_id[tool_name]
        # TODO: generalize when multiple cpo seqs are available
        seq = self.tool_config.tool_action_id_to_cpo_seqs[action_id][0]
        data.meta_info['tool_seq'] = seq
        data.meta_info['tool_idx'] = action_id
        # compute log probs as scores
        with self.ulysses_sharding_manager:
            tool_data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                scores = self.actor.compute_tool_scores(data=tool_data)
            output[f"{tool_name}_scores"] = scores
            call_tool_pos = top_k_top_p_sample(scores, top_k=top_k, top_p=top_p)
            all_pos.append(call_tool_pos)
            all_tool_ids.append(torch.tensor([action_id]).unsqueeze(1).repeat(len(data), 1))

        all_pos = torch.cat(all_pos, dim=1)
        all_tool_ids = torch.cat(all_tool_ids, dim=1)
        output["call_tool_pos"] = all_pos
        output["call_tool_id"] = all_tool_ids
        output = DataProto.from_dict(output)
        output = output.to("cpu")
        output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def compute_gt_sequences(self, prompts: DataProto):
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("compute_gt_sequences", timing_generate):
                output = self.rollout.compute_gt_sequences(prompts=prompts)
            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)
    
        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def rewrite_sequences(self, data: DataProto):
        data = data.to(get_device_id())
        assert self._is_rollout
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        data.meta_info.update(meta_info)
        timing_generate = {}
        id_to_score = {}
        id_to_indices = {}
        rewrite_mask = torch.zeros(len(data), dtype=torch.bool, device=get_device_id())
        for i in range(len(data)):
            uid = data.non_tensor_batch['uid'][i]
            score = data.batch['token_level_rewards'][i].sum().item()
            if uid not in id_to_score:
                id_to_score[uid] = score
                id_to_indices[uid] = [i]
            else:
                id_to_score[uid] += score
                id_to_indices[uid].append(i)
        for uid, score in id_to_score.items():
            if score == 0.:
                rewrite_mask[id_to_indices[uid]] = True
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            data = self.rollout_sharding_manager.preprocess_data(data)
            with simple_timer("rewrite_sequence", timing_generate):
                output = self.rollout.rewrite_sequences(prompts=data, mask=rewrite_mask)
            log_gpu_memory_usage("After rollout generation", logger=logger)
            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()
        return output
    
    def debug_batch(self, batch):
        idx = 0
        print_rollout_results(
            self.tokenizer, self.tool_config, batch, idx, f"debug_rollout_{idx}.txt"
        )