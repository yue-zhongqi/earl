import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
from verl.workers.actor.dp_actor import DataParallelPPOActor, logger
from verl.workers.earl.earl_functional import entropy_from_logits, compute_log_probs, entropy_from_logits_with_chunking
from transformers.cache_utils import DynamicCache


def get_dynamic_chunk_size(local_num_tokens, max_chunk_size=128):
    """
    Dynamically compute a chunk size such that all ranks have a similar number of chunks.

    Args:
        local_num_tokens (int): The total number of tokens in the current rank.
        target_chunk_size (int): The desired chunk size (default: 128).

    Returns:
        int: The dynamically computed chunk size.
    """
    import torch.distributed as dist
    import math
    # Step 1: Gather the total number of tokens from all ranks
    world_size = dist.get_world_size()
    local_num_tokens_tensor = torch.tensor([local_num_tokens], dtype=torch.long, device="cuda")
    all_num_tokens = [torch.zeros_like(local_num_tokens_tensor) for _ in range(world_size)]
    dist.all_gather(all_num_tokens, local_num_tokens_tensor)

    # Step 2: Compute the total number of tokens across all ranks
    local_batch_sizes = [t.item() for t in all_num_tokens]
    K = max(math.ceil(b / max_chunk_size) if b > 0 else 0
            for b in local_batch_sizes)
    K = max(K, 1)

    q, rem = divmod(local_num_tokens, K)
    return [q + 1] * rem + [q] * (K - rem)

class EarlActor(DataParallelPPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module, actor_optimizer)
        if self.config.entropy_from_logits_with_chunking:
            self.entropy_from_logits = entropy_from_logits_with_chunking
        else:
            self.entropy_from_logits = entropy_from_logits

    @GPUMemoryLogger(role="earl actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, force_label=None) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # breakpoint()
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = [
                "responses", "input_ids", "attention_mask",
                "position_ids", "seq_mask", "earl_action_mask",
                "tool_mask", "response_earl"
            ]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
            # with torch.inference_mode():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, force_label=force_label
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="earl actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = [
            "responses", "input_ids", "attention_mask",
            "position_ids", "old_log_probs", "advantages",
            "seq_mask", "earl_action_mask", "tool_mask", "response_earl"
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)

                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                # breakpoint()
                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, earl_ref_log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_earl_ref=True
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # breakpoint()
                    earl_tool_mask = (earl_ref_log_prob != 0.0)
                    pg_mask = torch.logical_and(earl_tool_mask, response_mask)
                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    elif self.config.policy_loss.loss_mode == "ce":
                        pg_loss = -agg_loss(loss_mat=log_prob, loss_mask=response_mask, loss_agg_mode='token-mean')
                        pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
                        ppo_kl = torch.tensor(0.0, device=pg_loss.device)
                        pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # breakpoint()
                        # get the mask where earl_ref_log_prob is zero
                        earl_ref_mask = (earl_ref_log_prob != 0.0)
                        kl_mask = torch.logical_and(earl_ref_mask, response_mask)
                        # compute kl loss
                        kld = kl_penalty(logprob=earl_ref_log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=kl_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    # breakpoint()
                    loss.backward()
                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    @torch.no_grad()
    def compute_insert_log_probs(self, input_ids, insert_ids, tool_id, logits_mask=None):
        assert len(insert_ids) >= 1
        model = self.actor_module
        device = input_ids.device
        # log_probs = []

        bsz = len(input_ids)
        input_ids = input_ids.unsqueeze(0)
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values  # tuple of (k,v) per layer, shapes: [1, H, P, D]
        assert past is not None
        
        logits = out.logits
        if logits_mask is not None:
            logits.masked_fill_(~logits_mask, float("-inf"))
        # breakpoint()
        # logp = logits.log_softmax(dim=-1)[0, :, insert_ids[0]]
        logp = logprobs_from_logits(
            logits=logits[0],
            labels=torch.tensor(insert_ids[0]).to(logits.device).repeat(logits.shape[1]),
            inplace_backward=True,
        )
        if tool_id > 0:
            tool_logp = logprobs_from_logits(
                logits=logits[0],
                labels=torch.tensor(tool_id).to(logits.device).repeat(logits.shape[1]),
                inplace_backward=True,
            )
        else:
            tool_logp = None

        if len(insert_ids) > 1:
            chunk_sizes = get_dynamic_chunk_size(bsz, max_chunk_size=128)
            # print(chunk_sizes)
            pos = torch.arange(bsz, device=device).unsqueeze(1)
            attn_mask = torch.tril(torch.ones((bsz, bsz), dtype=torch.bool, device=device))
            c = 0
            for chunk_size in chunk_sizes:
                pos_c = pos[c:c+chunk_size].clone()
                csz = len(pos_c)
                tiled = []
                for k, v in past:
                    # [1,H,P,D] -> [B,H,P,D]
                    tiled.append((k.expand(csz, -1, -1, -1).contiguous(),
                                v.expand(csz, -1, -1, -1).contiguous()))
                past_b = DynamicCache(tuple(tiled))
                # create a causal mask
                attn_mask_c = attn_mask[c:c+chunk_size].clone()
                true_mask = torch.ones((csz, 1), dtype=torch.bool, device=device)  # True for new token

                for i in range(0, len(insert_ids)-1):
                    attn_mask_c = torch.cat((attn_mask_c, true_mask), dim=1)  # add a True column for the new token
                    input_ids = torch.tensor(insert_ids[i:i+1], device=device).unsqueeze(0).repeat(csz,1)
                    # breakpoint()
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask_c,
                        past_key_values=past_b,
                        position_ids=pos_c + i + 1,
                        use_cache=True,
                    )
                    logits = out.logits
                    if logits_mask is not None:
                        logits.masked_fill_(~logits_mask, float("-inf"))
                    # logp = logits.log_softmax(dim=-1)[:, 0, insert_ids[i+1]]
                    # breakpoint()
                    logp[c:c+chunk_size] += logprobs_from_logits(
                        logits=logits,
                        labels=torch.tensor(insert_ids[i+1]).to(logits.device).repeat(logits.shape[0]),
                        inplace_backward=True,
                    ).squeeze(1)
                    past_b = out.past_key_values
                del past_b
                c += chunk_size
        output = torch.max(logp, tool_logp) if tool_logp is not None else logp
        return output

    @GPUMemoryLogger(role="earl actor", logger=logger)
    def compute_tool_scores(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()
        temperature = data.meta_info['temperature_tool']
        org_vocab_size = data.meta_info['org_vocab_size']
        insert_ids = data.meta_info['tool_seq']
        if len(insert_ids) == 1:
            # fast method
            prompt_length = data.batch['prompts'].size(1)
            response_length = data.batch['responses'].size(1)
            # part_scores compute the log prob of inserting ** ahead ** of each token in the response
            part_scores = self.compute_log_prob(data, calculate_entropy=False, force_label=insert_ids[0])[0]
            tool_id = data.meta_info['tool_idx']
            if tool_id > 0:
                part_scores2 = self.compute_log_prob(data, calculate_entropy=False, force_label=tool_id)[0]
                part_scores = torch.max(part_scores, part_scores2)
            # note that insertable mask no need roll left, since it is already aligned with response log probs
            insertable_mask = data.batch['insertable_mask'].bool()
            non_insertable_samples = (insertable_mask.sum(dim=-1) == 0)
            part_scores.masked_fill_(~insertable_mask, float("-inf"))
            part_scores[non_insertable_samples, 0] = 0.  # if no where to insert tool, put it at first position
            scores = torch.zeros(
                data.batch['attention_mask'].shape, dtype=part_scores.dtype, device=part_scores.device
            ).fill_(float("-inf"))
            # here, we roll the scores left so that scores = the log prob of inserting ** after ** each token
            scores[:, prompt_length-1:prompt_length+response_length-1] = part_scores.div_(temperature)
        else:
            # slow method for multiple token insertion
            scores = []
            for sample in data.batch:
                input_mask = sample['attention_mask'].bool()
                # breakpoint()
                non_tool_mask = sample['insertable_mask'].bool()
                
                # get first True position in non_tool_mask
                first_non_tool = torch.argmax(non_tool_mask.int(), dim=-1)
                # this is the logits mask outside tool mode
                logits_mask = sample['earl_action_mask'].bool()[first_non_tool]

                # prepare insertable mask
                prompt_mask = torch.zeros_like(sample['prompts'], dtype=torch.bool)
                prompt_length = len(sample['prompts'])
                # tool insertable ** before ** each token
                insertable_mask = torch.cat((prompt_mask, non_tool_mask), dim=-1)  # (bsz, seqlen)
                # exclude paddings
                insertable_mask = insertable_mask[input_mask]
                # to align with self.compute_insert_log_probs output
                insertable_mask = torch.roll(insertable_mask, shifts=-1, dims=-1)
                # prepare logits mask
                org_vocab_action_mask = torch.ones(org_vocab_size, dtype=torch.bool, device=input_mask.device)
                logits_mask = torch.cat((org_vocab_action_mask, logits_mask), dim=-1)

                # compute log probs
                input_ids = sample['input_ids'][input_mask]
                insert_ids = data.meta_info['tool_seq']
                tool_id = data.meta_info['tool_idx']
                
                # this computes the log probs of inserting after each token
                log_probs = self.compute_insert_log_probs(input_ids, insert_ids, tool_id, logits_mask).div_(temperature)
                log_probs.masked_fill_(~insertable_mask, float("-inf"))  # mask out the tool tokens

                if insertable_mask.sum() == 0:
                    log_probs[prompt_length - 1] = 0.    # if no where to insert tool, put it right after prompt

                full_log_probs = torch.zeros(
                    input_mask.shape, dtype=log_probs.dtype, device=input_mask.device
                )
                # set full_log_probs as -inf
                full_log_probs.fill_(float("-inf"))
                full_log_probs[input_mask] = log_probs

                scores.append(full_log_probs)
            scores = torch.stack(scores, dim=0)  # shape (bsz, seqlen)
        # breakpoint()
        return scores
    
    @GPUMemoryLogger(role="earl actor", logger=logger)
    def compute_tool_scores_old(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()
        temperature = data.meta_info['temperature']
        org_vocab_size = data.meta_info['org_vocab_size']

        scores = []
        for sample in data.batch:
            input_mask = sample['attention_mask'].bool()
            non_tool_mask = sample['response_earl'] < org_vocab_size
            # get first True position in non_tool_mask
            first_non_tool = torch.argmax(non_tool_mask.int(), dim=-1)
            logits_mask = sample['earl_action_mask'].bool()[first_non_tool]

            # prepare output mask
            prompt_mask = torch.zeros_like(sample['prompts'], dtype=torch.bool)
            non_tool_mask = torch.cat((prompt_mask, non_tool_mask), dim=-1)  # (bsz, seqlen)
            non_tool_mask = non_tool_mask[input_mask]
            # shift non_tool_mask to the left by 1
            non_tool_mask_l = torch.roll(non_tool_mask, shifts=-1, dims=-1)
            # logical and non_tool_mask, non_tool_mask_l, and input_mask

            output_mask = non_tool_mask & non_tool_mask_l
            # breakpoint()
            # assert output_mask.sum() > 0

            # prepare logits mask
            org_vocab_action_mask = torch.ones(org_vocab_size, dtype=torch.bool, device=input_mask.device)
            logits_mask = torch.cat((org_vocab_action_mask, logits_mask), dim=-1)

            # compute log probs
            input_ids = sample['input_ids'][input_mask]
            insert_ids = data.meta_info['tool_seq']
            tool_id = data.meta_info['tool_idx']
            
            log_probs = self.compute_insert_log_probs(input_ids, insert_ids, tool_id, logits_mask).div_(temperature)
            log_probs.masked_fill_(~output_mask, float("-inf"))  # mask out the tool tokens

            if output_mask.sum() == 0:
                log_probs[-1] = 0    # if no where to insert tool, put it at last position

            full_log_probs = torch.zeros(
                input_mask.shape, dtype=log_probs.dtype, device=input_mask.device
            )
            # set full_log_probs as -inf
            full_log_probs.fill_(float("-inf"))
            full_log_probs[input_mask] = log_probs

            scores.append(full_log_probs)
        # breakpoint()
        scores = torch.stack(scores, dim=0)  # shape (bsz, seqlen)
        # torch.distributed.barrier()
        return scores

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_earl_ref=False, force_label=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        prompt_length = micro_batch["input_ids"].size(-1) - response_length
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                # Special handling for MiniCPM-o model: pixel_values, image_bound, and tgt_sizes
                # need different concatenation strategies compared to other multimodal inputs
                if (key == "pixel_values" and isinstance(micro_batch["multi_modal_inputs"][0]["pixel_values"], list)) or key == "image_bound" or key == "tgt_sizes":
                    # For MiniCPM-o: keep as list structure instead of concatenating tensors
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            
            seq_mask = micro_batch['seq_mask']
            seq_mask_padding = torch.zeros((batch_size, prompt_length), dtype=torch.bool, device=seq_mask.device)
            seq_mask = torch.cat((seq_mask_padding, seq_mask), dim=1)
            
            earl_action_mask = micro_batch['earl_action_mask']
            action_size = earl_action_mask.size(-1)
            action_mask_padding = torch.zeros((batch_size, prompt_length, action_size), dtype=torch.bool, device=earl_action_mask.device)
            earl_action_mask = torch.cat((action_mask_padding, earl_action_mask), dim=1)
            
            response_earl = micro_batch['response_earl']
            response_earl = torch.cat((input_ids[:, :prompt_length], response_earl), dim=1)

            tool_mask = micro_batch['tool_mask']
            tool_mask_padding = torch.zeros((batch_size, prompt_length), dtype=torch.bool, device=tool_mask.device)
            tool_mask = torch.cat((tool_mask_padding, tool_mask), dim=1)
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                seq_mask_rmpad = index_first_axis(
                    rearrange(seq_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)                   # (1, total_nnz)
                response_earl_rmpad = index_first_axis(
                    rearrange(response_earl.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)                   # (1, total_nnz)
                tool_mask_rmpad = index_first_axis(
                    rearrange(tool_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)                   # (1, total_nnz)
                if earl_action_mask.shape[-1] == 0:
                    earl_action_mask_rmpad = None
                else:   
                    earl_action_mask_rmpad = index_first_axis(
                        rearrange(earl_action_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 2).transpose(1, 2)   # (1, total_nnz, action_size)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if "multi_modal_inputs" in micro_batch:
                    # MiniCPM-o specific processing for image bounds and pixel values
                    if "image_bound" in multi_modal_inputs:
                        # Adjust image bounds based on left padding and cumulative sequence lengths
                        # This is necessary for MiniCPM-o's vision-language alignment
                        left_padding_length = torch.argmax(attention_mask, dim=1)
                        image_bounds = []
                        for i in range(len(multi_modal_inputs["image_bound"])):
                            image_bound = multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
                            image_bounds.append(image_bound)
                        multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
                        # Flatten pixel values list for MiniCPM-o processing
                        pixel_values = []
                        for i in range(len(multi_modal_inputs["pixel_values"])):
                            pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])
                        multi_modal_inputs["pixel_values"] = [pixel_values]
                    # Handle target sizes for MiniCPM-o vision processing
                    if "tgt_sizes" in multi_modal_inputs:
                        multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]

                # for compute the log_prob
                response_rmpad_rolled = torch.roll(response_earl_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    raise NotImplementedError("Ulysses SP is not supported in EarlActor yet.")

                response_rmpad_rolled = response_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # earl-specific masks rolling
                if earl_action_mask_rmpad is not None:
                    earl_action_mask_rolled = torch.roll(earl_action_mask_rmpad, shifts=-1, dims=1).squeeze(0)
                else:
                    earl_action_mask_rolled = None
                tool_mask_rolled = torch.roll(tool_mask_rmpad, shifts=-1, dims=1).squeeze(0).to(torch.bool)
                seq_mask_rolled = torch.roll(seq_mask_rmpad, shifts=-1, dims=1).squeeze(0).to(torch.bool)
                
                # only pass input_ids and position_ids to enable flash_attn_varlen
                # extra_args = {
                #     'earl_head': self.earl_head,
                # }
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    output_hidden_states=True,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                # hidden_states = output.hidden_states[-1]  # (1, total_nnz, hidden_size)
                # tool_logits = self.earl_head(hidden_states)
                if self.use_fused_kernels:
                    # log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    # entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    raise NotImplementedError("Fused kernels are not supported in EarlActor yet.")
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    if earl_action_mask_rolled is None:
                        org_vocab_size = logits_rmpad.size(1)
                        tool_size = 0
                    else:
                        org_vocab_size = logits_rmpad.size(1) - earl_action_mask_rolled.size(1)
                        tool_size = earl_action_mask_rolled.size(1)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    # compute ref log probs at non-tool position
                    if return_earl_ref:
                        earl_ref_mask = (response_rmpad_rolled < org_vocab_size)
                        ref_logits = logits_rmpad[earl_ref_mask, :org_vocab_size]
                        earl_ref_log_probs = logprobs_from_logits(
                            logits=ref_logits,
                            labels=response_rmpad_rolled[earl_ref_mask],
                            inplace_backward=inplace_backward,
                        )   # uses flash attn optimization
                        earl_ref_log_probs_full = torch.zeros_like(response_rmpad_rolled, dtype=earl_ref_log_probs.dtype)
                        earl_ref_log_probs_full[earl_ref_mask] = earl_ref_log_probs
                        # breakpoint()

                    # apply action mask
                    org_vocab_action_mask = ~tool_mask_rolled.unsqueeze(-1)  # if tool_mask is True, then org vocab is not allowed
                    org_vocab_action_mask = org_vocab_action_mask.expand(-1, org_vocab_size)  # (total_nnz, vocab_size)
                    if earl_action_mask_rolled is not None:
                        logits_mask = torch.cat((org_vocab_action_mask, earl_action_mask_rolled), dim=-1)
                    else:
                        logits_mask = org_vocab_action_mask
                    logits_rmpad.masked_fill_(~logits_mask, float("-inf"))

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    # compute tool-specific log probs separately
                    if force_label is not None:
                        # create labels of shape response_rmpad_rolled filled with force_label
                        labels = torch.full_like(response_rmpad_rolled, force_label)
                    else:
                        labels = response_rmpad_rolled
                    log_probs = compute_log_probs(
                        logits=logits_rmpad,
                        labels=labels,
                        inplace_backward=inplace_backward,
                        tool_mask=tool_mask_rolled,
                        org_vocab_size=org_vocab_size,
                        tool_size=tool_size,
                    )   # compute location wiith tool_mask=True and False separately to prevent NaN output from flash attn chunking
                    log_probs.masked_fill_(~seq_mask_rolled, 0.0)
                    # breakpoint()

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.entropy_from_logits(logits_rmpad, logits_mask)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.entropy_from_logits, logits_rmpad, logits_mask
                            )
                        entropy_rmpad.masked_fill_(~seq_mask_rolled, 0.0)
                # breakpoint()

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    # log_probs = gather_outpus_and_unpad(
                    #     log_probs,
                    #     gather_dim=0,
                    #     unpad_dim=0,
                    #     padding_size=pad_size,
                    # )
                    # if calculate_entropy:
                    #     entropy_rmpad = gather_outpus_and_unpad(
                    #         entropy_rmpad,
                    #         gather_dim=0,
                    #         unpad_dim=0,
                    #         padding_size=pad_size,
                    #     )
                    raise NotImplementedError("Ulysses SP is not supported in EarlActor yet.")
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if return_earl_ref:
                    earl_ref_log_probs_full = pad_input(
                        hidden_states=earl_ref_log_probs_full.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    earl_ref_log_probs_full = earl_ref_log_probs_full.squeeze(-1)[:, -response_length - 1 : -1]
            else:  # not using rmpad and no ulysses sp
                # extra_args = {}
                # if self.use_fused_kernels:
                #     extra_args["temperature"] = temperature
                #     extra_args["return_dict"] = True

                # output = self.actor_module(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     position_ids=position_ids,
                #     **multi_modal_inputs,
                #     use_cache=False,
                #     **extra_args,
                # )  # prevent model thinks we are generating

                # if self.use_fused_kernels:
                #     log_probs = output.log_probs[:, -response_length - 1 : -1]
                #     entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                # else:
                #     logits = output.logits

                #     logits.div_(temperature)
                #     logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                #     log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                #     if calculate_entropy:
                #         entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                raise NotImplementedError("EarlActor does not support non-rmpad and non-ulysses sp yet.")
            if return_earl_ref:
                return entropy, log_probs, earl_ref_log_probs_full
            else:
                return entropy, log_probs