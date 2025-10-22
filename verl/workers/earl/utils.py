import torch
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import Dict, List
from verl.workers.earl.tools.base import Tool
from vllm.v1.sample.metadata import SamplingMetadata

@dataclass
class ToolConfig:
    """
    Configurations needed for VLLM sampler, logits processor, runner, scheduler.
    """
    id_to_seq: Dict = field(default_factory=dict)               # expanded id -> [original_ids]
    id_to_str: Dict = field(default_factory=dict)               # expanded id -> tokenizer.decode([original_ids])
    id_to_init_weight: Dict = field(default_factory=dict)        # expanded id -> init_weight_with
    seq_to_action_id: Dict = field(default_factory=dict)        # str([original ids]) -> [tool_action_ids]
    tool_idx_to_id: List = field(default_factory=list)          # [idx][action_desc] -> action_id, does not include enter tool action
    tool_action_ids: List = field(default_factory=list)
    tool_action_ids_relative: List = field(default_factory=list)  # tool action ids - org_vocab_size
    exit_action_ids: List = field(default_factory=list)
    tool_action_id_to_idx: Dict = field(default_factory=dict)
    tool_action_id_to_name: Dict = field(default_factory=dict)
    tool_action_id_to_non_earl_actions: Dict = field(default_factory=dict)  # for non-earl tools, map action id to the list of token ids to insert
    tool_action_id_to_exit_str: Dict = field(default_factory=dict)  # for exit action, map action id to the exit sequence
    tool_action_id_to_exit_seq: Dict = field(default_factory=dict)  # for exit action, map action id to the exit sequence token ids
    tool_action_id_to_cpo_seqs: Dict = field(default_factory=dict)  # for cpo, map tool action id to the list of token ids to insert
    tool_name_to_action_id: Dict = field(default_factory=dict)
    tool_name_to_output_template: Dict = field(default_factory=dict)  # map tool name to the output template
    total_size: int = 0
    org_vocab_size: int = 0
    org_tokenizer_size: int = 0
    starting_mode: str = 'default'  # default, tool, or exit
    tokenizer: object = None  # tokenizer
    is_sft_mode: str='default'


def build_tool_config(org_vocab_size: int, starting_mode: str, tool_config: DictConfig, tokenizer) -> ToolConfig:
    config = ToolConfig(
        org_vocab_size=org_vocab_size,
        starting_mode=starting_mode,
        org_tokenizer_size=len(tokenizer),
        tokenizer=tokenizer,
    )
    current_id = org_vocab_size
    tool_idx = 0
    non_earl_tool_idx = -1

    saved_action_groups = {}
    for tool in tool_config.tool_list:
        # CASE 1: non-earl tool, we only add the entry action and exit str
        if hasattr(tool, "earl") and tool.earl == False:
            seq = tokenizer(tool.entry_desc)['input_ids']
            cpo_ids = [tokenizer(words)['input_ids'] for words in tool.cpo_words]
            exit_str = tool.exit_desc if hasattr(tool, 'exit_desc') else ""
            exit_seq = tokenizer(exit_str)['input_ids'] if exit_str else []

            config.is_sft_mode=tool.is_sft_mode if hasattr(tool, 'is_sft_mode') else 'default'
            config.tool_name_to_action_id[tool.name] = non_earl_tool_idx
            config.tool_action_id_to_name[non_earl_tool_idx] = tool.name
            config.tool_action_id_to_idx[non_earl_tool_idx] = tool_idx
            config.tool_action_id_to_exit_seq[non_earl_tool_idx] = exit_seq
            config.tool_action_id_to_exit_str[non_earl_tool_idx] = exit_str
            config.tool_action_id_to_cpo_seqs[non_earl_tool_idx] = cpo_ids
            config.tool_action_id_to_non_earl_actions[non_earl_tool_idx] = {}
            for name in tool.actions:
                action_ids = tokenizer(name)['input_ids']
                # generalize to multi-token action ids?
                assert len(action_ids) == 1, "Non-EARL tool actions must be single token for now"
                action_id = action_ids[0]
                config.tool_action_id_to_non_earl_actions[non_earl_tool_idx][name] = action_id
            config.tool_name_to_output_template[tool.name] = tool.output_template
            config.id_to_seq[non_earl_tool_idx] = seq
            config.tool_idx_to_id.append({})    # no tool-specific actions, as action space is not expanded
            tool_idx += 1
            non_earl_tool_idx -= 1
            continue

        # CASE 2: earl tool
        seq = tokenizer(tool.entry_desc)['input_ids']
        cpo_ids = [tokenizer(words)['input_ids'] for words in tool.cpo_words]
        exit_str = ""   # earl tools do not have exit str, use output_template instead
        assert str(seq) not in config.seq_to_action_id, \
            f"Duplicate entry_desc found: {tool.entry_desc}"
        # assert len(seq) > 1, f"Entry desc must be longer than 1 token: {tool.entry_desc}"
        config.id_to_seq[current_id] = seq
        config.id_to_str[current_id] = tool.entry_desc
        config.id_to_init_weight[current_id] = tool.get('init_weight_with', 0)
        config.tool_idx_to_id.append({})
        # tool action specific config
        config.seq_to_action_id[str(seq)] = [current_id]
        config.tool_action_ids.append(current_id)
        config.tool_action_ids_relative.append(current_id - org_vocab_size)
        config.tool_action_id_to_idx[current_id] = tool_idx
        config.tool_action_id_to_name[current_id] = tool.name
        config.tool_action_id_to_cpo_seqs[current_id] = cpo_ids
        config.tool_action_id_to_exit_str[current_id] = exit_str
        config.tool_action_id_to_exit_seq[current_id] = []
        config.tool_name_to_action_id[tool.name] = current_id
        config.tool_name_to_output_template[tool.name] = tool.output_template
        current_id += 1
        
        for action_group in tool.action_groups:
            name = action_group.name
            if name not in saved_action_groups:
                saved_action_groups[name] = action_group.actions
                for action in action_group.actions:
                    desc = tokenizer(action.desc)['input_ids']
                    config.id_to_seq[current_id] = desc
                    config.id_to_str[current_id] = action.desc
                    config.id_to_init_weight[current_id] = action.get('init_weight_with', 0)
                    action.id = current_id
                    if 'exit' in name:
                        config.exit_action_ids.append(current_id)
                    current_id += 1
            actions = saved_action_groups[name]
            for action in actions:
                assert action.desc not in config.tool_idx_to_id[tool_idx], \
                        f"Duplicate action desc found: {action.desc}"
                config.tool_idx_to_id[tool_idx][action.desc] = action.id
            # end of action loop
        tool_idx += 1
        # end of tool loop
    config.total_size = current_id - org_vocab_size
    return config

def get_expansion_size(earl_config):
    """
    Get the size of additional action space.
    
    Args:
        earl_config (dict)
        
    Returns:
        int: Size of the expanded action space.
    """
    n = 0
    if earl_config is None or not hasattr(earl_config, 'tool_list'):
        return n
    
    for tool in earl_config.tool_list:
        n += 1
        if not hasattr(tool, 'action_groups'):
            continue
        for action_group in tool.action_groups:
            n += len(action_group.actions)
    return n

def get_output_seq_from_token_id(tool, current_seq, token_id: int, interaction_kwargs) -> list[int]:
    '''
    get description ids from pressing earl actions; note that current_seq SHOULD contain the tool action id
    '''
    return tool.press(current_seq, token_id, interaction_kwargs)

def _find_last_tool_action_index(output_seq, ids):
    for i, token_id in enumerate(reversed(output_seq)):
        if token_id in ids:
            return len(output_seq) - 1 - i
    return -1

def find_last_sublist(lst, sublist):
    sublist_len = len(sublist)
    for i in range(len(lst) - sublist_len, -1, -1):  # Iterate in reverse
        if lst[i:i + sublist_len] == sublist:
            return i
    return -1

def find_active_earl_tool_name(config: ToolConfig, output_seq):
    output_name = None
    last_tool_call_pos = _find_last_tool_action_index(output_seq, config.tool_action_ids)
    if last_tool_call_pos >= 0:
        output_name = config.tool_action_id_to_name[output_seq[last_tool_call_pos]]
    return output_name, last_tool_call_pos

def find_active_non_earl_tool_name(config: ToolConfig, output_seq):
    output_name = None
    last_tool_call_pos = -1
    for name, action_id in config.tool_name_to_action_id.items():
        if action_id < 0:
            # non-earl tool
            entry_seq = config.id_to_seq[action_id]
            exit_seq = config.tool_action_id_to_exit_seq[action_id]
            # note that model cannot spuriously gen entry without going to tool mode,
            # so this simple approach works
            last_entry = find_last_sublist(output_seq, entry_seq)
            if last_entry > find_last_sublist(output_seq, exit_seq):
                output_name = name
                last_tool_call_pos = last_entry + len(entry_seq)
                break
    return output_name, last_tool_call_pos

def find_completed_non_earl_tool_name(config: ToolConfig, output_seq):
    output_name = None
    tool_call_seq = None
    for name, action_id in config.tool_name_to_action_id.items():
        if action_id < 0:
            # non-earl tool
            entry_seq = config.id_to_seq[action_id]
            exit_seq = config.tool_action_id_to_exit_seq[action_id]
            last_exit = find_last_sublist(output_seq, exit_seq)
            last_last_exit = find_last_sublist(output_seq[:-len(exit_seq)], exit_seq)
            # model may generate exit str by itself without an entry str
            # e.g. <calc> </calc> <result> </result> *</calc>*, where inside * is spuriously generated by model
            # searching entry between last and second last exit covers this corner case
            if last_last_exit < 0:
                last_entry = find_last_sublist(output_seq, entry_seq)
            else:
                last_entry = find_last_sublist(output_seq[last_last_exit:], entry_seq)
            if last_entry >= 0 and last_exit >= 0 and last_entry < last_exit and (last_exit + len(exit_seq) == len(output_seq)):
                last_entry += (last_last_exit if last_last_exit >=0 else 0)
                output_name = name
                tool_call_seq = output_seq[last_entry+len(entry_seq):last_exit]
                for aid in tool_call_seq:
                    if aid not in config.tool_action_id_to_non_earl_actions[action_id].values():
                        raise ValueError(f"Token ID {aid} in non-EARL tool call does not belong to tool {name}")
                break
    return output_name, tool_call_seq

def get_allowed_token_ids_mask(
        tool_config: ToolConfig,
        tools,
        sampling_metadata: SamplingMetadata,
        starting_mode: str = 'default',
    ) -> torch.Tensor:
    """
    Get the mask for allowed token IDs based on the sampling metadata and EARL configuration.
    
    Args:
        sampling_metadata (SamplingMetadata): Metadata containing allowed token IDs.
        org_vocab_size (int): Original vocabulary size.
        earl_config (Optional[DictConfig]): Configuration for EARL, if available.
        
    Returns:
        torch.Tensor: Mask for allowed token IDs.
    """
    masks = []
    total_size = tool_config.total_size + tool_config.org_vocab_size
    for output_seq in sampling_metadata.output_token_ids:
        # handle starting mode
        if starting_mode != 'default' and len(output_seq) == 0:
            # If starting mode is not default, we limit first action to the tool action ids
            mask = torch.zeros(total_size, dtype=torch.bool)
            action_id = tool_config.tool_name_to_action_id[starting_mode]
            if action_id >= 0:
                # earl tool
                mask[tool_config.tool_name_to_action_id[starting_mode]] = True
            else:
                # non-earl tool
                entry_seq = tool_config.id_to_seq[action_id]
                mask[entry_seq[0]] = True
            masks.append(mask)
            continue
        if starting_mode != 'default':
            # check for non-earl tool mode
            action_id = tool_config.tool_name_to_action_id[starting_mode]
            entry_seq = tool_config.id_to_seq[action_id]
            if action_id < 0 and len(output_seq) < len(entry_seq):
                mask = torch.zeros(total_size, dtype=torch.bool)
                mask[entry_seq[len(output_seq)]] = True
                masks.append(mask)
                continue
        
        # if inside earl tool mode, cut output_seq from last tool call position
        if len(output_seq) == 0 or output_seq[-1] not in tool_config.id_to_seq.keys():
            # default mode
            tool = None
        else:
            # find EARL tool name
            tool_name, last_tool_index = find_active_earl_tool_name(tool_config, output_seq)
            tool = tools[tool_name]
            output_seq = output_seq[last_tool_index + 1:]   # Remove the tool action and everything before it
        
        # check non-EARL tools
        if tool is None:
            tool_name, last_tool_index = find_active_non_earl_tool_name(tool_config, output_seq)
            if tool_name is not None:
                # non-earl tool mode
                action_id = tool_config.tool_name_to_action_id[tool_name]
                exit_seq = tool_config.tool_action_id_to_exit_seq[action_id]
                output_seq = output_seq[last_tool_index:] # non-EARL tool action ids are already removed
                # if len(output_seq) > 0 and output_seq[-1] in exit_seq:
                if exit_seq[0] in output_seq:
                    # find position in exit seq
                    pos = exit_seq.index(output_seq[-1])
                    # only allow next token in exit seq
                    mask = torch.zeros(total_size, dtype=torch.bool)
                    mask[exit_seq[pos + 1]] = True
                else:
                    tool = tools[tool_name]
                    mask = tool.get_allowed_token_ids_mask_non_earl(output_seq)
                    mask[exit_seq[0]] = True  # always allow exit action
                masks.append(mask)
            else:
                # default mode
                mask = torch.zeros(total_size, dtype=torch.bool)
                mask[:tool_config.org_tokenizer_size] = True    # allow tokenizer vocab
                mask[tool_config.tool_action_ids] = True
                masks.append(mask)
        else:
            # earl tool mode
            masks.append(tool.get_allowed_token_ids_mask(output_seq))
    return torch.stack(masks, dim=0)

def decode_earl_response(response, tool_config: ToolConfig, tokenizer):
    org_vocab_size = tool_config.org_vocab_size
    decoded = []
    skip = 0
    for idx in response:
        idx = int(idx)
        if skip > 0:
            skip -= 1
            continue
        if idx < org_vocab_size:
            decoded.append(idx)
        else:
            seq = tool_config.id_to_seq[idx]
            decoded += seq
            skip = len(seq) - 1
    return tokenizer.decode(decoded)

def earl_action_mask_to_list(tool_config, action_mask: torch.Tensor) -> List[int]:
    action_list = []
    for i in range(len(action_mask)):
        if action_mask[i]:
            action_id = tool_config.org_vocab_size + i
            action_list.append(tool_config.id_to_str[action_id])
    return action_list

def print_rollout_results(tokenizer, tool_config, batch, idx, file_name):
    data = batch[idx]
    response_length = len(data['responses'])
    prompt_length = len(data['input_ids']) - response_length
    with open(file_name, "w") as f:
        f.write(f"Batch Index {idx}, starting mode {tool_config.starting_mode}:\n")
        f.write("Prompts: " + tokenizer.decode(data['input_ids'][:prompt_length]) + "\n")
        f.write(f"Responses: {tokenizer.decode(data['responses'])}\n")
        f.write(f"Earl response: {decode_earl_response(data['response_earl'], tool_config, tokenizer)}\n")
        f.write(f"Response breakdown:\n")
        for i in range(response_length):
            f.write(f"'{tokenizer.decode(data['responses'][:i+1])}': seq_mask-{data['seq_mask'][i]}, tool_mask-{data['tool_mask'][i]}, action_mask-{earl_action_mask_to_list(tool_config, data['earl_action_mask'][i])}\n")
        f.write("\n")