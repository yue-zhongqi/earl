import torch
from abc import ABC, abstractmethod
from copy import deepcopy

class Tool(ABC):
    def __init__(self, tool_config, tool_name: str):
        self.tool_config = tool_config
        self.tool_name = tool_name
        self.action_id = tool_config.tool_name_to_action_id[tool_name]
        self.is_earl = self.action_id >= 0
        self.exit_seq = tool_config.tool_action_id_to_exit_str[self.action_id]
        self.tool_idx = tool_config.tool_action_id_to_idx[self.action_id]
        self.desc_list = [seq for seq in tool_config.tool_idx_to_id[self.tool_idx].keys()]
        if not self.is_earl:
            self.desc_map = tool_config.tool_action_id_to_non_earl_actions[self.action_id]

    def get_allowed_token_ids_mask_non_earl(self, output_seq):
        allowed = []
        parsed_seq = self._parse_non_earl(output_seq)
        for name, action_id in self.desc_map.items():
            if not self._forbid(parsed_seq, name.strip()):
                allowed.append(action_id)
        total_size = self.tool_config.org_vocab_size
        # breakpoint()
        mask = torch.zeros(total_size, dtype=torch.bool)
        mask[allowed] = True
        return mask

    def get_allowed_token_ids_mask(self, output_seq):
        total_size = self.tool_config.total_size + self.tool_config.org_vocab_size
        mask = torch.zeros(total_size, dtype=torch.bool)
        allowed_descs = self._get_allowed_desc(output_seq)
        allowed_ids = [self.tool_config.tool_idx_to_id[self.tool_idx][desc] for desc in allowed_descs]
        mask[allowed_ids] = True
        if len(allowed_ids) == 0:
            assert False, f"No allowed ids for tool {self.tool_name} with output_seq {output_seq}"
        return mask
    
    def execute(self, output_seq, exit_token=None, interaction_kwargs=None):
        # breakpoint()
        if len(output_seq) == 0:
            return " "
        if output_seq[0] in self.tool_config.tool_action_ids:
            # If the first token is a tool action, we skip it
            output_seq = output_seq[1:]
        parsed_seq = self._parse(output_seq) if self.is_earl else self._parse_non_earl(output_seq)
        output = self._execute(parsed_seq, exit_token, interaction_kwargs)
        return output

    @abstractmethod
    def _execute(self, parsed_seq, exit_token, interaction_kwargs=None):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _forbid(self, parsed_seq, desc):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def press(self, current_seq, token_id, interaction_kwargs=None):
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def init_state(interaction_kwargs, action_seq, tool_config):
        return deepcopy(interaction_kwargs)
    
    def _parse(self, output_seq):
        """
        Parse the output sequence ids to tool/action desc.
        """
        parsed_seq = []
        for token_id in output_seq:
            parsed_seq.append(self.tool_config.id_to_str[token_id])
        return parsed_seq
    
    def _parse_non_earl(self, output_seq):
        parsed_seq = []
        # breakpoint()
        for i, token_id in enumerate(output_seq):
            matched = False
            for name, action_id in self.desc_map.items():
                if token_id == action_id:
                    matched = True
                    if name.strip() == '':
                        continue  # skip empty desc
                    parsed_seq.append(name.strip())  # remove spaces around
                    break
            if not matched:
                # breakpoint()
                raise ValueError(f"Token ID {token_id} does not belong to non-EARL tool {self.tool_name}")
        return parsed_seq

    def _get_allowed_desc(self, output_seq):
        parsed_seq = self._parse(output_seq)
        parsed_seq = [seq.strip() for seq in parsed_seq]
        # breakpoint()
        return [desc for desc in self.desc_list if not self._forbid(parsed_seq, desc.strip())]

    