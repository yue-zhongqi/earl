from numpy import char
from .base import Tool
from . import register_tool
from copy import deepcopy
@register_tool('compare')
class Compare(Tool):
    def __init__(self, tool_config, tool_name):
        super().__init__(tool_config, tool_name)
        if self.is_earl:
            self.insert_ids = tool_config.tokenizer(" and")['input_ids']
        else:
            self.insert_ids = []

    def _execute(self, parsed_seq, exit_token, interaction_kwargs=None):
        if self.is_earl:
            exit_str = self.tool_config.id_to_str[exit_token].strip()
            chosen_symbols = [parsed_seq[0], exit_str]
        else:
            assert exit_token==None
            chosen_symbols = [parsed_seq[0], parsed_seq[-1]]

        undefined_symbols = [s for s in chosen_symbols if not s.strip() in interaction_kwargs['symbols']]
        if len(undefined_symbols) > 0:
            return f"Cannot compare due to undefined symbols {', '.join(undefined_symbols)}."
        sym_1 = chosen_symbols[0].strip()
        sym_2 = chosen_symbols[1].strip()
        pos_1 = interaction_kwargs['symbols'].index(sym_1)
        pos_2 = interaction_kwargs['symbols'].index(sym_2)
        val_1 = interaction_kwargs['items'][pos_1]
        val_2 = interaction_kwargs['items'][pos_2]
        if val_1 < val_2:
            return f"{sym_1} < {sym_2}"
        elif val_1 > val_2:
            return f"{sym_1} > {sym_2}"
        else:
            return f"{sym_1} = {sym_2}"

    def _forbid(self, parsed_seq, desc):
        # you can compare any two symbols
        return False
    
    def press(self, current_seq, token_id, interaction_kwargs=None):
        exiting = (len(current_seq) == 2)
        if token_id < self.tool_config.org_vocab_size:
            return [token_id], exiting
        else:
            ret = self.tool_config.id_to_seq[token_id].copy()
            if len(current_seq) != 1:
                return ret, exiting
            else:
                return ret + self.insert_ids, exiting
        
@register_tool('swap')
class Swap(Tool):
    def __init__(self, tool_config, tool_name):
        super().__init__(tool_config, tool_name)
        if self.is_earl:
            self.insert_ids = tool_config.tokenizer(" and")['input_ids']
        else:
            self.insert_ids = []

    @staticmethod
    def init_state(interaction_kwargs, action_seq, tool_config):
        interaction_kwargs = deepcopy(interaction_kwargs)
        if len(action_seq) == 0:
            interaction_kwargs['state'] = interaction_kwargs['symbols'].copy()
        else:
            if tool_config.id_to_str=={}:
                # get swap action id
                swap_action_id=tool_config.tool_name_to_action_id['swap']

                # swap_exit_action_id=tool_config.tool_name_to_action_id['swap']

                entry_swap_action_id=tool_config.id_to_seq[swap_action_id]
                exit_swap_action_id=tool_config.tool_action_id_to_exit_seq[swap_action_id]
                # initialize state
                state = interaction_kwargs['symbols'].copy()
                swap_params = []
                swapping = 0
                action_seq_len=len(action_seq)
                if action_seq_len>=9:
                    for i in range(action_seq_len):
                        if action_seq[i:i+3] == entry_swap_action_id and action_seq[i+6:i+9] == exit_swap_action_id:
                            for sym,sym_index in tool_config.tool_action_id_to_non_earl_actions[swap_action_id].items():
                                if sym_index==action_seq[i+3]:
                                    sym_1=sym
                                if sym_index==action_seq[i+5]:
                                    sym_2=sym
                            state = Swap._swap(state, sym_1, sym_2)
                            swap_params = []
                interaction_kwargs['state'] = state
            else:
                # get swap action id
                swap_action_id=tool_config.tool_name_to_action_id['swap']
                # initialize state
                state = interaction_kwargs['symbols'].copy()
                swap_params = []
                swapping = 0

                for action_id in action_seq:
                    if action_id == swap_action_id:
                        swapping = 2
                        continue
                    elif swapping > 0:
                        swap_params.append(tool_config.id_to_str[action_id].strip())
                        swapping -= 1
                    if len(swap_params) == 2:
                        sym_1 = swap_params[0]
                        sym_2 = swap_params[1]
                        state = Swap._swap(state, sym_1, sym_2)
                        swap_params = []
                interaction_kwargs['state'] = state
                

        return interaction_kwargs
    
    @staticmethod
    def _swap(state, sym_1, sym_2):
        if sym_1 != sym_2 and sym_1 in state and sym_2 in state:
            pos_1 = state.index(sym_1)
            pos_2 = state.index(sym_2)
            state[pos_1], state[pos_2] = state[pos_2], state[pos_1]
        return state

    def _execute(self, parsed_seq, exit_token, interaction_kwargs=None):
        if self.is_earl:
            exit_str = self.tool_config.id_to_str[exit_token].strip()
            chosen_symbols = [parsed_seq[0], exit_str]
        else:
            assert exit_token==None
            chosen_symbols = [parsed_seq[0], parsed_seq[-1]]
        
        sym_1 = chosen_symbols[0].strip()
        sym_2 = chosen_symbols[1].strip()
        interaction_kwargs['state'] = Swap._swap(interaction_kwargs['state'], sym_1, sym_2)
        assert len(set(interaction_kwargs['state'])) == len(interaction_kwargs['symbols'])
        return ", ".join(interaction_kwargs['state'])

    def _forbid(self, parsed_seq, desc):
        # cannot swap the same symbol
        if len(parsed_seq) == 0:
            return False
        else:
            return parsed_seq[-1].strip() == desc.strip()

    
    def press(self, current_seq, token_id, interaction_kwargs=None):
        exiting = (len(current_seq) == 2)
        if token_id < self.tool_config.org_vocab_size:
            return [token_id], exiting
        else:
            ret = self.tool_config.id_to_seq[token_id].copy()
            if len(current_seq) != 1:
                return ret, exiting
            else:
                return ret + self.insert_ids, exiting
        
@register_tool('check')
class Check(Tool):
    def __init__(self, tool_config, tool_name):
        super().__init__(tool_config, tool_name)

    def _execute(self, parsed_seq, exit_token, interaction_kwargs=None):
        undefined_symbols = not (parsed_seq[0].strip() in interaction_kwargs['symbols'])
        sym = parsed_seq[0].strip()
        pos = interaction_kwargs['symbols'].index(sym)
        val = interaction_kwargs['items'][pos]
        if undefined_symbols:
            return f"Cannot check an undefined symbol {parsed_seq[0].strip()}."
        if self.is_earl:
            exit_str = self.tool_config.id_to_str[exit_token].strip()
        else:
            assert exit_token==None
            exit_str = parsed_seq
        
        if 'negative' in exit_str:
            return str(val < 0)
        elif 'positive' in exit_str:
            return str(val > 0)
        elif 'even' in exit_str:
            return str(val % 2 == 0)
        elif 'odd' in exit_str:
            return str(val % 2 == 1)
        else:
            return str(f"Unexpected property {exit_str}.")

    def _forbid(self, parsed_seq, desc):
        if len(parsed_seq) == 0:
            # must be symbol
            return 'is' in desc.strip()
        else:
            # must be a property
            return 'is' not in desc.strip()
    
    def press(self, current_seq, token_id, interaction_kwargs=None):
        exiting = (len(current_seq) == 2)
        if token_id < self.tool_config.org_vocab_size:
            return [token_id], exiting
        else:
            ret = self.tool_config.id_to_seq[token_id].copy()
            return ret, exiting