from numpy import char
from .base import Tool
from . import register_tool


@register_tool('calculator')
class Calculator(Tool):
    def _execute(self, parsed_seq, exit_token, interaction_kwargs=None):
        if exit_token is None:
            exit_symbol = '='
        else:
            exit_symbol = self.tool_config.id_to_str[exit_token]
        if exit_symbol.strip() == '=':
            # calculate from the string
            expression = ''.join(parsed_seq)
            try:
                result = eval(expression)
                if result == int(result):
                    result = str(int(result))
                else:
                    result = f"{result:.3f}"
            except Exception as e:
                result = f"Error: {e}"
            return result
        else:
            raise ValueError(f"Unexpected exit token: {exit_symbol}. Expected '=' for calculation.")

    def _forbid(self, parsed_seq, desc):
        def _is_digit_or_left_paren(char):
            return (char.isdigit()) or char == '('

        def _is_digit_or_right_paren(char):
            return (char.isdigit()) or char == ')'
        
        if len(parsed_seq) == 0:
            return (desc == '0') or (not _is_digit_or_left_paren(desc))
        
        match desc:
            case '+':
                return not _is_digit_or_right_paren(parsed_seq[-1])
            case '-':
                return not _is_digit_or_right_paren(parsed_seq[-1])
            case '*':
                return not _is_digit_or_right_paren(parsed_seq[-1])
            case '/':
                return not _is_digit_or_right_paren(parsed_seq[-1])
            case '0':
                return parsed_seq[-1] in '+-*/()'  # Forbid '0' if after '/'
            case '=':
                has_unclosed_paren = parsed_seq.count('(') > parsed_seq.count(')')
                return has_unclosed_paren or (not _is_digit_or_right_paren(parsed_seq[-1]))
            case '(':
                return _is_digit_or_right_paren(parsed_seq[-1]) or parsed_seq[-1] == '.'  # Forbid '(' if after a digit or ')' or '.'
            case ')':
                has_unclosed_paren = parsed_seq.count('(') > parsed_seq.count(')')
                return not has_unclosed_paren or parsed_seq[-1] in '(+-*/'
            case '.':
                return (not parsed_seq[-1].isdigit())
        
        if desc.isdigit():
            return parsed_seq[-1] == ')'  # Forbid digit if after ')'
        
        # else allow all
        return False
    
    def press(self, current_seq, token_id, interaction_kwargs=None):
        exiting = token_id in self.tool_config.exit_action_ids
        if token_id < self.tool_config.org_vocab_size:
            return [token_id], exiting
        else:
            return self.tool_config.id_to_seq[token_id].copy(), exiting