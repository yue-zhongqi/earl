import re
import math

def reverse_condition(expr: str) -> str:
    # Extract operands and operator
    import re
    match = re.match(r'(\w)([><=])(\w)', expr.strip())
    if not match:
        raise ValueError("Invalid expression format")
    
    left, op, right = match.groups()
    
    # Mapping of operator flips
    flip = {'>': '<', '<': '>', '=': '='}
    
    # Build reversed expression
    return f"{right}{flip[op]}{left}"


def equation_reward_func(solution_str, target, extra_info):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        solution_str ([str]): Generated outputs
        target ([int]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    try:
        if extra_info['task'] == 'compare':
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", solution_str)
            if match is None:
                return False
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()

            equation = equation.replace(" ", "")
            target = target.replace(" ", "")

            # comparison
            gts = [target, reverse_condition(target)]
            return equation in gts
        elif extra_info['task'] == 'sort' or extra_info['task'] == 'order':
            return ','.join(extra_info['interaction_kwargs']['state']) == target
    except Exception:
        # If evaluation fails, reward is 0
        return False

def minimal_swaps_from_to(src, dst):
    # dst is sorted(src) with stable tie-breaking
    # build mapping from src indices to dst indices (handle duplicates stably)
    from collections import defaultdict, deque
    pos = defaultdict(deque)
    for i, v in enumerate(dst):
        pos[v].append(i)
    perm = [None]*len(src)
    for i, v in enumerate(src):
        perm[i] = pos[v].popleft()
    # count cycles in permutation 'perm'
    n = len(src); seen = [False]*n; cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return n - cycles   # S_min

def approx_log2_fact(n):
    # Stirling: log2(n!) ≈ n*log2(n) - n/ln(2) + 0.5*log2(2πn)
    if n <= 1: return 0.0
    return n*math.log2(n) - n/math.log(2) + 0.5*math.log2(2*math.pi*n)

def efficiency_reward_func(ground_truth, extra_info):
    '''
    Return efficiency reward in [0,1]. Return 0 if insufficeint comparison was made.
    '''
    # breakpoint()
    if extra_info['task'] == 'compare':
        return 1. if len(extra_info['interaction_kwargs']['tool_call_names']) == 1 else 0.
    elif extra_info['task'] == 'sort':
        # number of 'compare' in extra_info['interaction_kwargs']['tool_call_names']
        C = extra_info['interaction_kwargs']['tool_call_names'].count('compare')
        S = extra_info['interaction_kwargs']['tool_call_names'].count('swap')
        n = len(extra_info['interaction_kwargs']['symbols'])
        INV_MAX = n*(n-1)//2  # max inversions

        C_min = math.ceil(approx_log2_fact(n))
        C_ref = max(C_min, n*math.ceil(math.log2(max(1,n))))
        C_norm = (C - C_min) / max(1, C_ref - C_min)
        insuff_compare = (C_norm < 0)
        C_norm = max(0.0, min(1.0, C_norm))
        if insuff_compare:
            return 0.

        S_min = minimal_swaps_from_to(extra_info['interaction_kwargs']['symbols'], ground_truth.split(','))
        S_ref = max(S_min+1, INV_MAX)  # safe broad upper ref
        S_norm = (S - S_min) / max(1, S_ref - S_min)
        S_norm = max(0.0, min(1.0, S_norm))
        return 0.5 * (1. - C_norm) + 0.5 * (1. - S_norm)
    elif extra_info['task'] == 'order':
        n = len(extra_info['interaction_kwargs']['symbols'])
        S = extra_info['interaction_kwargs']['tool_call_names'].count('swap')
        INV_MAX = n*(n-1)//2  # max inversions
        S_min = minimal_swaps_from_to(extra_info['interaction_kwargs']['symbols'], ground_truth.split(','))
        S_ref = max(S_min+1, INV_MAX)  # safe broad upper ref
        S_norm = (S - S_min) / max(1, S_ref - S_min)
        S_norm = max(0.0, min(1.0, S_norm))
        return (1. - S_norm)
    else:
        raise NotImplementedError(f"Efficiency reward not implemented for task {extra_info['task']}")

def compute_score(solution_str, ground_truth, score=1.0, extra_info=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # breakpoint()
    correct = equation_reward_func(solution_str, ground_truth, extra_info)
    final_score = score if correct else 0.0
    if correct:
        # correct answer
        final_score += efficiency_reward_func(ground_truth, extra_info)
    # breakpoint()
    return final_score / 2.