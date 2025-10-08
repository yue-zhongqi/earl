import re
import ast


def extract_solution(solution_str, extra_info):
    task = extra_info['task']
    # breakpoint()
    if task in ['arithmetic', 'gsm8kr', 'modulo', 'approximate', 'repeat', 'count','big_math']:
        # this also tests the formatting of the model
        # breakpoint()
        solutions = re.findall(r"####\s?(-?[0-9\.,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "").strip()
    elif task == 'countdown':
        # r"####\s*(.*)"
        match = re.search(r"####\s*(.*)", solution_str)
        if match:
            final_answer = match.group(1).strip()
        else:
            final_answer = None
    elif task == 'sudoku':
        match = re.search(r"####\s*(\[[\s\S]*\])", solution_str)
        if match:
            sudoku_json = match.group(1)
            final_answer = sudoku_json
        else:
            final_answer = None
    else:
        raise ValueError(f"Unknown task: {task}")
    return final_answer

def evaluate(answer, ground_truth, extra_info):
    task = extra_info['task']
    if task in ['arithmetic', 'gsm8kr', 'modulo', 'repeat', 'count','big_math']:
        correct = (float(answer) == float(ground_truth))
        return 1.0 if correct else 0.0
    elif task == 'approximate':
        diff = abs(float(answer) - float(ground_truth))
        score = 1. - diff
        return max(0.0, score)
    elif task == 'countdown':
        expr = extra_info['expr']
        used_numbers = list(map(int, re.findall(r'\d+', answer)))
        gt_numbers = list(map(int, re.findall(r'\d+', expr)))
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(gt_numbers):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, answer):
           return 0.0
        
        # Evaluate the equation with restricted globals and locals
        result = eval(answer, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(ground_truth)) < 1e-5:
            return 1.0
        else:
            return 0.0
    elif task == 'sudoku':
        answer_list = ast.literal_eval(answer)
        ground_truth_list = ast.literal_eval(ground_truth)
        correct = (answer_list == ground_truth_list)
        return 1.0 if correct else 0.0
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0, extra_info=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, extra_info=extra_info)
    if answer is None:
        return 0
    try:
        score = evaluate(answer, ground_truth, extra_info)
        # if correct:
        #     breakpoint()
        return score
    except Exception as e:
        return format_score
