import json
import pandas as pd
import copy
import re
LOCAL_DIR = "./val_results/0_autorunner/calc_bench/"

def extract_numbers_and_target(text):
    numbers_match = re.search(r"\[([0-9,\s\-]+)\]", text)
    target_match = re.search(r"equals\s*(-?\d+)", text)
    
    if numbers_match and target_match:
        numbers = [int(n.strip()) for n in numbers_match.group(1).split(",")]
        target = int(target_match.group(1))
        
        return numbers, target
    else:
        return None, None


def tool_use(text, valid_numbers):
    pattern = r"calculate\s+([0-9.\-+*/()\s]+)\s*=\s*(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text)

    tool_use_count = 0
    invalid_usage_count = 0
    details = []
    log_calc_result=[]
    redundancy=0
    for expr, calc_result in matches:
        tool_use_count += 1

        nums = re.findall(r"\d+(?:\.\d+)?", expr)
        nums = [float(n) if "." in n else int(n) for n in nums]

        invalid_nums = [n for n in nums if n not in valid_numbers]
        if invalid_nums:
            invalid_usage_count += 1

        details.append({
            "equation": f"calculate {expr.strip()} = {calc_result}",
            "numbers": nums,
            "invalid_numbers": invalid_nums
        })
        if calc_result not in log_calc_result:
            log_calc_result.append(calc_result)
        else:
            redundancy+=1
    result = {
        "valid_numbers": valid_numbers,
        "tool_use_count": tool_use_count,
        "invalid_usage_count": invalid_usage_count,
        "details": details,
    }

    return result

def tool_use_xml_mode(text, valid_numbers):
    pattern = r"<calculator>\s*([\d+\-*/()\s]+)\s*</calculator>\s*<result>\s*(-?\d+(?:\.\d+)?)\s*</result>"
    matches = re.findall(pattern, text)

    tool_use_count = 0
    invalid_usage_count = 0
    details = []
    log_calc_result=[]
    redundancy=0
    for expr, calc_result in matches:
        tool_use_count += 1

        nums = re.findall(r"\d+(?:\.\d+)?", expr)
        nums = [float(n) if "." in n else int(n) for n in nums]

        invalid_nums = [n for n in nums if n not in valid_numbers]
        if invalid_nums:
            invalid_usage_count += 1

        details.append({
            "equation": f"calculate {expr.strip()} = {calc_result}",
            "numbers": nums,
            "invalid_numbers": invalid_nums
        })
        if calc_result not in log_calc_result:
            log_calc_result.append(calc_result)
        else:
            redundancy+=1
    result = {
        "valid_numbers": valid_numbers,
        "tool_use_count": tool_use_count,
        "invalid_usage_count": invalid_usage_count,
        "details": details,
    }
    return result

def analyze_run(name, input_path, output_path, parser, 
                ref_line_list=None, reward_filter=None):
    """
    Analyze tool usage for a given method.
    
    Args:
        name (str): Method name (e.g. "earl_cpo").
        input_path (str): Input .jsonl file path.
        output_path (str): Output .jsonl file path.
        parser (callable): Function to parse tool use (e.g. tool_use or tool_use_xml_mode).
        ref_line_list (list[int] | None): Optional line number filter.
        reward_filter (int | None): If set, skip lines with reward == reward_filter.
    """
    count_line = 0
    count_valid = 0
    tool_use_count_total = 0
    hallucination_total = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            count_line += 1

            if ref_line_list and count_line not in ref_line_list:
                continue

            record = json.loads(line)
            input_text = record.get("input", "")
            output_text = record.get("output", "")
            score = record.get("score", None)
            reward = record.get("reward", None)

            # skip if reward matches filter
            if reward_filter is not None and reward == reward_filter:
                continue

            count_valid += 1
            valid_numbers, target = extract_numbers_and_target(input_text)
            result = parser(output_text, valid_numbers)

            new_record = {
                "line_num": count_line,
                "result": result,
                "score": score,
                "target": target
            }
            tool_use_count_total += result["tool_use_count"]
            hallucination_total += result["invalid_usage_count"]

            outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    print(f"------- {name} -------")
    print(f"Processing complete. Output saved to: {output_path}")
    n = count_valid if count_valid > 0 else count_line
    print("tool_use_count mean:", tool_use_count_total / n)
    print("hallucination mean:", hallucination_total / n)

if __name__ == "__main__":
    # earl-cpo
    ref_line_list = []
    input_path = f'{LOCAL_DIR}earl-cpo/275/calc_bench/countdown/0.jsonl'
    output_path = f'{LOCAL_DIR}earl-cpo/275/calc_bench/countdown/0_analysis.jsonl'
    analyze_run("earl_cpo", input_path, output_path, parser=tool_use, reward_filter=0)

    # collect ref_line_list
    with open(output_path, 'r', encoding='utf-8') as f:
        ref_line_list = [json.loads(line).get("line_num") for line in f]

    # prompt-cpo
    input_path = f'{LOCAL_DIR}prompt-cpo/100/calc_bench/countdown/0.jsonl'
    output_path = f'{LOCAL_DIR}prompt-cpo/100/calc_bench/countdown/0_analysis.jsonl'
    analyze_run("prompt_cpo", input_path, output_path, parser=tool_use_xml_mode,
                ref_line_list=ref_line_list, reward_filter=1)

    # prompt-grpo
    input_path = f'{LOCAL_DIR}prompt-grpo/250/calc_bench/countdown/0.jsonl'
    output_path = f'{LOCAL_DIR}prompt-grpo/250/calc_bench/countdown/0_analysis.jsonl'
    analyze_run("prompt_grpo", input_path, output_path, parser=tool_use_xml_mode,
                ref_line_list=ref_line_list, reward_filter=1)

    # sft-grpo
    input_path = f'{LOCAL_DIR}sft-grpo/240/calc_bench/countdown/0.jsonl'
    output_path = f'{LOCAL_DIR}sft-grpo/240/calc_bench/countdown/0_analysis.jsonl'
    analyze_run("sft_grpo", input_path, output_path, parser=tool_use_xml_mode)

    