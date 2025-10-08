import json
import pandas as pd
import copy
import re
import pickle
from utils import min_swaps_to_sort,aggregate_records,normalize_string
LOCAL_DIR = "./val_results/sorting/qwen2.5-7b/"

def judge_order(comparison, order_str):
    match = re.search(r"compare\s+(\w+)\s+and\s+(\w+):\s*(\w+)\s*([<>])\s*(\w+)", comparison)
    if not match:
        raise ValueError("Invalid comparison statement format")
    left, right, l2, op, r2 = match.groups()
    
    order = [x.strip() for x in order_str.split(",")]
    pos = {elem: i for i, elem in enumerate(order)}
    
    if op == "<":  # A < B
        if pos[l2] < pos[r2]:
            return "ascending"
        else:
            return "descending"
    else:  # A > B
        if pos[l2] < pos[r2]:
            return "descending"
        else:
            return "ascending"

# sort-4 / sort-5 / ...
def detect_sort_type(text: str) -> str:
    pattern = r' order:\s*([A-Z](?:,\s*[A-Z])*)\.?\s*While you do not know the values of the items'
    match = re.search(pattern, text)
    if match:
        items = match.group(1).split(',')
        count = len(items)
        return count,f'sort-{count}'
    return 0,'unknown'

def extract_compare_swap_ordered(text: str,data_analysis:dict,n:int):
    usage_list = []
    compare_pattern = r'<compare>\s*([A-Z]),\s*([A-Z])\s*</compare>\s*<result>\s*([A-Z])\s*([<>])\s*([A-Z])\s*</result>'
    swap_pattern = rf'<swap>\s*([A-Z]),\s*([A-Z])\s*</swap>\s*<result>\s*([A-Z](?:,\s*[A-Z]){{{n-1}}})\s*</result>'


    ground_truth = ""

    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()

        # compare
        compare_match = re.search(compare_pattern, line)
        if compare_match:
            usage_list.append(f"compare {compare_match.group(1)} and {compare_match.group(2)}: {compare_match.group(3)} {compare_match.group(4)} {compare_match.group(5)}")

        # swap
        swap_match = re.search(swap_pattern, line)
        if swap_match:
            usage_list.append(f"swap {swap_match.group(1)} and {swap_match.group(2)} => {swap_match.group(3)}")
            ground_truth = swap_match.group(3)

    compare_count = len([u for u in usage_list if u.startswith("compare")])
    swap_count = len([u for u in usage_list if u.startswith("swap")])

    data_analysis["usage_list"] = usage_list
    data_analysis["compare_count"] = compare_count
    data_analysis["swap_count"] = swap_count
    data_analysis["ground_truth"] = ground_truth

    if ground_truth=="":
        if n==4:ground_truth="A, B, C, D"
        else:ground_truth="A, B, C, D, E"
    if ground_truth not in data_analysis.keys():
        data_analysis[ground_truth]=[]
    data_analysis[ground_truth].append(usage_list)

    result_config={
        "level":min_swaps_to_sort(ground_truth),
        "compare_count":compare_count, 
        "swap_count":swap_count, 
        "compare_swap_usage":usage_list, 
        "result":ground_truth,
        "ground_truth":ground_truth,
    }

    return result_config

def convert_compare(sentence: str) -> str:
    """
    Convert a sentence like 'compare A and B: A < B' into 'if A < B'.
    """
    # Use regex to extract components
    match = re.search(r"compare\s+(\w+)\s+and\s+(\w+):\s*(\w+)\s*([<>]=?)\s*(\w+)", sentence)
    if not match:
        raise ValueError("Invalid input format. Expected format: 'compare A and B: A < B'")
    
    left, right, l2, op, r2 = match.groups()
    return f"if {l2} {op} {r2}"


def convert_swap(sentence: str) -> str:
    """
    Convert a sentence like 'swap A and B' into 'swap(A, B)'.
    """
    # Match the two characters after 'swap'
    match = re.search(r"swap\s+([A-Z])\s+and\s+([A-Z])", sentence)
    if not match:
        raise ValueError("Invalid input format. Expected format: 'swap X and Y'")
    
    a, b = match.groups()
    return f"swap({a}, {b})"


def convert_sentence(sentence):
    if "compare" in sentence:
        return(convert_compare(sentence))
    else:
        return(convert_swap(sentence))

def build_nested_log(sentences):
    grouped_log = {}
    order_tracker = []

    for sentence in sentences:
        if not sentence:
            continue

        key = convert_sentence(sentence.pop(0)) 

        if key not in order_tracker:
            order_tracker.append(key)

        if key not in grouped_log:
            grouped_log[key] = [sentence]
        else:
            grouped_log[key].append(sentence)

    for key in order_tracker:
        grouped_log[key] = build_nested_log(grouped_log[key])

    return grouped_log


if __name__ == "__main__":
    # Choose experiment mode here (only change this line)
    MODE = "prompt-cpo"   # or "prompt-grpo"
    
    input_path = f'{LOCAL_DIR}{MODE}-sort/36.jsonl'
    output_path = f'./val_results/analysis_results/sorting/{MODE}-sort_36_result.jsonl'
    data_analysis = {}
    record_total=[]
    swap_count_total = 0
    compare_count_total = 0
    
    df = pd.read_parquet('./data/sort_baseline_4_5/test.parquet')
    extracted_data=[]
    for _, data in df.iterrows():
        ground_truth= data.get("reward_model", {}).get("ground_truth", {})
        interaction = data.get("extra_info", {}).get("interaction_kwargs", {})
        line_num = data.get("extra_info", {}).get("index", 0)
        items = interaction.get("items", [])
        symbols = interaction.get("symbols", [])
        reverse = interaction.get("reverse", False)
        mapping = dict(zip(symbols,items))

        items = items.tolist() if hasattr(items, 'tolist') else items
        symbols = symbols.tolist() if hasattr(symbols, 'tolist') else symbols

        extracted_data.append({
            "line": line_num,
            "items": items,
            "reverse":reverse,
            "symbols": symbols,
            "ground_truth":ground_truth,
            "sort-type":f'sort-{len(symbols)}'
        })

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line,data in zip(infile,extracted_data):
            record = json.loads(line)
            input_text = record.get("input", "")
            output_text = record.get("output", "")
            score = record.get("score", None)
            step = record.get("step", None)
            reward = record.get("reward", None)
            n,sort_type = detect_sort_type(input_text)

            result_config = extract_compare_swap_ordered(output_text,data_analysis,n)
            
            acc = int(normalize_string(data["ground_truth"]) == normalize_string(result_config["result"]))
            new_record = {
                "method":MODE,
                "sort-type": data["sort-type"],
                "step": step,
                "reward": reward,
                "score": score,
                "level":result_config["level"],
                "compare_count": result_config["compare_count"],
                "swap_count": result_config["swap_count"],
                "compare_swap_usage": result_config["compare_swap_usage"],
                "ground_truth":normalize_string(data["ground_truth"]),
                "result":normalize_string(result_config["result"]),
                "acc":acc
            }
            record_total.append(new_record)
            swap_count_total += result_config["swap_count"]
            compare_count_total += result_config["compare_count"]
            outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
    print(f"Processing complete. Output saved to: {output_path}")
    print("swap_mean:",swap_count_total / 2000)
    print("compare_mean:",compare_count_total / 2000)
    result=aggregate_records(record_total)
    
    from pprint import pprint
    print("-------------------------")
    pprint(result)
