import json
import pandas as pd
import copy
import re
from earl_sort_algori import earl_sort4
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

    compare_pattern = r'compare\s+[A-Z]\s+and\s+[A-Z]:\s+[A-Z]\s*[<>]=?\s*[A-Z]'
    swap_pattern = r'swap\s+[A-Z]\s+and\s+[A-Z]\s*=>.*'

    ground_truth=""
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if re.search(compare_pattern, line):
            match = re.search(compare_pattern, line)
            usage_list.append(match.group(0))
        elif re.search(swap_pattern, line):
            match = re.search(swap_pattern, line)
            usage_list.append(match.group(0))
            ground_truth=match.group(0).split("=> ")[1]
    
    compare_count = len([u for u in usage_list if u.startswith("compare")])
    swap_count = len([u for u in usage_list if u.startswith("swap")])

    if ground_truth=="":
        if n==4:ground_truth="A, B, C, D"
        else:ground_truth="A, B, C, D, E"
    if ground_truth not in data_analysis.keys():
        data_analysis[ground_truth]=[]
    data_analysis[ground_truth].append(usage_list)

    result_config={
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

def analyze_and_display_structure(analysis_summary, order_type_label):
    print(f"\n{order_type_label.capitalize()} order:")
    summary_copy = copy.deepcopy(analysis_summary[order_type_label])
    content_sequences = [entry["content"] for entry in summary_copy]
    nested_structure = build_nested_log(content_sequences)
    from pprint import pprint
    pprint(nested_structure)

if __name__ == "__main__":
    input_path = f'{LOCAL_DIR}earl-cpo-sort/42.jsonl'
    output_path = f'./val_results/analysis_results/sorting/earl-cpo-sort_42_result.jsonl'
    data_analysis = {}
    record_total=[]
    swap_count_total = 0
    compare_count_total = 0
    count_line=0


    df = pd.read_parquet('./data/sort_earl_4_5/test.parquet')
    extracted_data=[]
    for _, data in df.iterrows():
        ground_truth= data.get("reward_model", {}).get("ground_truth", {})
        interaction = data.get("extra_info", {}).get("interaction_kwargs", {})
        line_num = data.get("extra_info", {}).get("index", 0)
        items = interaction.get("items", [])
        symbols = interaction.get("symbols", [])
        reverse = interaction.get("reverse", None)
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
            count_line+=1
            record = json.loads(line)
            input_text = record.get("input", "")
            output_text = record.get("output", "")
            score = record.get("score", None)
            step = record.get("step", None)
            reward = record.get("reward", None)
            n,sort_type = detect_sort_type(input_text)
            reverse=data["reverse"]
            
            if n==4:
                result_config=earl_sort4(data)
            else:
                result_config = extract_compare_swap_ordered(output_text,data_analysis,n)

            compare_count=result_config["compare_count"]
            swap_count=result_config["swap_count"]
            new_record = {
                "method":"earl-cpo",
                "sort-type": sort_type,
                "step": step,
                "reward": reward,
                "score": score,
                "level":min_swaps_to_sort(result_config["ground_truth"]),
                "compare_count": compare_count,
                "swap_count": swap_count,
                "compare_swap_usage": result_config["compare_swap_usage"],
                "ground_truth":normalize_string(data["ground_truth"]),
                "result":normalize_string(result_config["result"])
            }
            record_total.append(new_record)
            swap_count_total += swap_count
            compare_count_total += compare_count
            outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
    print(f"Processing complete. Output saved to: {output_path}")

    print("swap_mean:",swap_count_total / count_line)
    print("compare_mean:",compare_count_total / count_line)
    result=aggregate_records(record_total)
    
    from pprint import pprint
    print("-------------------------")
    pprint(result)
    
    # analysis_summary_by_groundtruth = {
    #     'ascending': [],
    #     'descending': []
    # }
    # for result_sort, traj in data_analysis.items():
    #     s_traj = pd.Series(traj)
    #     counts = s_traj.value_counts()
    #     for key, value in counts.items():
    #         order_type = judge_order(key[0], result_sort)
    #         analysis_summary_by_groundtruth[order_type].append(
    #             {
    #                 "result_sort": result_sort,
    #                 "content": key,
    #                 "count": value
    #             })
    # analyze_and_display_structure(analysis_summary_by_groundtruth, "ascending")
    # analyze_and_display_structure(analysis_summary_by_groundtruth, "descending")
    # print(record_total[0])
