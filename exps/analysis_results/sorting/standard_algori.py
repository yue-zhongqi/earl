import json
import copy
import pandas as pd
from collections import defaultdict
import os

LOCAL_DIR = "./data/sort_earl_4_5/"
GLOBAL_MAPPING = None
GLOBAL_LOG=None

def set_mapping(mapping):
    global GLOBAL_MAPPING
    GLOBAL_MAPPING = mapping

def set_log(log):
    global GLOBAL_LOG
    GLOBAL_LOG = log

def apply_swap(pos_list):
    GLOBAL_LOG["swap_count"] += 1

    pos1, pos2 = pos_list
    seq = GLOBAL_LOG["current_seq"]
    sym1, sym2 = seq[pos1], seq[pos2]
    seq[pos1], seq[pos2] = sym2, sym1

    GLOBAL_LOG["compare_swap_usage"].append(f"swap {sym1} and {sym2} => {seq}")

def log_compare(sym1, sym2, val1, val2):
    GLOBAL_LOG["compare_count"] += 1

    relation = "<" if val1 < val2 else ">"
    
    GLOBAL_LOG["compare_swap_usage"].append(f"compare {sym1} and {sym2}: {sym1} {relation} {sym2}")
    return val1 < val2


def sort2(pos_list, reverse):
    pos1, pos2 = pos_list
    seq = GLOBAL_LOG["current_seq"]

    sym1, sym2 = seq[pos1], seq[pos2]
    val1, val2 = GLOBAL_MAPPING[sym1], GLOBAL_MAPPING[sym2]

    compare_result=log_compare(sym1, sym2, val1, val2)

    need_swap = (not reverse and not compare_result) or (reverse and compare_result)
    if need_swap:
        apply_swap(pos_list)
        

def partial_sort3(pos_list, reverse):
    pos1, pos2, pos3 = pos_list
    seq = GLOBAL_LOG["current_seq"]

    sym1, sym2, sym3 = seq[pos1], seq[pos2], seq[pos3]
    val1, val2, val3 = GLOBAL_MAPPING[sym1], GLOBAL_MAPPING[sym2], GLOBAL_MAPPING[sym3]

    if not reverse:  # ascending
        if log_compare(sym1, sym3, val1, val3):
            if not log_compare(sym1, sym2, val1, val2):
                apply_swap([pos1, pos2])
        else:
            apply_swap([pos1, pos2])
            apply_swap([pos2, pos3])
    else:  # descending
        if log_compare(sym3, sym1, val3, val1):
            if not log_compare(sym3, sym2, val3, val2):
                apply_swap([pos2, pos3])
        else:
            apply_swap([pos2, pos3])
            apply_swap([pos1, pos2])

def sort3(pos_list, reverse):
    comparators_asc = [
        ("pair", pos_list[1:3]),   # compare last two elements
        ("triple", pos_list)       # sort all three
    ]
    comparators_desc = [
        ("pair", pos_list[0:2]),   # compare first two elements
        ("triple", pos_list)       # sort all three
    ]

    # Select comparator sequence
    comparators = comparators_desc if reverse else comparators_asc

    # Execute comparators
    for ctype, pos in comparators:
        if ctype == "pair":
            sort2(pos, reverse)
        elif ctype == "triple":
            partial_sort3(pos, reverse)


def sort4(reverse=False):
    # Define the comparator sequence for 4-element sorting network
    if not reverse:  # ascending order
        pairs = [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]
    else:  # descending order
        pairs = [(1, 3), (0, 2), (2, 3), (0, 1), (1, 2)]

    # Apply each comparison-swap step
    for p in pairs:
        sort2(list(p), reverse)


def sort5(reverse):
    comparators_asc = [
        ("pair", [0, 1]),       # compare first two
        ("pair", [3, 4]),       # compare last two
        ("triple", [2, 3, 4]),  # sort right triple
        ("pair", [1, 4]),       # compare 2nd and last
        ("triple", [0, 2, 3]),  # sort left-middle triple
        ("triple", [1, 2, 3])   # refine middle triple
    ]

    comparators_desc = [
        ("pair", [3, 4]),       # compare last two
        ("pair", [0, 1]),       # compare first two
        ("triple", [0, 1, 2]),  # sort left triple
        ("pair", [0, 3]),       # compare 1st and 4th
        ("triple", [1, 2, 4]),  # sort triple including last
        ("triple", [1, 2, 3])   # refine middle triple
    ]

    # Select comparator list depending on order
    comparators = comparators_desc if reverse else comparators_asc

    # Execute comparators
    for ctype, pos in comparators:
        if ctype == "pair":
            sort2(pos, reverse)
        elif ctype == "triple":
            partial_sort3(pos, reverse)

def network_sort(symbols, reverse, mapping):
    set_mapping(mapping)
    set_log({
            "compare_count": 0,
            "swap_count": 0,
            "compare_swap_usage": [],
            "current_seq":copy.deepcopy(symbols)
        })
    
    n = len(symbols)
    if n == 2:
        sort2([0,1], reverse)
    elif n == 3:
        sort3([0,1,2], reverse)
    elif n == 4:
        sort4(reverse)
    elif n == 5:
        sort5(reverse)
    result_symbols = GLOBAL_LOG["current_seq"]


def analyze_sort_data(extracted_data, verbose=True):
    
    summary_by_sort_type = {}  # Stores general stats per sort-type
    ground_truth_stats_by_sort_type = {}  # Stores per-ground-truth stats within each sort-type

    # Step 1: Group entries by `sort-type`
    grouped = defaultdict(list)
    for entry in extracted_data:
        stype = entry.get("sort-type")
        grouped[stype].append(entry)

    # Step 2: Compute overall stats per sort-type
    for stype, group in grouped.items():
        total = len(group)
        compare_sum = sum(e["compare_count"] for e in group)
        swap_sum = sum(e["swap_count"] for e in group)

        summary_by_sort_type[stype] = {
            "count": total,
            "compare_count_mean": compare_sum / total,
            "swap_count_mean": swap_sum / total
        }

    # Step 3: For each sort-type, group by ground_truth and compute stats
    for stype, group in grouped.items():
        gt_stats = defaultdict(lambda: {
            "count": 0,
            "compare_count_total": 0,
            "swap_count_total": 0
        })

        for e in group:
            gt = e["ground_truth"]
            gt_stats[gt]["count"] += 1
            gt_stats[gt]["compare_count_total"] += e["compare_count"]
            gt_stats[gt]["swap_count_total"] += e["swap_count"]

        # Calculate mean for each ground_truth
        gt_stats_final = {}
        for gt, stats in gt_stats.items():
            count = stats["count"]
            gt_stats_final[gt] = {
                "count": count,
                "compare_count_mean": stats["compare_count_total"] / count,
                "swap_count_mean": stats["swap_count_total"] / count
            }

        ground_truth_stats_by_sort_type[stype] = gt_stats_final

    # Optional: print result summaries
    if verbose:
        print("\n=== Summary by sort-type ===")
        print(json.dumps(summary_by_sort_type, indent=4))

        print("\n=== Ground truth stats by sort-type ===")
        for stype, gt_stats in ground_truth_stats_by_sort_type.items():
            print(f"\n--- sort-type: {stype} ---")
            for gt, stats in gt_stats.items():
                print(f"{gt}: {stats}")

    return summary_by_sort_type, ground_truth_stats_by_sort_type



if __name__ == "__main__":
    count=0
    swap_count_total=0
    compare_count_total=0
    output_path = f'./val_results/analysis_results/sorting/standard_algori_result.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extracted_data = []

    df = pd.read_parquet(f'{LOCAL_DIR}/test.parquet')
    for _, data in df.iterrows():
        count+=1
        ground_truth= data.get("reward_model", {}).get("ground_truth", {})
        interaction = data.get("extra_info", {}).get("interaction_kwargs", {})
        line_num = data.get("extra_info", {}).get("index", 0)
        items = interaction.get("items", [])
        symbols = interaction.get("symbols", [])
        reverse = interaction.get("reverse", False)
        mapping=dict(zip(symbols, items))

        items = items.tolist() if hasattr(items, 'tolist') else items
        symbols = symbols.tolist() if hasattr(symbols, 'tolist') else symbols

        network_sort(symbols, reverse,mapping)
        result=GLOBAL_LOG["current_seq"]
        compare_count=GLOBAL_LOG["compare_count"]
        swap_count=GLOBAL_LOG["swap_count"]
        extracted_data.append({
            "line": line_num,
            "items": items,
            "reverse":reverse,
            "symbols": symbols,
            "ground_truth":ground_truth,
            "result": result,
            "compare_count":compare_count,
            "swap_count": swap_count,
            "compare_swap_usage": GLOBAL_LOG["compare_swap_usage"],
            "sort-type":len(symbols)
        })
        swap_count_total += swap_count
        compare_count_total += compare_count


    print("swap_mean:",swap_count_total/count)
    print("compare_mean:",compare_count_total/count)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in extracted_data:
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write('\n')
    # analyze_sort_data(extracted_data)
    print(f"Output saved to: {output_path}")