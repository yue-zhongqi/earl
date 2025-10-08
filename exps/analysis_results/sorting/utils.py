from collections import defaultdict
import statistics

def normalize_string(seq_str):
    return ",".join([s.strip() for s in seq_str.split(",")])

def min_swaps_to_sort(groundtruth):
    items = [x.strip() for x in groundtruth.split(",")]
    mapping = {letter: idx for idx, letter in enumerate(items)}
    arr=[mapping[k] for k in sorted(mapping)]
    
    n = len(arr)
    visited = [False] * n
    swap_count_min = 0

    arr_pos = list(enumerate(arr))
    arr_pos.sort(key=lambda x: x[1])  

    # Traverse array and count cycles
    for i in range(n):
        if visited[i] or arr_pos[i][0] == i:
            continue  

        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arr_pos[j][0]  
            cycle_size += 1

        if cycle_size > 0:
            swap_count_min += (cycle_size - 1)

    return swap_count_min


def aggregate_records(records):
    sort_types = set(record["sort-type"] for record in records)

    acc_by_level = {st: defaultdict(list) for st in sort_types}
    swap_by_level = {st: defaultdict(list) for st in sort_types}
    compare_by_level = {st: defaultdict(list) for st in sort_types}

    # Iterate over all records and collect metrics
    for record in records:
        sort_type = record["sort-type"]
        level = record["level"]

        success = record["result"] == record["ground_truth"]
        acc_by_level[sort_type][level].append(1 if success else 0)

        if success:
            swap_by_level[sort_type][level].append(record["swap_count"])
            compare_by_level[sort_type][level].append(record["compare_count"])

    # Aggregate statistics into a clean result structure
    result = []
    for sort_type in sort_types:
        result.append({
            "sort-type": sort_type,
            "acc": {
                f"level-{lvl}": round(statistics.mean(vals), 3)
                for lvl, vals in acc_by_level[sort_type].items()
            },
            "compare_count": {
                f"level-{lvl}": round(statistics.mean(vals), 3)
                for lvl, vals in compare_by_level[sort_type].items() if vals
            },
            "swap_count": {
                f"level-{lvl}": round(statistics.mean(vals), 3)
                for lvl, vals in swap_by_level[sort_type].items() if vals
            }
        })
    return result
