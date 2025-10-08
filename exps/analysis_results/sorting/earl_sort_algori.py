import json
import copy
GLOBAL_MAPPING = None
GLOBAL_LOG=None

def set_mapping(mapping):
    global GLOBAL_MAPPING
    GLOBAL_MAPPING = mapping

def set_log(log):
    global GLOBAL_LOG
    GLOBAL_LOG = log


def apply_swap(symbol_list):
    GLOBAL_LOG["swap_count"] += 1

    sym1, sym2 = symbol_list
    seq = GLOBAL_LOG["current_seq"]

    idx1, idx2 = seq.index(sym1), seq.index(sym2)

    seq[idx1], seq[idx2] = sym2, sym1

    GLOBAL_LOG["compare_swap_usage"].append(
        f"swap {sym1} and {sym2} => {seq}"
    )

def cmp(sym1, sym2,op="<"):
    GLOBAL_LOG["compare_count"] += 1
    v1, v2 = GLOBAL_MAPPING[sym1], GLOBAL_MAPPING[sym2]
    if op == "<":
        return v1 < v2
    else:
        return v1 > v2


def sort4(reverse):
    if not reverse:
        if cmp("A", "B", "<"):
            if cmp("A", "C", "<"):
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                None
                            else:
                                apply_swap(["D","C"])
                        else:
                            if cmp("C", "D", ">"):
                                apply_swap(["B","D"])
                                apply_swap(["B","C"])
                    else:
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["B","C"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["B","C"])
                                apply_swap(["B","D"])
                            else:
                                apply_swap(["B","D"])
                else:
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", ">"):
                            apply_swap(["A","D"])
                            apply_swap(["A","B"])
                            apply_swap(["C","B"])
                    else:
                        if cmp("B", "D", ">"):
                            apply_swap(["A","D"])
                            apply_swap(["A","B"])
            else:
                if cmp("B", "C", ">"):
                    if cmp("A", "D", "<"):
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["A","B"])
                        else:
                            apply_swap(["A","C"])
                            apply_swap(["A","B"])
                            apply_swap(["B","D"])
                    else:
                        if cmp("B", "D", ">"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","C"])
                                apply_swap(["B",'D'])
                            else:
                                apply_swap(["A","D"])
                                apply_swap(["B","C"])
                                apply_swap(["B","A"])
        else:
            if cmp("A", "C", "<"):
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                            else:
                                apply_swap(["A","B"])
                                apply_swap(["C","D"])
                else:
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            apply_swap(["A","B"])
                            apply_swap(["A","D"])
                            apply_swap(["A","C"])
                        else:
                            apply_swap(["A","D"])
                            apply_swap(["A","C"])
            else:
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                    else:
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","C"])
                else:
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                                apply_swap(["A","D"])
                            else:
                                apply_swap(["A","B"])
                                apply_swap(["A","D"])
                        else:
                            if cmp("C", "D", ">"):
                                apply_swap(["A","D"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["A","D"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["A","C"])
                                apply_swap(["B",'D'])
                                apply_swap(["A","B"])
                            else:
                                apply_swap(["A","D"])
                                apply_swap(["B","C"])

    else:
        if cmp("A", "B", "<"):
            if cmp("B", "C", "<"):
                if cmp("C", "D", "<"):
                    if cmp("A", "B", "<"):
                        if cmp("A", "D", "<"):
                            apply_swap(["A","D"])
                            apply_swap(["B","C"])
                    
                else:
                    if cmp("A", "B", "<"):
                        if cmp("A", "D", "<"):
                            if cmp("B", "D", "<"):
                                apply_swap(["A","C"])
                                apply_swap(["B","D"])
                                apply_swap(["A","B"])
                            else:
                                apply_swap(["A","C"])
                                apply_swap(["A","D"])
                        else:
                            if cmp("B", "D", "<"):
                                apply_swap(["A","C"])
            else:
                if cmp("B", "D", "<"):
                    if cmp("A", "C", "<"):
                        if cmp("A", "D", "<"):
                            apply_swap(["A","D"])
                    else:
                        if cmp("A", "D", "<"):
                            apply_swap(["A","D"])
                            apply_swap(["A","C"])
                else:
                    if cmp("A", "C", "<"):
                        if cmp("C", "D", "<"):
                            if cmp("A", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","D"])
                        else:
                            if cmp("A", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                                apply_swap(["A","D"])
                            else:
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                    else:
                        if cmp("A", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","D"])
                                apply_swap(["A","C"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["C","D"])
                            else:
                                apply_swap(["A","B"])
        else:
            if cmp("A", "C", "<"):
                if cmp("C", "D", "<"):
                    if cmp("A", "B", ">"):
                        if cmp("A", "D", "<"):
                            apply_swap(["A","D"])
                            apply_swap(["B","C"])
                            apply_swap(["A","B"])
                else:
                    if cmp("A", "D", "<"):
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["B","D"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["A","B"])
                            apply_swap(["B","D"])
                        else:
                            apply_swap(["A","C"])
                            apply_swap(["A","B"])
            else:
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            apply_swap(["A","D"])
                            apply_swap(["A","B"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["A","D"])
                            apply_swap(["A","B"])
                            apply_swap(["B","C"])
                else:
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["B","D"])
                            else:
                                apply_swap(["B","C"])
                                apply_swap(["B","D"])
                        else:
                            apply_swap(["B","C"])
                    else:
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["B","D"])
                                apply_swap(["B","C"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["C","D"])
                            else:
                                None




def optimal_sort4(reverse):
    if not reverse:
        if cmp("A", "B", "<"):
            if cmp("A", "C", "<"):
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                None
                            else:
                                apply_swap(["D","C"])
                        else:
                            apply_swap(["B","D"])
                            apply_swap(["B","C"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["B","C"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["B","C"])
                                apply_swap(["B","D"])
                            else:
                                apply_swap(["B","D"])
                else:
                    if cmp("B", "C", "<"):
                        apply_swap(["A","D"])
                        apply_swap(["A","B"])
                        apply_swap(["C","B"])
                    else:
                        apply_swap(["A","D"])
                        apply_swap(["A","B"])
            else:
                if cmp("A", "D", "<"):
                    if cmp("B", "D", "<"):
                        apply_swap(["A","C"])
                        apply_swap(["A","B"])
                    else:
                        apply_swap(["A","C"])
                        apply_swap(["A","B"])
                        apply_swap(["B","D"])
                else:
                    if cmp("C", "D", "<"):
                        apply_swap(["A","C"])
                        apply_swap(["B",'D'])
                    else:
                        apply_swap(["A","D"])
                        apply_swap(["B","C"])
                        apply_swap(["B","A"])
        else:
            if cmp("A", "C", "<"):
                if cmp("A", "D", "<"):
                    if cmp("C", "D", "<"):
                        apply_swap(["A","B"])
                    else:
                        apply_swap(["A","B"])
                        apply_swap(["C","D"])
                else:
                    if cmp("B", "D", "<"):
                        apply_swap(["A","B"])
                        apply_swap(["A","D"])
                        apply_swap(["A","C"])
                    else:
                        apply_swap(["A","D"])
                        apply_swap(["A","C"])
            else:
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        apply_swap(["A","B"])
                        apply_swap(["A","C"])
                    else:
                        apply_swap(["A","C"])
                else:
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                                apply_swap(["A","D"])
                            else:
                                apply_swap(["A","B"])
                                apply_swap(["A","D"])
                        else:
                            apply_swap(["A","D"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["A","D"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["A","C"])
                                apply_swap(["B",'D'])
                                apply_swap(["A","B"])
                            else:
                                apply_swap(["A","D"])
                                apply_swap(["B","C"])

    else:
        if cmp("A", "B", "<"):
            if cmp("B", "C", "<"):
                if cmp("C", "D", "<"):
                    apply_swap(["A","D"])
                    apply_swap(["B","C"])
                    
                else:
                    if cmp("A", "D", "<"):
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["B","D"])
                            apply_swap(["A","B"])
                        else:
                            apply_swap(["A","C"])
                            apply_swap(["A","D"])
                    else:
                        apply_swap(["A","C"])
            else:
                if cmp("B", "D", "<"):
                    if cmp("A", "C", "<"):
                        apply_swap(["A","D"])
                    else:
                        apply_swap(["A","D"])
                        apply_swap(["A","C"])
                else:
                    if cmp("A", "C", "<"):
                        if cmp("C", "D", "<"):
                            apply_swap(["A","B"])
                            apply_swap(["A","D"])
                        else:
                            if cmp("A", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                                apply_swap(["A","D"])
                            else:
                                apply_swap(["A","B"])
                                apply_swap(["A","C"])
                    else:
                        if cmp("A", "D", "<"):
                            apply_swap(["A","B"])
                            apply_swap(["A","D"])
                            apply_swap(["A","C"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["A","B"])
                                apply_swap(["C","D"])
                            else:
                                apply_swap(["A","B"])
        else:
            if cmp("A", "C", "<"):
                if cmp("C", "D", "<"):
                    apply_swap(["A","D"])
                    apply_swap(["B","C"])
                    apply_swap(["A","B"])
                else:
                    if cmp("A", "D", "<"):
                        apply_swap(["A","C"])
                        apply_swap(["B","D"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["A","C"])
                            apply_swap(["A","B"])
                            apply_swap(["B","D"])
                        else:
                            apply_swap(["A","C"])
                            apply_swap(["A","B"])
            else:
                if cmp("A", "D", "<"):
                    if cmp("B", "C", "<"):
                        apply_swap(["A","D"])
                        apply_swap(["A","B"])
                    else:
                        apply_swap(["A","D"])
                        apply_swap(["A","B"])
                        apply_swap(["B","C"])
                else:
                    if cmp("B", "C", "<"):
                        if cmp("B", "D", "<"):
                            if cmp("C", "D", "<"):
                                apply_swap(["B","D"])
                            else:
                                apply_swap(["B","C"])
                                apply_swap(["B","D"])
                        else:
                            apply_swap(["B","C"])
                    else:
                        if cmp("B", "D", "<"):
                            apply_swap(["B","D"])
                            apply_swap(["B","C"])
                        else:
                            if cmp("C", "D", "<"):
                                apply_swap(["C","D"])
                            else:
                                None


def network_sort(symbols, reverse, mapping):
    set_mapping(mapping)
    set_log({
            "compare_count": 0,
            "swap_count": 0,
            "compare_swap_usage": [],
            "current_seq":copy.deepcopy(symbols)
        })
    sort4(reverse)


def earl_sort4(data):
    ground_truth= data.get("ground_truth", {})
    items = data.get("items", [])
    symbols = data.get("symbols", [])
    reverse = data.get("reverse", )
    mapping = dict(zip(symbols,items))
    network_sort(symbols, reverse=reverse, mapping=mapping)
    result=GLOBAL_LOG["current_seq"]
    result_config={
        "compare_count":GLOBAL_LOG["compare_count"], 
        "swap_count":GLOBAL_LOG["swap_count"],
        "compare_swap_usage":GLOBAL_LOG["compare_swap_usage"],
        "result":",".join(result),
        "ground_truth":ground_truth
    }
    return result_config