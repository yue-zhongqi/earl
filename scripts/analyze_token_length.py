import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import time


model_name = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


all_files = {
    "earl": "val_results/0_autorunner/calc_bench/earl-cpo/275/calc_bench/gsm8kr/0.jsonl",
    "prompt-cpo": "val_results/0_autorunner/calc_bench/prompt-cpo/100/calc_bench/gsm8kr/0.jsonl",
    "sft-grpo": "scripts/0.jsonl",
    "prompt-grpo": "val_results/0_autorunner/calc_bench/prompt-grpo/250/calc_bench/gsm8kr/0.jsonl",

}


def print_results(model_name, jsonl_file):
    # read jsonl as an array of dictionary
    data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # group the len(entry['output']) into <100, <200, <300, <400, <500, <600, >600
    length_groups = {
        "<200": [0,0],
        "<250": [0,0],
        "<300": [0,0],
        "<350": [0,0],
        "<400": [0,0],
        ">=400": [0,0],
    }
    total_correct = 0
    total_wrong = 0
    for entry in data:
        length = len(tok(entry["output"])['input_ids'])
        cat = 0 if entry['reward'] == 0.0 else 1
        total_correct += cat
        total_wrong += 1 - cat
        if length < 200:
            length_groups["<200"][cat] += 1
        elif length < 250:
            length_groups["<250"][cat] += 1
        elif length < 300:
            length_groups["<300"][cat] += 1
        elif length < 350:
            length_groups["<350"][cat] += 1
        elif length < 400:
            length_groups["<400"][cat] += 1
        else:
            length_groups[">=400"][cat] += 1
    correct_str = ','.join([str(item[1] / total_correct) for item in length_groups.values()])
    wrong_str = ','.join([str(item[0] / total_wrong) for item in length_groups.values()])
    print(f"{model_name} correct: {correct_str}")
    print(f"{model_name} wrong: {wrong_str}")

for model_name, jsonl_file in all_files.items():
    print_results(model_name, jsonl_file)